"""Agent: holds settings and performs the actions that need a model and tools.

The contract is in ARCHITECTURE.md "Agent contract". Agent does not print (it notifies the Reporter), does not
read human input (it asks the Human), does not hold history (State's job), and does not decide the order of
steps (the Loop's job).
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import warnings
from collections.abc import Awaitable, Callable, Generator, Iterable, Mapping
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

from . import _structured
from ._async import ASYNC_RUN, check_sync_call, is_async_callable, run_in_thread
from ._runner import arun_calls, run_calls
from ._tokens import estimate_overhead_tokens
from .errors import NoHumanError, OutputError, ResumeWarning, fix_message
from .human import check_returns as check_human_returns
from .mcp_tools import MCP, MCPTool, MCPToolRef, check_tool_names, wait_sync
from .mcp_tools import _Loop as _MCPLoop
from .models.resolve import resolve_model
from .reporter import Reporter
from .state import State
from .store import Store
from .tool import FunctionTool, Tool, ToolMap, collect_tools
from .types import Message, ModelEvent, Request, TextBlock

if TYPE_CHECKING:
    from .human import Human
    from .models.base import Model
    from .types import Reply, ToolCall, ToolSpec

__all__ = ["Agent", "DEFAULT"]


class _Default:
    """Marker for the ``reporter``/``human`` default. The default is a single ``Terminal()`` shared by all Agents."""

    _instance: _Default | None = None

    def __new__(cls) -> _Default:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "DEFAULT"


DEFAULT: Any = _Default()

#: Setting names ``copy`` accepts (same as the constructor arguments).
SETTINGS = ("model", "system", "tools", "skills", "loop", "reporter", "human", "store", "name", "description")

#: Notification methods a Reporter must have: every ``on_*`` the Reporter base class defines.
_REPORTER_METHODS = tuple(name for name in vars(Reporter) if name.startswith("on_"))


def _as_tuple(value: Any, setting: str, example: str) -> tuple[Any, ...]:
    """Converts ``tools=``/``skills=`` to a tuple. If it is not a list, ``TypeError`` with a fix."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(
            fix_message(
                f"{setting}= takes a list (got: {value!r})",
                "Wrap it in square brackets, even if there is only one",
                example,
            )
        )
    return tuple(value)


def _check_state(state: Any, method: str) -> None:
    if not isinstance(state, State):
        raise TypeError(
            fix_message(
                f"The first argument of agent.{method}() must be a State (got: {state!r})",
                "Pass along the state your loop or block received",
                f"def my_loop(agent: Agent, state: State):\n    agent.{method}(state)",
            )
        )


def _same_tool(a: Tool, b: Tool) -> bool:
    """Whether two tools are the same. ``obj.method`` gives a new FunctionTool on every access, so compare the
    original function and the bound object."""
    if a is b:
        return True
    return isinstance(a, FunctionTool) and isinstance(b, FunctionTool) and a.fn is b.fn and a.bound_to is b.bound_to


class Agent:
    """An agent's settings, and the actions that need a model, tools or a human.

    Settings do not change after creation. Use ``copy`` to get an Agent with different settings.
    The history of a run lives in a ``State``, and the order of steps is decided by the loop.

    Example:
        ```python
        agent = Agent(model="claude-sonnet-5", system="You are a coding assistant", tools=[read_file])
        answer = agent.run("Find the bug in main.py")
        ```
    """

    def __init__(
        self,
        model: str | Model,
        *,
        system: str | None = None,
        tools: Iterable[Any] = (),
        skills: Iterable[str] = (),
        loop: Any = None,
        reporter: Reporter | None = DEFAULT,
        human: Human | None = DEFAULT,
        store: Store | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Creates an Agent. Mistakes in the settings are reported here, with how to fix them.

        Creating an Agent uses no network and needs no API key.

        Args:
            model: A model name such as ``"claude-sonnet-5"``, ``"anthropic/claude-sonnet-5"`` or
                ``"ollama/llama3:8b"``, or a ``Model`` object such as ``Anthropic(...)`` when you need settings.
            system: The system prompt.
            tools: ``@tool`` functions, objects with ``@tool`` methods, single ``@tool`` methods, objects of ``Tool``
                subclasses, ``MCP`` servers and single MCP tools (``github.create_issue``).
            skills: Not implemented yet. A non-empty value raises ``NotImplementedError``.
            loop: The loop ``run`` calls with ``(agent, state)``. Defaults to ``default_loop``.
            reporter: Receives progress notifications. Defaults to the shared ``Terminal``. ``None`` is silent.
            human: Answers ``ask_human``. Defaults to the shared ``Terminal``. ``None`` means no human.
            store: Saves the States this Agent runs as they go, so they can be continued with
                ``store.load(id)``. ``None`` (the default) saves nothing.
            name: The Agent's name.
            description: What the Agent does.

        Raises:
            TypeError: A setting has the wrong type, for example a function without ``@tool`` in ``tools=``,
                a ``loop`` that is not callable, a Reporter, Human or Store class instead of an object.
            ValueError: Two tools share a name, or the model string is ambiguous.
        """
        # - model: resolve_model(model), no network.
        # - tools: Agent items are checked first: missing name or description is a ValueError; with both,
        #   NotImplementedError (subagents are outside the MVP). MCP items are grouped by server (_add_mcp).
        #   The rest are flattened with collect_tools (duplicate names and missing @tool error there).
        # - skills: NotImplementedError if not empty (outside the MVP).
        # - loop: default_loop if None. TypeError if not callable.
        # - reporter/human: DEFAULT means terminal.default_terminal() (one per process, the same object for
        #   both). reporter needs the Reporter on_* methods, human needs ask or aask.
        # - The original arguments are kept as given in _kwargs, for copy (including the DEFAULT marker).
        tools = _as_tuple(tools, "tools", "Agent(model=..., tools=[web_search])")
        skills = (skills,) if isinstance(skills, str) else _as_tuple(
            skills, "skills", 'Agent(model=..., skills=["./skills"])'
        )
        self._kwargs: dict[str, Any] = {
            "model": model,
            "system": system,
            "tools": tools,
            "skills": skills,
            "loop": loop,
            "reporter": reporter,
            "human": human,
            "store": store,
            "name": name,
            "description": description,
        }

        for setting, value in (("system", system), ("name", name), ("description", description)):
            if value is not None and not isinstance(value, str):
                raise TypeError(
                    fix_message(
                        f"{setting}= takes a string (got: {value!r})",
                        f"Pass a string like {setting}=\"...\", or leave it out",
                    )
                )

        self._model: Model = resolve_model(model)
        self._system = system
        self._name = name
        self._description = description

        plain: list[Any] = []
        mcp_uses: dict[str, _MCPUse] = {}
        for item in tools:
            if isinstance(item, Agent):
                missing = [
                    attr for attr in ("name", "description") if not getattr(item, attr)
                ]
                if missing:
                    raise ValueError(
                        fix_message(
                            f"{item!r} passed in tools= has no {', '.join(missing)}. "
                            "A subagent's name becomes the tool name and its description the tool description",
                            "Give an Agent used as a subagent both name and description",
                            "researcher = Agent(\n"
                            '    name="researcher",\n'
                            '    description="Finds material on the web and summarizes it",\n'
                            '    model="claude-haiku-4-5",\n'
                            "    tools=[web_search],\n"
                            ")",
                        )
                    )
                raise NotImplementedError(
                    f"Subagents are not supported yet (Agent {item.name!r} passed in tools=)"
                )
            if isinstance(item, (MCP, MCPToolRef)):
                _add_mcp(mcp_uses, item)
                continue
            if isinstance(item, MCPTool):
                raise TypeError(
                    fix_message(
                        f"{item!r} passed in tools= is a connected tool of another Agent's MCP server",
                        "Pass the server, or pick the tool from it",
                        f"Agent(model=..., tools=[{item.server.name}])  # or [{item.server.name}"
                        f"[{item.remote_name!r}]]",
                    )
                )
            plain.append(item)
        self._tools = tools
        self._tool_by_name: dict[str, Tool] = collect_tools(plain)

        #: MCP servers in tools=, by name. Their tools are known only once connected (``_mcp_tools``).
        self._mcp_uses: tuple[_MCPUse, ...] = tuple(mcp_uses.values())
        self._mcp_lock = threading.Lock()
        self._mcp_users = 0
        #: Servers this Agent holds (``MCP._acquire``) while it has users. Given back by the last ``_mcp_exit``.
        self._mcp_held: list[MCP] = []
        self._mcp_ready: Future[dict[str, MCPTool]] | None = None
        self._mcp_tools: dict[str, MCPTool] = {}

        if skills:
            raise NotImplementedError("skills= is not supported yet")
        self._skills: tuple[str, ...] = skills

        if loop is None:
            from .loop import default_loop

            loop = default_loop
        if not callable(loop):
            raise TypeError(
                fix_message(
                    f"loop= takes a function that receives (agent, state) (got: {loop!r})",
                    "Pass a function decorated with @loop(until=..., limit=...) or a callable taking (agent, state)",
                    "@loop(until=State.is_answered, limit=50)\n"
                    "def coding(agent: Agent, state: State):\n"
                    "    agent.think(state)\n"
                    "    if state.wants_tools():\n"
                    "        agent.use_tools(state)\n"
                    "\n"
                    "agent = Agent(model=..., loop=coding)",
                )
            )
        self._loop = loop

        if reporter is not None and reporter is not DEFAULT:
            missing = [m for m in _REPORTER_METHODS if not callable(getattr(reporter, m, None))]
            if isinstance(reporter, type) or missing:
                raise TypeError(
                    fix_message(
                        f"reporter={reporter!r} is not a Reporter object"
                        + (f" (missing methods: {', '.join(missing)})" if missing else ""),
                        "Pass an object of a class that subclasses alpineagents.Reporter (an object, not the class)",
                        "class MyReporter(Reporter):\n"
                        "    def on_tool_start(self, state, call): ...\n"
                        "\n"
                        "agent = Agent(model=..., reporter=MyReporter())",
                    )
                )
        self._reporter = reporter

        if human is not None and human is not DEFAULT:
            if isinstance(human, type) or not (
                callable(getattr(human, "ask", None)) or callable(getattr(human, "aask", None))
            ):
                raise TypeError(
                    fix_message(
                        f"human={human!r} is not a Human object",
                        "Pass an object of an alpineagents.Human subclass that implements ask(state, prompt, returns) "
                        "or async aask(...)",
                        'agent = Agent(model=..., human=FakeHuman(["yes"]))',
                    )
                )
        self._human = human

        if store is not None and not isinstance(store, Store):
            raise TypeError(
                fix_message(
                    f"store={store!r} is not a Store object",
                    "Pass an object of an alpineagents.Store subclass (an object, not the class), such as FileStore",
                    'agent = Agent(model=..., store=FileStore(".agent-runs"))',
                )
            )
        self._store = store
        # Agent._summary() for snapshots, computed at the first save (settings do not change).
        self._saved_summary: dict[str, Any] | None = None

    # ------------------------------------------------------------ settings (read-only)

    @property
    def model(self) -> Model:
        """The Model object. A model string given to ``Agent(model=...)`` is already turned into one."""
        return self._model

    @property
    def system(self) -> str | None:
        """The system prompt, or ``None``."""
        return self._system

    @property
    def tools(self) -> tuple[Any, ...]:
        """The items passed in ``tools=``, as given. ``tool_map`` has the tools they stand for, by name."""
        return self._tools

    @property
    def tool_map(self) -> Mapping[str, Tool | MCPTool]:
        """Every tool the model can call, by the name it calls it: ``@tool`` functions, the ``@tool`` methods of
        objects in ``tools=``, and the tools of MCP servers (``MCPTool``, named ``{server}__{tool}``).

        Use it to look up the tool of a call, and read what the tool does from its hints (``read_only``,
        ``destructive``, ``idempotent``, ``open_world``). Read-only: to change the tools, use ``copy(tools=...)``.
        MCP servers' tools are listed only while the servers are connected: during a run, or inside ``with agent:``.
        Looking up a missing name before that raises a ``KeyError`` that says so.

        Example:
            ```python
            for call in state.pending_calls:
                tool = agent.tool_map.get(call.name)  # None for a name the model made up
                if tool is not None and not tool.read_only and not confirm(call):
                    state.deny(call, "The user declined")
            ```
        """
        ready = self._mcp_ready
        connected = ready is not None and ready.done() and not ready.cancelled() and ready.exception() is None
        waiting = () if connected else tuple(use.server.name for use in self._mcp_uses)
        return ToolMap(self._tool_map(), waiting)

    @property
    def skills(self) -> tuple[str, ...]:
        """Reserved for skills, which are not implemented yet. Always empty."""
        return self._skills

    @property
    def loop(self) -> Any:
        """The loop ``run`` calls. ``default_loop`` unless ``loop=`` was given."""
        return self._loop

    @property
    def reporter(self) -> Reporter | None:
        """The Reporter that receives progress notifications, or ``None``. Defaults to the shared ``Terminal``."""
        # DEFAULT resolves to terminal.default_terminal() on each access.
        if self._reporter is DEFAULT:
            from .terminal import default_terminal

            return default_terminal()
        return self._reporter

    @property
    def human(self) -> Human | None:
        """The Human that answers ``ask_human``, or ``None``. Defaults to the shared ``Terminal``."""
        if self._human is DEFAULT:
            from .terminal import default_terminal

            return default_terminal()
        return self._human

    @property
    def store(self) -> Store | None:
        """The Store that saves this Agent's States as they run, or ``None``."""
        return self._store

    @property
    def name(self) -> str | None:
        """The Agent's name, or ``None``."""
        return self._name

    @property
    def description(self) -> str | None:
        """What the Agent does, or ``None``."""
        return self._description

    # ------------------------------------------------------------ actions

    def run(self, task: str | State) -> Any:
        """Runs the loop on a task and returns what the loop returns, usually the answer.

        MCP servers in ``tools=`` connect for the run and disconnect after it, unless ``with agent:`` keeps
        them open. If an exception leaves the run, it is recorded in ``state.history``, tool calls still
        waiting for a result are closed (``(interrupted by user)`` after Ctrl+C), and the exception is
        raised as is. After Ctrl+C you can call ``run`` again with the same State.

        With ``store=``, the State is saved as the run goes: at the start and end of the run, before and after
        each ``think``, after each tool result, and after ``use_tools`` and ``compact``. If saving fails while
        the run is already failing, the original exception is raised with a note about the failed save.

        Args:
            task: A task string, which starts a new ``State``, or a ``State`` to continue.

        Returns:
            What the loop returns. The default loop returns ``state.answer``.

        Raises:
            TypeError: The loop is async (use ``arun``), ``task`` is neither a string nor a State, a value in the
                State cannot be saved as JSON, or the store only works asynchronously.
            ValueError: The State was already ended with ``state.finish()``, or it is new and the store already
                has a State with its id.

        Warns:
            ResumeWarning: The State was saved by an Agent with a different name, model, system prompt, tools
                or MCP servers.

        Example:
            ```python
            state = State("Find the bug in this repo")
            answer = agent.run(state)
            print(state.stopped_by, state.usage.cost)
            ```
        """
        # Order (ARCHITECTURE.md "run order and the context rules after exceptions"):
        # 1. _start_run: str -> State(task); State as is; anything else TypeError. A finish()ed State is a
        #    ValueError. Clears state.stopped_by so a run ending in an exception keeps no stale reason. A loaded
        #    State saved by a different Agent gives a ResumeWarning (once, outside the State lock).
        # 2. try: on_run_start; connect MCP (_mcp_open); state._note_window(...) records the window size with the
        #    full tool list so compact_if_full works on the first turn (the owner is not set); return loop(...)
        #    on_run_start is inside the try, so on_run_end is called whichever step raises.
        # 3. except BaseException: state._record_error(e); state._close_pending(e); raise (KeyboardInterrupt too)
        # 4. finally: close the MCP connections this run opened, then on_run_end(state, error) (None on success).
        # With a store: save after step 1 (before on_run_start: a failed first save means the run never started),
        # after the loop returns, and in step 3 after closing the pending calls (_save_after).
        # An async loop (or a store that only works asynchronously) raises TypeError before step 1.
        check_sync_call("run")
        if is_async_callable(self._loop):
            raise TypeError(
                fix_message(
                    f"This Agent's loop {_loop_name(self._loop)} is async, so it cannot run with run()",
                    "Await it with arun() inside async code, or write the loop as a regular def",
                    'answer = await agent.arun("...")',
                )
            )
        if self._store is not None and type(self._store).write is Store.write:
            raise self._store._async_only("write", "awrite")
        state = self._start_run(task)
        self._save(state)
        reporter = self.reporter
        error: BaseException | None = None
        connected = False
        try:
            if reporter is not None:
                reporter.on_run_start(state)
            self._mcp_open()
            connected = True
            self._note_window(state)
            result = self._loop(self, state)
            self._save(state)
            return result
        except BaseException as e:
            error = e
            state._record_error(e)
            state._close_pending(e)
            self._save_after(state, e)
            raise
        finally:
            try:
                if connected:
                    self._mcp_close()
            finally:
                if reporter is not None:
                    reporter.on_run_end(state, error)

    async def arun(self, task: str | State) -> Any:
        """The async version of ``run``.

        Uses this Agent's loop if it is async, or ``adefault_loop`` if the Agent uses the default loop.
        Cancelling the task follows the same rules as Ctrl+C in ``run``.

        Args:
            task: A task string, which starts a new ``State``, or a ``State`` to continue.

        Returns:
            What the loop returns. The default loop returns ``state.answer``.

        Raises:
            TypeError: The loop is a sync loop other than ``default_loop``, which would block the event loop.
            ValueError: The State was already ended with ``state.finish()``.

        Warns:
            ResumeWarning: The State was saved by an Agent with a different name, model, system prompt, tools
                or MCP servers.
        """
        # Same steps as run. Sets ASYNC_RUN so sync methods called on this event loop thread raise TypeError.
        loop = self._async_loop()
        state = self._start_run(task)
        await self._asave(state)
        reporter = self.reporter
        error: BaseException | None = None
        token = ASYNC_RUN.set(asyncio.get_running_loop())
        connected = False
        try:
            if reporter is not None:
                reporter.on_run_start(state)
            await self._amcp_open()
            connected = True
            self._note_window(state)
            result = await loop(self, state)
            await self._asave(state)
            return result
        except BaseException as e:
            error = e
            state._record_error(e)
            state._close_pending(e)
            await self._asave_after(state, e)
            raise
        finally:
            ASYNC_RUN.reset(token)
            try:
                if connected:
                    await self._amcp_close()
            finally:
                if reporter is not None:
                    reporter.on_run_end(state, error)

    def think(self, state: State, tools: Iterable[Any] | None = None) -> None:
        """Sends the context to the model once and records its reply in the State.

        The reply may ask for tool calls. Check ``state.wants_tools()`` and run them with ``use_tools``.
        If anything fails, the context goes back to how it was before this call, and the error is raised.

        Args:
            state: The State to continue. The first Agent that thinks on a State owns it.
            tools: Which of this Agent's tools the model may see this turn. ``None`` shows all of them,
                ``[]`` shows none. Takes the items of ``tools=`` and the values of ``tool_map``, so
                ``[t for t in agent.tool_map.values() if t.read_only]`` works.

        Raises:
            ValueError: The State is finished, still has tool calls waiting for results, belongs to another
                Agent, or ``tools`` has a tool this Agent does not have.
            ProviderError: The model provider failed (``RateLimitError``, ``AuthError``,
                ``ContextTooLongError`` or another ``ProviderError``).
        """
        # 1. _begin_think: state._claim(self, context_window=, overhead_tokens=estimate_overhead_tokens(system,
        #    specs shown)); cp = state._begin_think() (finish and pending checks, late result notices, turn + 1).
        # 2. try: request = model.mark_cache(Request(system, state.context, specs)); on_think_start;
        #    reply = model.respond(request, on_text, on_event); state._record_reply(reply); on_think_end.
        # 3. except BaseException: state._rollback(cp, e); _save_after; raise.
        # With a store: save between _claim and state._begin_think (the snapshot _replay starts from), and after
        # step 2 outside its try, so a failed save never rolls back a reply that already arrived.
        # tools= items are flattened with collect_tools and looked up by name; each must be the same tool.
        check_sync_call("think")
        self._mcp_ensure()
        specs = self._prepare_think(state, tools, "think")
        self._save(state)
        checkpoint = state._begin_think()
        reporter = self.reporter
        try:
            request = self._think_request(state, specs, reporter)
            reply = self._model.respond(request, self._on_text(state, reporter), self._on_event(state, reporter))
            self._end_think(state, reply, reporter)
        except BaseException as e:
            state._rollback(checkpoint, e)
            self._save_after(state, e)
            raise
        self._save(state)

    async def athink(self, state: State, tools: Iterable[Any] | None = None) -> None:
        """The async version of ``think``. Cancelling it rolls the context back like any other failure."""
        # Same steps as think, with model.arespond.
        await self._amcp_ensure()
        specs = self._prepare_think(state, tools, "athink")
        await self._asave(state)
        checkpoint = state._begin_think()
        reporter = self.reporter
        try:
            request = self._think_request(state, specs, reporter)
            reply = await self._model.arespond(
                request, self._on_text(state, reporter), self._on_event(state, reporter)
            )
            self._end_think(state, reply, reporter)
        except BaseException as e:
            state._rollback(checkpoint, e)
            await self._asave_after(state, e)
            raise
        await self._asave(state)

    def use_tools(self, state: State) -> None:
        """Runs the tool calls from the last reply and records their results in the State.

        Calls to tools made with ``parallel=True`` (the default) run at the same time, then the rest run one
        by one. Does nothing if there are no calls. A tool that raises does not become an error result:
        results of the calls that finished are recorded, and the exception is raised as is. Invalid arguments
        and unknown tool names go back to the model as ``(input error: ...)`` results.

        Args:
            state: The State whose ``pending_calls`` to run.

        Raises:
            ValueError: The State is finished or belongs to another Agent.
        """
        # state._ensure_open("use_tools") and state._claim(...) in _begin_use_tools, then _runner.run_calls.
        # Concurrency, exception and interrupt rules are in _runner.run_calls.
        # With a store: save after each batch of recorded results. Those saves do not stop the tools: an
        # Exception is dropped, and the save at the end (which writes everything not saved yet) raises instead.
        check_sync_call("use_tools")
        self._mcp_ensure()
        calls = self._begin_use_tools(state, "use_tools")
        try:
            if calls:
                run_calls(state, calls, self._tool_map(), self.reporter, self._tool_saver(state))
        except BaseException as e:
            self._save_after(state, e)
            raise
        self._save(state)

    async def ause_tools(self, state: State) -> None:
        """The async version of ``use_tools``.

        ``async def`` tools run as tasks on the running event loop, other tools on worker threads.
        Cancelling it follows the same rules as Ctrl+C.
        """
        # _runner.arun_calls.
        await self._amcp_ensure()
        calls = self._begin_use_tools(state, "ause_tools")
        try:
            if calls:
                await arun_calls(state, calls, self._tool_map(), self.reporter, self._atool_saver(state))
        except BaseException as e:
            await self._asave_after(state, e)
            raise
        await self._asave(state)

    def ask(self, state: State, prompt: str, returns: Any = str, *, retries: int = 2) -> Any:
        """Asks the model a question about the current context and returns the answer in the ``returns`` format.

        The question and answer are recorded in ``state.history`` but do not enter the context, so the run
        continues as if the question was never asked. The model cannot call tools while answering. Any Agent
        can ask about a State, not only the one running it.

        Args:
            state: The State whose context the model reads.
            prompt: The question.
            returns: The answer format: ``str``, a dataclass or a Pydantic model.
            retries: How many more times to ask when the answer does not fit ``returns``.

        Returns:
            The answer, converted to ``returns``.

        Raises:
            TypeError: ``returns`` is not a supported format.
            ValueError: The State is finished.
            OutputError: The answer still did not fit ``returns`` after ``retries`` more tries.
            ProviderError: The model provider failed.

        Example:
            ```python
            @dataclass
            class Review:
                approved: bool
                reason: str

            review = agent.ask(state, "Is this change safe to merge?", returns=Review)
            ```
        """
        # - state._ensure_open("ask"). No owner check.
        # - _structured.check_returns(returns) (TypeError if unsupported).
        # - messages = state._context_for_question() + Message.user(_structured.question_text(prompt, returns)).
        # - Request(system, messages, tools=all of this Agent's specs, tool_choice="none").
        #   on_think_start -> respond(request, on_text, on_event) -> on_think_end (no _record_reply).
        # - state._add_usage(reply.usage) for every reply.
        # - If _structured.parse_reply(reply.text, returns) raises ValueError: append that reply (text only, so
        #   no tool_use without a result goes into the request) and Message.user(_structured.retry_text(err)),
        #   then ask again, up to retries more times.
        # - Success: state._record_ask(prompt, value). Failure: state._record_ask(prompt, None), then
        #   OutputError with the last ValueError as __cause__.
        # - Model exceptions propagate as is (the context was not changed, so there is nothing to roll back).
        # The steps live in _ask_steps, shared with aask.
        check_sync_call("ask")
        self._mcp_ensure()
        reporter = self.reporter
        on_text, on_event = self._on_text(state, reporter), self._on_event(state, reporter)
        steps = self._ask_steps(state, prompt, returns, retries, reporter, "ask")
        request = next(steps)
        while True:
            reply = self._model.respond(request, on_text, on_event)
            try:
                request = steps.send(reply)
            except StopIteration as done:
                return done.value

    async def aask(self, state: State, prompt: str, returns: Any = str, *, retries: int = 2) -> Any:
        """The async version of ``ask``."""
        # Same steps as ask (_ask_steps), with model.arespond.
        await self._amcp_ensure()
        reporter = self.reporter
        on_text, on_event = self._on_text(state, reporter), self._on_event(state, reporter)
        steps = self._ask_steps(state, prompt, returns, retries, reporter, "aask")
        request = next(steps)
        while True:
            reply = await self._model.arespond(request, on_text, on_event)
            try:
                request = steps.send(reply)
            except StopIteration as done:
                return done.value

    def ask_human(self, state: State, prompt: str, returns: Any = str) -> Any:
        """Asks the person through ``human`` and returns the answer in the ``returns`` format.

        The question and answer are recorded in ``state.history``. The context does not change.

        Args:
            state: The State of the current run.
            prompt: The question.
            returns: The answer format: ``str``, ``bool`` or ``Literal[...]`` for a fixed set of choices.

        Returns:
            The answer, converted to ``returns``.

        Raises:
            NoHumanError: The Agent was created with ``human=None``.
            TypeError: ``returns`` is not a supported format, or the Human only answers asynchronously
                (use ``aask_human``).

        Example:
            ```python
            choice = agent.ask_human(state, "Run the command?", returns=Literal["yes", "no", "always"])
            ```
        """
        # - NoHumanError if self.human is None; human.check_returns(returns) (in _human_to_ask).
        # - value = human.ask(state, prompt, returns); state._record_human(prompt, value); return value.
        # - No finish check. Never called while holding the State lock.
        check_sync_call("ask_human")
        human = self._human_to_ask(state, prompt, returns, "ask_human")
        ask = getattr(human, "ask", None)
        if not callable(ask):
            raise TypeError(
                fix_message(
                    f"human={human!r} answers only asynchronously (it has aask, not ask)",
                    "Ask it from an async loop with await agent.aask_human(...)",
                    'ok = await agent.aask_human(state, "Run it?", returns=bool)',
                )
            )
        value = ask(state, prompt, returns)
        state._record_human(prompt, value)
        return value

    async def aask_human(self, state: State, prompt: str, returns: Any = str) -> Any:
        """The async version of ``ask_human``.

        Awaits ``human.aask`` when the Human has it, and otherwise runs ``human.ask`` on a worker thread.
        """
        # A Human subclass's default aask runs ask on a worker thread too.
        human = self._human_to_ask(state, prompt, returns, "aask_human")
        aask = getattr(human, "aask", None)
        if callable(aask):
            value = await aask(state, prompt, returns)
        else:
            value = await run_in_thread(human.ask, state, prompt, returns)
        state._record_human(prompt, value)
        return value

    def compact(self, state: State, instructions: str | None = None) -> None:
        """Asks the model to summarize the context, then replaces the context with the task and that summary.

        ``state.history`` keeps everything. If anything fails, the context is left unchanged.
        In a loop, ``compact_if_full`` calls this only when the context is getting full.

        Args:
            state: The State to compact.
            instructions: What the summary should keep, for example ``"Keep file paths and failing tests"``.

        Raises:
            ValueError: The State still has tool calls waiting for results, or belongs to another Agent.
            OutputError: The model returned an empty summary.
            ProviderError: The model provider failed.
        """
        # 1. _begin_compact: state._claim(...); state._begin_compact()
        # 2. reply = model.compact(mark_cache(Request(system, state.context, all specs, tool_choice="none")),
        #    instructions, on_event)
        # 3. _apply_summary: state._add_usage(reply.usage); empty summary -> OutputError with the context
        #    unchanged; otherwise state._replace_context(reply.text, "compact") (records, on_context_change)
        # 4. state._end_compact(), also when a step above raised
        # 5. With a store: save (_save_after if a step above raised)
        check_sync_call("compact", " (in a loop: await acompact_if_full(agent, state))")
        self._mcp_ensure()
        self._begin_compact(state, instructions, "compact")
        try:
            try:
                request = self._compact_request(state)
                reply = self._model.compact(request, instructions, self._on_event(state, self.reporter))
                self._apply_summary(state, reply)
            finally:
                state._end_compact()
        except BaseException as e:
            self._save_after(state, e)
            raise
        self._save(state)

    async def acompact(self, state: State, instructions: str | None = None) -> None:
        """The async version of ``compact``."""
        # Same steps as compact, with model.acompact.
        await self._amcp_ensure()
        self._begin_compact(state, instructions, "acompact")
        try:
            try:
                request = self._compact_request(state)
                reply = await self._model.acompact(request, instructions, self._on_event(state, self.reporter))
                self._apply_summary(state, reply)
            finally:
                state._end_compact()
        except BaseException as e:
            await self._asave_after(state, e)
            raise
        await self._asave(state)

    def save(self, state: State) -> None:
        """Saves what the store does not have yet. For changes made outside the Agent's own steps.

        ``run``, ``think``, ``use_tools`` and ``compact`` save by themselves. Call this to save something else
        right away, such as the person's next message in a chat loop, before the next ``run``.

        Args:
            state: The State to save.

        Raises:
            ValueError: The Agent has no store, the State is saved in another store, or it is new and the store
                already has a State with its id.
            TypeError: A value in the State cannot be saved as JSON, or the store only works asynchronously.

        Example:
            ```python
            state.add_user_message(input("> "))
            agent.save(state)
            ```
        """
        check_sync_call("save")
        _check_state(state, "save")
        self._need_store("save")
        self._save(state)

    async def asave(self, state: State) -> None:
        """The async version of ``save``."""
        _check_state(state, "asave")
        self._need_store("asave")
        await self._asave(state)

    def copy(self, **changes: Any) -> Agent:
        """Returns a new Agent with some settings changed. The original Agent does not change.

        The new Agent goes through the same checks as ``Agent(...)``.

        Args:
            **changes: Settings to change, by the same names as the ``Agent(...)`` arguments.

        Returns:
            The new Agent.

        Raises:
            TypeError: A name in ``changes`` is not a setting.

        Example:
            ```python
            quiet = agent.copy(model=FakeModel(["Done"]), reporter=None)
            ```
        """
        # Agent(**{**self._kwargs, **changes}); keys must be in SETTINGS.
        unknown = [key for key in changes if key not in SETTINGS]
        if unknown:
            raise TypeError(
                fix_message(
                    f"copy() got unknown settings: {', '.join(unknown)}",
                    f"Settings you can change: {', '.join(SETTINGS)}",
                    'agent.copy(system="You are a careful reviewer")',
                )
            )
        return Agent(**{**self._kwargs, **changes})

    # ------------------------------------------------------------ outside the MVP (extension points)

    def run_tool(self, state: State, tool: Any, /, **args: Any) -> Any:
        """Not implemented yet. Raises ``NotImplementedError``."""
        raise NotImplementedError("agent.run_tool() is not implemented yet")

    def load_skill(self, state: State, name: str) -> None:
        """Not implemented yet. Raises ``NotImplementedError``."""
        raise NotImplementedError("agent.load_skill() is not implemented yet")

    def __enter__(self) -> Agent:
        """``with agent:`` keeps the MCP connections open across the runs inside it (without it, each ``run``
        connects and disconnects). Nothing to do without MCP servers."""
        self._mcp_open()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._mcp_close()

    async def __aenter__(self) -> Agent:
        """``async with agent:``, the async version of ``with agent:``."""
        await self._amcp_open()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self._amcp_close()

    # ------------------------------------------------------------ steps shared by the sync and async versions

    def _start_run(self, task: str | State) -> State:
        """Step 1 of ``run``: the State to run, checked and prepared. Warns with ``ResumeWarning`` if the State
        was saved by a different Agent."""
        if isinstance(task, str):
            state = State(task)
        elif isinstance(task, State):
            state = task
        else:
            raise TypeError(
                fix_message(
                    f"run() takes a task string or a State (got: {task!r})",
                    "agent.run(\"...\") or agent.run(State(\"...\"))",
                    'state = State("Find the bug in this repo")\nanswer = agent.run(state)',
                )
            )
        if state.is_finished():
            raise ValueError(
                fix_message(
                    "Cannot run() again with a State that was finish()ed",
                    "Create a new State and pass that",
                    'agent.run(State("Next task"))',
                )
            )

        state._set_stopped_by(None)

        saved = state._take_saved_agent()
        if saved is not None:
            changes = _resume_changes(saved, self._summary())
            if changes:
                try:
                    # stacklevel 3: the user's run()/arun() call (warn <- _start_run <- run <- caller).
                    warnings.warn(ResumeWarning(_describe_changes(changes), changes), stacklevel=3)
                except BaseException:
                    # A warnings filter turned it into an exception: the run did not start, so the next run
                    # compares again.
                    state._set_saved_agent(saved)
                    raise
        return state

    def _note_window(self, state: State) -> None:
        """Records the context window size so context_used means something even before the first turn (the owner
        is not set). After connecting, so MCP tools count toward the overhead."""
        state._note_window(
            self,
            context_window=self._model.context_window,
            overhead_tokens=estimate_overhead_tokens(self._system, self._specs(None)),
        )

    # ------------------------------------------------------------ saving to the store

    def _save(self, state: State) -> None:
        """A save point: writes what the store does not have yet. Nothing without a store, or when the store has
        everything. The State is marked saved only after the write returns."""
        store = self._store
        if store is None:
            return
        batch = state._unsaved(store, self._store_summary())
        if batch is None:
            return
        store.write(state.id, batch.entries, batch.snapshot, create=batch.create)
        state._mark_saved(store, batch)

    async def _asave(self, state: State) -> None:
        """The async version of ``_save``."""
        store = self._store
        if store is None:
            return
        batch = state._unsaved(store, self._store_summary())
        if batch is None:
            return
        await store.awrite(state.id, batch.entries, batch.snapshot, create=batch.create)
        state._mark_saved(store, batch)

    def _save_after(self, state: State, error: BaseException) -> None:
        """A save point while ``error`` is being raised (called in the ``except`` block that re-raises it).

        A save that fails with an ``Exception`` becomes a note on ``error``, so callers still catch the original.
        Any other ``BaseException`` (Ctrl+C, cancellation, ``SystemExit``) propagates, with ``error`` as its
        ``__context__``.
        """
        try:
            self._save(state)
        except Exception as failure:
            _note_failed_save(error, failure)

    async def _asave_after(self, state: State, error: BaseException) -> None:
        """The async version of ``_save_after``."""
        try:
            await self._asave(state)
        except Exception as failure:
            _note_failed_save(error, failure)

    def _tool_saver(self, state: State) -> Callable[[], None] | None:
        """The per-result save for ``run_calls``: an ``Exception`` is dropped (the tools keep running, and the save
        at the end of ``use_tools`` writes what is missing or raises)."""
        if self._store is None:
            return None

        def save() -> None:
            try:
                self._save(state)
            except Exception:
                pass

        return save

    def _atool_saver(self, state: State) -> Callable[[], Awaitable[None]] | None:
        """The async version of ``_tool_saver``."""
        if self._store is None:
            return None

        async def save() -> None:
            try:
                await self._asave(state)
            except Exception:
                pass

        return save

    def _store_summary(self) -> dict[str, Any]:
        """``_summary()`` for snapshots, computed once."""
        if self._saved_summary is None:
            self._saved_summary = self._summary()
        return self._saved_summary

    def _need_store(self, method: str) -> None:
        if self._store is None:
            raise ValueError(
                fix_message(
                    f"{method}() needs a store, and this Agent has none",
                    "create the Agent with store=",
                    'agent = Agent(model=..., store=FileStore(".agent-runs"))',
                )
            )

    def _async_loop(self) -> Any:
        """The loop ``arun`` awaits: this Agent's loop if async, ``adefault_loop`` for the default loop."""
        from .loop import adefault_loop, default_loop

        if self._loop is default_loop:
            return adefault_loop
        if is_async_callable(self._loop):
            return self._loop
        raise TypeError(
            fix_message(
                f"arun() needs an async loop, but this Agent's loop {_loop_name(self._loop)} is a regular def",
                "Write the loop as async def and await the agent.a* methods and async blocks in it "
                "(the async default loop is adefault_loop; adefault_loop.copy(limit=...) changes its limit), "
                "or call run() instead",
                "@loop(until=State.is_answered, limit=50)\n"
                "async def coding(agent: Agent, state: State):\n"
                "    await acompact_if_full(agent, state)\n"
                "    await agent.athink(state)\n"
                "    if state.wants_tools():\n"
                "        await agent.ause_tools(state)",
            )
        )

    def _prepare_think(self, state: State, tools: Iterable[Any] | None, method: str) -> tuple[ToolSpec, ...]:
        """Step 1 of ``think`` up to ``state._begin_think``: the specs to show."""
        _check_state(state, method)
        specs = self._specs(tools)
        state._claim(
            self,
            context_window=self._model.context_window,
            overhead_tokens=estimate_overhead_tokens(self._system, specs),
        )
        return specs

    def _think_request(self, state: State, specs: tuple[ToolSpec, ...], reporter: Reporter | None) -> Request:
        request = self._model.mark_cache(Request(self._system, state.context, specs))
        if reporter is not None:
            reporter.on_think_start(state)
        return request

    @staticmethod
    def _end_think(state: State, reply: Reply, reporter: Reporter | None) -> None:
        state._record_reply(reply)
        if reporter is not None:
            reporter.on_think_end(state, reply)

    def _begin_use_tools(self, state: State, method: str) -> tuple[ToolCall, ...]:
        """The checks of ``use_tools``; returns the calls to run."""
        _check_state(state, method)
        state._ensure_open(method)
        # Keep the overhead as computed for the tools the last think showed.
        state._claim(self, context_window=self._model.context_window)
        return state.pending_calls

    def _ask_steps(
        self, state: State, prompt: str, returns: Any, retries: int, reporter: Reporter | None, method: str
    ) -> Generator[Request, Reply, Any]:
        """``ask`` without the model call: yields each request, receives its reply, returns the value (or raises
        ``OutputError``). ``ask`` and ``aask`` send the replies."""
        _check_state(state, method)
        if not isinstance(prompt, str):
            raise TypeError(
                fix_message(
                    f"The prompt of {method}() takes a string (got: {prompt!r})",
                    "Pass the question as a string and set the format with returns=",
                    'plan = agent.ask(state, "Split the task into 3-7 steps", returns=Plan)',
                )
            )
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError(
                fix_message(
                    f"retries={retries!r} is not allowed",
                    "retries takes an integer of 0 or more (how many times to ask again)",
                    'agent.ask(state, "...", returns=Plan, retries=2)',
                )
            )
        state._ensure_open(method)
        _structured.check_returns(returns)

        specs = self._specs(None)
        messages = list(state._context_for_question())
        messages.append(Message.user(_structured.question_text(prompt, returns)))

        last_error: ValueError | None = None
        for _ in range(retries + 1):
            request = self._model.mark_cache(Request(self._system, tuple(messages), specs, tool_choice="none"))
            if reporter is not None:
                reporter.on_think_start(state)
            reply = yield request
            state._add_usage(reply.usage)
            if reporter is not None:
                reporter.on_think_end(state, reply)
            try:
                value = _structured.parse_reply(reply.text, returns)
            except ValueError as err:
                last_error = err
                messages.append(Message("assistant", (TextBlock(reply.text or "(empty reply)"),)))
                messages.append(Message.user(_structured.retry_text(err)))
                continue
            state._record_ask(prompt, value)
            return value

        state._record_ask(prompt, None)
        raise OutputError(
            fix_message(
                f"All {retries + 1} answers from {method}() failed to match returns={_type_name(returns)}: "
                f"{last_error}",
                "State the format more clearly in the question, raise retries=, or use a simpler returns format",
            )
        ) from last_error

    def _human_to_ask(self, state: State, prompt: str, returns: Any, method: str) -> Any:
        """The checks of ``ask_human``; returns the Human."""
        _check_state(state, method)
        human = self.human
        if human is None:
            raise NoHumanError(
                fix_message(
                    f"{method}() was called on an Agent with human=None (question: {prompt!r})",
                    "Give it a human to ask with Agent(human=...), or write the rule in code so no human is needed",
                    'agent = Agent(model=..., human=Terminal())\n'
                    "# in tests\n"
                    'agent.copy(human=FakeHuman(["yes"]))',
                )
            )
        check_human_returns(returns)
        return human

    def _begin_compact(self, state: State, instructions: str | None, method: str) -> None:
        """Step 1 of ``compact``. After this, the caller must call ``state._end_compact()`` in ``finally``."""
        _check_state(state, method)
        if instructions is not None and not isinstance(instructions, str):
            raise TypeError(
                fix_message(
                    f"The instructions of {method}() take a string (got: {instructions!r})",
                    "Write in a sentence what to keep",
                    'agent.compact(state, instructions="Keep architecture decisions and remaining bugs")',
                )
            )
        specs = self._specs(None)
        state._claim(
            self,
            context_window=self._model.context_window,
            overhead_tokens=estimate_overhead_tokens(self._system, specs),
        )
        state._begin_compact()

    def _compact_request(self, state: State) -> Request:
        return self._model.mark_cache(Request(self._system, state.context, self._specs(None), tool_choice="none"))

    @staticmethod
    def _apply_summary(state: State, reply: Reply) -> None:
        """Steps 3 and 4 of ``compact``: an empty summary leaves the context unchanged and raises ``OutputError``."""
        state._add_usage(reply.usage)
        summary = reply.text.strip()
        if not summary:
            raise OutputError(
                fix_message(
                    "compact() got an empty summary, so the context was not changed",
                    "Call it again, or get a summary with agent.ask(state, ...) and use state.start_from(summary)",
                )
            )
        state._replace_context(summary, "compact")

    # ------------------------------------------------------------ internal

    # ------------------------------------------------------------ MCP connections

    def _mcp_open(self) -> None:
        """One more user of this Agent's MCP servers (``run``, ``with``). The first connects them all; every user
        waits until they are connected and checked. On failure the use is given back and the error raised."""
        ready = self._mcp_enter()
        if ready is None:
            return
        try:
            wait_sync(ready)
        except BaseException:
            self._mcp_close()
            raise

    async def _amcp_open(self) -> None:
        ready = self._mcp_enter()
        if ready is None:
            return
        try:
            # Shielded: other users wait on the same connection; a cancel here only gives back this use.
            await asyncio.shield(asyncio.wrap_future(ready))
        except BaseException:
            await self._amcp_close()
            raise

    def _mcp_close(self) -> None:
        """One user less. After the last, disconnects (waiting up to ``_MCP_CLOSE_WAIT`` seconds)."""
        for closing in self._mcp_exit():
            try:
                closing.result(timeout=_MCP_CLOSE_WAIT)
            except Exception:  # a failed or slow shutdown does not hide what the run did (Ctrl+C still does)
                pass

    async def _amcp_close(self) -> None:
        closing = [asyncio.wrap_future(f) for f in self._mcp_exit()]
        if closing:
            await asyncio.wait(closing, timeout=_MCP_CLOSE_WAIT)
            for waiter in closing:
                if waiter.done() and not waiter.cancelled():
                    waiter.exception()  # seen

    def _mcp_ensure(self) -> None:
        """``think``/``use_tools``/``ask``/``compact`` outside ``run`` and ``with``: connect once and stay
        connected (the servers' tools must be known before the first think)."""
        if self._mcp_uses and self._mcp_users == 0:
            self._mcp_open()

    async def _amcp_ensure(self) -> None:
        if self._mcp_uses and self._mcp_users == 0:
            await self._amcp_open()

    def _mcp_enter(self) -> Future[dict[str, MCPTool]] | None:
        if not self._mcp_uses:
            return None
        with self._mcp_lock:
            self._mcp_users += 1
            if self._mcp_ready is None:
                self._mcp_held = [use.server for use in self._mcp_uses]
                lists = [server._acquire() for server in self._mcp_held]
                self._mcp_ready = _MCPLoop.submit(self._mcp_connect(lists))
            return self._mcp_ready

    def _mcp_exit(self) -> list[Future[None]]:
        """Gives back one use. After the last: stops a connect still going and gives back every server this Agent
        holds, whatever the connect's outcome. Returns the futures of the disconnects."""
        if not self._mcp_uses:
            return []
        with self._mcp_lock:
            self._mcp_users -= 1
            if self._mcp_users > 0 or self._mcp_ready is None:
                return []
            ready, self._mcp_ready = self._mcp_ready, None
            held, self._mcp_held = self._mcp_held, []
            self._mcp_tools = {}
        ready.cancel()  # no effect once done
        return [closing for server in held if (closing := server._release()) is not None]

    async def _mcp_connect(self, lists: list[Future[tuple[Any, ...]]]) -> dict[str, MCPTool]:
        """On the MCP loop: waits for every server's tool list, then checks them against this Agent's tools (the
        names in ``tools=[gh.x]`` exist, no name collides with another tool). The servers were acquired by
        ``_mcp_enter`` and are given back by ``_mcp_exit``."""
        # Shielded: the tool lists are shared with other Agents; cancelling this connect must not cancel them.
        remote_lists = await asyncio.gather(*(asyncio.shield(asyncio.wrap_future(f)) for f in lists))
        found: dict[str, MCPTool] = {}
        for use, remote_tools in zip(self._mcp_uses, remote_lists):
            for mcp_tool in check_tool_names(use.server, remote_tools, use.only):
                if mcp_tool.name in self._tool_by_name:
                    raise ValueError(
                        fix_message(
                            f"Duplicate tool name {mcp_tool.name!r}: tool {mcp_tool.remote_name!r} of "
                            f"{use.server!r} and {self._tool_by_name[mcp_tool.name]!r}",
                            'Rename the other tool with @tool(name="..."), or give the server another name=',
                        )
                    )
                found[mcp_tool.name] = mcp_tool
        self._mcp_tools = found
        return found

    @staticmethod
    def _on_text(state: State, reporter: Reporter | None) -> Callable[[str], None] | None:
        if reporter is None:
            return None
        return lambda chunk: reporter.on_text(state, chunk)

    @staticmethod
    def _on_event(state: State, reporter: Reporter | None) -> Callable[[ModelEvent], None]:
        """Records what the Model reported in history (``model_event``) and shows it to the Reporter. Records even
        without a Reporter."""

        def on_event(event: ModelEvent) -> None:
            if not isinstance(event, ModelEvent):
                raise TypeError(
                    fix_message(
                        f"on_event takes a ModelEvent (got: {event!r})",
                        "Create a ModelEvent(kind, message) and pass that",
                        'on_event(ModelEvent("fallback", "qwen rate limited → switched to laguna"))',
                    )
                )
            state._record_model_event(event)
            if reporter is not None:
                reporter.on_model_event(state, event)

        return on_event

    def _summary(self) -> dict[str, Any]:
        """What a saved State records about the Agent that saved it, compared when the State is resumed
        (``_start_run``). Plain JSON values only. Needs no MCP connection."""
        model = self._model
        return {
            "name": self._name,
            "model": f"{model.provider}/{model.name}" if model.provider else model.name,
            # A hash, not the text: enough to see that the prompt changed, without copying long prompts or
            # internal instructions into storage.
            "system_sha256": (
                hashlib.sha256(self._system.encode("utf-8")).hexdigest() if self._system is not None else None
            ),
            # Only this Agent's own tools (MCP tools are known only while connected); servers are listed by name.
            "tools": sorted(self._tool_by_name),
            "mcp_servers": sorted(use.server.name for use in self._mcp_uses),
        }

    def _tool_map(self) -> dict[str, Any]:
        """Name → Tool: the ``collect_tools`` result, then the connected MCP servers' tools."""
        return {**self._tool_by_name, **self._mcp_tools}

    def _specs(self, tools: Iterable[Any] | None = None) -> tuple[ToolSpec, ...]:
        """Specs of the tools to show, following the ``think(tools=...)`` rules."""
        if tools is None:
            return tuple(t.spec for t in self._tool_map().values())
        items = _as_tuple(tools, "think(tools", "agent.think(state, tools=[read_file])")
        if not items:
            return ()
        mcp_items = (MCP, MCPToolRef, MCPTool)
        mcp_specs = [t.spec for t in self._chosen_mcp_tools(i for i in items if isinstance(i, mcp_items))]
        items = tuple(i for i in items if not isinstance(i, mcp_items))
        for item in items:
            if isinstance(item, Agent):
                raise ValueError(
                    fix_message(
                        f"{item!r} passed in think(tools=...) is not one of this Agent's tools",
                        "think(tools=...) only takes tools that were passed in Agent(tools=...)",
                    )
                )
        chosen = collect_tools(items)
        specs = []
        for tool_name, chosen_tool in chosen.items():
            mine = self._tool_by_name.get(tool_name)
            if mine is None or not _same_tool(mine, chosen_tool):
                raise ValueError(
                    fix_message(
                        f"Tool {tool_name!r} passed in think(tools=...) is not one of this Agent's tools "
                        f"(this Agent's tools: {', '.join(self._tool_by_name) or '(none)'})",
                        "Add it to Agent(tools=...) first. think(tools=...) only shows a subset of those",
                        "agent = Agent(model=..., tools=[read_file, write_file])\n"
                        "agent.think(state, tools=[read_file])",
                    )
                )
            specs.append(mine.spec)
        return tuple(specs + mcp_specs)

    def _chosen_mcp_tools(self, items: Iterable[MCP | MCPToolRef | MCPTool]) -> list[MCPTool]:
        """``think(tools=[gh, gh.search_code, agent.tool_map["gh__x"]])``: the connected tools these stand for."""
        chosen: list[MCPTool] = []
        for item in items:
            if isinstance(item, MCPTool):
                matches = [item] if self._mcp_tools.get(item.name) is item else []
            elif isinstance(item, MCPToolRef):
                found = self._mcp_tools.get(item.name)
                matches = [found] if found is not None and found.server is item.server else []
            else:
                matches = [t for t in self._mcp_tools.values() if t.server is item]
            if not matches:
                raise ValueError(
                    fix_message(
                        f"{item!r} passed in think(tools=...) is not a connected MCP tool of this Agent "
                        f"(its MCP tools: {', '.join(self._mcp_tools) or '(none connected)'})",
                        "Add the server (or the tool) to Agent(tools=...) first",
                        'gh = MCP("...", name="github")\nagent = Agent(model=..., tools=[gh])\n'
                        "agent.think(state, tools=[gh.search_code])",
                    )
                )
            chosen.extend(t for t in matches if t not in chosen)
        return chosen

    def __repr__(self) -> str:
        parts = [f"model={self._model!r}"]
        if self._name is not None:
            parts.insert(0, f"name={self._name!r}")
        names = [*self._tool_by_name, *(repr(use.server) for use in self._mcp_uses)]
        if names:
            parts.append(f"tools=[{', '.join(names)}]")
        return f"Agent({', '.join(parts)})"


#: Seconds ``run`` waits for MCP servers to shut down.
_MCP_CLOSE_WAIT = 5.0


class _MCPUse:
    """One MCP server in ``tools=``: all its tools (``only is None``) or the ones picked with ``gh.x``."""

    def __init__(self, server: MCP) -> None:
        self.server = server
        self.only: frozenset[str] | None = frozenset()

    def add(self, item: MCP | MCPToolRef) -> None:
        if isinstance(item, MCP):
            self.only = None
        elif self.only is not None:
            self.only = self.only | {item.tool_name}


def _add_mcp(uses: dict[str, _MCPUse], item: MCP | MCPToolRef) -> None:
    """Groups ``tools=`` items by server. Two different servers with the same name are a ``ValueError``."""
    server = item if isinstance(item, MCP) else item.server
    use = uses.get(server.name)
    if use is None:
        use = uses[server.name] = _MCPUse(server)
    elif use.server is not server:
        raise ValueError(
            fix_message(
                f"Two MCP servers are named {server.name!r}: {use.server!r} and {server!r}",
                "Give each server its own name=",
                'MCP("...", name="github"), MCP("...", name="github_enterprise")',
            )
        )
    use.add(item)


#: Keys of ``Agent._summary`` that hold lists of names. Their changes are shown as removed and added names.
_SUMMARY_LISTS = ("tools", "mcp_servers")


def _resume_changes(saved: dict[str, Any], current: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """``{key: (saved, current)}`` for each ``Agent._summary`` key that differs. A key missing from ``saved`` is
    not compared, so an older or partial summary does not warn."""
    changes: dict[str, tuple[Any, Any]] = {}
    if not isinstance(saved, dict):
        return changes
    for key, now in current.items():
        if key not in saved:
            continue
        before = saved[key]
        if key in _SUMMARY_LISTS and _is_name_list(before):
            same = set(before) == set(now)
        else:
            same = before == now
        if not same:
            changes[key] = (before, now)
    return changes


def _describe_changes(changes: dict[str, tuple[Any, Any]]) -> str:
    """The ``ResumeWarning`` message, e.g. ``... (tools: removed search_web, added search_docs; model: a -> b)``.
    The system prompt is only reported as changed."""
    parts = []
    for key, (before, now) in changes.items():
        if key == "system_sha256":
            parts.append("system prompt changed")
        elif key in _SUMMARY_LISTS and _is_name_list(before):
            moves = []
            removed = [name for name in before if name not in now]
            added = [name for name in now if name not in before]
            if removed:
                moves.append(f"removed {', '.join(map(str, removed))}")
            if added:
                moves.append(f"added {', '.join(map(str, added))}")
            parts.append(f"{key}: {', '.join(moves)}")
        else:
            parts.append(f"{key}: {_or_none(before)} -> {_or_none(now)}")
    return (
        f"This State was saved by a different Agent ({'; '.join(parts)}). "
        "The run continues, but may behave differently than before it was saved"
    )


def _note_failed_save(error: BaseException, failure: Exception) -> None:
    """Adds ``saving the state also failed: ...`` to ``error`` once, unless ``failure`` says the same as ``error``
    (the failed save being raised was tried once more)."""
    if type(failure) is type(error) and str(failure) == str(error):
        return
    note = f"saving the state also failed: {type(failure).__name__}: {failure}"
    if note not in getattr(error, "__notes__", ()):  # think and then run both try to save
        error.add_note(note)


def _is_name_list(value: Any) -> bool:
    """A saved summary comes from storage, so a list of names is checked before it is compared as a set."""
    return isinstance(value, list) and all(isinstance(name, str) for name in value)


def _or_none(value: Any) -> str:
    return "(none)" if value is None else str(value)


def _loop_name(loop: Any) -> str:
    return repr(getattr(loop, "__name__", None) or loop)


def _type_name(returns: Any) -> str:
    return getattr(returns, "__name__", None) or repr(returns)
