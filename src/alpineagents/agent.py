"""Agent: holds settings and performs the actions that need a model and tools.

The contract is in ARCHITECTURE.md "Agent contract". Agent does not print (it notifies the Reporter), does not
read human input (it asks the Human), does not hold history (State's job), and does not decide the order of
steps (the Loop's job).
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

from . import _structured
from ._async import ASYNC_RUN, check_sync_call, is_async_callable, run_in_thread
from ._runner import arun_calls, run_calls
from ._tokens import estimate_overhead_tokens
from .errors import NoHumanError, OutputError, fix_message
from .human import check_returns as check_human_returns
from .mcp_tools import MCP, MCPTool, MCPToolRef, check_tool_names, wait_sync
from .mcp_tools import _Loop as _MCPLoop
from .models.resolve import resolve_model
from .reporter import Reporter
from .state import State
from .tool import Tool, collect_tools
from .types import Message, ModelEvent, Request, TextBlock

if TYPE_CHECKING:
    from .human import Human
    from .models.base import Model
    from .state import Checkpoint
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
SETTINGS = ("model", "system", "tools", "skills", "loop", "reporter", "human", "name", "description")

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
    """Whether two tools are the same. ``obj.method`` gives a new Tool on every access, so compare the original
    function and the bound object."""
    return a is b or (a.fn is b.fn and a.bound_to is b.bound_to)


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
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Creates an Agent. Mistakes in the settings are reported here, with how to fix them.

        Creating an Agent uses no network and needs no API key.

        Args:
            model: A model name such as ``"claude-sonnet-5"``, ``"anthropic/claude-sonnet-5"`` or
                ``"ollama/llama3:8b"``, or a ``Model`` object such as ``Anthropic(...)`` when you need settings.
            system: The system prompt.
            tools: ``@tool`` functions, objects with ``@tool`` methods, single ``@tool`` methods, ``MCP`` servers
                and single MCP tools (``github.create_issue``).
            skills: Not implemented yet. A non-empty value raises ``NotImplementedError``.
            loop: The loop ``run`` calls with ``(agent, state)``. Defaults to ``default_loop``.
            reporter: Receives progress notifications. Defaults to the shared ``Terminal``. ``None`` is silent.
            human: Answers ``ask_human``. Defaults to the shared ``Terminal``. ``None`` means no human.
            name: The Agent's name.
            description: What the Agent does.

        Raises:
            TypeError: A setting has the wrong type, for example a function without ``@tool`` in ``tools=``,
                a ``loop`` that is not callable, or a Reporter or Human class instead of an object.
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
        """The items passed in ``tools=``, as given."""
        return self._tools

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

        Args:
            task: A task string, which starts a new ``State``, or a ``State`` to continue.

        Returns:
            What the loop returns. The default loop returns ``state.answer``.

        Raises:
            TypeError: The loop is async (use ``arun``), or ``task`` is neither a string nor a State.
            ValueError: The State was already ended with ``state.finish()``.

        Example:
            ```python
            state = State("Find the bug in this repo")
            answer = agent.run(state)
            print(state.stopped_by, state.usage.cost)
            ```
        """
        # Order (ARCHITECTURE.md "run order and the context rules after exceptions"):
        # 1. _start_run: str -> State(task); State as is; anything else TypeError. A finish()ed State is a
        #    ValueError. Clears state.stopped_by so a run ending in an exception keeps no stale reason.
        # 2. try: on_run_start; connect MCP (_mcp_open); state._note_window(...) records the window size with the
        #    full tool list so compact_if_full works on the first turn (the owner is not set); return loop(...)
        #    on_run_start is inside the try, so on_run_end is called whichever step raises.
        # 3. except BaseException: state._record_error(e); state._close_pending(e); raise (KeyboardInterrupt too)
        # 4. finally: close the MCP connections this run opened, then on_run_end(state, error) (None on success).
        # An async loop raises TypeError before step 1.
        check_sync_call("run")
        if is_async_callable(self._loop):
            raise TypeError(
                fix_message(
                    f"This Agent's loop {_loop_name(self._loop)} is async, so it cannot run with run()",
                    "Await it with arun() inside async code, or write the loop as a regular def",
                    'answer = await agent.arun("...")',
                )
            )
        state = self._start_run(task)
        reporter = self.reporter
        error: BaseException | None = None
        connected = False
        try:
            if reporter is not None:
                reporter.on_run_start(state)
            self._mcp_open()
            connected = True
            self._note_window(state)
            return self._loop(self, state)
        except BaseException as e:
            error = e
            state._record_error(e)
            state._close_pending(e)
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
        """
        # Same steps as run. Sets ASYNC_RUN so sync methods called on this event loop thread raise TypeError.
        loop = self._async_loop()
        state = self._start_run(task)
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
            return await loop(self, state)
        except BaseException as e:
            error = e
            state._record_error(e)
            state._close_pending(e)
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
                ``[]`` shows none.

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
        # 3. except BaseException: state._rollback(cp, e); raise.
        # tools= items are flattened with collect_tools and looked up by name; each must be the same tool.
        check_sync_call("think")
        self._mcp_ensure()
        specs, checkpoint = self._begin_think(state, tools, "think")
        reporter = self.reporter
        try:
            request = self._think_request(state, specs, reporter)
            reply = self._model.respond(request, self._on_text(state, reporter), self._on_event(state, reporter))
            self._end_think(state, reply, reporter)
        except BaseException as e:
            state._rollback(checkpoint, e)
            raise

    async def athink(self, state: State, tools: Iterable[Any] | None = None) -> None:
        """The async version of ``think``. Cancelling it rolls the context back like any other failure."""
        # Same steps as think, with model.arespond.
        await self._amcp_ensure()
        specs, checkpoint = self._begin_think(state, tools, "athink")
        reporter = self.reporter
        try:
            request = self._think_request(state, specs, reporter)
            reply = await self._model.arespond(
                request, self._on_text(state, reporter), self._on_event(state, reporter)
            )
            self._end_think(state, reply, reporter)
        except BaseException as e:
            state._rollback(checkpoint, e)
            raise

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
        check_sync_call("use_tools")
        self._mcp_ensure()
        calls = self._begin_use_tools(state, "use_tools")
        if calls:
            run_calls(state, calls, self._tool_map(), self.reporter)

    async def ause_tools(self, state: State) -> None:
        """The async version of ``use_tools``.

        ``async def`` tools run as tasks on the running event loop, other tools on worker threads.
        Cancelling it follows the same rules as Ctrl+C.
        """
        # _runner.arun_calls.
        await self._amcp_ensure()
        calls = self._begin_use_tools(state, "ause_tools")
        if calls:
            await arun_calls(state, calls, self._tool_map(), self.reporter)

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
        check_sync_call("compact", " (in a loop: await acompact_if_full(agent, state))")
        self._mcp_ensure()
        self._begin_compact(state, instructions, "compact")
        try:
            request = self._compact_request(state)
            reply = self._model.compact(request, instructions, self._on_event(state, self.reporter))
            self._apply_summary(state, reply)
        finally:
            state._end_compact()

    async def acompact(self, state: State, instructions: str | None = None) -> None:
        """The async version of ``compact``."""
        # Same steps as compact, with model.acompact.
        await self._amcp_ensure()
        self._begin_compact(state, instructions, "acompact")
        try:
            request = self._compact_request(state)
            reply = await self._model.acompact(request, instructions, self._on_event(state, self.reporter))
            self._apply_summary(state, reply)
        finally:
            state._end_compact()

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
        """Steps 1 and 2 of ``run``: the State to run, checked and prepared."""
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
        return state

    def _note_window(self, state: State) -> None:
        """Records the context window size so context_used means something even before the first turn (the owner
        is not set). After connecting, so MCP tools count toward the overhead."""
        state._note_window(
            self,
            context_window=self._model.context_window,
            overhead_tokens=estimate_overhead_tokens(self._system, self._specs(None)),
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

    def _begin_think(
        self, state: State, tools: Iterable[Any] | None, method: str
    ) -> tuple[tuple[ToolSpec, ...], Checkpoint]:
        """Steps 1 and 2 of ``think``: the specs to show and the checkpoint to roll back to."""
        _check_state(state, method)
        specs = self._specs(tools)
        state._claim(
            self,
            context_window=self._model.context_window,
            overhead_tokens=estimate_overhead_tokens(self._system, specs),
        )
        return specs, state._begin_think()

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
        mcp_specs = [t.spec for t in self._chosen_mcp_tools(i for i in items if isinstance(i, (MCP, MCPToolRef)))]
        items = tuple(i for i in items if not isinstance(i, (MCP, MCPToolRef)))
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

    def _chosen_mcp_tools(self, items: Iterable[MCP | MCPToolRef]) -> list[MCPTool]:
        """``think(tools=[gh, gh.search_code])``: the connected tools these stand for."""
        chosen: list[MCPTool] = []
        for item in items:
            if isinstance(item, MCPToolRef):
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


def _loop_name(loop: Any) -> str:
    return repr(getattr(loop, "__name__", None) or loop)


def _type_name(returns: Any) -> str:
    return getattr(returns, "__name__", None) or repr(returns)
