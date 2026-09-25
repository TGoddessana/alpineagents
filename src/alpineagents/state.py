"""State: manages this run's history and the context the model sees.

Internal rules (the user-facing contract is in the public docstrings):

- It is mutable, but only through State's methods. Every method changes data inside the reentrant lock
  ``self.lock``, and Reporter notifications (``on_context_change``, ``on_tool_end`` for a deny) are sent after
  the lock is released.
- ``history`` is append-only. Shrinking or rolling back the context never deletes from it.
- Blocks of received messages (``Reply.message``) go into the context unchanged.
- It never calls the model, runs tools or prints.
- Methods starting with ``_`` are for Agent (and Loop) only. User code does not call them.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import _tokens
from .errors import fix_message
from .types import (
    ContextChange,
    ContextChangeKind,
    Exchange,
    HistoryEntry,
    Message,
    ModelEvent,
    Reply,
    ToolCall,
    ToolOutcome,
    ToolResultBlock,
    Usage,
    format_call,
)

if TYPE_CHECKING:
    from .agent import Agent
    from .reporter import Reporter


__all__ = ["State", "Checkpoint"]

#: Text left in place of a result cleared by ``clear_tool_results``.
CLEARED = "(cleared: kept in history)"
#: Result of a call closed by an interrupt (KeyboardInterrupt, CancelledError).
INTERRUPTED = "(interrupted by user)"
#: Notice prefix. ``add_notice`` adds it if missing.
NOTICE_PREFIX = "[notice] "
#: Text ``start_from``/``compact`` put before the summary.
SUMMARY_PREFIX = "[notice] Summary so far:\n"

#: Text put in place of a pending call's result in the context for ``ask``.
NOT_RUN_YET = "(not run yet)"
#: Notice that puts a late result into the context.
LATE_RESULT = NOTICE_PREFIX + "interrupted {call} finished later: {content}"


def closed_result(error: BaseException) -> str:
    """Result text for a call closed by an exception.

    ``"(interrupted by user)"`` for ``KeyboardInterrupt``/``asyncio.CancelledError``,
    otherwise ``"(aborted: {type(error).__name__})"`` (e.g. ``"(aborted: TimeoutError)"``).
    """
    if _is_interrupt(error):
        return INTERRUPTED
    return f"(aborted: {type(error).__name__})"


def closed_outcome(error: BaseException) -> ToolOutcome:
    """The ``ToolOutcome`` matching ``closed_result(error)``: ``"interrupted"`` or ``"aborted"``, with ``error``."""
    return ToolOutcome("interrupted" if _is_interrupt(error) else "aborted", error)


def _is_interrupt(error: BaseException) -> bool:
    return isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))


@dataclass(frozen=True)
class Checkpoint:
    """Rollback point returned by ``_begin_think``. Only ``_rollback`` uses it."""

    context_len: int
    turn: int
    last_answer: str | None


class _LockedDict(dict):
    """A dict whose single reads and writes are guarded by the State lock (``state.data``).

    For multi-step updates and consistent iteration, the user holds ``with state.lock:``.
    Copying or pickling it gives a plain dict without the lock.

    ``__iter__`` and ``__len__`` are deliberately not overridden. If ``__iter__`` is overridden, CPython reads
    ``dict(d)``, ``{**d}`` and ``{} | d`` as ``keys()`` plus ``d[k]`` per key, so another thread deleting a key
    in between raises ``KeyError``. If ``__len__`` is Python code, the thread can switch when ``list(d)`` asks
    for the length after creating the iterator, raising ``RuntimeError``. With both left as the C
    implementation, these copies read the table in one go.
    On the other hand, another thread changing the dict while a ``for k in d:`` body runs can raise
    ``RuntimeError``: iterate inside ``with state.lock:``, or iterate over ``list(d)``/``d.copy()``.
    """

    __slots__ = ("_lock",)

    def __init__(self, lock: Any) -> None:
        super().__init__()
        self._lock = lock

    def __getitem__(self, key: Any) -> Any:
        with self._lock:
            return super().__getitem__(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        with self._lock:
            super().__setitem__(key, value)

    def __delitem__(self, key: Any) -> None:
        with self._lock:
            super().__delitem__(key)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return super().__contains__(key)

    def __eq__(self, other: object) -> bool:
        with self._lock:
            return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        with self._lock:
            return super().__ne__(other)

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        with self._lock:
            return super().__repr__()

    def __or__(self, other: Any) -> Any:
        with self._lock:
            return dict(self.items()) | other

    def __ior__(self, other: Any) -> _LockedDict:
        self.update(other)
        return self

    def __reduce__(self) -> Any:
        with self._lock:
            return (dict, (dict(super().items()),))

    def get(self, key: Any, default: Any = None) -> Any:
        with self._lock:
            return super().get(key, default)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        with self._lock:
            return super().setdefault(key, default)

    def pop(self, key: Any, *default: Any) -> Any:
        with self._lock:
            return super().pop(key, *default)

    def popitem(self) -> tuple[Any, Any]:
        with self._lock:
            return super().popitem()

    def update(self, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            super().update(*args, **kwargs)

    def clear(self) -> None:
        with self._lock:
            super().clear()

    def copy(self) -> dict[Any, Any]:
        with self._lock:
            return dict(super().items())

    def keys(self) -> Any:
        with self._lock:
            return super().keys()

    def values(self) -> Any:
        with self._lock:
            return super().values()

    def items(self) -> Any:
        with self._lock:
            return super().items()


def _short(text: Any, limit: int = 60) -> str:
    """Display string shortened to one line."""
    one_line = " ".join(str(text).split())
    return one_line if len(one_line) <= limit else one_line[: limit - 1] + "…"


def _size(text: str) -> str:
    """Display of UTF-8 byte size (``512B``, ``1.2KB``, ``3.4MB``)."""
    n = len(text.encode("utf-8"))
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


def _error_text(error: BaseException) -> str:
    message = str(error)
    name = type(error).__name__
    return f"{name}: {message}" if message else name


class State:
    """One run's record: everything that happened (``history``) and what the model sees next (``context``).

    Pass a State to ``agent.run`` to keep it after the run, or to continue it after ``Ctrl+C``. Every method is
    thread-safe, so tools may call them from worker threads.

    Example:
        ```python
        state = State("Find the bug in this repo")
        agent.run(state)
        print(state.answer, state.stopped_by, state.usage.cost)
        ```
    """

    # A new State starts with history [HistoryEntry("user", task, turn=0)], context (Message.user(task),),
    # turn 0, Usage(), no pending calls, stopped_by None, parent None, root itself, depth 0
    # (subagents are an extension, see _child).

    def __init__(self, task: str) -> None:
        """Start a run with one user message.

        Args:
            task: What the agent should do. The first user message the model sees.

        Raises:
            TypeError: ``task`` is not a string.
            ValueError: ``task`` is empty or whitespace only.
        """
        if not isinstance(task, str):
            raise TypeError(
                fix_message(
                    f"State(task) needs task to be a string (got: {type(task).__name__})",
                    "pass the task as a single sentence",
                    'state = State("Find the bug in this repo")',
                )
            )
        if not task.strip():
            raise ValueError(
                fix_message(
                    f"State(task) got an empty task (got: {task!r}). Providers reject empty messages",
                    "pass the task as a single sentence",
                    'state = State("Find the bug in this repo")',
                )
            )
        self._lock = threading.RLock()
        self._task = task
        self._history: list[HistoryEntry] = [HistoryEntry("user", task, turn=0)]
        self._context: list[Message] = [Message.user(task)]
        self._turn = 0
        self._usage = Usage()
        # This turn: all calls the model requested (request order), pending calls, collected results, deferred messages.
        self._turn_calls: tuple[ToolCall, ...] = ()
        self._pending: list[ToolCall] = []
        self._results: dict[str, ToolResultBlock] = {}
        self._deferred: list[Message] = []
        # think is waiting on the model (_begin_think .. _record_reply/_rollback): added messages are deferred.
        self._thinking = False
        # Messages added while compact waits on the model (_begin_compact .. _end_compact).
        self._compact_added: list[Message] | None = None
        # Late result notices to put into the context at the next _begin_think.
        self._late_notices: list[str] = []
        self._finished = False
        self._finish_answer: Any = None
        self._last_answer: str | None = None
        self._stopped_by: str | None = None
        self._stopped_limit_value: int | None = None
        # Owner and context window size (set by _claim).
        self._owner: Agent | None = None
        self._reporter: Reporter | None = None
        self._context_window: int | None = None
        self._overhead = 0
        self._parent: State | None = None
        self._root: State = self
        self._depth = 0
        self._data = _LockedDict(self._lock)

    # ------------------------------------------------------------ yes/no questions

    def is_answered(self) -> bool:
        """Whether the model's last reply is a final answer.

        True when the last message in the context is a model reply without tool calls. A message added after it
        (a notice, a user message) makes it false, and so do pending calls.

        Use it as a stop condition: ``@loop(until=State.is_answered, limit=50)``.
        """
        with self._lock:
            if self._pending or self._deferred or not self._context:
                return False
            last = self._context[-1]
            return last.role == "assistant" and not last.tool_calls

    def wants_tools(self) -> bool:
        """Whether the model requested tool calls that have not run yet (``pending_calls`` is not empty)."""
        with self._lock:
            return bool(self._pending)

    def is_finished(self) -> bool:
        """Whether ``finish()`` was called. A finished State stops its loop before the next turn."""
        with self._lock:
            return self._finished

    # ------------------------------------------------------------ values (read-only)

    @property
    def task(self) -> str:
        """The task this State was created with."""
        return self._task

    @property
    def answer(self) -> Any:
        """The run's answer.

        The value given to ``finish(answer)`` if there is one, otherwise the text of the latest model reply
        without tool calls, otherwise ``None``. A ``think`` that fails does not change it.
        """
        # _rollback restores _last_answer from the Checkpoint.
        with self._lock:
            if self._finish_answer is not None:
                return self._finish_answer
            return self._last_answer

    @property
    def turn(self) -> int:
        """Number of turns so far: each ``think`` adds 1, and a ``think`` that fails does not count."""
        # _begin_think adds 1 and _rollback reverts it.
        with self._lock:
            return self._turn

    @property
    def usage(self) -> Usage:
        """Total tokens, requests and cost of every ``think``, ``ask`` and ``compact`` on this State."""
        # Added with _add_usage.
        with self._lock:
            return self._usage

    @property
    def pending_calls(self) -> tuple[ToolCall, ...]:
        """Tool calls the model requested this turn that have no result yet, in request order.

        A call leaves this tuple when it gets a result, is denied with ``deny``, or is closed by an exception.
        """
        with self._lock:
            return tuple(self._pending)

    @property
    def context_tokens(self) -> int:
        """Estimated token count of the context, including the system prompt and tool definitions."""
        # _tokens.context_tokens(context, overhead). overhead is the value _claim received (0 if none yet).
        with self._lock:
            return _tokens.context_tokens(tuple(self._context), self._overhead)

    @property
    def context_used(self) -> float:
        """Fraction of the model's context window in use: ``context_tokens / context_window``.

        0.0 before the first run or ``think`` (the window size is unknown until then). Can go above 1.0.
        """
        # The window size is stored by _claim or Agent.run's _note_window.
        with self._lock:
            window = self._context_window
            if not window or window <= 0:
                return 0.0
            return self.context_tokens / window

    @property
    def stopped_by(self) -> str | None:
        """Why the last loop stopped: ``"finish"``, ``"limit"``, or the name of the ``until`` function that returned
        true (e.g. ``"is_answered"``). ``None`` while running, or if the run ended with an exception."""
        with self._lock:
            return self._stopped_by

    @property
    def stopped_limit(self) -> int | None:
        """The ``limit`` the loop reached if ``stopped_by == "limit"``, otherwise ``None``."""
        with self._lock:
            return self._stopped_limit_value if self._stopped_by == "limit" else None

    @property
    def parent(self) -> State | None:
        """Reserved for subagents, which are not implemented yet. Always ``None``."""
        return self._parent

    @property
    def root(self) -> State:
        """Reserved for subagents, which are not implemented yet. Always this State."""
        return self._root

    @property
    def depth(self) -> int:
        """Reserved for subagents, which are not implemented yet. Always ``0``."""
        return self._depth

    @property
    def history(self) -> tuple[HistoryEntry, ...]:
        """Everything that happened in this run, oldest first: messages, replies, tool results, errors and context
        changes. Entries are only ever added, so shrinking the context never removes anything from here.

        Returns a copy.
        """
        with self._lock:
            return tuple(self._history)

    @property
    def context(self) -> tuple[Message, ...]:
        """The messages the model will see at the next ``think``. Returns a copy.

        Results of this turn's tool calls go in together once the last call has a result. Messages added in the
        meantime, or while ``think`` waits on the model, go in after them.
        """
        with self._lock:
            return tuple(self._context)

    @property
    def data(self) -> dict[str, Any]:
        """A dict for your own data. The model never sees it.

        Single reads and writes (``d[k]``, ``d[k] = v``, ``get``, ``setdefault``, ``pop``) are thread-safe. For
        an update that takes several steps, or to loop over it, hold ``state.lock``.

        Example:
            ```python
            with state.lock:
                state.data["calls"] = state.data.get("calls", 0) + 1
            ```
        """
        # A _LockedDict guarded by self._lock. See its docstring for why __iter__/__len__ stay as dict's.
        return self._data

    @property
    def lock(self) -> threading.RLock:
        """The reentrant lock (``threading.RLock``) every State method uses. Hold it to make several changes as one."""
        return self._lock

    # ------------------------------------------------------------ adding messages

    def add_user_message(self, text: str) -> None:
        """Add a message from the human, for example the next request in a chat loop.

        While tool calls are pending, or while ``think`` waits on the model, the message is held and goes into
        the context right after the results or the reply. ``is_answered()`` is false until the model answers it.

        Args:
            text: The message.

        Raises:
            ValueError: ``text`` is empty or whitespace only.
        """
        # History records it right away; the context gets it via _add_to_context, which defers it while
        # _pending or _thinking so a message the model never saw does not land before its reply.
        _check_text("add_user_message", "text", text, 'state.add_user_message("Next step: run the tests")')
        with self._lock:
            self._append("user", text)
            self._add_to_context(Message.user(text))

    def add_notice(self, text: str) -> None:
        """Tell the model something from your loop or a tool, for example ``"The tests failed"``.

        The model sees it as a user message starting with ``"[notice] "`` (added if missing). It is held while
        calls are pending, like ``add_user_message``.

        Args:
            text: The notice.

        Raises:
            ValueError: ``text`` is empty or whitespace only.
        """
        _check_text("add_notice", "text", text, 'state.add_notice("The tests failed")')
        if not text.startswith(NOTICE_PREFIX):
            text = NOTICE_PREFIX + text
        with self._lock:
            self._append("notice", text)
            self._add_to_context(Message.user(text))

    # ------------------------------------------------------------ stopping, denying

    def finish(self, answer: Any = None) -> None:
        """End the run. The loop stops before its next turn with ``stopped_by == "finish"``.

        Tools may call it (a ``submit`` tool, for example), even while other calls are pending. Calling it again
        is allowed; the last ``answer`` given wins.

        Args:
            answer: If not ``None``, becomes ``state.answer``. Any value, not only a string.
        """
        with self._lock:
            self._finished = True
            if answer is not None:
                self._finish_answer = answer

    def deny(self, call: ToolCall, reason: str) -> None:
        """Refuse a pending tool call. The model gets ``reason`` as that call's error result.

        Args:
            call: One of ``state.pending_calls``.
            reason: What the model is told, for example ``"The user denied it"``.

        Raises:
            ValueError: ``call`` is not pending, or ``reason`` is empty or whitespace only.

        Example:
            ```python
            for call in state.pending_calls:
                if call.name == "delete_file":
                    state.deny(call, "Deleting files is not allowed")
            agent.use_tools(state)
            ```
        """
        # Compared by id. History gets "denied" (content=reason, call=call); the context gets
        # ToolResultBlock(call.id, reason, name=call.name, is_error=True) under the same buffer rules as
        # _record_tool_result. After the lock is released, the Reporter gets
        # on_tool_end(self, call, reason, ToolOutcome("denied")).
        _check_text("deny", "reason", reason, 'state.deny(call, "The user denied it")')
        with self._lock:
            index = self._pending_index(call)
            if index is None:
                shown = format_call(call) if isinstance(call, ToolCall) else repr(call)
                raise ValueError(
                    fix_message(
                        f"cannot deny a call that is not pending: {shown}",
                        "only calls in state.pending_calls can be denied "
                        "(this one already has a result or belongs to another turn)",
                        'for call in state.pending_calls:\n    state.deny(call, "The user denied it")',
                    )
                )
            stored = self._pending[index]
            self._append("denied", reason, call=stored)
            self._resolve(index, reason, is_error=True)
            reporter = self._reporter
        if reporter is not None:
            reporter.on_tool_end(self, call, reason, ToolOutcome("denied"))

    # ------------------------------------------------------------ shrinking the context

    def start_from(self, summary: str) -> None:
        """Replace the context with two messages: the original task and ``summary``. History is kept.

        Args:
            summary: What the model should know about the work so far.

        Raises:
            TypeError: ``summary`` is not a string.
            ValueError: Tool calls are pending, or ``summary`` is empty or whitespace only.
        """
        # Same as _replace_context(summary, "start_from").
        self._replace_context(summary, "start_from")

    def clear_tool_results(self, keep_last: int = 5) -> None:
        """Shrink the context by blanking old tool results. History keeps the full results.

        Every tool result except the last ``keep_last`` is replaced with ``"(cleared: kept in history)"``.
        Results that are already cleared still count toward ``keep_last``.

        Args:
            keep_last: How many of the most recent tool results to keep.

        Raises:
            ValueError: Tool calls are pending, or ``keep_last`` is not an integer >= 0.
        """
        # Only ToolResultBlock changes (dataclasses.replace(block, content=CLEARED)); thinking blocks etc. stay.
        # If anything changed, every context message is rebuilt with tokens=None (the anchor is no longer
        # valid), ContextChange("clear_tool_results", before, after) goes to history and the Reporter is told.
        # If nothing changed, it does nothing.
        if isinstance(keep_last, bool) or not isinstance(keep_last, int) or keep_last < 0:
            raise ValueError(
                fix_message(
                    f"keep_last must be an integer >= 0 (got: {keep_last!r})",
                    "pass the number of recent tool results to keep as an integer",
                    "state.clear_tool_results(keep_last=5)",
                )
            )
        with self._lock:
            self._ensure_no_pending("clear_tool_results")
            positions = [
                (m, b)
                for m, message in enumerate(self._context)
                for b, block in enumerate(message.content)
                if isinstance(block, ToolResultBlock)
            ]
            to_clear = positions[: max(len(positions) - keep_last, 0)]
            targets: dict[int, set[int]] = {}
            for m, b in to_clear:
                if self._context[m].content[b].content != CLEARED:
                    targets.setdefault(m, set()).add(b)
            if not targets:
                return
            before = self.context_tokens
            new_context: list[Message] = []
            for m, message in enumerate(self._context):
                blocks = message.content
                if m in targets:
                    blocks = tuple(
                        dataclasses.replace(block, content=CLEARED) if b in targets[m] else block
                        for b, block in enumerate(blocks)
                    )
                new_context.append(Message(message.role, blocks, tokens=None))
            self._context = new_context
            change = ContextChange("clear_tool_results", before, self.context_tokens)
            self._append("context_change", change)
            reporter = self._reporter
        self._notify_context_change(reporter, change)

    # ------------------------------------------------------------ saving (outside MVP)

    def save(self, path: str) -> None:
        """Not implemented yet. Raises ``NotImplementedError``, or ``ValueError`` while calls are pending."""
        self._ensure_no_pending("save")
        raise NotImplementedError("state.save() is not implemented yet")

    @classmethod
    def load(cls, path: str) -> State:
        """Not implemented yet. Raises ``NotImplementedError``."""
        raise NotImplementedError("State.load() is not implemented yet")

    # ------------------------------------------------------------ display

    def __str__(self) -> str:
        """Several lines summarizing history for humans. One line per entry; long content is shortened.

        E.g. ``[turn 0] user: Find the bug`` / ``[turn 1] reply: read_file(path="main.py")`` /
        ``[turn 1] tool_result read_file: 1.2KB`` / ``done: is_answered (2 turns)``
        """
        with self._lock:
            lines = [_describe_entry(entry) for entry in self._history]
            if self._stopped_by is not None:
                reason = self._stopped_by
                if reason == "limit" and self._stopped_limit_value is not None:
                    reason = f"limit({self._stopped_limit_value})"
                lines.append(f"done: {reason} ({_turns(self._turn)})")
            elif self._finished:
                lines.append(f"done: finish ({_turns(self._turn)})")
            return "\n".join(lines)

    def __repr__(self) -> str:
        with self._lock:
            extra = " finished" if self._finished else ""
            return (
                f"<State task={_short(self._task, 30)!r} turn={self._turn} "
                f"history={len(self._history)} context={len(self._context)} "
                f"pending={len(self._pending)}{extra}>"
            )

    # ================================================================ Agent only (internal contract)

    def _claim(
        self, agent: Agent, *, context_window: int, overhead_tokens: int | None = None
    ) -> None:
        """Called by Agent before it uses the context (``think``, ``use_tools``, ``compact``).

        - With no owner, ``agent`` becomes the owner. If there is an owner and it is not ``agent`` (``is``
          comparison), ``ValueError``: another Agent cannot think with someone else's State. Fix: read it with
          ``other.ask(state, ...)`` or call it as a subagent.
        - Stores ``context_window`` and ``overhead_tokens`` (estimate for system + tool definitions) for
          ``context_used``. ``overhead_tokens=None`` keeps the previous value (``use_tools`` does not know which
          tools were shown).
        - Sets the Reporter to notify to ``agent.reporter`` (may be ``None``).
        """
        # agent.reporter may create the default Terminal, so read it outside the lock.
        reporter = agent.reporter
        with self._lock:
            if self._owner is None:
                self._owner = agent
            elif self._owner is not agent:
                raise ValueError(
                    fix_message(
                        f"this State is owned by another Agent ({_agent_label(self._owner)}). "
                        "An Agent cannot think with another Agent's State (the two conversations must not mix)",
                        "let the other Agent only read it with ask, or call it as a subagent to work in a new State",
                        'review = reviewer.ask(state, "Review the changes so far")',
                    )
                )
            self._context_window = context_window
            if overhead_tokens is not None:
                self._overhead = overhead_tokens
            self._reporter = reporter

    def _note_window(self, agent: Agent, *, context_window: int, overhead_tokens: int) -> None:
        """Called by ``Agent.run`` before the loop: with no owner, or if ``agent`` is the owner, only records the
        window size and overhead.

        It does not set the owner (the owner is the first Agent to ``think``). This way ``context_used`` means
        something before the first turn, and ``compact_if_full`` can compact a context that is already full at
        the first turn. If another Agent is the owner, it does nothing (owner checks are ``_claim``'s job).
        """
        with self._lock:
            if self._owner is not None and self._owner is not agent:
                return
            self._context_window = context_window
            self._overhead = overhead_tokens

    def _ensure_open(self, action: str) -> None:
        """``ValueError`` after ``finish()``: ``{action}`` cannot be called after ``finish()``.
        Fix: if a block called ``finish()``, ``return`` from the loop body. (Called by think, use_tools, ask)
        """
        with self._lock:
            if not self._finished:
                return
        raise ValueError(
            fix_message(
                f"cannot call {action}() on a finished State (finish() ends the whole run)",
                "if a block called state.finish(), return from the loop body right away",
                "submit(agent, state)\nif state.is_finished():\n    return\nagent.think(state)",
            )
        )

    def _ensure_no_pending(self, action: str) -> None:
        """``ValueError`` if there are pending calls: ``{action}`` was called while calls (list of names) are
        pending. Fix: close them first with ``agent.use_tools(state)`` or ``state.deny(call, reason)``.
        No side effects.
        """
        with self._lock:
            if not self._pending:
                return
            names = ", ".join(format_call(call) for call in self._pending)
        raise ValueError(
            fix_message(
                f"cannot call {action}() while calls are pending ({names})",
                "run them with agent.use_tools(state) or close them first with state.deny(call, reason)",
                "agent.think(state)\nif state.wants_tools():\n    agent.use_tools(state)",
            )
        )

    def _begin_compact(self) -> None:
        """Start of ``Agent.compact``, right before it reads the context. Inside the lock:
        ``_ensure_no_pending("compact")``, then start collecting messages added via
        ``add_user_message``/``add_notice``: the model will not see them, so ``_replace_context(.., "compact")``
        re-adds them after the summary. Every call is paired with ``_end_compact``.
        """
        with self._lock:
            self._ensure_no_pending("compact")
            self._compact_added = []

    def _end_compact(self) -> None:
        """End of ``Agent.compact``, success or not: stop collecting (``_replace_context`` already consumed them
        on success; on failure they are already in the context and nothing else is needed)."""
        with self._lock:
            self._compact_added = None

    def _begin_think(self) -> Checkpoint:
        """Start of ``think``. Inside the lock, in order:

        1. ``_ensure_open("think")``, ``_ensure_no_pending("think")``
        2. Put late result notices (collected by ``_record_late_result``) into the context as user messages
           (history already has ``tool_result`` (late=True), so no new history entry is added)
        3. Create ``Checkpoint(len(context), turn, last_answer)`` (the state after step 2)
        4. ``turn += 1``
        5. Mark as waiting on the model: until ``_record_reply``/``_rollback``, ``add_user_message``/``add_notice``
           are deferred instead of going into the context (so a message the model never saw does not land
           before its reply)
        6. Return the Checkpoint
        """
        with self._lock:
            self._ensure_open("think")
            self._ensure_no_pending("think")
            for notice in self._late_notices:
                self._context.append(Message.user(notice))
            self._late_notices.clear()
            checkpoint = Checkpoint(len(self._context), self._turn, self._last_answer)
            self._turn += 1
            self._thinking = True
            return checkpoint

    def _record_reply(self, reply: Reply) -> None:
        """Record a model reply.

        - ``"reply"`` in history (content=reply)
        - ``Message("assistant", reply.message.content, tokens=reply.context_tokens)`` in the context
          (blocks unchanged; the new Message differs only in ``tokens``)
        - ``pending_calls`` = ``reply.tool_calls``
        - With no tool calls, ``last_answer`` = ``reply.text``
        - ``usage += reply.usage``
        - Messages deferred during ``think``: with no tool calls, they go right after the reply
          (``is_answered()`` false); otherwise they stay deferred and go after this turn's result message.
        """
        with self._lock:
            self._thinking = False
            self._append("reply", reply)
            self._context.append(
                Message("assistant", reply.message.content, tokens=reply.context_tokens)
            )
            calls = reply.tool_calls
            self._turn_calls = calls
            self._pending = list(calls)
            self._results = {}
            if not calls:
                self._last_answer = reply.text
                self._context.extend(self._deferred)
                self._deferred = []
            self._usage = self._usage + reply.usage

    def _rollback(self, checkpoint: Checkpoint, error: BaseException) -> None:
        """Exception during ``think``: roll the context back to just before that ``think``.

        - Truncate the context to ``checkpoint.context_len``, restore ``turn`` and ``last_answer``, and empty
          ``pending_calls`` (it was empty when ``think`` started).
        - ``_record_error(error)``
        - If any messages were dropped, record ``ContextChange("rollback", before, after)`` and notify the Reporter.
        - Usage is not rolled back (the tokens were already spent). The ``reply`` entry in history is not
          deleted either.

        User messages and notices that other threads added during ``think`` (not made by ``think``) are kept
        and re-added after the truncated context. Only this ``think``'s reply and its results are dropped.
        """
        with self._lock:
            before = self.context_tokens
            kept = self._context[: checkpoint.context_len]
            dropped = 0
            # Keep human messages and notices that came in during think (drop the reply and tool results).
            for message in self._context[checkpoint.context_len :]:
                if message.role == "user" and not any(
                    isinstance(block, ToolResultBlock) for block in message.content
                ):
                    kept.append(message)
                else:
                    dropped += 1
            kept.extend(self._deferred)
            self._context = kept
            self._thinking = False
            self._turn_calls = ()
            self._pending = []
            self._results = {}
            self._deferred = []
            # Record the error and rollback under the failed think's turn number, then restore the turn.
            self._record_error(error)
            change = None
            if dropped:
                change = ContextChange("rollback", before, self.context_tokens)
                self._append("context_change", change)
            self._turn = checkpoint.turn
            self._last_answer = checkpoint.last_answer
            reporter = self._reporter
        if change is not None:
            self._notify_context_change(reporter, change)

    def _record_tool_result(self, call: ToolCall, content: str, *, is_error: bool = False) -> None:
        """Record the result of one call. Safe to call from multiple threads.

        - ``ValueError`` if ``call`` is not in ``pending_calls`` (compared by id).
        - ``"tool_result"`` in history (content, call=call). Removed from ``pending_calls``.
        - Results collect in this turn's buffer. When the last call closes (``pending_calls`` becomes empty),
          one user message (the ``ToolResultBlock``s, **in the order the model requested them**) goes into the
          context, followed by the deferred user messages and notices in arrival order.
        """
        with self._lock:
            index = self._pending_index(call)
            if index is None:
                shown = format_call(call) if isinstance(call, ToolCall) else repr(call)
                raise ValueError(f"cannot record a result for a call that is not pending: {shown}")
            stored = self._pending[index]
            self._append("tool_result", content, call=stored)
            self._resolve(index, content, is_error=is_error)

    def _record_late_result(self, call: ToolCall, content: str, *, is_error: bool = False) -> None:
        """The result of a sync tool that was closed without waiting on interrupt arrives later (called from the
        worker thread).

        - If the ``call`` **object itself** (``is`` comparison) is still in ``pending_calls`` (the loop caught
          the interrupt), same as ``_record_tool_result``. Not compared by id: a call in the next turn can reuse
          the same id (servers that do not give ids), and last turn's result must not attach to it.
        - Otherwise record ``"tool_result"`` (content, call, late=True) in history, and collect the notice
          ``"[notice] interrupted {format_call(call)} finished later: {content}"`` to put into the context at the
          next ``_begin_think``.
        """
        with self._lock:
            index = next((i for i, pending in enumerate(self._pending) if pending is call), None)
            if index is not None:
                self._append("tool_result", content, call=call)
                self._resolve(index, content, is_error=is_error)
                return
            self._append("tool_result", content, call=call, late=True)
            self._late_notices.append(LATE_RESULT.format(call=format_call(call), content=content))

    def _close_pending(self, error: BaseException) -> list[ToolCall]:
        """Close all pending calls when an exception leaves ``run()``.

        Records each call with ``_record_tool_result(call, closed_result(error), is_error=True)``
        (in request order). Returns the list of closed calls, or an empty list if there were none.
        """
        text = closed_result(error)
        with self._lock:
            closed = list(self._pending)
            for call in closed:
                self._record_tool_result(call, text, is_error=True)
            return closed

    def _record_error(self, error: BaseException, call: ToolCall | None = None) -> None:
        """``"error"`` in history (content=``f"{type(error).__name__}: {error}"``, error=error, call=call).

        If the same exception object (``is``) is already an ``error`` entry, it is not added again (so an
        exception recorded by think/use_tools is not recorded again by ``run``). An exception with an empty
        message (``KeyboardInterrupt()``) records only its name.
        """
        with self._lock:
            for entry in self._history:
                if entry.kind == "error" and entry.error is error:
                    return
            self._append("error", _error_text(error), call=call, error=error)

    def _record_ask(self, question: str, answer: Any) -> None:
        """``"ask"`` in history (content=Exchange(question, answer)). The context is unchanged."""
        with self._lock:
            self._append("ask", Exchange(question, answer))

    def _record_human(self, question: str, answer: Any) -> None:
        """``"human"`` in history (content=Exchange(question, answer)). The context is unchanged."""
        with self._lock:
            self._append("human", Exchange(question, answer))

    def _record_model_event(self, event: ModelEvent) -> None:
        """``"model_event"`` in history (content=event). The context is unchanged."""
        with self._lock:
            self._append("model_event", event)

    def _add_usage(self, usage: Usage) -> None:
        """Add usage from ``ask`` and ``compact`` (``_record_reply`` adds it for ``think``)."""
        with self._lock:
            self._usage = self._usage + usage

    def _replace_context(self, summary: str, kind: ContextChangeKind) -> None:
        """Replace the context with ``(Message.user(task), Message.user(SUMMARY_PREFIX + summary))``.

        ``kind`` is ``"start_from"`` or ``"compact"``. ``ValueError`` if there are pending calls
        (``_ensure_no_pending(kind)``). Records ``ContextChange(kind, before, after, summary=summary)`` in
        history and notifies the Reporter. The summary is a user message, so ``is_answered()`` is false afterwards.

        For ``"compact"``, user messages and notices added while compaction waited on the model (after
        ``_begin_compact``) are re-added after the summary in arrival order (the model never saw
        them, so the summary does not cover them).
        """
        _check_text(
            kind, "summary", summary,
            'summary = agent.ask(state, "Summarize the work so far")\nstate.start_from(summary)',
        )
        with self._lock:
            self._ensure_no_pending(kind)
            added = self._compact_added if kind == "compact" else None
            self._compact_added = None
            before = self.context_tokens
            self._context = [Message.user(self._task), Message.user(SUMMARY_PREFIX + summary)]
            self._context.extend(added or ())
            change = ContextChange(kind, before, self.context_tokens, summary=summary)
            self._append("context_change", change)
            reporter = self._reporter
        self._notify_context_change(reporter, change)

    def _context_for_question(self) -> tuple[Message, ...]:
        """A copy of the context for ``ask``. Does not change the context.

        With no pending calls, ``context`` as is. Otherwise one more user message is appended to ``context``:
        for every call this turn, in request order, its result if it has one, otherwise a ``ToolResultBlock``
        with ``"(not run yet)"``. (Providers reject a tool_use that has no result after it)
        """
        with self._lock:
            context = tuple(self._context)
            if not self._pending:
                return context
            blocks = tuple(
                self._results.get(call.id)
                or ToolResultBlock(call.id, NOT_RUN_YET, name=call.name)
                for call in self._turn_calls
            )
            return context + (Message("user", blocks),)

    def _set_stopped_by(self, reason: str | None, *, limit: int | None = None) -> None:
        """Called when Loop stops. For ``reason="limit"``, pass the number as ``limit``
        (read back as ``stopped_limit``). For other reasons ``stopped_limit`` is ``None``.
        ``Agent.run`` resets it to ``None`` at start (so the previous run's reason does not linger).
        """
        with self._lock:
            self._stopped_by = reason
            self._stopped_limit_value = limit if reason == "limit" else None

    @classmethod
    def _child(cls, parent: State, task: str) -> State:
        """State for a subagent (extension). ``parent``, ``root=parent.root``, ``depth=parent.depth+1``."""
        child = cls(task)
        child._parent = parent
        child._root = parent.root
        child._depth = parent.depth + 1
        return child

    # ------------------------------------------------------------ internal helpers (called inside the lock)

    def _append(self, kind: Any, content: Any, **fields: Any) -> None:
        self._history.append(HistoryEntry(kind, content, turn=self._turn, **fields))

    def _add_to_context(self, message: Message) -> None:
        """Defer if there are pending calls or think is waiting on the model; otherwise add to the context now.
        During compaction, also collect it separately for ``_replace_context`` to re-add after the summary."""
        if self._compact_added is not None:
            self._compact_added.append(message)
        if self._pending or self._thinking:
            self._deferred.append(message)
        else:
            self._context.append(message)

    def _pending_index(self, call: Any) -> int | None:
        call_id = getattr(call, "id", None)
        if call_id is None:
            return None
        for index, pending in enumerate(self._pending):
            if pending.id == call_id:
                return index
        return None

    def _resolve(self, index: int, content: str, *, is_error: bool) -> None:
        """Put the result in the turn buffer; on the last call, add the result message and deferred messages."""
        call = self._pending.pop(index)
        self._results[call.id] = ToolResultBlock(call.id, content, name=call.name, is_error=is_error)
        if self._pending:
            return
        blocks = tuple(self._results[c.id] for c in self._turn_calls if c.id in self._results)
        if blocks:
            self._context.append(Message("user", blocks))
        self._context.extend(self._deferred)
        self._deferred = []
        self._results = {}
        self._turn_calls = ()

    def _notify_context_change(self, reporter: Reporter | None, change: ContextChange) -> None:
        """Called outside the lock."""
        if reporter is not None:
            reporter.on_context_change(self, change)


def _check_text(action: str, name: str, value: Any, example: str) -> None:
    if not isinstance(value, str):
        raise TypeError(
            fix_message(
                f"{action}() needs {name} to be a string (got: {type(value).__name__})",
                f"pass {name} as a str",
                example,
            )
        )
    if not value.strip():
        raise ValueError(
            fix_message(
                f"{action}() got an empty {name} (got: {value!r}). Providers reject empty messages",
                f"pass a non-empty {name} (if the human's input was empty, ask again)",
                example,
            )
        )


def _turns(n: int) -> str:
    return f"{n} turn" if n == 1 else f"{n} turns"


def _agent_label(agent: Any) -> str:
    try:
        name = agent.name
    except Exception:
        name = None
    return repr(name) if name else f"id={id(agent):#x}"


def _describe_entry(entry: HistoryEntry) -> str:
    """One history entry as one line."""
    head = f"[turn {entry.turn}] {entry.kind}"
    content = entry.content
    kind = entry.kind
    if kind == "reply" and isinstance(content, Reply):
        parts = []
        if content.text:
            parts.append(_short(content.text))
        parts.extend(_short(format_call(call)) for call in content.tool_calls)
        return f"{head}: {'; '.join(parts) if parts else '(empty reply)'}"
    if kind == "tool_result":
        name = entry.call.name if entry.call else "?"
        late = " (late)" if entry.late else ""
        return f"{head} {name}: {_size(str(content))}{late}"
    if kind == "denied":
        name = entry.call.name if entry.call else "?"
        return f"{head} {name}: {_short(content)}"
    if kind in ("ask", "human") and isinstance(content, Exchange):
        return f"{head}: {_short(content.question, 40)} → {_short(content.answer, 40)}"
    if kind == "context_change" and isinstance(content, ContextChange):
        return f"{head} {content.kind}: {content.before_tokens} → {content.after_tokens} tokens"
    if kind == "error" and entry.call is not None:
        return f"{head} {entry.call.name}: {_short(content)}"
    return f"{head}: {_short(content)}"
