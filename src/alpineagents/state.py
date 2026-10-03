"""State: one conversation's record, and the value (``StateSnapshot``) every public property reads.

Internal rules (the user-facing contract is in the public docstrings):

- **History is the only source of truth.** ``fold(snapshot, entry)`` is a pure function that adds one history
  entry to a ``StateSnapshot``; every public value of a snapshot is what folding its history from an empty
  snapshot gives. A State is a lock around one current snapshot: every change builds one entry and does
  ``self._snap = fold(self._snap, entry)`` (``State._commit``). There is no second "record" path: loading a saved
  State and ``State(history=...)`` replay entries through the same ``fold``.
- A State is mutable only through its methods. Commands validate first (raising before anything is recorded), then
  build the entry and commit it, all inside the reentrant lock ``self._lock``. Reporter notifications
  (``on_context_change``) are sent after the lock is released. Nothing here calls the model, tools, a Human, a
  Permission or a Store.
- ``history`` only grows. Shrinking the context (``compact``, ``clear_tool_results``) or going back
  (``restore``) records an entry that says so.
- Every entry gets its ``at`` when it is created inside the lock, right before it is committed (or in ``__init__``,
  before the State is shared), so ``at`` does not go backwards in history order (unless the system clock does).
  A State loaded from a Store keeps the saved ``at`` values.
- Saving: a State turns the entries a store does not have yet into JSON values (``_unsaved``) but never calls a
  Store. ``Store.save`` writes them outside the lock and then calls ``_mark_saved``.
- Methods starting with ``_`` are for Agent, Loop, runner, Store and the permissions only. User code does not call
  them.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import math
import re
import threading
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Never

from . import _serial, _tokens
from ._frozen import FrozenDict, thaw
from .errors import fix_message
from .types import (
    NOTICE_PREFIX,
    AgentInfo,
    ContextChange,
    ContextChangeEntry,
    ErrorEntry,
    Exchange,
    ExchangeEntry,
    ExtraDataEntry,
    HistoryEntry,
    Image,
    Message,
    MessageEntry,
    ModelEvent,
    ModelEventEntry,
    ModelReplyEntry,
    ModelRequestEntry,
    Reply,
    RunStartEntry,
    Stopped,
    StoppedByFinish,
    StoppedByPermission,
    StopEntry,
    TextBlock,
    ToolCall,
    ToolOutcome,
    ToolOutcomeKind,
    ToolResultBlock,
    ToolResultContent,
    ToolResultEntry,
    Usage,
    _byte_size,
    _content_bytes,
    format_call,
    result_text,
)

if TYPE_CHECKING:
    from .agent import Agent

__all__ = ["State", "StateSnapshot"]

#: Text left in place of a result cleared by ``clear_tool_results``.
CLEARED = "(cleared: kept in history)"
#: Result of a call closed by an interrupt (KeyboardInterrupt, CancelledError).
INTERRUPTED = "(interrupted by user)"
#: Text ``compact`` puts before the summary.
SUMMARY_PREFIX = "[notice] Summary so far:\n"

#: Result of a call not run because a permission denied another call of the same turn with ``stop=True``.
STOPPED_TURN = "(not run: the user stopped this turn)"

#: Text put in place of a pending call's result in the context for ``ask``.
NOT_RUN_YET = "(not run yet)"
#: Notice that puts a late result into the context. Images in the result appear as ``(image/png, 34.2KB)``.
LATE_RESULT = NOTICE_PREFIX + "interrupted {call} finished later: {content}"

#: Result a loaded State gives a call whose result was never saved (the process stopped while it ran).
UNSAVED_RESULT = (
    "(unknown: the run stopped before this result was saved, so the tool may or may not have run. "
    "Check whether it took effect before relying on it or running it again)"
)
#: The ``error`` entry a loaded State gets when it was saved while the model was being waited on.
STOPPED_WAITING = "(process stopped while waiting for the model)"

#: What a State id may contain. It becomes a file name in FileStore, so no dots or slashes.
_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}")


def closed_result(error: BaseException) -> str:
    """Result text for a call closed by an exception.

    ``"(interrupted by user)"`` for ``KeyboardInterrupt``/``asyncio.CancelledError``,
    otherwise ``"(aborted: {type(error).__name__})"`` (e.g. ``"(aborted: TimeoutError)"``).
    """
    if _is_interrupt(error):
        return INTERRUPTED
    return f"(aborted: {type(error).__name__})"


def closed_outcome(error: BaseException) -> ToolOutcome:
    """The ``ToolOutcome`` matching ``closed_result(error)``: ``INTERRUPTED`` or ``ABORTED``, with ``error``."""
    kind = ToolOutcomeKind.INTERRUPTED if _is_interrupt(error) else ToolOutcomeKind.ABORTED
    return ToolOutcome(kind, error)


def _is_interrupt(error: BaseException) -> bool:
    return isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))


def check_id(value: Any, where: str) -> str:
    """A State id: letters, digits, ``_`` and ``-``, 1 to 128 characters. ``TypeError``/``ValueError`` otherwise."""
    if not isinstance(value, str):
        raise TypeError(
            fix_message(
                f"{where} needs the id as a string (got: {type(value).__name__})",
                "use letters, digits, _ and -",
                'State(id="bug-hunt-1")',
            )
        )
    if not _ID_PATTERN.fullmatch(value):
        raise ValueError(
            fix_message(
                f"{where} got the id {value!r}, which may only have letters, digits, _ and - (1 to 128 characters)",
                "leave out dots, slashes and spaces (a FileStore uses the id as a file name)",
                'State(id="bug-hunt-1")',
            )
        )
    return value


# ---------------------------------------------------------------- StateSnapshot


# No slots=True: on Python 3.13 and earlier a frozen slots dataclass answers ``snap.new_name = 1`` with a confusing
# "super(type, obj)" TypeError (the generated __setattr__ names the pre-slots class) instead of FrozenInstanceError.
@dataclass(frozen=True, kw_only=True, repr=False)
class StateSnapshot:
    """The value of a State at one moment: what ``state.snapshot()`` returns, and what every ``state.<property>``
    reads. It is frozen, so it never changes while you hold it, and it has no methods that change anything.

    Every field is a pure function of ``history``: building a State from the same history (``State(history=...)``)
    gives an equal snapshot. ``==`` compares the public fields, so equal histories give equal snapshots. Pass a
    snapshot to ``state.restore(snapshot)`` to go back to that point.

    Example:
        ```python
        before = state.snapshot()
        agent.run(state)
        if before.usage != state.usage:
            state.restore(before)  # undo the run's effect on the context
        ```
    """

    history: tuple[HistoryEntry, ...] = ()
    """Everything that happened, oldest first. Entries are only ever added."""
    messages: tuple[Message, ...] = ()
    """What the model will see at the next ``think``. Results of this turn's tool calls go in together once the
    last call has a result. Messages added in the meantime, or while the model is being waited on, go in after
    them."""
    pending_calls: tuple[ToolCall, ...] = ()
    """The tool calls of the open turn that have no result yet, in request order."""
    turn: int = 0
    """Completed model replies, plus one while a request is waiting on the model."""
    usage: Usage = field(default_factory=Usage)
    """Tokens, requests and cost of every ``think``, ``ask`` and ``compact``. It is cumulative: ``restore`` does not
    lower it, because the tokens were spent."""
    extra_data: FrozenDict[str, Any] = field(default_factory=FrozenDict)
    """Your own values (JSON), which the model never sees. Read-only: change them with
    ``state.edit_extra_data()``."""
    finished: bool = False
    """Whether ``finish()`` was called (and not undone by ``restore``). A finished State cannot run again."""
    answer: Any = None
    """The value given to ``finish(answer)`` if there is one, otherwise the text of the latest reply without tool
    calls, otherwise ``None``."""
    stopped: Stopped | None = None
    """Why the current or last run is set to stop, or ``None`` (a new run clears it)."""
    # The times are left out of ``==`` because entries ignore ``at`` in ``==``: two States with equal histories
    # recorded at different moments must still give equal snapshots.
    created_at: datetime | None = field(default=None, compare=False)
    """The ``at`` of the first history entry, or ``None`` while the history is empty. Not compared by ``==``."""
    updated_at: datetime | None = field(default=None, compare=False)
    """The ``at`` of the latest history entry, or ``None`` while the history is empty. Not compared by ``==``."""

    # Bookkeeping of the fold. Not part of the value: ``compare=False`` keeps it out of ``==``, and the public
    # fields above are enough to tell two snapshots apart (they come from the same history).
    _turn_calls: tuple[ToolCall, ...] = field(default=(), compare=False, repr=False)
    """Every call the model requested in the open turn, in request order."""
    _turn_results: tuple[ToolResultBlock | None, ...] = field(default=(), compare=False, repr=False)
    """The result of each call of ``_turn_calls`` (same positions), ``None`` while it is pending."""
    _queued: tuple[Message, ...] = field(default=(), compare=False, repr=False)
    """Messages added while the turn was open or the model was being waited on. They go into ``messages`` when
    it ends."""
    _waiting: bool = field(default=False, compare=False, repr=False)
    """A request is waiting on the model (``ModelRequestEntry`` with no reply or error yet)."""
    _late_notices: tuple[str, ...] = field(default=(), compare=False, repr=False)
    """Notices of results that came after their call was closed. They go into ``messages`` at the next request."""
    _first_user_message: Message | None = field(default=None, compare=False, repr=False)
    """The first user message: ``compact`` keeps it in front of the summary."""
    _last_reply_text: str | None = field(default=None, compare=False, repr=False)
    """The text of the latest reply without tool calls."""
    _finish_answer: Any = field(default=None, compare=False, repr=False)
    """The latest ``finish`` answer that was not ``None``."""
    _memo: dict[int, StateSnapshot] = field(default_factory=dict, compare=False, repr=False)
    """The snapshots ``restore`` went back to, by history length. A cache of folds this State already did: all
    snapshots of one State share it, so a history with many restores replays in linear time. Never part of
    ``==``."""

    def __repr__(self) -> str:
        extra = " finished" if self.finished else ""
        return (
            f"<StateSnapshot turn={self.turn} history={len(self.history)} messages={len(self.messages)} "
            f"pending={len(self.pending_calls)}{extra}>"
        )


# ---------------------------------------------------------------- fold
# One function per rule of the spec's section 4. Each returns the fields it changes; ``fold`` adds the fields every
# entry changes (history, created_at, updated_at) and the two derived ones (``pending_calls``, ``answer``).


def fold(snap: StateSnapshot, entry: HistoryEntry) -> StateSnapshot:
    """The snapshot after ``entry`` was recorded on top of ``snap``. Pure: nothing is changed, and the same inputs
    give an equal result.

    Raises:
        ValueError: ``entry`` cannot follow what ``snap`` holds (a result for a call that is not pending, a reply
            nobody asked for, ...). ``State(history=...)`` adds the index of the entry to the message.
    """
    changes: dict[str, Any]
    match entry:
        case MessageEntry():
            changes = _on_message(snap, entry)
        case ModelRequestEntry():
            changes = _on_request(snap)
        case ModelReplyEntry():
            changes = _on_reply(snap, entry)
        case ErrorEntry():
            changes = _on_error(snap, entry)
        case ToolResultEntry():
            changes = _on_tool_result(snap, entry)
        case StopEntry():
            changes = _on_stop(entry.content)
        case RunStartEntry():
            changes = {"stopped": None}
        case ExtraDataEntry():
            changes = _on_extra_data(snap, entry)
        case ExchangeEntry():
            changes = {} if entry.usage is None else {"usage": snap.usage + entry.usage}
        case ModelEventEntry():
            changes = {}
        case ContextChangeEntry():
            changes = _on_context_change(snap, entry)
        case _:
            raise ValueError(f"cannot record an entry of type {type(entry).__name__}")
    calls = changes.get("_turn_calls", snap._turn_calls)
    results = changes.get("_turn_results", snap._turn_results)
    finish_answer = changes.get("_finish_answer", snap._finish_answer)
    last_text = changes.get("_last_reply_text", snap._last_reply_text)
    return dataclasses.replace(
        snap,
        **changes,
        history=snap.history + (entry,),
        pending_calls=tuple(call for call, result in zip(calls, results, strict=True) if result is None),
        answer=finish_answer if finish_answer is not None else last_text,
        created_at=entry.at if snap.created_at is None else snap.created_at,
        updated_at=entry.at,
    )


def _replay(entries: Iterable[HistoryEntry], memo: dict[int, StateSnapshot]) -> StateSnapshot:
    """Folds ``entries`` from an empty snapshot that shares ``memo``."""
    snap = StateSnapshot(_memo=memo)
    for entry in entries:
        snap = fold(snap, entry)
    return snap


def _on_message(snap: StateSnapshot, entry: MessageEntry) -> dict[str, Any]:
    message = entry.message
    changes: dict[str, Any] = {}
    if snap._first_user_message is None and entry.kind == "user":
        changes["_first_user_message"] = message
    # While the model has not seen the last reply (or the calls have no results yet), the message waits: it must
    # not land between a tool call and its result, or before the reply it was written after.
    if snap._waiting or snap.pending_calls:
        changes["_queued"] = snap._queued + (message,)
    else:
        changes["messages"] = snap.messages + (message,)
    return changes


def _on_request(snap: StateSnapshot) -> dict[str, Any]:
    if snap._waiting:
        raise ValueError("a request is already waiting on the model")
    if snap.pending_calls:
        raise ValueError("calls of the last reply have no result yet")
    late = tuple(Message.user(text) for text in snap._late_notices)
    return {"messages": snap.messages + late, "_late_notices": (), "_waiting": True, "turn": snap.turn + 1}


def _on_reply(snap: StateSnapshot, entry: ModelReplyEntry) -> dict[str, Any]:
    if not snap._waiting:
        raise ValueError("no request is waiting for this reply")
    reply = entry.content
    reply_message = Message("assistant", reply.message.content, tokens=reply.context_tokens)
    changes: dict[str, Any] = {
        "_waiting": False,
        "messages": snap.messages + (reply_message,),
        "usage": snap.usage + reply.usage,
    }
    calls = reply.tool_calls
    if calls:
        # The turn stays open until every call has a result; queued messages wait for them.
        changes["_turn_calls"] = calls
        changes["_turn_results"] = (None,) * len(calls)
    else:
        changes["_last_reply_text"] = reply.text
        changes["messages"] = changes["messages"] + snap._queued
        changes["_queued"] = ()
    return changes


def _on_error(snap: StateSnapshot, entry: ErrorEntry) -> dict[str, Any]:
    # A request that fails before its reply changes nothing: the turn is taken back. An error with a call is a
    # tool's, never the model request's.
    if snap._waiting and entry.call is None:
        return {"_waiting": False, "turn": snap.turn - 1, "messages": snap.messages + snap._queued, "_queued": ()}
    return {}


def _pending_slot(snap: StateSnapshot, call_id: str | None) -> int | None:
    """The position in the open turn of the first call with this id that has no result."""
    if call_id is None:
        return None
    for index, (call, result) in enumerate(zip(snap._turn_calls, snap._turn_results, strict=True)):
        if result is None and call.id == call_id:
            return index
    return None


def _on_tool_result(snap: StateSnapshot, entry: ToolResultEntry) -> dict[str, Any]:
    if entry.late:
        # The call was closed already. The model hears of it at the next request.
        text = LATE_RESULT.format(call=format_call(entry.call), content=result_text(entry.content))
        return {"_late_notices": snap._late_notices + (text,)}
    slot = _pending_slot(snap, entry.call.id)
    if slot is None:
        raise ValueError(f"{format_call(entry.call)} is not a pending call, so it cannot get a result")
    call = snap._turn_calls[slot]
    block = ToolResultBlock(call.id, entry.content, name=call.name, is_error=entry.is_error)
    results = snap._turn_results[:slot] + (block,) + snap._turn_results[slot + 1 :]
    changes: dict[str, Any] = {"_turn_results": results}
    if all(result is not None for result in results):
        # The last call closed: the results go in as one message in the order the model asked for them, then the
        # messages that waited.
        blocks = tuple(result for result in results if result is not None)
        changes["messages"] = snap.messages + (Message("user", blocks),) + snap._queued
        changes["_queued"] = ()
        changes["_turn_calls"] = ()
        changes["_turn_results"] = ()
    return changes


def _on_stop(stopped: Stopped | None) -> dict[str, Any]:
    changes: dict[str, Any] = {"stopped": stopped}
    if isinstance(stopped, StoppedByFinish):
        changes["finished"] = True
        if stopped.answer is not None:
            changes["_finish_answer"] = stopped.answer
    return changes


def _on_extra_data(snap: StateSnapshot, entry: ExtraDataEntry) -> dict[str, Any]:
    data = {key: value for key, value in snap.extra_data.items() if key not in entry.removed}
    data.update(entry.content)
    return {"extra_data": FrozenDict(data)}


def _on_context_change(snap: StateSnapshot, entry: ContextChangeEntry) -> dict[str, Any]:
    change = entry.content
    if change.kind == "import":
        if snap._waiting or snap.pending_calls:
            raise ValueError("messages can only be imported between turns")
        changes: dict[str, Any] = {"messages": snap.messages + change.messages}
        if snap._first_user_message is None:
            changes["_first_user_message"] = next((m for m in change.messages if m.role == "user"), None)
        return changes
    if change.kind == "compact":
        if snap.pending_calls:
            raise ValueError("cannot compact while calls are pending")
        if change.summary is None:
            raise ValueError("a compact entry needs its summary")
        if change.kept < 0:
            raise ValueError(f"a compact entry cannot keep {change.kept} messages")
        changes = {"messages": _compacted(snap.messages, snap._first_user_message, change.summary, change.kept)}
        if change.usage is not None:
            changes["usage"] = snap.usage + change.usage
        return changes
    if change.kind == "clear_tool_results":
        return {"messages": _cleared(snap.messages, frozenset(change.cleared))}
    if change.kind == "restore":
        return _on_restore(snap, change)
    raise ValueError(f"unknown context change kind {change.kind!r}")


def _on_restore(snap: StateSnapshot, change: ContextChange) -> dict[str, Any]:
    length = change.restored_to
    if length is None or not 0 <= length <= len(snap.history):
        raise ValueError(f"cannot restore to {length!r}: the history has {len(snap.history)} entries before it")
    target = _restored(snap, length)
    # Everything goes back to its value at that point, except what cannot be undone: the tokens that were spent
    # (usage), and the history, which keeps everything.
    keep = {"history", "usage", "created_at", "updated_at", "pending_calls", "answer", "_memo"}
    return {f.name: getattr(target, f.name) for f in dataclasses.fields(StateSnapshot) if f.name not in keep}


def _restored(snap: StateSnapshot, length: int) -> StateSnapshot:
    """The snapshot of ``snap.history[:length]``. Cached in ``snap._memo``: the restores inside a prefix would
    otherwise be replayed again for every restore after them, which grows exponentially."""
    target = snap._memo.get(length)
    if target is None:
        target = snap if length == len(snap.history) else _replay(snap.history[:length], snap._memo)
        snap._memo[length] = target
    return target


def _compacted(
    messages: Sequence[Message], first: Message | None, summary: str, kept: int
) -> tuple[Message, ...]:
    """The context after ``compact``: the first user message, the summary, and the last ``kept`` messages (the ones
    added while a model was writing the summary, which it never saw)."""
    head = (first,) if first is not None else ()
    tail = tuple(messages[max(len(messages) - kept, 0) :]) if kept > 0 else ()
    return head + (Message.user(SUMMARY_PREFIX + summary),) + tail


def _cleared(messages: Sequence[Message], call_ids: frozenset[str]) -> tuple[Message, ...]:
    """``messages`` with the results of ``call_ids`` blanked. Every message loses its ``tokens`` anchor, which is no
    longer true once content changed."""
    rebuilt = []
    for message in messages:
        blocks = tuple(
            dataclasses.replace(block, content=CLEARED)
            if isinstance(block, ToolResultBlock) and block.call_id in call_ids
            else block
            for block in message.content
        )
        rebuilt.append(Message(message.role, blocks, tokens=None))
    return tuple(rebuilt)


# ---------------------------------------------------------------- the State


def _short(text: Any, limit: int = 60) -> str:
    """Display string shortened to one line."""
    one_line = " ".join(str(text).split())
    return one_line if len(one_line) <= limit else one_line[: limit - 1] + "…"


def _size(content: ToolResultContent) -> str:
    """Display of a result's byte size, text in UTF-8 plus images (``512B``, ``1.2KB``, ``3.4MB``)."""
    return _byte_size(_content_bytes(content))


def _error_text(error: BaseException) -> str:
    message = str(error)
    name = type(error).__name__
    return f"{name}: {message}" if message else name


@dataclass(frozen=True)
class _Batch:
    """What a store does not have yet (``State._unsaved``): JSON-ready entries and the info dict.

    Typed as the ``Store.write`` arguments: stores take plain dicts.
    """

    entries: list[dict[str, Any]]
    info: _serial.InfoDict
    #: How many history entries the store has once this batch is written.
    end: int
    #: The first write of this State: the store must refuse an id it already has.
    create: bool


class State:
    """One conversation's record: the messages the model sees (``messages``) and everything that happened
    (``history``).

    Create it with the messages to start from, pass it to ``agent.run`` to keep it after the run, or to continue it
    after ``Ctrl+C``. Its methods are the only way to change it, and every one is thread-safe, so tools may call
    them from worker threads. Read values with the properties, or take ``snapshot()`` for several values that must
    belong together.

    Example:
        ```python
        state = State(messages=[Message.user("Find the bug in this repo")], extra_data={"repo": "api"})
        agent.run(state)
        print(state.answer, state.stopped, state.usage.cost)
        ```
    """

    def __init__(
        self,
        *args: Never,
        id: str | None = None,
        messages: Iterable[Message] = (),
        extra_data: Mapping[str, Any] | None = None,
        history: Iterable[HistoryEntry] | None = None,
    ) -> None:
        """Start a State, empty or from messages, extra data or a history. Everything is keyword-only.

        Args:
            id: The name a store saves this State under: letters, digits, ``_`` and ``-``, at most 128
                characters. A random id when left out. Continue a saved State with ``store.load(id)``, not
                with a new State of the same id.
            messages: The conversation to start from, as ``Message`` objects. Recorded as one ``import`` entry.
                May be empty: add messages later with ``add_message``.
            extra_data: Your own values (JSON: dicts, lists, str, int, float, bool, None). The model never sees
                them. Recorded as one ``extra_data`` entry.
            history: Entries to rebuild a State from (``other.history``, or what a store loaded). It cannot be
                combined with ``messages`` or ``extra_data``: the history already holds them.

        Raises:
            TypeError: A positional argument, an item of ``messages`` that is not a ``Message``, an
                ``extra_data`` value that is not JSON, an item of ``history`` that is not a history entry, or
                ``history`` together with ``messages`` or ``extra_data``.
            ValueError: ``id`` has other characters, or ``history`` is inconsistent (it gives the index of the
                entry that cannot follow the ones before it).
        """
        if args:
            raise TypeError(
                fix_message(
                    f"State takes keyword arguments only (got the positional argument {args[0]!r})",
                    "give the starting messages with messages=",
                    'State(messages=[Message.user("Find the bug in this repo")])',
                )
            )
        self._id = uuid.uuid4().hex if id is None else check_id(id, "State(id=...)")
        self._lock = threading.RLock()
        # Held for a whole ``edit_extra_data()`` block, so concurrent edits take turns. Never taken inside
        # ``_lock``: the order is always ``_edit_lock``, then ``_lock``.
        self._edit_lock = threading.RLock()

        # Runtime values. They are not history, so they are not in the snapshot and are never saved.
        self._agent: Agent | None = None
        self._running = False
        self._compacting: int | None = None
        self._saved_store: object | None = None
        self._saved_len = 0
        self._saved_agent: AgentInfo | None = None
        # Reserved for subagents (see _child).
        self._parent: State | None = None
        self._root: State = self
        self._depth = 0

        self._snap = StateSnapshot()
        if history is not None:
            self._replay_history(self._check_history_args(history, messages, extra_data))
            return
        self._import(self._check_messages(messages), extra_data)

    # ------------------------------------------------------------ building

    @staticmethod
    def _check_messages(messages: Iterable[Message]) -> tuple[Message, ...]:
        example = 'State(messages=[Message.user("Find the bug in this repo")])'
        if isinstance(messages, (str, bytes, Message)):
            raise TypeError(
                fix_message(
                    f"State(messages=...) takes a list of Message objects (got: {type(messages).__name__})",
                    "wrap each message with Message.user(...), Message.notice(...) or Message.assistant(...)",
                    example,
                )
            )
        try:
            items = tuple(messages)
        except TypeError:
            raise TypeError(
                fix_message(
                    f"State(messages=...) takes a list of Message objects (got: {type(messages).__name__})",
                    "pass a list of messages, wrapping each one with Message.user(...), Message.notice(...) or "
                    "Message.assistant(...)",
                    example,
                )
            ) from None
        for item in items:
            if not isinstance(item, Message):
                raise TypeError(
                    fix_message(
                        f"State(messages=...) takes Message objects only (got: {item!r})",
                        "wrap text with Message.user(...), Message.notice(...) or Message.assistant(...)",
                        example,
                    )
                )
        return items

    @staticmethod
    def _check_history_args(
        history: Iterable[HistoryEntry], messages: Iterable[Message], extra_data: Mapping[str, Any] | None
    ) -> tuple[HistoryEntry, ...]:
        if extra_data is not None or tuple(messages):
            raise TypeError(
                fix_message(
                    "State(history=...) cannot be combined with messages= or extra_data=: the history already holds them",
                    "pass the history alone, or leave it out and build the State from messages=",
                    "State(history=other.history)\nState(messages=[Message.user('...')], extra_data={'repo': 'api'})",
                )
            )
        entries = tuple(history)
        for index, entry in enumerate(entries):
            if not isinstance(entry, _ENTRY_TYPES):
                raise TypeError(
                    fix_message(
                        f"State(history=...) takes history entries (history[{index}] is {entry!r})",
                        "pass state.history, or the entries a store loaded",
                        "State(history=state.history)",
                    )
                )
        return entries

    def _replay_history(self, entries: tuple[HistoryEntry, ...]) -> None:
        snap = self._snap
        for index, entry in enumerate(entries):
            try:
                snap = fold(snap, entry)
            except ValueError as e:
                raise ValueError(
                    fix_message(
                        f"history entry {index} ({entry.kind}) cannot follow the entries before it: {e}",
                        "pass a history that a State recorded (state.history), without leaving entries out",
                        "State(history=state.history)",
                    )
                ) from e
        self._snap = snap

    def _import(self, messages: tuple[Message, ...], extra_data: Mapping[str, Any] | None) -> None:
        """The two entries of ``State(messages=, extra_data=)``: the ``import`` of the messages, then every
        ``extra_data`` key. An empty argument records nothing."""
        data: dict[str, Any] = {}
        if extra_data is not None:
            if not isinstance(extra_data, Mapping):
                raise TypeError(
                    fix_message(
                        f"State(extra_data=...) takes a dict (got: {type(extra_data).__name__})",
                        "pass your values as a dict with string keys",
                        'State(extra_data={"repo": "api"})',
                    )
                )
            data = _json(extra_data, "State(extra_data=...)")
        if messages:
            change = ContextChange(
                "import", 0, _tokens.context_tokens(messages, 0), messages=messages
            )
            self._commit(ContextChangeEntry(content=change, turn=0))
        if data:
            self._commit(ExtraDataEntry(content=data, turn=0))

    # ------------------------------------------------------------ values (read-only)

    # Each property is one atomic read of the current snapshot, so none of them takes the lock. For several values
    # that must belong together, read them from one ``snapshot()``.

    @property
    def id(self) -> str:
        """The name a store saves this State under. Given with ``State(id=...)``, or random."""
        return self._id

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
        """Everything that happened, oldest first: messages, model requests and replies, tool results, errors,
        context changes, runs and stops. Entries are only ever added, so shrinking the context never removes
        anything from here.

        Each entry is one of the ``HistoryEntry`` classes (``ModelReplyEntry``, ``ToolResultEntry``, ...). Checking
        ``kind`` tells which, so type checkers know what ``content`` is.

        Example:
            ```python
            for entry in state.history:
                if entry.kind == "model_reply":
                    print(entry.content.tool_calls)  # a Reply
                elif entry.kind == "tool_result":
                    print(entry.call.name, entry.outcome)
            ```
        """
        return self._snap.history

    @property
    def messages(self) -> tuple[Message, ...]:
        """The messages the model will see at the next ``think``.

        Results of this turn's tool calls go in together once the last call has a result. Messages added in the
        meantime, or while ``think`` waits on the model, go in after them.
        """
        return self._snap.messages

    @property
    def pending_calls(self) -> tuple[ToolCall, ...]:
        """Tool calls the model requested this turn that have no result yet, in request order.

        A call leaves this tuple when it gets a result, is denied or cancelled by a permission, or is closed by an
        exception.
        """
        return self._snap.pending_calls

    @property
    def turn(self) -> int:
        """Number of turns so far: each ``think`` adds 1 while it waits on the model, and a ``think`` that fails
        takes it back."""
        return self._snap.turn

    @property
    def usage(self) -> Usage:
        """Total tokens, requests and cost of every ``think``, ``ask`` and ``compact`` on this State. ``restore``
        does not lower it."""
        return self._snap.usage

    @property
    def extra_data(self) -> FrozenDict[str, Any]:
        """A read-only dict for your own values (JSON). The model never sees it.

        Change it with ``edit_extra_data()``: ``state.extra_data["x"] = 1`` raises ``TypeError``.
        """
        return self._snap.extra_data

    @property
    def finished(self) -> bool:
        """Whether ``finish()`` was called. A finished State stops its loop before the next turn, and cannot run
        again."""
        return self._snap.finished

    @property
    def answer(self) -> Any:
        """The run's answer.

        The value given to ``finish(answer)`` if there is one, otherwise the text of the latest model reply
        without tool calls, otherwise ``None``. A ``think`` that fails does not change it.
        """
        return self._snap.answer

    @property
    def stopped(self) -> Stopped | None:
        """Why this run is set to stop, or ``None`` until something decides that.

        - ``StoppedByFinish(answer)``: ``finish()`` was called (set right away)
        - ``StoppedByPermission(call, permission)``: a permission denied ``call`` with ``stop=True`` (set right
          away, when ``use_tools`` records the results)
        - ``StoppedByUntil(name)``: an ``until`` function returned true, e.g.
          ``StoppedByUntil("waiting_for_user")`` (set when ``@loop`` checks it before a turn)
        - ``StoppedByLimit(turns)``: the loop ran ``limit`` turns in this call; ``turns`` is the limit (set when
          ``@loop`` checks it before a turn)

        ``run``/``arun`` reset it to ``None`` at the start, and it is ``None`` after a run that ended with an
        exception. A ``@loop`` stops before its next turn once it is set. A loop written without ``@loop`` checks
        it itself. A ``@loop`` called inside such a loop leaves its ``until``/limit reason set, so a loop without
        ``@loop`` that calls one checks only for ``StoppedByFinish``/``StoppedByPermission``.

        Example:
            ```python
            agent.run(state)
            if isinstance(state.stopped, StoppedByLimit):
                print(f"gave up after {state.stopped.turns} turns")

            def my_loop(agent, state):
                while state.stopped is None and not state.finished:
                    agent.think(state)
                    if state.pending_calls:
                        agent.use_tools(state)
            ```
        """
        return self._snap.stopped

    @property
    def created_at(self) -> datetime | None:
        """When the first thing was recorded: the ``at`` of the first history entry, in UTC. ``None`` for a State
        with no history yet."""
        return self._snap.created_at

    @property
    def updated_at(self) -> datetime | None:
        """When something was last recorded: the ``at`` of the latest history entry, in UTC. ``None`` for a State
        with no history yet.

        Anything added to ``history`` moves it, including changes to ``extra_data``.
        """
        return self._snap.updated_at

    def snapshot(self) -> StateSnapshot:
        """The current value of this State: a frozen ``StateSnapshot`` that never changes, and that
        ``restore`` can go back to.

        Example:
            ```python
            before = state.snapshot()
            state.add_message(Message.user("Try the other approach"))
            state.restore(before)
            ```
        """
        return self._snap

    # ------------------------------------------------------------ commands

    def add_message(self, message: Message) -> None:
        """Add a user message or a notice to the conversation, for example the next request in a chat loop.

        While tool calls are pending, or while ``think`` waits on the model, the message is held and goes into
        ``messages`` right after the results or the reply. A loop waiting for the user stays waiting until the
        model answers it.

        Args:
            message: ``Message.user(text, *images)``, or ``Message.notice(text)`` to tell the model something from
                your loop or a tool (a user message whose text starts with ``"[notice] "``).

        Raises:
            TypeError: ``message`` is not a ``Message``.
            ValueError: ``message`` is not a user message, has blocks other than text and images, or has no text
                and no image.

        Example:
            ```python
            state.add_message(Message.user("Now fix it"))
            state.add_message(Message.notice("The tests failed"))
            ```
        """
        example = 'state.add_message(Message.user("Next step: run the tests"))'
        if not isinstance(message, Message):
            raise TypeError(
                fix_message(
                    f"add_message() takes a Message (got: {type(message).__name__})",
                    "wrap the text with Message.user(...) or Message.notice(...)",
                    example,
                )
            )
        if message.role != "user":
            raise ValueError(
                fix_message(
                    f"add_message() takes user messages only (got a {message.role} message)",
                    "assistant messages come from the model; to start from a conversation that has them, use "
                    "State(messages=[...])",
                    example,
                )
            )
        # A notice's text starts with the prefix ``Message.notice`` added, so look at what follows it.
        text = message.text
        if text.startswith(NOTICE_PREFIX):
            text = text[len(NOTICE_PREFIX) :]
        if not text.strip() and not any(isinstance(block, Image) for block in message.content):
            raise ValueError(
                fix_message(
                    f"add_message() got a message with no text and no image (text: {message.text!r}). Providers "
                    "reject empty messages",
                    "pass a message with text (if the human's input was empty, ask again)",
                    example,
                )
            )
        with self._lock:
            before = len(self._snap.messages)
            self._commit(MessageEntry.from_message(message, turn=self._snap.turn))
            # A message that went into ``messages`` while a model writes a summary is one the model never saw, so
            # ``compact`` keeps it after the summary (see ``_begin_compact``). A held message is not counted: it
            # goes in later, after the summary.
            if self._compacting is not None and len(self._snap.messages) > before:
                self._compacting += 1

    def finish(self, answer: Any = None) -> None:
        """End the run. ``state.stopped`` becomes ``StoppedByFinish(answer)`` right away, and the loop stops
        before its next turn.

        Tools may call it (a ``submit`` tool, for example), even while other calls are pending. Calling it again
        is allowed; the last ``answer`` given wins.

        Args:
            answer: If not ``None``, becomes ``state.answer``. Any JSON value, a Pydantic model or a dataclass
                (turned into a dict), not only a string. ``None`` keeps an earlier answer.

        Raises:
            TypeError: ``answer`` cannot be turned into JSON.
        """
        value = _serial.answer(answer, "the finish() answer")
        with self._lock:
            self._commit(StopEntry(content=StoppedByFinish(answer=value), turn=self._snap.turn))

    def compact(self, summary: str) -> None:
        """Replace the messages with the first user message and ``summary``. History keeps everything.

        Use it with a summary you wrote or got from ``agent.ask``. ``agent.compact(state)`` asks the model for the
        summary.

        Args:
            summary: What the model should know about the work so far.

        Raises:
            TypeError: ``summary`` is not a string.
            ValueError: Tool calls are pending, or ``summary`` is empty or whitespace only.
        """
        self._compact(summary, usage=None, in_flight=False)

    def clear_tool_results(self, keep_last: int = 5) -> None:
        """Shrink the context by blanking old tool results. History keeps the full results.

        Every tool result except the last ``keep_last`` is replaced with ``"(cleared: kept in history)"``.
        Results that are already cleared still count toward ``keep_last``. Nothing is recorded when nothing
        changes.

        Args:
            keep_last: How many of the most recent tool results to keep.

        Raises:
            ValueError: Tool calls are pending, or ``keep_last`` is not an integer >= 0.
        """
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
            snap = self._snap
            blocks = [
                block for message in snap.messages for block in message.content if isinstance(block, ToolResultBlock)
            ]
            doomed = blocks[: max(len(blocks) - keep_last, 0)]
            # By call id, which is what the entry records; ids are unique within a conversation unless a server
            # gives none.
            cleared = tuple(dict.fromkeys(block.call_id for block in doomed if block.content != CLEARED))
            if not cleared:
                return
            after = _cleared(snap.messages, frozenset(cleared))
            change = ContextChange(
                "clear_tool_results",
                _tokens.context_tokens(snap.messages, 0),
                _tokens.context_tokens(after, 0),
                cleared=cleared,
            )
            self._commit(ContextChangeEntry(content=change, turn=snap.turn))
        self._notify_context_change(change)

    @contextlib.contextmanager
    def edit_extra_data(self) -> Iterator[dict[str, Any]]:
        """Change ``extra_data`` inside a ``with`` block. It gives you an editable copy (plain dicts and lists), and
        when the block ends normally the changes are recorded as one history entry.

        Edits take turns: a lock held for the whole block makes two threads' edits run one after the other, so
        ``data["n"] = data.get("n", 0) + 1`` never loses an update. If the block raises, nothing changes and the
        exception propagates. If nothing changed, nothing is recorded.

        Raises:
            TypeError: At the end of the block, a value is not JSON (a set, an object, a non-string key, NaN). The
                message names the key.

        Example:
            ```python
            with state.edit_extra_data() as data:
                data.setdefault("notes", {})["bug"] = "b.py"
                data.pop("draft", None)
            ```
        """
        with self._edit_lock:
            before = self._snap.extra_data
            data: dict[str, Any] = thaw(before)
            yield data
            after = _json(data, "state.extra_data")
            changed = {key: value for key, value in after.items() if key not in before or not _same(before[key], value)}
            removed = tuple(key for key in before if key not in after)
            if changed or removed:
                with self._lock:
                    self._commit(ExtraDataEntry(content=changed, removed=removed, turn=self._snap.turn))

    def fork(self) -> State:
        """A new State with the same history and so the same content, a new id, and nothing shared with this one:
        no store, no running Agent. Use it to try something on a copy.

        Example:
            ```python
            attempt = state.fork()
            agent.run(attempt)  # `state` is untouched
            ```
        """
        return State(history=self.history)

    def restore(self, snapshot: StateSnapshot) -> None:
        """Go back to ``snapshot``, a ``state.snapshot()`` taken earlier from this State.

        Every value goes back to what it was then (``messages``, ``turn``, ``extra_data``, ``finished``,
        ``answer``, ``stopped``), except ``usage``, which stays cumulative (the tokens were spent), and
        ``history``, which keeps everything and gets one ``context_change`` entry saying the State went back.

        Args:
            snapshot: A snapshot of this State, taken between turns.

        Raises:
            TypeError: ``snapshot`` is not a ``StateSnapshot``.
            ValueError: ``snapshot`` is not from this State, tool calls are pending, the model is being waited on,
                a compaction is under way, or the snapshot was taken while the model was being waited on.
        """
        if not isinstance(snapshot, StateSnapshot):
            raise TypeError(
                fix_message(
                    f"restore() takes a StateSnapshot (got: {type(snapshot).__name__})",
                    "pass what state.snapshot() returned",
                    "before = state.snapshot()\n...\nstate.restore(before)",
                )
            )
        with self._lock:
            snap = self._snap
            length = len(snapshot.history)
            if length > len(snap.history) or snapshot.history != snap.history[:length]:
                raise ValueError(
                    fix_message(
                        "restore takes a snapshot taken from this State (its history is not the start of this "
                        "State's history)",
                        "for a snapshot of another State, build a new State from it: State(history=snapshot.history)",
                        "copy = State(history=snapshot.history)",
                    )
                )
            self._ensure_no_pending("restore")
            if snap._waiting or self._compacting is not None:
                what = "the model is being waited on" if snap._waiting else "a compaction is under way"
                raise ValueError(
                    fix_message(
                        f"cannot call restore() while {what}",
                        "wait for the think or compact to end, then restore",
                        "agent.think(state)\nstate.restore(before)",
                    )
                )
            target = _restored(snap, length)
            if target._waiting:
                raise ValueError(
                    fix_message(
                        "this snapshot was taken while the model was being waited on, so there is no complete "
                        "turn to go back to",
                        "take snapshots between turns (before think, or after use_tools)",
                        "before = state.snapshot()  # not inside a think",
                    )
                )
            change = ContextChange(
                "restore",
                _tokens.context_tokens(snap.messages, 0),
                _tokens.context_tokens(target.messages, 0),
                restored_to=length,
            )
            self._commit(ContextChangeEntry(content=change, turn=snap.turn))
        self._notify_context_change(change)

    # ------------------------------------------------------------ display

    def __str__(self) -> str:
        """Several lines summarizing history for humans. One line per entry; long content is shortened.

        E.g. ``[turn 0] user: Find the bug`` / ``[turn 1] model_reply: read_file(path="main.py")`` /
        ``[turn 1] tool_result read_file: 1.2KB`` / ``[turn 1] tool_result fetch_url (error): HTTP 404: …`` /
        ``done: stopped by is_answered (2 turns)``
        """
        snap = self._snap
        lines = [_describe_entry(entry) for entry in snap.history]
        if snap.stopped is not None:
            lines.append(f"done: {snap.stopped} ({_turns(snap.turn)})")
        elif snap.finished:
            lines.append(f"done: stopped by finish ({_turns(snap.turn)})")
        return "\n".join(lines)

    def __repr__(self) -> str:
        snap = self._snap
        extra = " finished" if snap.finished else ""
        return (
            f"<State id={self._id!r} turn={snap.turn} history={len(snap.history)} "
            f"messages={len(snap.messages)} pending={len(snap.pending_calls)}{extra}>"
        )

    # ================================================================ internal (Agent, loop, runner, Store)
    #
    # Every change goes through ``_commit(entry)``. The methods below validate, build the entry for one fact, and
    # commit it, inside the lock. The Agent decides when to call them; none of them calls the model, tools, a Human,
    # a Reporter, a Permission or a Store.

    def _commit(self, entry: HistoryEntry) -> None:
        """The one way anything changes: ``self._snap = fold(self._snap, entry)``. ``ValueError`` (from ``fold``)
        for an entry that cannot follow the current snapshot, with nothing changed."""
        with self._lock:
            self._snap = fold(self._snap, entry)

    # ------------------------------------------------------------ running

    def _attach(self, agent: Agent) -> None:
        """Called by Agent before it thinks, runs tools or compacts: ``agent`` becomes the Agent this State
        notifies (``on_context_change``) and the permissions ask (``DecideByHuman``). Any Agent may use any State, so
        this is only a link to the latest one, not an owner."""
        self._agent = agent

    def _start_run(self, agent: Agent, info: AgentInfo) -> None:
        """Start of ``Agent.run``/``arun``, after the checks that need no lock: the one-run-at-a-time guard, the
        link to ``agent`` (``_attach``) and ``RunStartEntry(info)``, which clears ``stopped``, as one change.

        Every call that returned is paired with ``_end_run``.

        Raises:
            ValueError: This State is already being run (by any Agent, on any thread).
        """
        with self._lock:
            if self._running:
                raise ValueError(
                    fix_message(
                        "this State is already being run by another run() or arun() call, and a State runs one "
                        "at a time",
                        "wait for that run to end, or run a copy: state.fork()",
                        "attempt = state.fork()\nagent.run(attempt)",
                    )
                )
            self._commit(RunStartEntry(content=info, turn=self._snap.turn))
            self._running = True
            self._agent = agent

    def _end_run(self) -> None:
        """End of a run, whether it ended or raised (also when the first save of the run failed): the State may be run
        again. The link to the Agent stays, so later notifications still reach its Reporter."""
        with self._lock:
            self._running = False

    def _take_saved_agent(self) -> AgentInfo | None:
        """Called by ``Agent.run`` at start: the Agent that last ran this State before it was saved, only for a State
        a Store loaded (its last ``RunStartEntry``), and only once. It clears the value, so the comparison
        (``ResumeWarning``) happens at the first run after loading and never after that."""
        with self._lock:
            info, self._saved_agent = self._saved_agent, None
            return info

    def _set_saved_agent(self, info: AgentInfo | None) -> None:
        """Puts back what ``_take_saved_agent`` returned, when the run did not start (a warnings filter made the
        ``ResumeWarning`` an exception), so the next run compares again."""
        with self._lock:
            self._saved_agent = info

    # ------------------------------------------------------------ checks

    def _ensure_open(self, action: str) -> None:
        """``ValueError`` after ``finish()``: ``{action}`` cannot be called after ``finish()``.
        Fix: if a block called ``finish()``, ``return`` from the loop body. (Called by think, use_tools, ask)
        """
        if not self._snap.finished:
            return
        raise ValueError(
            fix_message(
                f"cannot call {action}() on a finished State (finish() ends the whole run)",
                "if a block called state.finish(), return from the loop body right away",
                "submit(agent, state)\nif state.finished:\n    return\nagent.think(state)",
            )
        )

    def _ensure_no_pending(self, action: str) -> None:
        """``ValueError`` if there are pending calls: ``{action}`` was called while calls (list of names) are
        pending. Fix: run them first with ``agent.use_tools(state)``.
        No side effects.
        """
        pending = self._snap.pending_calls
        if not pending:
            return
        names = ", ".join(format_call(call) for call in pending)
        raise ValueError(
            fix_message(
                f"cannot call {action}() while calls are pending ({names})",
                "run them first with agent.use_tools(state) (its permissions decide which ones run)",
                "agent.think(state)\nif state.pending_calls:\n    agent.use_tools(state)",
            )
        )

    # ------------------------------------------------------------ think

    def _begin_think(self, model: str) -> None:
        """Start of ``think``, right before the request is sent: ``ModelRequestEntry(model)``. The late-result
        notices become messages, ``turn`` goes up by one, and messages added from now on are held until the reply or
        the failure.

        ``ValueError`` (nothing recorded) if the State is finished, has pending calls, is already waiting on a model,
        or has no messages.
        """
        with self._lock:
            self._ensure_open("think")
            self._ensure_no_pending("think")
            snap = self._snap
            if snap._waiting:
                raise ValueError(
                    fix_message(
                        "cannot call think() while another think() is waiting on the model",
                        "wait for it to end, or think on a copy: state.fork()",
                        "agent.think(state)\nagent.think(state)  # one after the other",
                    )
                )
            if not snap.messages:
                raise ValueError(
                    fix_message(
                        "cannot call think() on a State with no messages (the model would get an empty request)",
                        "start the State with messages, or add one first",
                        'state = State(messages=[Message.user("Find the bug")])\n'
                        '# or\nstate.add_message(Message.user("Find the bug"))',
                    )
                )
            self._commit(ModelRequestEntry(content=model, turn=snap.turn + 1))

    def _record_reply(self, reply: Reply) -> None:
        """The reply of the request ``_begin_think`` recorded: ``ModelReplyEntry(reply)``. The reply message joins
        ``messages`` with its token anchor, ``usage`` grows, and its tool calls (if any) become ``pending_calls``.
        With no tool calls, held messages go in right after it. Nothing here can undo it afterwards: an exception
        raised after this (``on_think_end``) is recorded with ``_record_error`` and leaves the reply in place.

        ``ValueError`` if no request is waiting.
        """
        with self._lock:
            self._commit(ModelReplyEntry(content=reply, turn=self._snap.turn))

    def _record_model_event(self, event: ModelEvent) -> None:
        """``ModelEventEntry(event)``. Nothing else changes."""
        with self._lock:
            self._commit(ModelEventEntry(content=event, turn=self._snap.turn))

    # ------------------------------------------------------------ tool results

    def _tool_result(
        self,
        call: ToolCall,
        content: ToolResultContent,
        outcome: ToolOutcomeKind,
        *,
        error: BaseException | None = None,
        if_pending: bool = False,
    ) -> bool:
        """The result of one call of the open turn: ``ToolResultEntry(content, call, outcome, error)``. Safe to call
        from multiple threads. This is how every way a call ends is recorded: ``DONE``, ``ERROR``, ``INPUT_ERROR``,
        ``ABORTED``, ``INTERRUPTED``, and ``DENIED``/``CANCELLED`` for a permission's refusal (``content`` is the
        text the model gets).

        When it was the last pending call, the results go into ``messages`` as one user message in the order the
        model requested the calls, followed by the messages that were held.

        Returns ``True`` if recorded. If ``call`` (compared by id) is not pending: ``False`` with ``if_pending``
        (the check and the record are one step, so nothing can close the call in between), else ``ValueError``.
        """
        with self._lock:
            snap = self._snap
            slot = _pending_slot(snap, getattr(call, "id", None))
            if slot is None:
                if if_pending:
                    return False
                shown = format_call(call) if isinstance(call, ToolCall) else repr(call)
                raise ValueError(f"cannot record a result for a call that is not pending: {shown}")
            self._commit(
                ToolResultEntry(
                    content=content, call=snap._turn_calls[slot], outcome=outcome, error=error, turn=snap.turn
                )
            )
            return True

    def _late_result(
        self,
        call: ToolCall,
        content: ToolResultContent,
        outcome: ToolOutcomeKind,
        *,
        error: BaseException | None = None,
    ) -> None:
        """The result of a sync tool that was closed without waiting on interrupt arrives later (called from the
        worker thread).

        - If the ``call`` **object itself** (``is`` comparison) is still in ``pending_calls`` (the loop caught the
          interrupt), it is recorded like ``_tool_result``. Not compared by id: a call in the next turn can reuse
          the same id (servers that do not give ids), and last turn's result must not attach to it.
        - Otherwise ``ToolResultEntry(late=True)``: ``messages`` get the notice ``"[notice] interrupted
          {call} finished later: {content}"`` at the next ``_begin_think``.
        """
        with self._lock:
            snap = self._snap
            still_pending = any(
                result is None and pending is call for pending, result in zip(snap._turn_calls, snap._turn_results, strict=True)
            )
            self._commit(
                ToolResultEntry(
                    content=content, call=call, outcome=outcome, late=not still_pending, error=error, turn=snap.turn
                )
            )

    def _stop_turn(self, call: ToolCall, reason: str, permission: str) -> list[ToolCall]:
        """A permission denied ``call`` with ``stop=True``. As one change under the lock: ``call`` gets ``reason``
        (``DENIED``), every other pending call of the turn gets ``STOPPED_TURN`` (``CANCELLED``), and then
        ``StopEntry(StoppedByPermission(call, permission))`` is recorded, unless ``finish()`` already ended the State
        (finish wins). Returns the cancelled calls, in request order.

        ``ValueError`` (nothing recorded) if ``call`` is not pending.
        """
        with self._lock:
            snap = self._snap
            slot = _pending_slot(snap, getattr(call, "id", None))
            if slot is None:
                shown = format_call(call) if isinstance(call, ToolCall) else repr(call)
                raise ValueError(f"cannot deny a call that is not pending: {shown}")
            self._commit(
                ToolResultEntry(
                    content=reason, call=snap._turn_calls[slot], outcome=ToolOutcomeKind.DENIED, turn=snap.turn
                )
            )
            cancelled = list(self._snap.pending_calls)
            for other in cancelled:
                self._commit(
                    ToolResultEntry(
                        content=STOPPED_TURN, call=other, outcome=ToolOutcomeKind.CANCELLED, turn=self._snap.turn
                    )
                )
            if not self._snap.finished:
                self._commit(StopEntry(content=StoppedByPermission(call, permission), turn=self._snap.turn))
            return cancelled

    def _close_pending(self, error: BaseException) -> list[ToolCall]:
        """Close all pending calls when an exception leaves ``run()``.

        Records each one with ``closed_result(error)`` and the outcome ``closed_outcome(error).kind``
        (``INTERRUPTED`` or ``ABORTED``), in request order. Returns the list of closed calls, or an empty list if
        there were none.
        """
        text = closed_result(error)
        kind = closed_outcome(error).kind
        with self._lock:
            closed = list(self._snap.pending_calls)
            for call in closed:
                self._commit(
                    ToolResultEntry(content=text, call=call, outcome=kind, error=error, turn=self._snap.turn)
                )
            return closed

    def _record_error(self, error: BaseException, call: ToolCall | None = None) -> None:
        """``ErrorEntry`` (content=``f"{type(error).__name__}: {error}"``, error=error, call=call).

        If the model request of ``_begin_think`` is waiting (no reply came) and ``call`` is ``None``, this is what
        takes the request back: ``turn`` goes down by one, held messages go in, and the context is as before the
        ``think``. Recorded after a reply, it takes nothing back.

        If the same exception object (``is``) is already an ``error`` entry, it is not added again (so an
        exception recorded by think/use_tools is not recorded again by ``run``). An exception with an empty
        message (``KeyboardInterrupt()``) records only its name.
        """
        with self._lock:
            for entry in self._snap.history:
                if entry.kind == "error" and entry.error is error:
                    return
            self._commit(ErrorEntry(content=_error_text(error), error=error, call=call, turn=self._snap.turn))

    # ------------------------------------------------------------ questions

    def _record_ask(self, question: str, answer: Any, usage: Usage | None = None) -> None:
        """``ExchangeEntry("ask", Exchange(question, answer))``: the answer is turned into JSON (a Pydantic model or
        dataclass becomes a dict; anything else that is not JSON is recorded as its ``repr``, so a paid answer is never
        lost to a recording error). ``usage`` is the usage of the model requests behind it (all the retries), which
        the State adds to its own. The context does not change."""
        with self._lock:
            self._commit(
                ExchangeEntry(
                    kind="ask", content=Exchange(question, _json_answer(question, answer)), usage=usage, turn=self._snap.turn
                )
            )

    def _record_human(self, question: str, answer: Any) -> None:
        """``ExchangeEntry("human", Exchange(question, answer))``, with no usage. The context does not change."""
        with self._lock:
            self._commit(
                ExchangeEntry(
                    kind="human", content=Exchange(question, _json_answer(question, answer)), turn=self._snap.turn
                )
            )

    def _context_for_question(self) -> tuple[Message, ...]:
        """A copy of the messages for ``ask``. Does not change anything.

        With no pending calls, ``messages`` as is. Otherwise one more user message is appended: for every call this
        turn, in request order, its result if it has one, otherwise a ``ToolResultBlock`` with ``"(not run yet)"``.
        (Providers reject a tool_use that has no result after it.) Held messages are left out.
        """
        snap = self._snap
        if not snap.pending_calls:
            return snap.messages
        blocks = tuple(
            result if result is not None else ToolResultBlock(call.id, NOT_RUN_YET, name=call.name)
            for call, result in zip(snap._turn_calls, snap._turn_results, strict=True)
        )
        return snap.messages + (Message("user", blocks),)

    # ------------------------------------------------------------ compaction

    def _begin_compact(self) -> None:
        """Start of ``Agent.compact``, right before it reads ``messages``: ``ValueError`` with pending calls or if a
        compaction is already under way. Until ``_end_compact``, ``add_message`` counts the messages that go into
        ``messages``: the model will not see them, so ``_compact_done`` keeps them after the summary. Every call is
        paired with ``_end_compact``."""
        with self._lock:
            self._ensure_no_pending("compact")
            if self._compacting is not None:
                raise ValueError(
                    fix_message(
                        "cannot call compact() while another compaction of this State is under way",
                        "wait for it to end",
                        "agent.compact(state)\nagent.compact(state)  # one after the other",
                    )
                )
            self._compacting = 0

    def _end_compact(self) -> None:
        """End of ``Agent.compact``, success or not: stop counting. ``_compact_done`` already did it on success; on
        failure the messages are already in ``messages`` and nothing else is needed."""
        with self._lock:
            self._compacting = None

    def _compact_done(self, summary: str, usage: Usage | None) -> None:
        """The model wrote ``summary`` (``Agent.compact``): the same change as ``compact(summary)`` with ``usage``
        (the usage of that request, which the State adds to its own) and ``kept`` = the number of messages
        ``add_message`` added since ``_begin_compact``."""
        self._compact(summary, usage=usage, in_flight=True)

    def _compact(self, summary: str, *, usage: Usage | None, in_flight: bool) -> None:
        _check_text(
            "compact",
            "summary",
            summary,
            'summary = agent.ask(state, "Summarize the work so far")\nstate.compact(summary)',
        )
        with self._lock:
            self._ensure_no_pending("compact")
            snap = self._snap
            kept = (self._compacting or 0) if in_flight else 0
            if not in_flight and self._compacting is not None:
                # A summary written by hand while a model is writing one: what the model's summary will be put
                # after is this new context, so only messages added from now on are kept.
                self._compacting = 0
            after = _compacted(snap.messages, snap._first_user_message, summary, kept)
            change = ContextChange(
                "compact",
                _tokens.context_tokens(snap.messages, 0),
                _tokens.context_tokens(after, 0),
                summary=summary,
                kept=kept,
                usage=usage,
            )
            self._commit(ContextChangeEntry(content=change, turn=snap.turn))
            if in_flight:
                self._compacting = None
        self._notify_context_change(change)

    # ------------------------------------------------------------ stopping

    def _record_stop(self, stopped: Stopped | None) -> None:
        """``StopEntry(stopped)``: called when Loop stops on an ``until`` function or its limit, with why (read back
        as ``stopped``). ``None`` clears ``stopped``: Loop records it when it goes on after a nested loop stopped on
        its own ``until``/limit."""
        with self._lock:
            self._commit(StopEntry(content=stopped, turn=self._snap.turn))

    def _clear_stop(self) -> None:
        """``stopped`` goes back to ``None``, with a ``StopEntry(None)`` if it was set. Called by ``Agent.run``/``arun``
        when a run raises (a run ending in an exception has no reason). A run clears it at its start with
        ``RunStartEntry`` (``_start_run``)."""
        with self._lock:
            if self._snap.stopped is not None:
                self._commit(StopEntry(content=None, turn=self._snap.turn))

    # ------------------------------------------------------------ saving (Store only)

    def _unsaved(self, store: object) -> _Batch | None:
        """What ``store`` does not have yet, as JSON values, or ``None`` if it has everything. Called by
        ``Store.save``, outside any State method (the write happens after the lock is released).

        - ``ValueError`` if another store saved this State first (a State is saved in one store only). Stores are
          compared with ``==``, so two ``FileStore`` objects for the same folder are the same store.
        - ``entries``: history from the first entry the store does not have, each with its index as ``seq``.
        - ``info``: the small dict a store lists States from (``_serial.info_to_dict``).
        - ``create``: this State was never saved, so the store must refuse an id it already has.
        - ``TypeError`` (from ``_serial``) if a value cannot be saved as JSON.
        """
        with self._lock:
            if self._saved_store is not None and self._saved_store != store:
                raise ValueError(
                    fix_message(
                        f"State {self._id!r} is saved in another store ({self._saved_store!r}), and a State is "
                        f"saved in one store only (this one is {store!r})",
                        "save it in the store it was saved in, or start a new State (state.fork() keeps the content)",
                        'state = store.load("task-1")\nstore.save(state)',
                    )
                )
            snap = self._snap
            end = len(snap.history)
            create = self._saved_store is None
            if end == self._saved_len and not create:
                return None
            entries = [_serial.entry_to_dict(snap.history[i], i) for i in range(self._saved_len, end)]
            first = snap._first_user_message
            info = _serial.info_to_dict(
                first_message=(first.text or None) if first is not None else None,
                created_at=snap.created_at,
                updated_at=snap.updated_at,
                turn=snap.turn,
                stopped=snap.stopped,
                finished=snap.finished,
            )
            return _Batch(entries, info, end, create)

    def _mark_saved(self, store: object, batch: _Batch) -> None:
        """``store`` now has ``batch``. Only called after the write succeeded, so a failed write is sent again at
        the next save (stores skip entries they already have by ``seq``)."""
        with self._lock:
            self._saved_store = store
            self._saved_len = max(self._saved_len, batch.end)

    @classmethod
    def _from_store(cls, state_id: str, entries: Sequence[HistoryEntry], store: object) -> State:
        """Rebuilds a saved State from the entries a Store read (``Store.load``). The result is bound to ``store``,
        which already has all of ``entries``, and its first run is compared with the Agent of its last
        ``RunStartEntry`` (``ResumeWarning``).

        Left in the middle of a turn, it is closed with entries that are facts and get saved with the next save:
        waiting on the model gets ``ErrorEntry(STOPPED_WAITING)`` (which takes the request back), and every call
        without a result gets ``ToolResultEntry(outcome=ABORTED, content=UNSAVED_RESULT)``, in request order. The
        tool is never run again.

        ``ValueError`` if the entries cannot be replayed.
        """
        try:
            state = cls(id=state_id, history=entries)
        except ValueError as e:
            raise ValueError(
                fix_message(
                    f"saved State {state_id!r} is damaged: {e}",
                    "restore the store's data from a backup, or start a new State",
                )
            ) from e
        state._saved_store = store
        state._saved_len = len(entries)
        for entry in reversed(entries):
            if isinstance(entry, RunStartEntry):
                state._saved_agent = entry.content
                break
        with state._lock:
            if state._snap._waiting:
                state._commit(ErrorEntry(content=STOPPED_WAITING, turn=state._snap.turn))
            for call in state._snap.pending_calls:
                state._commit(
                    ToolResultEntry(
                        content=UNSAVED_RESULT,
                        call=call,
                        outcome=ToolOutcomeKind.ABORTED,
                        turn=state._snap.turn,
                    )
                )
        return state

    @classmethod
    def _child(cls, parent: State, messages: Iterable[Message] = ()) -> State:
        """State for a subagent (extension). ``parent``, ``root=parent.root``, ``depth=parent.depth+1``."""
        child = cls(messages=messages)
        child._parent = parent
        child._root = parent.root
        child._depth = parent.depth + 1
        return child

    # ------------------------------------------------------------ helpers

    def _notify_context_change(self, change: ContextChange) -> None:
        """Called outside the lock: tells the Reporter of the Agent linked to this State (if any)."""
        agent = self._agent
        # agent.reporter may create the default Terminal, so it is read here, outside the lock.
        reporter = agent.reporter if agent is not None else None
        if reporter is not None:
            reporter.on_context_change(self, change)


#: The classes a history entry may be an instance of (the members of ``HistoryEntry``).
_ENTRY_TYPES = (
    MessageEntry,
    ModelRequestEntry,
    ModelReplyEntry,
    ModelEventEntry,
    ToolResultEntry,
    ExchangeEntry,
    ContextChangeEntry,
    RunStartEntry,
    StopEntry,
    ExtraDataEntry,
    ErrorEntry,
)


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


def _json(value: Any, where: str) -> Any:
    """A plain deep copy of ``value`` if it is JSON (dict with string keys, list or tuple, str, int, float, bool,
    None), else a ``TypeError`` that names ``where``."""
    example = "with state.edit_extra_data() as data:\n    data['seen'] = sorted(seen)  # a list, not a set"
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(
                fix_message(
                    f"{where} must be JSON (got {value!r}, which JSON cannot hold)",
                    "use a finite number, or None for \"no value\"",
                    example,
                )
            )
        return value
    if isinstance(value, Mapping):
        copy: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    fix_message(
                        f"{where} must be JSON, and its key {key!r} is a {type(key).__name__} (JSON keys are strings)",
                        "use str(key) as the key",
                        example,
                    )
                )
            copy[key] = _json(item, f"{where}[{key!r}]")
        return copy
    if isinstance(value, (list, tuple)):
        return [_json(item, f"{where}[{i}]") for i, item in enumerate(value)]
    hint = {
        "set": "use a list: sorted(seen)",
        "frozenset": "use a list: sorted(seen)",
        "bytes": "use text: base64.b64encode(data).decode()",
        "datetime": "use value.isoformat()",
    }.get(type(value).__name__, "turn it into a dict, list, str, int, float, bool or None first (a dataclass: dataclasses.asdict)")
    raise TypeError(fix_message(f"{where} must be JSON (got {type(value).__name__}); {hint}", hint, example))


def _same(a: Any, b: Any) -> bool:
    """Whether two JSON values are the same, telling ``1``, ``1.0`` and ``True`` apart (Python's ``==`` does not),
    and ignoring the order of dict keys."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _json_answer(question: str, answer: Any) -> Any:
    """The JSON form of an ``ask``/``ask_human`` answer for history, frozen. A value that cannot be turned into JSON
    is recorded as its ``repr``."""
    try:
        return _serial.answer(answer, f"the answer to {question!r}")
    except TypeError:
        return repr(answer)


def _turns(n: int) -> str:
    return f"{n} turn" if n == 1 else f"{n} turns"


def _message_text(entry: MessageEntry) -> str:
    if isinstance(entry.content, str):
        return entry.content
    return " ".join(
        f"({block.media_type})" if isinstance(block, Image) else block.text
        for block in entry.content
        if isinstance(block, (TextBlock, Image))
    )


def _describe_entry(entry: HistoryEntry) -> str:
    """One history entry as one line."""
    head = f"[turn {entry.turn}] {entry.kind}"
    match entry:
        case MessageEntry():
            return f"{head}: {_short(_message_text(entry))}"
        case ModelReplyEntry(content=reply):
            parts = [_short(reply.text)] if reply.text else []
            parts.extend(_short(format_call(call)) for call in reply.tool_calls)
            return f"{head}: {'; '.join(parts) if parts else '(empty reply)'}"
        case ToolResultEntry(content=content, call=call):
            late = " (late)" if entry.late else ""
            if entry.outcome in (ToolOutcomeKind.DENIED, ToolOutcomeKind.CANCELLED):
                return f"{head} {call.name}{late} ({entry.outcome}): {_short(result_text(content))}"
            if entry.is_error:
                return f"{head} {call.name}{late} (error): {_short(result_text(content))}"
            return f"{head} {call.name}: {_size(content)}{late}"
        case ExchangeEntry(content=exchange):
            return f"{head}: {_short(exchange.question, 40)} → {_short(exchange.answer, 40)}"
        case ContextChangeEntry(content=change):
            if change.kind == "import":
                return f"{head} import: {len(change.messages)} messages"
            if change.kind == "restore":
                return f"{head} restore: back to {change.restored_to} history entries"
            return f"{head} {change.kind}: {change.before_tokens} → {change.after_tokens} tokens"
        case RunStartEntry(content=info):
            return f"{head}: {info.model}" + (f" ({info.name})" if info.name else "")
        case StopEntry(content=stopped):
            return f"{head}: {stopped if stopped is not None else '(cleared)'}"
        case ExtraDataEntry(content=content):
            parts = [f"set {', '.join(content)}"] if content else []
            if entry.removed:
                parts.append(f"removed {', '.join(entry.removed)}")
            return f"{head}: {'; '.join(parts)}"
        case ErrorEntry(call=ToolCall() as call):
            return f"{head} {call.name}: {_short(entry.content)}"
    return f"{head}: {_short(entry.content)}"

