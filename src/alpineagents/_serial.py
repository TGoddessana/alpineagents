"""Converts State records to plain JSON values and back (used by ``State`` for Stores).

Only State calls these. The formats are the saved formats, so a change here needs a new ``VERSION``.

- Every value becomes ``dict``/``list``/``str``/``int``/``float``/``bool``/``None``. Tuples become lists.
- ``plain`` is strict: a value JSON cannot hold (a set, an object, a non-string key, NaN) is a ``TypeError`` that
  names where it was. ``answer`` also turns Pydantic models and dataclasses into dicts first (``ask`` and
  ``finish`` answers), so they come back as dicts.
- A tool result with images is a list of ``{"type": "text", "text"}`` and ``{"type": "image", "media_type", "data"}``
  (``data`` in base64); a string result stays a string.
- ``HistoryEntry.error`` (the exception object) is not saved. An ``error`` entry's text keeps its type and message;
  a ``tool_result`` from a ``ToolError`` keeps the message the model saw.
"""

# No ``from __future__ import annotations``: ``Snapshot.__required_keys__`` needs real ``NotRequired`` annotations.
import dataclasses
import math
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal, NotRequired, TypeAlias, TypedDict, cast

from .errors import fix_message
from .types import (
    ContextChange,
    ContextChangeEntry,
    ErrorEntry,
    Exchange,
    ExchangeEntry,
    HistoryEntry,
    Image,
    Message,
    MessageEntry,
    ModelEvent,
    ModelEventEntry,
    NotRunEntry,
    RawBlock,
    Reply,
    ReplyEntry,
    Stopped,
    StoppedByFinish,
    StoppedByLimit,
    StoppedByPermission,
    StoppedByUntil,
    TextBlock,
    ToolCall,
    ToolResultBlock,
    ToolResultContent,
    ToolResultEntry,
    Usage,
)

__all__ = [
    "VERSION",
    "Snapshot",
    "AgentSummary",
    "read_snapshot",
    "plain",
    "answer",
    "entry_to_dict",
    "entry_from_dict",
    "message_to_dict",
    "message_from_dict",
    "usage_to_dict",
    "usage_from_dict",
    "time_to_str",
    "time_from_str",
    "stopped_to_dict",
    "stopped_from_snapshot",
]

#: Version of the saved formats. Records with a newer version are refused when loading.
#: 2: tool results can hold images.
#: 3: the snapshot has ``stopped`` (a dict, see ``stopped_to_dict``) instead of ``stopped_by`` and ``stopped_limit``.
VERSION = 3


# ---------------------------------------------------------------- saved shapes

#: ``Agent._summary()``: ``name``, ``model``, ``system_sha256``, ``tools``, ``mcp_servers``. One loaded from a store
#: may miss keys or hold odd values, so it stays a plain dict (``Agent._start_run`` compares it without raising).
AgentSummary: TypeAlias = dict[str, Any]
#: One message block: ``{"type": "text" | "tool_call" | "tool_result" | "raw", ...}`` (see ``_block_to_dict``).
BlockDict: TypeAlias = dict[str, Any]


class CallDict(TypedDict):
    name: str
    args: dict[str, Any]
    id: str


class MessageDict(TypedDict):
    role: Literal["user", "assistant"]
    content: list[BlockDict]
    tokens: int | None


class UsageDict(TypedDict):
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    requests: int
    cost: float | None


class StoppedDict(TypedDict):
    kind: Literal["until", "limit", "finish", "permission"]
    name: NotRequired[str]
    """``until``: the until function's name."""
    turns: NotRequired[int]
    """``limit``: the limit."""
    call: NotRequired[CallDict]
    """``permission``: the denied call."""
    permission: NotRequired[str]
    """``permission``: the permission's ``repr()``."""


class Snapshot(TypedDict):
    """Everything in a State but its history, as JSON values (``State._snapshot``). A store keeps the latest one.

    Taken only when no call is pending and ``think`` is not waiting on the model, so there is no half-done turn in
    it. History entries saved after it are replayed on load (``State._replay``).
    """

    v: int
    """Format version (``VERSION`` when written)."""
    id: str
    task: str
    history_len: int
    """How many history entries it covers (``seq`` 0 to ``history_len - 1``)."""
    created_at: str
    updated_at: str
    """ISO ``at`` of the first and of the last covered history entry."""
    context: list[MessageDict]
    turn: int
    usage: UsageDict
    late_notices: list[str]
    finished: bool
    finish_answer: Any
    """The ``finish()`` answer as JSON (see ``answer``), or ``None``."""
    last_answer: str | None
    stopped: NotRequired[StoppedDict | None]
    """Version 3 and later."""
    stopped_by: NotRequired[str | None]
    """Versions 1 and 2: ``"finish"``, ``"limit"`` or an ``until`` name."""
    stopped_limit: NotRequired[int | None]
    """Versions 1 and 2: the limit, when ``stopped_by`` is ``"limit"``."""
    data: dict[str, Any]
    """``state.data``."""
    agent: AgentSummary | None
    """The Agent that saved it, compared when the State is resumed."""


def read_snapshot(state_id: str, data: Mapping[str, Any]) -> Snapshot:
    """A snapshot a store returned, checked: ``ValueError`` if it was saved by a newer version or misses keys."""
    version = data.get("v")
    if not isinstance(version, int) or version > VERSION:
        raise ValueError(
            f"saved State {state_id!r} has format version {version!r}, and this alpineagents reads up to "
            f"{VERSION}. Upgrade alpineagents to load it"
        )
    missing = sorted(Snapshot.__required_keys__ - data.keys())
    if missing:
        raise ValueError(f"saved State {state_id!r} is damaged: its snapshot has no {', '.join(missing)}")
    return cast(Snapshot, data)


def plain(value: Any, where: str) -> Any:
    """A deep copy of ``value`` made only of JSON values. ``TypeError`` naming ``where`` (e.g. ``state.data['x']``)
    for anything else."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _not_json(where, value)
        return value
    if isinstance(value, Mapping):
        copy: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise _not_json(f"{where} key {key!r}", key)
            copy[key] = plain(item, f"{where}[{key!r}]")
        return copy
    if isinstance(value, (list, tuple)):
        return [plain(item, f"{where}[{i}]") for i, item in enumerate(value)]
    raise _not_json(where, value)


def answer(value: Any, where: str) -> Any:
    """Like ``plain``, but a Pydantic model or a dataclass instance is first turned into a dict."""
    dump = getattr(value, "model_dump", None)
    if callable(dump) and not isinstance(value, type):
        value = dump(mode="json")
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    return plain(value, where)


def _not_json(where: str, value: Any) -> TypeError:
    return TypeError(
        fix_message(
            f"{where} holds {type(value).__name__} {_short_repr(value)}, which cannot be saved as JSON",
            "with a store, keep dicts (string keys), lists, str, int, float, bool or None there "
            "(turn other values into one of these first)",
            'state.data["seen"] = sorted(seen)  # a list, not a set',
        )
    )


def _short_repr(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:59] + "…"


# ---------------------------------------------------------------- time


def time_to_str(at: datetime) -> str:
    return at.isoformat()


def time_from_str(text: str) -> datetime:
    return datetime.fromisoformat(text)


# ---------------------------------------------------------------- messages


def usage_to_dict(usage: Usage) -> UsageDict:
    return cast(UsageDict, dataclasses.asdict(usage))


def usage_from_dict(data: Mapping[str, Any]) -> Usage:
    return Usage(**data)


def _call_to_dict(call: ToolCall) -> CallDict:
    return {"name": call.name, "args": plain(call.args, f"tool call {call.name} args"), "id": call.id}


def _call_from_dict(data: Mapping[str, Any]) -> ToolCall:
    return ToolCall(data["name"], dict(data["args"]), data["id"])


def _result_to_plain(content: ToolResultContent) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    return [
        {"type": "image", "media_type": b.media_type, "data": b.base64}
        if isinstance(b, Image)
        else {"type": "text", "text": b.text}
        for b in content
    ]


def _result_from_plain(data: Any) -> ToolResultContent:
    if isinstance(data, str):
        return data
    return tuple(
        Image.from_base64(b["data"], b["media_type"]) if b["type"] == "image" else TextBlock(b["text"])
        for b in data
    )


def _block_to_dict(block: Any) -> BlockDict:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolCall):
        return {"type": "tool_call", **_call_to_dict(block)}
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "call_id": block.call_id,
            "content": _result_to_plain(block.content),
            "name": block.name,
            "is_error": block.is_error,
        }
    if isinstance(block, RawBlock):
        return {"type": "raw", "provider": block.provider, "data": plain(block.data, f"{block.provider} block")}
    raise TypeError(f"cannot save a message block of type {type(block).__name__}")


def _block_from_dict(data: Mapping[str, Any]) -> Any:
    kind = data["type"]
    if kind == "text":
        return TextBlock(data["text"])
    if kind == "tool_call":
        return _call_from_dict(data)
    if kind == "tool_result":
        return ToolResultBlock(
            data["call_id"], _result_from_plain(data["content"]), name=data["name"], is_error=data["is_error"]
        )
    if kind == "raw":
        return RawBlock(data["provider"], dict(data["data"]))
    raise ValueError(f"unknown message block type {kind!r}")


def message_to_dict(message: Message) -> MessageDict:
    return {
        "role": message.role,
        "content": [_block_to_dict(block) for block in message.content],
        "tokens": message.tokens,
    }


def message_from_dict(data: Mapping[str, Any]) -> Message:
    return Message(data["role"], tuple(_block_from_dict(b) for b in data["content"]), tokens=data["tokens"])


# ---------------------------------------------------------------- why a loop stopped


def stopped_to_dict(stopped: Stopped | None) -> StoppedDict | None:
    """``{"kind": "until", "name": ...}``, ``{"kind": "limit", "turns": ...}``, ``{"kind": "finish"}``,
    ``{"kind": "permission", "call": ..., "permission": ...}``, or ``None``."""
    if stopped is None:
        return None
    if isinstance(stopped, StoppedByUntil):
        return {"kind": "until", "name": stopped.name}
    if isinstance(stopped, StoppedByLimit):
        return {"kind": "limit", "turns": stopped.turns}
    if isinstance(stopped, StoppedByFinish):
        return {"kind": "finish"}
    if isinstance(stopped, StoppedByPermission):
        return {"kind": "permission", "call": _call_to_dict(stopped.call), "permission": stopped.permission}
    raise TypeError(f"cannot save a stop reason of type {type(stopped).__name__}")


def stopped_from_snapshot(snapshot: Mapping[str, Any]) -> Stopped | None:
    """The ``stopped`` value of a snapshot. Version 1 and 2 snapshots have ``stopped_by`` (``"finish"``,
    ``"limit"`` or an ``until`` name) and ``stopped_limit`` instead."""
    if "stopped" not in snapshot:
        return _stopped_from_old(snapshot.get("stopped_by"), snapshot.get("stopped_limit"))
    data = snapshot["stopped"]
    if data is None:
        return None
    kind = data["kind"]
    if kind == "until":
        return StoppedByUntil(data["name"])
    if kind == "limit":
        return StoppedByLimit(data["turns"])
    if kind == "finish":
        return StoppedByFinish()
    if kind == "permission":
        return StoppedByPermission(_call_from_dict(data["call"]), data["permission"])
    raise ValueError(f"unknown stop kind {kind!r}")


def _stopped_from_old(stopped_by: str | None, stopped_limit: int | None) -> Stopped | None:
    # "finish" and "limit" could not be until names then (they were reserved), so the mapping is exact.
    if stopped_by is None:
        return None
    if stopped_by == "finish":
        return StoppedByFinish()
    if stopped_by == "limit":
        return StoppedByLimit(stopped_limit or 0)
    return StoppedByUntil(stopped_by)


# ---------------------------------------------------------------- history entries
# Every kind is saved with the same keys: seq, kind, turn, at, content, and call/late/is_error when the entry has
# them set. So the format does not depend on which class holds the entry.


def _reply_to_plain(reply: Reply) -> dict[str, Any]:
    return {
        "message": message_to_dict(reply.message),
        "usage": usage_to_dict(reply.usage),
        "context_tokens": reply.context_tokens,
        "stop_reason": reply.stop_reason,
        "model": reply.model,
    }


def _reply_from_plain(data: Mapping[str, Any]) -> Reply:
    return Reply(
        message=message_from_dict(data["message"]),
        usage=usage_from_dict(data["usage"]),
        context_tokens=data["context_tokens"],
        stop_reason=data["stop_reason"],
        model=data["model"],
    )


def _content_to_plain(entry: HistoryEntry) -> Any:
    match entry:
        case ReplyEntry(content=reply):
            return _reply_to_plain(reply)
        case ExchangeEntry(content=exchange):
            return {
                "question": exchange.question,
                "answer": answer(exchange.answer, f"answer to {exchange.question!r}"),
            }
        case ContextChangeEntry(content=change):
            return dataclasses.asdict(change)
        case ModelEventEntry(content=event):
            return {"kind": event.kind, "message": event.message, "data": plain(event.data, "ModelEvent.data")}
        case ToolResultEntry(content=content):
            return _result_to_plain(content)
        case MessageEntry(content=text) | NotRunEntry(content=text) | ErrorEntry(content=text):
            return plain(text, f"{entry.kind} history entry")


def entry_to_dict(entry: HistoryEntry, seq: int) -> dict[str, Any]:
    """``seq`` is the entry's index in history, which Stores use to skip entries they already have."""
    data: dict[str, Any] = {
        "seq": seq,
        "kind": entry.kind,
        "turn": entry.turn,
        "at": time_to_str(entry.at),
        "content": _content_to_plain(entry),
    }
    if isinstance(entry, (ToolResultEntry, NotRunEntry, ErrorEntry)) and entry.call is not None:
        data["call"] = _call_to_dict(entry.call)
    if isinstance(entry, ToolResultEntry) and entry.late:
        data["late"] = True
    if isinstance(entry, (ToolResultEntry, NotRunEntry)) and entry.is_error:
        data["is_error"] = True
    return data


def entry_from_dict(data: Mapping[str, Any]) -> HistoryEntry:
    kind = data["kind"]
    content = data["content"]
    turn = data["turn"]
    at = time_from_str(data["at"])
    call = _call_from_dict(data["call"]) if data.get("call") is not None else None
    if kind in ("user", "notice"):
        return MessageEntry(kind=kind, content=content, turn=turn, at=at)
    if kind == "reply":
        return ReplyEntry(content=_reply_from_plain(content), turn=turn, at=at)
    if kind == "tool_result":
        return ToolResultEntry(
            content=_result_from_plain(content),
            call=_needs_call(kind, call),
            is_error=bool(data.get("is_error", False)),
            late=bool(data.get("late", False)),
            turn=turn,
            at=at,
        )
    if kind in ("denied", "cancelled"):
        return NotRunEntry(kind=kind, content=content, call=_needs_call(kind, call), turn=turn, at=at)
    if kind in ("ask", "human"):
        return ExchangeEntry(kind=kind, content=Exchange(content["question"], content["answer"]), turn=turn, at=at)
    if kind == "context_change":
        return ContextChangeEntry(content=ContextChange(**content), turn=turn, at=at)
    if kind == "model_event":
        event = ModelEvent(content["kind"], content["message"], dict(content["data"]))
        return ModelEventEntry(content=event, turn=turn, at=at)
    if kind == "error":
        return ErrorEntry(content=content, call=call, turn=turn, at=at)
    raise ValueError(f"unknown history entry kind {kind!r}")


def _needs_call(kind: str, call: ToolCall | None) -> ToolCall:
    if call is None:
        raise ValueError(f"a saved {kind} history entry has no call")
    return call
