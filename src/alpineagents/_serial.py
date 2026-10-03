"""Converts history entries to plain JSON values and back (used by ``State`` for Stores).

Only State and Store call these. The formats are the saved formats, so a change here needs a new ``VERSION``.

- Every value becomes ``dict``/``list``/``str``/``int``/``float``/``bool``/``None``. Tuples become lists.
- What a Store saves is the history entries (``entry_to_dict``) and a small info dict (``info_to_dict``) for
  listing. Nothing else: a State is rebuilt by replaying its entries (``State(history=...)``).
- ``plain`` is strict: a value JSON cannot hold (a set, an object, a non-string key, NaN) is a ``TypeError`` that
  names where it was. ``answer`` also turns Pydantic models and dataclasses into dicts first (``ask`` and
  ``finish`` answers), and gives the result frozen, as history keeps it.
- A message block or tool result with an image is ``{"type": "image", "media_type", "data"}`` (``data`` in base64);
  a tool result that is a string stays a string.
- ``HistoryEntry.error`` (the exception object) is not saved. An ``error`` entry's text keeps its type and message;
  a ``tool_result`` from a ``ToolError`` keeps the message the model saw.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal, NotRequired, TypeAlias, TypedDict, cast

from ._frozen import freeze
from .errors import fix_message
from .types import (
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
    RawBlock,
    Reply,
    RunStartEntry,
    Stopped,
    StoppedByFinish,
    StoppedByLimit,
    StoppedByPermission,
    StoppedByUntil,
    StopEntry,
    TextBlock,
    ToolCall,
    ToolOutcomeKind,
    ToolResultBlock,
    ToolResultContent,
    ToolResultEntry,
    Usage,
)

__all__ = [
    "VERSION",
    "InfoDict",
    "check_version",
    "info_to_dict",
    "info_from_dict",
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
    "stopped_from_dict",
    "agent_info_to_dict",
    "agent_info_from_dict",
]

#: Version of the saved formats. A record with any other version is refused when loading (there is no converter).
#: 2: tool results can hold images.
#: 3: the snapshot has ``stopped`` (a dict, see ``stopped_to_dict``) instead of ``stopped_by`` and ``stopped_limit``.
#: 4 (alpineagents 0.5): a Store keeps history entries and a small info dict only. New entry kinds and fields
#: (``model_request``, ``run_start``, ``stop``, ``extra_data``, outcomes on ``tool_result``, ...), and user
#: messages can hold images.
VERSION = 4


# ---------------------------------------------------------------- saved shapes

#: One message block: ``{"type": "text" | "image" | "tool_call" | "tool_result" | "raw", ...}`` (see
#: ``_block_to_dict``).
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
    answer: NotRequired[Any]
    """``finish``: the answer as JSON (``None`` without one)."""
    call: NotRequired[CallDict]
    """``permission``: the denied call."""
    permission: NotRequired[str]
    """``permission``: the permission's ``repr()``."""


class AgentInfoDict(TypedDict):
    name: str | None
    model: str
    system_sha256: str | None
    tools: list[str]
    mcp_servers: list[str]


class InfoDict(TypedDict):
    """What a Store keeps next to the entries, for listing: exactly the ``StateInfo`` fields except ``id``, plus the
    format version. Small, with no messages and no history."""

    v: int
    """Format version (``VERSION`` when written)."""
    first_message: str | None
    created_at: str | None
    updated_at: str | None
    """ISO ``at`` of the first and of the last history entry (``None`` for a State with no entries)."""
    turn: int
    stopped: StoppedDict | None
    finished: bool


# ---------------------------------------------------------------- version and info


def check_version(state_id: str, data: Mapping[str, Any]) -> None:
    """``ValueError`` unless the info dict ``data`` a Store returned has the current format ``VERSION``. There is no
    converter: States saved by another version cannot be loaded."""
    version = data.get("v")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError(
            fix_message(
                f"saved State {state_id!r} has no format version in its info, so it was not saved by alpineagents "
                "or is damaged",
                "check that the Store returns what it was given to write",
            )
        )
    if version == VERSION:
        return
    if version < VERSION:
        release = ", 0.4.x" if version == 3 else ""
        raise ValueError(
            fix_message(
                f"saved State {state_id!r} was saved by an older alpineagents (format {version}{release}); "
                f"0.5 cannot load it",
                "open it with the alpineagents version that saved it, or start a new State",
            )
        )
    raise ValueError(
        fix_message(
            f"saved State {state_id!r} was saved by a newer alpineagents (format {version}); "
            f"this alpineagents reads format {VERSION}",
            "upgrade alpineagents to load it",
        )
    )


def info_to_dict(
    *,
    first_message: str | None,
    created_at: datetime | None,
    updated_at: datetime | None,
    turn: int,
    stopped: Stopped | None,
    finished: bool,
) -> InfoDict:
    """The info dict of a State (see ``InfoDict``). ``id`` is not in it: the Store knows it from where it keeps
    the State."""
    return {
        "v": VERSION,
        "first_message": first_message,
        "created_at": None if created_at is None else time_to_str(created_at),
        "updated_at": None if updated_at is None else time_to_str(updated_at),
        "turn": turn,
        "stopped": stopped_to_dict(stopped),
        "finished": finished,
    }


_INFO_KEYS = ("first_message", "created_at", "updated_at", "turn", "stopped", "finished")


def info_from_dict(state_id: str, data: Mapping[str, Any]) -> dict[str, Any]:
    """The ``StateInfo`` fields (all but ``id``) from an info dict a Store returned, as keyword arguments:
    ``first_message``, ``created_at``, ``updated_at`` (``datetime`` or ``None``), ``turn``, ``stopped``
    (``Stopped`` or ``None``) and ``finished``.

    Raises:
        ValueError: the format version is not ``VERSION`` (``check_version``), or a key is missing.
    """
    check_version(state_id, data)
    missing = [key for key in _INFO_KEYS if key not in data]
    if missing:
        raise ValueError(
            fix_message(
                f"saved State {state_id!r} is damaged: its info has no {', '.join(missing)}",
                "check that the Store returns what it was given to write",
            )
        )
    created_at, updated_at = data["created_at"], data["updated_at"]
    return {
        "first_message": data["first_message"],
        "created_at": None if created_at is None else time_from_str(created_at),
        "updated_at": None if updated_at is None else time_from_str(updated_at),
        "turn": data["turn"],
        "stopped": stopped_from_dict(data["stopped"]),
        "finished": data["finished"],
    }


# ---------------------------------------------------------------- JSON values


def plain(value: Any, where: str) -> Any:
    """A deep copy of ``value`` made only of plain JSON values (``dict``/``list``, never the frozen containers).
    ``TypeError`` naming ``where`` (e.g. ``state.extra_data['x']``) for anything else."""
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
    """Like ``plain``, but a Pydantic model or a dataclass instance is first turned into a dict, and the result is
    frozen (a ``FrozenDict``/``FrozenList``), the form history keeps an answer in."""
    dump = getattr(value, "model_dump", None)
    if callable(dump) and not isinstance(value, type):
        value = dump(mode="json")
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    return freeze(plain(value, where))


def _not_json(where: str, value: Any) -> TypeError:
    return TypeError(
        fix_message(
            f"{where} holds {type(value).__name__} {_short_repr(value)}, which cannot be saved as JSON",
            "with a store, keep dicts (string keys), lists, str, int, float, bool or None there "
            "(turn other values into one of these first)",
            "with state.edit_extra_data() as data:\n    data['seen'] = sorted(seen)  # a list, not a set",
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
    return ToolCall(data["name"], data["args"], data["id"])


def _image_to_dict(image: Image) -> dict[str, Any]:
    return {"type": "image", "media_type": image.media_type, "data": image.base64}


def _image_from_dict(data: Mapping[str, Any]) -> Image:
    return Image.from_base64(data["data"], data["media_type"])


def _result_to_plain(content: ToolResultContent) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    return [_image_to_dict(b) if isinstance(b, Image) else {"type": "text", "text": b.text} for b in content]


def _result_from_plain(data: Any) -> ToolResultContent:
    if isinstance(data, str):
        return data
    return tuple(_image_from_dict(b) if b["type"] == "image" else TextBlock(b["text"]) for b in data)


def _block_to_dict(block: Any) -> BlockDict:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, Image):
        return _image_to_dict(block)
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
    if kind == "image":
        return _image_from_dict(data)
    if kind == "tool_call":
        return _call_from_dict(data)
    if kind == "tool_result":
        return ToolResultBlock(
            data["call_id"], _result_from_plain(data["content"]), name=data["name"], is_error=data["is_error"]
        )
    if kind == "raw":
        return RawBlock(data["provider"], data["data"])
    raise ValueError(f"unknown message block type {kind!r}")


def message_to_dict(message: Message) -> MessageDict:
    return {
        "role": message.role,
        "content": [_block_to_dict(block) for block in message.content],
        "tokens": message.tokens,
    }


def message_from_dict(data: Mapping[str, Any]) -> Message:
    return Message(data["role"], tuple(_block_from_dict(b) for b in data["content"]), tokens=data["tokens"])


# ---------------------------------------------------------------- why a run stopped


def stopped_to_dict(stopped: Stopped | None) -> StoppedDict | None:
    """``{"kind": "until", "name": ...}``, ``{"kind": "limit", "turns": ...}``, ``{"kind": "finish", "answer": ...}``,
    ``{"kind": "permission", "call": ..., "permission": ...}``, or ``None``."""
    if stopped is None:
        return None
    if isinstance(stopped, StoppedByUntil):
        return {"kind": "until", "name": stopped.name}
    if isinstance(stopped, StoppedByLimit):
        return {"kind": "limit", "turns": stopped.turns}
    if isinstance(stopped, StoppedByFinish):
        return {"kind": "finish", "answer": plain(stopped.answer, "the finish() answer")}
    if isinstance(stopped, StoppedByPermission):
        return {"kind": "permission", "call": _call_to_dict(stopped.call), "permission": stopped.permission}
    raise TypeError(f"cannot save a stop reason of type {type(stopped).__name__}")


def stopped_from_dict(data: Mapping[str, Any] | None) -> Stopped | None:
    """The inverse of ``stopped_to_dict``."""
    if data is None:
        return None
    kind = data["kind"]
    if kind == "until":
        return StoppedByUntil(data["name"])
    if kind == "limit":
        return StoppedByLimit(data["turns"])
    if kind == "finish":
        return StoppedByFinish(data.get("answer"))
    if kind == "permission":
        return StoppedByPermission(_call_from_dict(data["call"]), data["permission"])
    raise ValueError(f"unknown stop kind {kind!r}")


# ---------------------------------------------------------------- the Agent of a run


def agent_info_to_dict(info: AgentInfo) -> AgentInfoDict:
    return {
        "name": info.name,
        "model": info.model,
        "system_sha256": info.system_sha256,
        "tools": list(info.tools),
        "mcp_servers": list(info.mcp_servers),
    }


def agent_info_from_dict(data: Mapping[str, Any]) -> AgentInfo:
    return AgentInfo(
        name=data["name"],
        model=data["model"],
        system_sha256=data["system_sha256"],
        tools=tuple(data["tools"]),
        mcp_servers=tuple(data["mcp_servers"]),
    )


# ---------------------------------------------------------------- history entries
# Every kind is saved with the same keys: seq, kind, turn, at, content, plus the extra fields the class has
# (call, outcome, late, usage, removed) when they are set. So the format does not depend on which class holds the
# entry.


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


def _change_to_plain(change: ContextChange) -> dict[str, Any]:
    return {
        "kind": change.kind,
        "before_tokens": change.before_tokens,
        "after_tokens": change.after_tokens,
        "summary": change.summary,
        "kept": change.kept,
        "cleared": list(change.cleared),
        "messages": [message_to_dict(m) for m in change.messages],
        "restored_to": change.restored_to,
        "usage": None if change.usage is None else usage_to_dict(change.usage),
    }


def _change_from_plain(data: Mapping[str, Any]) -> ContextChange:
    usage = data["usage"]
    return ContextChange(
        kind=data["kind"],
        before_tokens=data["before_tokens"],
        after_tokens=data["after_tokens"],
        summary=data["summary"],
        kept=data["kept"],
        cleared=tuple(data["cleared"]),
        messages=tuple(message_from_dict(m) for m in data["messages"]),
        restored_to=data["restored_to"],
        usage=None if usage is None else usage_from_dict(usage),
    )


def _content_to_plain(entry: HistoryEntry) -> Any:
    match entry:
        case MessageEntry(content=content):
            if isinstance(content, str):
                return content
            return [_block_to_dict(block) for block in content]
        case ModelRequestEntry(content=text) | ErrorEntry(content=text):
            return text
        case ModelReplyEntry(content=reply):
            return _reply_to_plain(reply)
        case ModelEventEntry(content=event):
            return {"kind": event.kind, "message": event.message, "data": plain(event.data, "ModelEvent.data")}
        case ToolResultEntry(content=content):
            return _result_to_plain(content)
        case ExchangeEntry(content=exchange):
            where = f"answer to {exchange.question!r}"
            # The answer is JSON already when State recorded it; this also covers an Exchange built by hand.
            return {"question": exchange.question, "answer": plain(answer(exchange.answer, where), where)}
        case ContextChangeEntry(content=change):
            return _change_to_plain(change)
        case RunStartEntry(content=info):
            return agent_info_to_dict(info)
        case StopEntry(content=stopped):
            return stopped_to_dict(stopped)
        case ExtraDataEntry(content=content):
            return plain(content, "state.extra_data")


def entry_to_dict(entry: HistoryEntry, seq: int) -> dict[str, Any]:
    """``seq`` is the entry's index in history, which Stores use to skip entries they already have."""
    data: dict[str, Any] = {
        "seq": seq,
        "kind": entry.kind,
        "turn": entry.turn,
        "at": time_to_str(entry.at),
        "content": _content_to_plain(entry),
    }
    if isinstance(entry, (ToolResultEntry, ErrorEntry)) and entry.call is not None:
        data["call"] = _call_to_dict(entry.call)
    if isinstance(entry, ToolResultEntry):
        data["outcome"] = entry.outcome.value
        if entry.late:
            data["late"] = True
    if isinstance(entry, ExchangeEntry) and entry.usage is not None:
        data["usage"] = usage_to_dict(entry.usage)
    if isinstance(entry, ExtraDataEntry) and entry.removed:
        data["removed"] = list(entry.removed)
    return data


def entry_from_dict(data: Mapping[str, Any]) -> HistoryEntry:
    """The inverse of ``entry_to_dict``. ``ValueError`` for an unknown kind or a damaged entry."""
    try:
        return _entry_from_dict(data)
    except (KeyError, TypeError, AttributeError) as e:
        seq = data.get("seq") if isinstance(data, Mapping) else None
        raise ValueError(f"saved history entry {seq!r} is damaged ({type(e).__name__}: {e})") from e


def _entry_from_dict(data: Mapping[str, Any]) -> HistoryEntry:
    kind = data["kind"]
    content = data["content"]
    turn = data["turn"]
    at = time_from_str(data["at"])
    call = _call_from_dict(data["call"]) if data.get("call") is not None else None
    if kind in ("user", "notice"):
        if isinstance(content, str):
            return MessageEntry(kind=kind, content=content, turn=turn, at=at)
        blocks = tuple(_block_from_dict(b) for b in content)
        return MessageEntry(kind=kind, content=blocks, turn=turn, at=at)
    if kind == "model_request":
        return ModelRequestEntry(content=content, turn=turn, at=at)
    if kind == "model_reply":
        return ModelReplyEntry(content=_reply_from_plain(content), turn=turn, at=at)
    if kind == "model_event":
        event = ModelEvent(content["kind"], content["message"], content["data"])
        return ModelEventEntry(content=event, turn=turn, at=at)
    if kind == "tool_result":
        return ToolResultEntry(
            content=_result_from_plain(content),
            call=_needs_call(kind, call),
            outcome=ToolOutcomeKind(data["outcome"]),
            late=bool(data.get("late", False)),
            turn=turn,
            at=at,
        )
    if kind in ("ask", "human"):
        usage = data.get("usage")
        return ExchangeEntry(
            kind=kind,
            content=Exchange(content["question"], content["answer"]),
            usage=None if usage is None else usage_from_dict(usage),
            turn=turn,
            at=at,
        )
    if kind == "context_change":
        return ContextChangeEntry(content=_change_from_plain(content), turn=turn, at=at)
    if kind == "run_start":
        return RunStartEntry(content=agent_info_from_dict(content), turn=turn, at=at)
    if kind == "stop":
        return StopEntry(content=stopped_from_dict(content), turn=turn, at=at)
    if kind == "extra_data":
        return ExtraDataEntry(content=content, removed=tuple(data.get("removed", ())), turn=turn, at=at)
    if kind == "error":
        return ErrorEntry(content=content, call=call, turn=turn, at=at)
    raise ValueError(f"unknown history entry kind {kind!r}")


def _needs_call(kind: str, call: ToolCall | None) -> ToolCall:
    if call is None:
        raise ValueError(f"a saved {kind} history entry has no call")
    return call
