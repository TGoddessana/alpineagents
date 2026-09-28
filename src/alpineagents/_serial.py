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

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from .errors import fix_message
from .types import (
    ContextChange,
    Exchange,
    HistoryEntry,
    Image,
    Message,
    ModelEvent,
    RawBlock,
    Reply,
    TextBlock,
    ToolCall,
    ToolResultBlock,
    ToolResultContent,
    Usage,
)

__all__ = [
    "VERSION",
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
]

#: Version of the saved formats. Records with a newer version are refused when loading.
#: 2: tool results can hold images.
VERSION = 2


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


def usage_to_dict(usage: Usage) -> dict[str, Any]:
    return dataclasses.asdict(usage)


def usage_from_dict(data: Mapping[str, Any]) -> Usage:
    return Usage(**data)


def _call_to_dict(call: ToolCall) -> dict[str, Any]:
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


def _block_to_dict(block: Any) -> dict[str, Any]:
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


def message_to_dict(message: Message) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": [_block_to_dict(block) for block in message.content],
        "tokens": message.tokens,
    }


def message_from_dict(data: Mapping[str, Any]) -> Message:
    return Message(data["role"], tuple(_block_from_dict(b) for b in data["content"]), tokens=data["tokens"])


# ---------------------------------------------------------------- history entries


def _content_to_plain(entry: HistoryEntry) -> Any:
    content = entry.content
    if isinstance(content, Reply):
        return {
            "message": message_to_dict(content.message),
            "usage": usage_to_dict(content.usage),
            "context_tokens": content.context_tokens,
            "stop_reason": content.stop_reason,
            "model": content.model,
        }
    if isinstance(content, Exchange):
        return {"question": content.question, "answer": answer(content.answer, f"answer to {content.question!r}")}
    if isinstance(content, ContextChange):
        return dataclasses.asdict(content)
    if isinstance(content, ModelEvent):
        return {"kind": content.kind, "message": content.message, "data": plain(content.data, "ModelEvent.data")}
    if entry.kind == "tool_result":
        return _result_to_plain(content)
    return plain(content, f"{entry.kind} history entry")


def _content_from_plain(kind: str, data: Any) -> Any:
    if kind == "reply":
        return Reply(
            message=message_from_dict(data["message"]),
            usage=usage_from_dict(data["usage"]),
            context_tokens=data["context_tokens"],
            stop_reason=data["stop_reason"],
            model=data["model"],
        )
    if kind in ("ask", "human"):
        return Exchange(data["question"], data["answer"])
    if kind == "context_change":
        return ContextChange(**data)
    if kind == "model_event":
        return ModelEvent(data["kind"], data["message"], dict(data["data"]))
    if kind == "tool_result":
        return _result_from_plain(data)
    if kind in ("user", "notice", "denied", "error"):
        return data
    raise ValueError(f"unknown history entry kind {kind!r}")


def entry_to_dict(entry: HistoryEntry, seq: int) -> dict[str, Any]:
    """``seq`` is the entry's index in history, which Stores use to skip entries they already have."""
    data: dict[str, Any] = {
        "seq": seq,
        "kind": entry.kind,
        "turn": entry.turn,
        "at": time_to_str(entry.at),
        "content": _content_to_plain(entry),
    }
    if entry.call is not None:
        data["call"] = _call_to_dict(entry.call)
    if entry.late:
        data["late"] = True
    if entry.is_error:
        data["is_error"] = True
    return data


def entry_from_dict(data: Mapping[str, Any]) -> HistoryEntry:
    call = data.get("call")
    return HistoryEntry(
        data["kind"],
        _content_from_plain(data["kind"], data["content"]),
        turn=data["turn"],
        call=_call_from_dict(call) if call is not None else None,
        late=bool(data.get("late", False)),
        is_error=bool(data.get("is_error", False)),
        at=time_from_str(data["at"]),
    )
