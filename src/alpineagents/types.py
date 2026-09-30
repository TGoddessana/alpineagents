"""Data types passed between modules. All are immutable (frozen) dataclasses.

Provider-neutral message representation (ARCHITECTURE.md "Message representation"):

- In ``Message(role, content)``, ``role`` is only ever ``"user"`` or ``"assistant"``.
  The system prompt lives separately in ``Request.system``.
- ``content`` is a tuple of blocks: ``TextBlock``, ``ToolCall`` (assistant only), ``ToolResultBlock`` (user only),
  ``RawBlock`` (provider-specific data, e.g. thinking blocks).
- A tool result's ``content`` is a string, or a tuple of ``TextBlock`` and ``Image`` when the tool returned an image.
- ``RawBlock.data`` is exactly what the provider gave, and nobody modifies it. Only the adapter for the same
  ``provider`` sends it back as is; other adapters drop it.
- Messages with the same role may come in a row. The adapter merges them to fit the provider's rules.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from functools import cached_property
from typing import Any, Literal, TypeAlias

from .errors import fix_message

__all__ = [
    "TextBlock",
    "Image",
    "ToolCall",
    "ToolResultBlock",
    "ToolResultContent",
    "RawBlock",
    "Block",
    "Message",
    "ToolSpec",
    "Request",
    "Reply",
    "Usage",
    "Price",
    "HistoryKind",
    "HistoryEntry",
    "MessageEntry",
    "ReplyEntry",
    "ToolResultEntry",
    "NotRunEntry",
    "ExchangeEntry",
    "ContextChangeEntry",
    "ModelEventEntry",
    "ErrorEntry",
    "Exchange",
    "ContextChangeKind",
    "ContextChange",
    "ModelEvent",
    "ToolOutcomeKind",
    "ToolOutcome",
    "StoppedByUntil",
    "StoppedByLimit",
    "StoppedByFinish",
    "StoppedByPermission",
    "Stopped",
    "INVALID_ARGS_KEY",
    "TRUNCATED_ARGS_MESSAGE",
    "format_call",
    "result_text",
    "IMAGE_TYPES",
]

#: The only key the adapter puts in ``ToolCall.args`` when it cannot read the tool argument JSON the model gave.
#: The value is usually the raw string received, but it may be ``TRUNCATED_ARGS_MESSAGE`` (below).
#: The Agent sees this and records ``(input error: ...)`` as the result.
INVALID_ARGS_KEY = "__invalid_json__"

#: The text used as the ``INVALID_ARGS_KEY`` value when the reply was cut off by max_tokens (Anthropic) or
#: length (OpenAI-compatible) before the tool call arguments were complete. When ``Tool.prepare`` sees this value
#: it uses it as is, without wrapping it in "arguments are not valid JSON" -- a truncated tool call must not
#: run, so this value overwrites the arguments even if they happen to parse as valid JSON.
TRUNCATED_ARGS_MESSAGE = (
    "The response hit the output token limit (max_tokens), so these arguments are incomplete. "
    "Split the work into smaller calls and try again."
)


# ---------------------------------------------------------------- content blocks


@dataclass(frozen=True)
class TextBlock:
    """A piece of text."""

    text: str


#: The image types every provider accepts, as media types.
IMAGE_TYPES = ("image/png", "image/jpeg", "image/gif", "image/webp")

# The first bytes of each type in IMAGE_TYPES. WebP is "RIFF", 4 size bytes, then "WEBP".
_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def _sniff(data: bytes) -> str | None:
    """The media type ``data`` starts like, or ``None``."""
    for signature, media_type in _SIGNATURES:
        if data.startswith(signature):
            return media_type
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


@dataclass(frozen=True, repr=False)
class Image:
    """An image, as a tool result the model sees. Return it from a tool, alone or in a list with text:

    ``return Image.from_path("chart.png")`` or ``return ["Page loaded in 1.2s", Image(png_bytes)]``

    ``data`` is the file's bytes. The adapters encode it (base64) when they send it, and a Store saves it the
    same way. PNG, JPEG, GIF and WebP are supported.
    """

    data: bytes
    """The image file's bytes, not encoded."""
    media_type: str | None = None
    """``"image/png"``, ``"image/jpeg"``, ``"image/gif"`` or ``"image/webp"``. Read from ``data`` when not
    given."""

    def __post_init__(self) -> None:
        if not isinstance(self.data, (bytes, bytearray, memoryview)):
            raise TypeError(
                fix_message(
                    f"Image needs the image file's bytes (got: {type(self.data).__name__})",
                    "Pass bytes, or use Image.from_path(path) or Image.from_base64(text, media_type)",
                    'Image(open("chart.png", "rb").read())',
                )
            )
        if not isinstance(self.data, bytes):
            object.__setattr__(self, "data", bytes(self.data))
        if self.media_type is None:
            media_type = _sniff(self.data)
            if media_type is None:
                raise ValueError(
                    fix_message(
                        f"Image data is not a PNG, JPEG, GIF or WebP image (it starts with {self.data[:8]!r})",
                        "Pass the bytes of one of those formats, or convert the image first",
                    )
                )
            object.__setattr__(self, "media_type", media_type)
        elif self.media_type not in IMAGE_TYPES:
            raise ValueError(
                fix_message(
                    f"Image media_type {self.media_type!r} is not supported",
                    f"Use one of {', '.join(IMAGE_TYPES)}, or leave media_type out to read it from the data",
                )
            )

    @classmethod
    def from_path(cls, path: str | os.PathLike[str]) -> Image:
        """Reads an image file. The type is read from the file's bytes, not its name."""
        with open(path, "rb") as file:
            return cls(file.read())

    @classmethod
    def from_base64(cls, data: str, media_type: str | None = None) -> Image:
        """An image from base64 text, such as an MCP server's image content."""
        return cls(base64.b64decode(data, validate=True), media_type)

    @cached_property
    def base64(self) -> str:
        """``data`` as base64 text, the form providers take."""
        return base64.b64encode(self.data).decode("ascii")

    def __repr__(self) -> str:
        return f"Image({self.media_type}, {_byte_size(len(self.data))})"


@dataclass(frozen=True)
class ToolCall:
    """One tool call the model requested. Also a block of an assistant message.

    Calls are told apart by ``id``, and two calls with the same ``id`` hash the same.
    """

    name: str
    """The tool name the model called."""
    args: dict[str, Any]
    """The arguments the model gave, as a dict."""
    id: str
    """The call id. Tool results refer to the call by this id."""

    # args is a dict, so the hash uses id only.
    def __hash__(self) -> int:
        return hash(self.id)


ToolResultContent: TypeAlias = str | tuple[TextBlock | Image, ...]
"""What the model gets as a tool result: a string, or text and images in order when the tool returned an
image. ``result_text`` turns either into one string for display."""


@dataclass(frozen=True)
class ToolResultBlock:
    """The result of one tool call. Goes in user messages only.

    ``content`` is exactly what is sent to the model: a string (e.g. ``"(done)"``, ``"(input error: ...)"``,
    ``"(aborted: TimeoutError)"``, ``"(cleared: kept in history)"``, a denial reason), or a tuple of
    ``TextBlock`` and ``Image`` when the tool returned an image.
    """

    call_id: str
    content: ToolResultContent
    name: str = ""
    is_error: bool = False


@dataclass(frozen=True)
class RawBlock:
    """A provider-specific block (thinking blocks, redacted_thinking, server tools, etc.).

    ``data`` is kept exactly as received and sent back as is to the same ``provider``. Never modified.
    """

    provider: str
    data: dict[str, Any]


Block: TypeAlias = TextBlock | ToolCall | ToolResultBlock | RawBlock


@dataclass(frozen=True)
class Message:
    """One message in the context (``state.context``).

    The system prompt is not a message; it goes in ``Request.system``.
    """

    # tokens is the anchor for estimating the context size (alpineagents._tokens.context_tokens).
    role: Literal["user", "assistant"]
    """``"user"`` or ``"assistant"``. Notices and tool results are user messages."""
    content: tuple[Block, ...]
    """The blocks in order: text, tool calls (assistant only), tool results (user only) and provider-specific
    blocks such as thinking."""
    tokens: int | None = None
    """The token count of the whole context up to and including this message, from API usage. Set on assistant
    replies only; ``None`` if unknown."""

    @classmethod
    def user(cls, text: str) -> Message:
        """A user message with a single text."""
        return cls("user", (TextBlock(text),))

    @property
    def text(self) -> str:
        """The text of all TextBlocks joined together."""
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """The tool calls in this message, in order."""
        return tuple(b for b in self.content if isinstance(b, ToolCall))


# ---------------------------------------------------------------- requests and replies


@dataclass(frozen=True)
class ToolSpec:
    """One tool shown to the model."""

    name: str
    """The tool name the model calls."""
    description: str
    """What the tool does, as the model reads it."""
    input_schema: dict[str, Any]
    """The arguments as a JSON Schema object."""


@dataclass(frozen=True)
class Request:
    """A request the Agent sends to the Model. Model settings (``max_tokens`` etc.) belong to the Model object."""

    system: str | None
    """The system prompt, or ``None``."""
    messages: tuple[Message, ...]
    """The context to send."""
    tools: tuple[ToolSpec, ...] = ()
    """The tools shown to the model."""
    # tool_choice="none" is used by ask and compact: some providers need tool definitions when the context has
    # tool_use blocks, even when no tool may be called.
    tool_choice: Literal["auto", "none"] = "auto"
    """``"auto"`` lets the model call tools. ``"none"`` still sends the tool definitions but forbids calling
    them."""


@dataclass(frozen=True)
class Reply:
    """One reply from the Model."""

    message: Message
    """The assistant message. Blocks stay in the order the provider gave."""
    usage: Usage
    """The usage of this one request (``requests=1``)."""
    context_tokens: int | None = None
    """The token count of the whole context up to and including this reply (input + cache read + cache write +
    output). ``None`` if unknown."""
    stop_reason: str | None = None
    """Exactly what the provider gave, e.g. ``"end_turn"``, ``"tool_use"`` or ``"max_tokens"``."""
    model: str | None = None
    """The model that answered, as the provider reported it. The requested name if the provider did not say.
    ``None`` only for a custom Model that does not set it. It can differ from the requested name (an alias, a
    router, a fallback); ``usage.cost`` is still computed from the requested Model's price."""

    @property
    def text(self) -> str:
        """The text of the reply message."""
        return self.message.text

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """The tool calls in the reply message, in order."""
        return self.message.tool_calls


# ---------------------------------------------------------------- usage and price


@dataclass(frozen=True)
class Price:
    """Dollars per million tokens. Cache prices default to the input price when not given.

    Example:
        ``Anthropic("claude-sonnet-5", price=Price(input=3, output=15, cache_read=0.3))``
    """

    input: float
    """Dollars per million input tokens that did not go through the cache."""
    output: float
    """Dollars per million output tokens."""
    cache_read: float | None = None
    """Dollars per million tokens read from the cache. ``None`` uses ``input``."""
    cache_write: float | None = None
    """Dollars per million tokens written to the cache. ``None`` uses ``input``."""

    def cost(self, usage: Usage) -> float:
        """The cost of ``usage`` in dollars."""
        cache_read = self.input if self.cache_read is None else self.cache_read
        cache_write = self.input if self.cache_write is None else self.cache_write
        return (
            usage.input_tokens * self.input
            + usage.output_tokens * self.output
            + usage.cache_read_tokens * cache_read
            + usage.cache_write_tokens * cache_write
        ) / 1_000_000


@dataclass(frozen=True)
class Usage:
    """Token usage. Every provider's counts are mapped to the same meanings, so total input is
    ``input_tokens + cache_read_tokens + cache_write_tokens``.

    Adding two Usages (``+``) returns a new one. Its ``cost`` is ``None`` if either side has requests but an
    unknown cost.
    """

    input_tokens: int = 0
    """Input tokens that did not go through the cache."""
    output_tokens: int = 0
    """Output tokens."""
    cache_read_tokens: int = 0
    """Input tokens read from the cache."""
    cache_write_tokens: int = 0
    """Input tokens newly written to the cache."""
    requests: int = 0
    """The number of model requests."""
    cost: float | None = None
    """Estimated dollars. ``None`` when the price is unknown, which is different from 0."""

    def __add__(self, other: Usage) -> Usage:
        if not isinstance(other, Usage):
            return NotImplemented
        if self.requests == 0:
            cost = other.cost
        elif other.requests == 0:
            cost = self.cost
        elif self.cost is None or other.cost is None:
            cost = None
        else:
            cost = self.cost + other.cost
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            requests=self.requests + other.requests,
            cost=cost,
        )

    @property
    def cache_hit_rate(self) -> float | None:
        """The fraction of total input read from the cache. ``None`` when there is no input."""
        total = self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
        return None if total == 0 else self.cache_read_tokens / total


# ---------------------------------------------------------------- history


HistoryKind: TypeAlias = Literal[
    "user",
    "reply",
    "tool_result",
    "notice",
    "denied",
    "cancelled",
    "ask",
    "human",
    "context_change",
    "model_event",
    "error",
]


@dataclass(frozen=True)
class Exchange:
    """The question and answer of ``ask``/``ask_human``. ``answer`` is ``None`` if no answer came (OutputError)."""

    question: str
    answer: Any


ContextChangeKind: TypeAlias = Literal["compact", "start_from", "clear_tool_results", "rollback"]


@dataclass(frozen=True)
class ContextChange:
    """A record that the context changed. Shown by ``Reporter.on_context_change`` and kept in history."""

    kind: ContextChangeKind
    """``"compact"``, ``"start_from"``, ``"clear_tool_results"`` or ``"rollback"``."""
    before_tokens: int
    """The estimated context size before the change (``state.context_tokens``)."""
    after_tokens: int
    """The estimated context size after the change."""
    summary: str | None = None
    """The summary text ``compact`` or ``start_from`` put in the context. ``None`` for other kinds."""


@dataclass(frozen=True)
class ModelEvent:
    """Something the Model went through while handling a request, such as falling back to another model or a
    retry. It is not part of the reply.

    A Model reports it by calling ``on_event`` in ``respond`` or ``compact``. The Agent records it in history
    (``model_event``) and shows it with ``Reporter.on_model_event``.
    """

    kind: str
    """A short type name, e.g. ``"fallback"``."""
    message: str
    """One line to show a person."""
    data: Mapping[str, Any] = field(default_factory=dict)
    """Extra values the Model adds."""


class ToolOutcomeKind(StrEnum):
    """How a tool call ended (``ToolOutcome.kind``). A ``StrEnum``, so ``outcome.kind == "denied"`` works too."""

    DONE = "done"
    """The tool returned, and the result is its output."""
    ERROR = "error"
    """The tool returned an error result, which the model sees as an error (an MCP server's ``isError``)."""
    INPUT_ERROR = "input_error"
    """The tool was not called because the tool is unknown or the arguments failed validation."""
    ABORTED = "aborted"
    """The tool raised."""
    INTERRUPTED = "interrupted"
    """Closed by ``KeyboardInterrupt`` or ``CancelledError``."""
    DENIED = "denied"
    """Not run because a permission denied it, and the result is the reason."""
    CANCELLED = "cancelled"
    """Not run because a permission stopped the turn (another call of the same turn was denied with
    ``stop=True``)."""


@dataclass(frozen=True)
class ToolOutcome:
    """How a tool call ended, passed to ``Reporter.on_tool_end`` next to the result.

    The result is written for the model and may change. Branch on ``kind`` instead of reading it.
    """

    kind: ToolOutcomeKind
    """One of:

    - ``DONE``: the tool returned, and the result is its output
    - ``ERROR``: the tool returned an error result, which the model sees as an error (an MCP server's
      ``isError``)
    - ``INPUT_ERROR``: the tool was not called because the tool is unknown or the arguments failed validation
    - ``ABORTED``: the tool raised
    - ``INTERRUPTED``: closed by ``KeyboardInterrupt`` or ``CancelledError``
    - ``DENIED``: not run because a permission denied it, and the result is the reason
    - ``CANCELLED``: not run because a permission stopped the turn (another call of the same turn was denied
      with ``stop=True``)

    A plain string such as ``"denied"`` given here becomes the matching ``ToolOutcomeKind``.
    """
    error: BaseException | None = None
    """The exception for ``ABORTED`` and ``INTERRUPTED``, and the ``ToolError`` a permission raised for a
    ``DENIED`` call. Otherwise ``None``."""
    decided_by: str | None = None
    """The ``repr()`` of the permission that decided a ``DENIED`` or ``CANCELLED`` outcome. ``None`` for other
    kinds, and when no permission decided."""

    def __post_init__(self) -> None:
        # ValueError for a string that is not a kind.
        object.__setattr__(self, "kind", ToolOutcomeKind(self.kind))


# ---------------------------------------------------------------- why a loop stopped


@dataclass(frozen=True)
class StoppedByUntil:
    """``state.stopped`` when an ``until`` function of ``@loop`` returned true. ``str()`` gives
    ``stopped by is_answered``."""

    name: str
    """The function's ``__name__``, e.g. ``"is_answered"`` (``"<lambda>"`` for a lambda)."""

    def __str__(self) -> str:
        return f"stopped by {self.name}"


@dataclass(frozen=True)
class StoppedByLimit:
    """``state.stopped`` when the loop ran ``limit`` turns in this call. ``str()`` gives ``stopped at limit 30``."""

    turns: int
    """The loop's ``limit``."""

    def __str__(self) -> str:
        return f"stopped at limit {self.turns}"


@dataclass(frozen=True)
class StoppedByFinish:
    """``state.stopped`` when ``state.finish()`` was called. ``str()`` gives ``stopped by finish``."""

    def __str__(self) -> str:
        return "stopped by finish"


@dataclass(frozen=True)
class StoppedByPermission:
    """``state.stopped`` when a permission denied a tool call with ``stop=True``. The run ended normally and the
    State is not finished: add a user message and run again to continue. ``str()`` gives
    ``stopped by permission DecideByHuman()``."""

    call: ToolCall
    """The tool call that was denied."""
    permission: str
    """The ``repr()`` of the permission that denied it."""

    def __str__(self) -> str:
        return f"stopped by permission {self.permission}"


#: Why a loop stopped: the type of ``state.stopped`` (without ``None``).
Stopped: TypeAlias = StoppedByUntil | StoppedByLimit | StoppedByFinish | StoppedByPermission


def _now() -> datetime:
    """The current time in UTC, used to stamp ``at`` on history entries."""
    return datetime.now(timezone.utc)


def _stamp() -> Any:
    # A lambda rather than _now itself, so _now is looked up at call time (tests monkeypatch types._now).
    return field(default_factory=lambda: _now(), compare=False)


# Every entry class has ``kind``, ``content``, ``turn`` and ``at``; ``kind`` tells them apart, so checking it
# narrows the type (``if entry.kind == "reply": entry.content`` is a ``Reply``). Other fields exist only where they
# mean something. Store records keep the same keys for every kind (``_serial.entry_to_dict``).
#
# ``at`` is when the entry was added to history, not when the work started: a ``reply`` is stamped when the reply
# finished, a ``human`` entry when the answer arrived, and a ``user`` message added while tools run when
# ``add_user_message`` was called, even though it goes into the context later. ``==`` ignores it.


@dataclass(frozen=True, kw_only=True)
class MessageEntry:
    """A user message or a notice. The first entry of every history is the task (``kind="user"``, turn 0)."""

    kind: Literal["user", "notice"]
    """``"user"``: from the person (``add_user_message``). ``"notice"``: from your loop or a tool
    (``add_notice``)."""
    content: str
    """The text. A notice starts with ``"[notice] "``."""
    turn: int
    """``state.turn`` when the entry was recorded."""
    at: datetime = _stamp()
    """When the entry was recorded, as a timezone-aware UTC ``datetime``."""


@dataclass(frozen=True, kw_only=True)
class ReplyEntry:
    """A model reply from ``think``."""

    kind: Literal["reply"] = "reply"
    content: Reply
    turn: int
    """``state.turn`` when the entry was recorded."""
    at: datetime = _stamp()
    """When the entry was recorded, as a timezone-aware UTC ``datetime``."""


@dataclass(frozen=True, kw_only=True)
class ToolResultEntry:
    """The result of a tool call, as the model got it. Also input errors, ``ToolError`` results and calls closed by
    an exception (``"(aborted: TimeoutError)"``, ``"(interrupted by user)"``)."""

    kind: Literal["tool_result"] = "tool_result"
    content: ToolResultContent
    """The result sent to the model: a ``str``, or a tuple of ``TextBlock`` and ``Image``."""
    call: ToolCall
    """The call this is the result of."""
    is_error: bool = False
    """``True`` if the model got it as an error result."""
    late: bool = False
    """``True`` if it arrived after its call was already closed (the model gets it as a notice at the next
    ``think``)."""
    error: BaseException | None = None
    """The ``ToolError`` behind an error result (its ``__cause__`` is the original exception when
    ``exception_handler`` made it). Not saved by a store."""
    turn: int
    """``state.turn`` when the entry was recorded."""
    at: datetime = _stamp()
    """When the entry was recorded, as a timezone-aware UTC ``datetime``."""


@dataclass(frozen=True, kw_only=True)
class NotRunEntry:
    """A tool call a permission kept from running. The model got ``content`` as an error result."""

    kind: Literal["denied", "cancelled"]
    """``"denied"``: a permission refused this call. ``"cancelled"``: a permission stopped the turn
    (``Denied(..., stop=True)`` on another call)."""
    content: str
    """What the model was told: the reason, or ``"(not run: the user stopped this turn)"``."""
    call: ToolCall
    """The call that did not run."""
    is_error: Literal[True] = True
    """Always ``True``, like an error result of ``ToolResultEntry``."""
    error: BaseException | None = None
    """The ``ToolError`` a permission raised to deny it, if that is how it was denied. Not saved by a store."""
    turn: int
    """``state.turn`` when the entry was recorded."""
    at: datetime = _stamp()
    """When the entry was recorded, as a timezone-aware UTC ``datetime``."""


@dataclass(frozen=True, kw_only=True)
class ExchangeEntry:
    """A question and its answer. The context does not change."""

    kind: Literal["ask", "human"]
    """``"ask"``: ``agent.ask`` asked the model. ``"human"``: the Human was asked (``ask_human``,
    ``DecideByHuman``)."""
    content: Exchange
    turn: int
    """``state.turn`` when the entry was recorded."""
    at: datetime = _stamp()
    """When the entry was recorded (when the answer came), as a timezone-aware UTC ``datetime``."""


@dataclass(frozen=True, kw_only=True)
class ContextChangeEntry:
    """The context was replaced or shrunk (``compact``, ``start_from``, ``clear_tool_results``) or rolled back."""

    kind: Literal["context_change"] = "context_change"
    content: ContextChange
    turn: int
    """``state.turn`` when the entry was recorded."""
    at: datetime = _stamp()
    """When the entry was recorded, as a timezone-aware UTC ``datetime``."""


@dataclass(frozen=True, kw_only=True)
class ModelEventEntry:
    """Something the Model went through, such as a fallback or a retry."""

    kind: Literal["model_event"] = "model_event"
    content: ModelEvent
    turn: int
    """``state.turn`` when the entry was recorded."""
    at: datetime = _stamp()
    """When the entry was recorded, as a timezone-aware UTC ``datetime``."""


@dataclass(frozen=True, kw_only=True)
class ErrorEntry:
    """An exception raised in ``think``, ``use_tools`` or ``run``. An exception is recorded once."""

    kind: Literal["error"] = "error"
    content: str
    """Its type and message, e.g. ``"TimeoutError: timed out"`` (only the type if the message is empty)."""
    error: BaseException | None = None
    """The exception. ``None`` in a State loaded from a store (``content`` keeps the text)."""
    call: ToolCall | None = None
    """The call whose tool or permission raised it, or ``None``."""
    turn: int
    """``state.turn`` when the entry was recorded."""
    at: datetime = _stamp()
    """When the entry was recorded, as a timezone-aware UTC ``datetime``."""


#: One entry of ``state.history``: one of the classes above. Check ``kind`` (or use ``isinstance``/``match``) to
#: know which, and so what ``content`` is. History only grows; shrinking the context never removes entries.
HistoryEntry: TypeAlias = (
    MessageEntry
    | ReplyEntry
    | ToolResultEntry
    | NotRunEntry
    | ExchangeEntry
    | ContextChangeEntry
    | ModelEventEntry
    | ErrorEntry
)


# ---------------------------------------------------------------- display helpers


def format_call(call: ToolCall) -> str:
    """The ``read_file(path="main.py")`` form. Shared by Terminal and notice texts."""
    parts = []
    for key, value in call.args.items():
        try:
            shown = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            shown = repr(value)
        parts.append(f"{key}={shown}")
    return f"{call.name}({', '.join(parts)})"


def result_text(content: ToolResultContent) -> str:
    """A tool result as one string for display: the string itself, or the text blocks joined by newlines with
    each image as ``(image/png, 34.2KB)``."""
    if isinstance(content, str):
        return content
    return "\n".join(
        f"({block.media_type}, {_byte_size(len(block.data))})" if isinstance(block, Image) else block.text
        for block in content
    )


def _content_bytes(content: ToolResultContent) -> int:
    """The size of a tool result: its text in UTF-8 plus each image's bytes."""
    if isinstance(content, str):
        return len(content.encode("utf-8"))
    return sum(len(b.data) if isinstance(b, Image) else len(b.text.encode("utf-8")) for b in content)


def _byte_size(n: int) -> str:
    """``512B``, ``1.2KB``, ``3.4MB``."""
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"
