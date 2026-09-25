"""Data types passed between modules. All are immutable (frozen) dataclasses.

Provider-neutral message representation (ARCHITECTURE.md "Message representation"):

- In ``Message(role, content)``, ``role`` is only ever ``"user"`` or ``"assistant"``.
  The system prompt lives separately in ``Request.system``.
- ``content`` is a tuple of blocks: ``TextBlock``, ``ToolCall`` (assistant only), ``ToolResultBlock`` (user only),
  ``RawBlock`` (provider-specific data, e.g. thinking blocks).
- ``RawBlock.data`` is exactly what the provider gave, and nobody modifies it. Only the adapter for the same
  ``provider`` sends it back as is; other adapters drop it.
- Messages with the same role may come in a row. The adapter merges them to fit the provider's rules.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

__all__ = [
    "TextBlock",
    "ToolCall",
    "ToolResultBlock",
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
    "Exchange",
    "ContextChangeKind",
    "ContextChange",
    "ModelEvent",
    "ToolOutcomeKind",
    "ToolOutcome",
    "INVALID_ARGS_KEY",
    "TRUNCATED_ARGS_MESSAGE",
    "format_call",
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


@dataclass(frozen=True)
class ToolResultBlock:
    """The result of one tool call. Goes in user messages only.

    ``content`` is exactly the string sent to the model (e.g. ``"(done)"``, ``"(input error: ...)"``,
    ``"(aborted: TimeoutError)"``, ``"(cleared: kept in history)"``, a denial reason).
    """

    call_id: str
    content: str
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
    "user", "reply", "tool_result", "notice", "denied", "ask", "human", "context_change", "model_event", "error"
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


ToolOutcomeKind: TypeAlias = Literal["done", "error", "input_error", "aborted", "interrupted", "denied"]


@dataclass(frozen=True)
class ToolOutcome:
    """How a tool call ended, passed to ``Reporter.on_tool_end`` next to the result string.

    The result string is written for the model and may change. Branch on ``kind`` instead of reading it.
    """

    kind: ToolOutcomeKind
    """One of:

    - ``"done"``: the tool returned, and the result is its output
    - ``"error"``: the tool returned an error result, which the model sees as an error (an MCP server's
      ``isError``)
    - ``"input_error"``: the tool was not called because the tool is unknown or the arguments failed validation
    - ``"aborted"``: the tool raised
    - ``"interrupted"``: closed by ``KeyboardInterrupt`` or ``CancelledError``
    - ``"denied"``: closed by ``state.deny``, and the result is the reason
    """
    error: BaseException | None = None
    """The exception for ``"aborted"`` and ``"interrupted"``, otherwise ``None``."""


@dataclass(frozen=True)
class HistoryEntry:
    """One entry of ``state.history``. History only grows; shrinking the context never removes entries."""

    kind: HistoryKind
    """What happened. ``content`` depends on it:

    | kind | content |
    | --- | --- |
    | ``user`` | ``str``, what the person said. The first entry is the task |
    | ``reply`` | ``Reply`` |
    | ``tool_result`` | ``str``, the result sent to the model. Has ``call`` |
    | ``notice`` | ``str`` starting with ``"[notice] "`` |
    | ``denied`` | ``str``, the denial reason. Has ``call`` |
    | ``ask`` | an object with ``question`` and ``answer`` (``answer`` is ``None`` if no answer came) |
    | ``human`` | same as ``ask`` |
    | ``context_change`` | ``ContextChange`` |
    | ``model_event`` | ``ModelEvent`` |
    | ``error`` | ``str`` such as ``"TimeoutError: ..."``. ``error`` holds the exception |
    """
    content: Any
    """The recorded value. Its type depends on ``kind``."""
    turn: int
    """``state.turn`` when the entry was recorded."""
    call: ToolCall | None = None
    """The tool call, for ``tool_result``, ``denied`` and errors raised by a tool."""
    error: BaseException | None = None
    """The exception, for ``error`` entries."""
    late: bool = False
    """``True`` for a tool result that arrived after its call was already closed."""
    substate: Any = None
    """Reserved for subagents, which are not implemented yet. Always ``None``."""


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
