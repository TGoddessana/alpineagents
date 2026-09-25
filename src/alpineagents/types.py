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
    """One tool call the model requested (``call``). Also a block of an assistant message.

    Calls are told apart by ``id``. ``args`` is a dict, so the hash uses ``id`` only.
    """

    name: str
    args: dict[str, Any]
    id: str

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

    ``tokens``: the token count of the whole context up to and including this message, when known from API usage
    (the Agent sets ``Reply.context_tokens`` on assistant replies only). ``None`` if unknown.
    It is the anchor for estimating the context size (``alpineagents._tokens.context_tokens``).
    """

    role: Literal["user", "assistant"]
    content: tuple[Block, ...]
    tokens: int | None = None

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
        return tuple(b for b in self.content if isinstance(b, ToolCall))


# ---------------------------------------------------------------- requests and replies


@dataclass(frozen=True)
class ToolSpec:
    """One tool shown to the model. ``input_schema`` is a JSON Schema (object)."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class Request:
    """A request the Agent sends to the Model. Model settings (max_tokens etc.) belong to the Model object.

    With ``tool_choice="none"``, tool definitions are still sent but the model cannot call tools
    (used by ``ask`` and ``compact``; some providers need tool definitions when the context has tool_use blocks).
    """

    system: str | None
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    tool_choice: Literal["auto", "none"] = "auto"


@dataclass(frozen=True)
class Reply:
    """One reply from the Model.

    - ``message``: a message with role ``"assistant"``. Blocks stay in the order the provider gave.
    - ``usage``: the usage of this one request (``requests=1``).
    - ``context_tokens``: the token count of the whole context up to and including this reply (computed from API
      usage: input + cache read + cache write + output). ``None`` if unknown.
    - ``stop_reason``: exactly what the provider gave (e.g. ``"end_turn"``, ``"tool_use"``, ``"max_tokens"``).
    """

    message: Message
    usage: Usage
    context_tokens: int | None = None
    stop_reason: str | None = None

    @property
    def text(self) -> str:
        return self.message.text

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return self.message.tool_calls


# ---------------------------------------------------------------- usage and price


@dataclass(frozen=True)
class Price:
    """Dollars per million tokens. Cache prices default to the input price when not given."""

    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None

    def cost(self, usage: Usage) -> float:
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
    """Token usage. Token counts are what the API returned; the adapter maps them to the meanings below.

    - ``input_tokens``: input tokens that did not go through the cache
    - ``cache_read_tokens``: input tokens read from the cache
    - ``cache_write_tokens``: input tokens newly written to the cache
    - total input = the sum of the three
    - ``cost``: estimated dollars. ``None`` when the price is unknown (distinct from 0).

    Adding (``+``) returns a new Usage. ``cost`` is ``None`` if either side is unknown (has requests but ``None``).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    requests: int = 0
    cost: float | None = None

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
    """A record that the context changed. Before/after token counts are ``state.context_tokens`` estimates.

    ``summary`` is the raw summary text ``compact``/``start_from`` put in the context (``None`` otherwise).
    """

    kind: ContextChangeKind
    before_tokens: int
    after_tokens: int
    summary: str | None = None


@dataclass(frozen=True)
class ModelEvent:
    """Something the Model went through while handling a request (e.g. falling back to another model, a retry).
    Not part of the reply.

    When the Model reports it via ``on_event`` in ``respond``/``compact``, the Agent records it in history
    (``model_event``) and shows it via ``Reporter.on_model_event``. ``kind`` is a short type name
    (e.g. ``"fallback"``), ``message`` is one line to show a person, ``data`` is extra values the implementation adds.
    """

    kind: str
    message: str
    data: Mapping[str, Any] = field(default_factory=dict)


ToolOutcomeKind: TypeAlias = Literal["done", "error", "input_error", "aborted", "interrupted", "denied"]


@dataclass(frozen=True)
class ToolOutcome:
    """How a tool call ended, for ``Reporter.on_tool_end``. The ``result`` string next to it is what the model
    sees; this is what a display branches on, so it never has to read that string.

    - ``"done"``: the tool returned (the result is its output)
    - ``"error"``: the tool returned an error result, which the model sees as an error (an MCP server's
      ``isError``)
    - ``"input_error"``: the tool was not called: unknown tool or arguments that failed validation
    - ``"aborted"``: the tool raised; ``error`` is the exception
    - ``"interrupted"``: closed by an interrupt (``KeyboardInterrupt``/``CancelledError``); ``error`` is it
    - ``"denied"``: closed by ``state.deny`` (the result is the reason)
    """

    kind: ToolOutcomeKind
    error: BaseException | None = None


@dataclass(frozen=True)
class HistoryEntry:
    """One entry of ``state.history``. ``content`` by ``kind``:

    ============== ==============================================================================================
    kind           content
    ============== ==============================================================================================
    user           str (what the person said; the first entry is the task)
    reply          Reply
    tool_result    str (the result sent to the model). Has ``call``. ``late=True`` for a late result
    notice         str (starts with ``"[notice] "``)
    denied         str (the denial reason). Has ``call``
    ask            Exchange
    human          Exchange
    context_change ContextChange
    model_event    ModelEvent
    error          str (``"TimeoutError: ..."``). ``error`` holds the exception; has ``call`` if raised by a tool
    ============== ==============================================================================================

    ``turn`` is ``state.turn`` at record time. ``substate`` is for a future subagent extension (always ``None`` in
    the MVP).
    """

    kind: HistoryKind
    content: Any
    turn: int
    call: ToolCall | None = None
    error: BaseException | None = None
    late: bool = False
    substate: Any = None


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
