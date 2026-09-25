"""Context size estimation. The only rule the core uses: State (``context_tokens``) calls it directly, and the
default Model ``count_tokens`` (a utility the core never calls) shares it.

Rule (the default ``count_tokens`` implementation): take the last message whose API usage is known
(one with ``Message.tokens``) as the anchor, and add a character-count estimate for only the messages after it.
With no anchor, estimate the system prompt, the tool definitions and every message.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

from .types import Message, RawBlock, TextBlock, ToolCall, ToolResultBlock, ToolSpec

__all__ = [
    "estimate_text_tokens",
    "estimate_message_tokens",
    "estimate_overhead_tokens",
    "context_tokens",
]


def estimate_text_tokens(text: str) -> int:
    """Character-count estimate. A generous 1 token per 3 characters, to cover languages such as Korean."""
    return (len(text) + 2) // 3


def _block_text(block: object) -> str:
    if isinstance(block, TextBlock):
        return block.text
    if isinstance(block, ToolCall):
        return block.name + json.dumps(block.args, ensure_ascii=False, default=str)
    if isinstance(block, ToolResultBlock):
        return block.content
    if isinstance(block, RawBlock):
        return json.dumps(block.data, ensure_ascii=False, default=str)
    return str(block)


def estimate_message_tokens(message: Message) -> int:
    return 4 + sum(estimate_text_tokens(_block_text(b)) for b in message.content)


def estimate_overhead_tokens(system: str | None, tools: Iterable[ToolSpec]) -> int:
    """Estimate of what every request carries outside the messages (system prompt, tool definitions)."""
    total = estimate_text_tokens(system or "")
    for spec in tools:
        total += estimate_text_tokens(
            spec.name + spec.description + json.dumps(spec.input_schema, ensure_ascii=False)
        )
    return total


def context_tokens(messages: Sequence[Message], overhead: int) -> int:
    """Estimated token count of the whole context."""
    for index in range(len(messages) - 1, -1, -1):
        anchor = messages[index].tokens
        if anchor is not None:
            return anchor + sum(estimate_message_tokens(m) for m in messages[index + 1 :])
    return overhead + sum(estimate_message_tokens(m) for m in messages)
