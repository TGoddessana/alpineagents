"""Anthropic Messages API adapter (rules in the ``Model`` docstring).

SDK: ``anthropic`` (the exact API is taken from the installed version's source). The SDK is imported on the first
request.
"""

from __future__ import annotations

import asyncio
import threading
import weakref
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from ..errors import AuthError, ContextTooLongError, ProviderError, RateLimitError, fix_message
from ..types import (
    INVALID_ARGS_KEY,
    TRUNCATED_ARGS_MESSAGE,
    Message,
    Price,
    RawBlock,
    Reply,
    Request,
    TextBlock,
    ToolCall,
    ToolResultBlock,
    Usage,
)
from .base import Model, OnEvent, OnText

__all__ = ["Anthropic"]

#: Placeholder schema used when tool_use/tool_result blocks are present without tool definitions.
_PLACEHOLDER_SCHEMA: dict[str, Any] = {"type": "object"}

#: Text used when a message's content is empty or whitespace only (the provider rejects such content).
_EMPTY_CONTENT_PLACEHOLDER = "(empty reply)"

#: Minimum thinking budget Anthropic requires (``ThinkingConfigEnabledParam.budget_tokens``).
_MIN_THINKING_BUDGET_TOKENS = 1024

#: Hints found in the ``TypeError`` messages the SDK raises when it cannot find credentials.
_MISSING_CREDENTIALS_HINTS = ("authentication method", "api_key", "auth_token", "credentials")

#: Hints for guessing a context overflow from a ``BadRequestError`` message (there is no standard code).
_CONTEXT_OVERFLOW_HINTS = (
    "prompt is too long",
    "context length",
    "maximum context",
    "too many tokens",
    "exceeds the model",
    "exceeds context",
)


def _looks_like_context_overflow(message: str) -> bool:
    lowered = message.lower()
    return any(hint in lowered for hint in _CONTEXT_OVERFLOW_HINTS)


def _looks_like_missing_credentials(message: str) -> bool:
    lowered = message.lower()
    return any(hint in lowered for hint in _MISSING_CREDENTIALS_HINTS)


def _referenced_tool_names(messages: Iterable[Message]) -> set[str]:
    """Tool names that appear in the messages (for placeholders when there are no tool definitions)."""
    names: set[str] = set()
    for message in messages:
        for block in message.content:
            if isinstance(block, ToolCall):
                names.add(block.name)
            elif isinstance(block, ToolResultBlock) and block.name:
                names.add(block.name)
    return names


class _Emit:
    """Calls ``on_text`` and remembers what it raised, so a ``TypeError`` from a Reporter is not taken for the SDK's
    missing-credentials ``TypeError``."""

    def __init__(self, on_text: OnText | None) -> None:
        self.on_text = on_text
        self.error: BaseException | None = None

    def __call__(self, chunk: str) -> None:
        if self.on_text is None:
            return
        try:
            self.on_text(chunk)
        except BaseException as e:
            self.error = e
            raise


class Anthropic(Model):
    """A Model for the Anthropic Messages API. Uses ``ANTHROPIC_API_KEY`` unless ``api_key=`` is given.

    Building it uses no network and needs no API key; the SDK client is created on the first request.

    Args:
        name: Model name, e.g. ``"claude-sonnet-5"``. A leading ``"anthropic/"`` is removed.
        max_tokens: Maximum output tokens per reply.
        temperature: Sampling temperature. Cannot be combined with ``thinking``.
        timeout: Request timeout in seconds. ``None`` uses the SDK default.
        retries: How many times the SDK retries a failed request.
        price: Token prices for ``usage.cost``.
        thinking: ``False`` turns extended thinking off, ``True`` turns on adaptive thinking, and an int turns it on
            with that token budget (at least 1024 and below ``max_tokens``). ``0`` does not turn it off.
        cache: Prompt caching. ``None`` turns it on when ``supports`` has ``"cache"``.
        context_window: Context window size in tokens.
        api_key: API key. ``None`` reads ``ANTHROPIC_API_KEY``.
        base_url: API base URL, for a proxy or a compatible gateway.
        supports: Features the model supports. Defaults to ``{"thinking", "cache", "vision"}``.

    Raises:
        ValueError: A setting the model does not support, ``thinking`` together with ``temperature``, or a
            ``thinking`` budget out of range.

    Example:
        ```python
        agent = Agent(model=Anthropic("claude-sonnet-5", thinking=True))
        ```
    """

    provider = "anthropic"
    """Always ``"anthropic"``."""
    name: str
    """Model name sent to the API."""
    max_tokens: int | None
    """Maximum output tokens per reply."""
    temperature: float | None
    """Sampling temperature, or ``None`` to leave it to the provider."""
    timeout: float | None
    """Request timeout in seconds, or ``None`` for the SDK default."""
    retries: int
    """How many times the SDK retries a failed request."""
    price: Price | None
    """Token prices used for ``usage.cost``."""
    supports: frozenset[str]
    """Features this model supports."""
    cache: bool
    """Whether prompt caching is on."""

    def __init__(
        self,
        name: str,
        *,
        max_tokens: int = 8192,
        temperature: float | None = None,
        timeout: float | None = None,
        retries: int = 2,
        price: Price | None = None,
        thinking: bool | int = False,
        cache: bool | None = None,
        context_window: int = 200_000,
        api_key: str | None = None,
        base_url: str | None = None,
        supports: Iterable[str] | None = None,
    ) -> None:
        # Only stores settings: no network, no credentials. The client is created on the first respond.
        # thinking: True -> {"type": "adaptive"}, int n -> {"type": "enabled", "budget_tokens": n}. When on,
        # _check_supported("thinking", ...). The budget checks (n >= 1024, n < max_tokens) catch at construction
        # what the API would reject later.
        # cache: None follows supports; an explicit True runs _check_supported("cache", "cache=True").
        # retries -> SDK max_retries, timeout -> SDK timeout.
        if name.startswith("anthropic/"):
            name = name[len("anthropic/") :]
        self.name = name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries
        self.price = price
        self._context_window = context_window
        self._api_key = api_key
        self._base_url = base_url
        self.supports = (
            frozenset(supports) if supports is not None else frozenset({"thinking", "cache", "vision"})
        )

        if thinking is False:
            self._thinking: dict[str, Any] | None = None
        else:
            self._check_supported("thinking", f"thinking={thinking!r}")
            if temperature is not None:
                raise ValueError(
                    fix_message(
                        "thinking is on, but temperature was also given",
                        "Thinking mode does not accept temperature. Remove temperature",
                        "Anthropic('claude-sonnet-5', thinking=True)",
                    )
                )
            if thinking is True:
                self._thinking = {"type": "adaptive"}
            else:
                budget_tokens = int(thinking)
                if budget_tokens < _MIN_THINKING_BUDGET_TOKENS:
                    raise ValueError(
                        fix_message(
                            f"thinking={budget_tokens}, but it must be at least {_MIN_THINKING_BUDGET_TOKENS}",
                            f"Set thinking to {_MIN_THINKING_BUDGET_TOKENS} or more, or use thinking=False "
                            "to turn it off (0 does not turn it off)",
                            "Anthropic('claude-sonnet-5', thinking=1024)",
                        )
                    )
                if budget_tokens >= max_tokens:
                    raise ValueError(
                        fix_message(
                            f"thinking={budget_tokens}, but max_tokens={max_tokens}",
                            "Set thinking below max_tokens, or raise max_tokens",
                            "Anthropic('claude-sonnet-5', thinking=4096, max_tokens=8192)",
                        )
                    )
                self._thinking = {"type": "enabled", "budget_tokens": budget_tokens}

        if cache is None:
            self.cache = "cache" in self.supports
        elif cache:
            self._check_supported("cache", "cache=True")
            self.cache = True
        else:
            self.cache = False

        self._sdk_client: Any = None
        self._async_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Any] = (
            weakref.WeakKeyDictionary()
        )
        self._client_lock = threading.Lock()

    @property
    def context_window(self) -> int:
        """Context window size in tokens, as passed to ``context_window=``."""
        return self._context_window

    def respond(
        self, request: Request, on_text: OnText | None = None, on_event: OnEvent | None = None
    ) -> Reply:
        """Sends one request as a stream and returns the reply.

        Text chunks go to ``on_text`` as they arrive. Thinking and other Anthropic-only blocks are kept as
        ``RawBlock``s and sent back unchanged on later requests. If the reply was cut off at ``max_tokens`` in the
        middle of a tool call, that call is marked invalid so the tool is not run with partial arguments.

        Raises:
            RateLimitError: The API returned 429 after the SDK's retries.
            AuthError: The API key is missing or rejected.
            ContextTooLongError: The request is larger than the context window.
            ProviderError: Any other API error.
        """
        import anthropic

        client = self._client()
        kwargs = self._build_request_kwargs(request)
        emit = _Emit(on_text)

        try:
            with client.messages.stream(**kwargs) as stream:
                for chunk in stream.text_stream:
                    emit(chunk)
                message = stream.get_final_message()
        except (anthropic.AnthropicError, TypeError) as e:
            if e is emit.error:
                raise  # raised by on_text, not by the SDK: propagates as is
            raise self._wrap_error(e) from e

        return self._to_reply(message)

    async def arespond(
        self, request: Request, on_text: OnText | None = None, on_event: OnEvent | None = None
    ) -> Reply:
        """The async version of ``respond``, with the same reply and errors. Cancelling it closes the stream."""
        # Uses AsyncAnthropic, one client per event loop (see _async_client).
        import anthropic

        client = self._async_client()
        kwargs = self._build_request_kwargs(request)
        emit = _Emit(on_text)

        try:
            async with client.messages.stream(**kwargs) as stream:
                async for chunk in stream.text_stream:
                    emit(chunk)
                message = await stream.get_final_message()
        except (anthropic.AnthropicError, TypeError) as e:
            if e is emit.error:
                raise  # raised by on_text, not by the SDK: propagates as is
            raise self._wrap_error(e) from e

        return self._to_reply(message)

    @staticmethod
    def _wrap_error(e: Exception) -> Exception:
        """The common error type for an SDK exception.

        ``anthropic.RateLimitError`` -> ``RateLimitError``; ``AuthenticationError``/``PermissionDeniedError`` or no
        API key -> ``AuthError``; ``RequestTooLargeError``, or a ``BadRequestError`` whose message looks like a
        context overflow ("prompt is too long", etc.) -> ``ContextTooLongError``; any other ``AnthropicError`` ->
        ``ProviderError``. The message is a short summary plus the original; the caller raises ``from e``.

        Without credentials (``ANTHROPIC_API_KEY`` not set) the SDK raises a bare ``TypeError`` at request time
        (header validation runs lazily), not an ``AnthropicError``. That message is recognized and wrapped in
        ``AuthError``; any other ``TypeError`` becomes ``ProviderError``. ``respond`` re-raises a ``TypeError``
        that ``on_text`` raised as is (``_Emit``).
        """
        import anthropic

        if isinstance(e, anthropic.RateLimitError):
            return RateLimitError(f"Anthropic API rate limit (429): {e}")
        if isinstance(e, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
            return AuthError(f"Anthropic authentication failed: {e}")
        if isinstance(e, anthropic.RequestTooLargeError):
            return ContextTooLongError(f"Anthropic request too large (413): {e}")
        if isinstance(e, anthropic.BadRequestError):
            if _looks_like_context_overflow(str(e)):
                return ContextTooLongError(f"The context exceeds the model's context window: {e}")
            return ProviderError(f"Anthropic bad request (400): {e}")
        if isinstance(e, TypeError):
            if _looks_like_missing_credentials(str(e)):
                return AuthError(f"Anthropic authentication failed (no credentials): {e}")
            return ProviderError(f"Error while building the Anthropic request: {e}")
        return ProviderError(f"Anthropic API error: {e}")

    # ------------------------------------------------------------ sending

    def _build_request_kwargs(self, request: Request) -> dict[str, Any]:
        """The ``client.messages.stream(...)`` arguments for ``request``.

        - ``system`` -> ``system=`` (omitted if None)
        - Messages go through ``_to_anthropic_messages`` (see ``_to_anthropic_content``).
        - ``tools`` -> ``[{"name", "description", "input_schema"}]``; ``tool_choice="none"`` ->
          ``{"type": "none"}``. Without tool definitions but with tool_use/tool_result in the messages, one
          placeholder definition (``{"type": "object"}`` schema) per tool name that appears, with
          ``tool_choice=none`` (``_tool_defs_and_choice``).
        - ``cache`` on -> a top-level ``cache_control={"type": "ephemeral"}`` marks the last cacheable block
          (the SDK's automatic caching).
        """
        kwargs: dict[str, Any] = {
            "model": self.name,
            "max_tokens": self.max_tokens,
            "messages": self._to_anthropic_messages(request.messages),
        }
        if request.system is not None:
            kwargs["system"] = request.system

        tools, forced_choice = self._tool_defs_and_choice(request)
        if tools:
            kwargs["tools"] = tools
        tool_choice = forced_choice
        if tool_choice is None and request.tool_choice == "none":
            tool_choice = {"type": "none"}
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        if self.temperature is not None:
            # anthropic 1.8's Messages.stream()/MessageCreateParams has no top-level temperature --
            # send it via extra_body so the SDK merges it into the request body.
            kwargs.setdefault("extra_body", {})["temperature"] = self.temperature
        if self._thinking is not None:
            kwargs["thinking"] = self._thinking
        if self.cache:
            kwargs["cache_control"] = {"type": "ephemeral"}
        return kwargs

    def _to_anthropic_messages(self, messages: Iterable[Message]) -> list[dict[str, Any]]:
        """Consecutive messages with the same role are merged first (``Model._merge_same_role``)."""
        merged = self._merge_same_role(messages)
        return [{"role": m.role, "content": self._to_anthropic_content(m)} for m in merged]

    def _to_anthropic_content(self, message: Message) -> list[dict[str, Any]]:
        """Blocks in the Anthropic format.

        - In a user message, ``tool_result`` blocks come before text.
        - ``TextBlock`` -> ``{"type": "text", "text"}``; empty or whitespace-only text is dropped. If the content
          ends up empty (e.g. an assistant reply that ended without tools), one placeholder text
          (``"(empty reply)"``) is added: Anthropic rejects empty content in any message but the last.
        - ``ToolCall`` -> ``{"type": "tool_use", "id", "name", "input": args}``
        - ``ToolResultBlock`` -> ``{"type": "tool_result", "tool_use_id", "content", "is_error"}``
        - ``RawBlock(provider="anthropic")`` -> ``data`` as is; RawBlocks of other providers are dropped.
        """
        blocks = list(message.content)
        if message.role == "user":
            # Put tool_result before text (a stable sort, so the order within each group is kept).
            blocks.sort(key=lambda b: 0 if isinstance(b, ToolResultBlock) else 1)

        out: list[dict[str, Any]] = []
        for block in blocks:
            if isinstance(block, TextBlock):
                # Drop whitespace-only text too -- Anthropic requires non-whitespace characters in text blocks.
                if block.text.strip():
                    out.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolCall):
                out.append({"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.args)})
            elif isinstance(block, ToolResultBlock):
                out.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.call_id,
                        "content": block.content,
                        "is_error": block.is_error,
                    }
                )
            elif isinstance(block, RawBlock):
                if block.provider == self.provider:
                    out.append(dict(block.data))
                # RawBlocks of other providers are dropped.
        if not out and self._should_placeholder_empty_content(blocks):
            # Anthropic rejects empty ([]) or whitespace-only content (especially when it is not the last
            # assistant message). Add placeholder text so the shape is always valid.
            out.append({"type": "text", "text": _EMPTY_CONTENT_PLACEHOLDER})
        return out

    @staticmethod
    def _should_placeholder_empty_content(blocks: list[Any]) -> bool:
        """Whether content that ended up empty needs a placeholder: only when the message was empty to begin with
        (``content=()``), or held only text that was all empty/whitespace. If it held only other providers'
        ``RawBlock``s that were deliberately all dropped ("RawBlocks of other providers are dropped"), it is left
        empty."""
        if not blocks:
            return True
        return all(isinstance(b, TextBlock) for b in blocks)

    def _tool_defs_and_choice(self, request: Request) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
        """Returns the list of tool definitions and the ``tool_choice`` to force (``None`` if none)."""
        if request.tools:
            tools = [
                {"name": spec.name, "description": spec.description, "input_schema": spec.input_schema}
                for spec in request.tools
            ]
            return tools, None

        names = _referenced_tool_names(request.messages)
        if not names:
            return [], None
        placeholders = [
            {"name": name, "description": "", "input_schema": _PLACEHOLDER_SCHEMA} for name in sorted(names)
        ]
        return placeholders, {"type": "none"}

    # ------------------------------------------------------------ receiving

    def _to_reply(self, message: Any) -> Reply:
        """The final stream message as a ``Reply``.

        - ``text`` -> ``TextBlock``, ``tool_use`` -> ``ToolCall(name, dict(input), id)``, any other block
          (thinking, redacted_thinking, server_tool_use ...) ->
          ``RawBlock("anthropic", block.model_dump(mode="json", exclude_none=True))``, a shape that can be sent
          back as is.
        - ``stop_reason == "max_tokens"`` with a tool_use last: its ``args`` become
          ``{INVALID_ARGS_KEY: TRUNCATED_ARGS_MESSAGE}`` so ``Tool.prepare`` treats it as an input error. Earlier
          complete blocks are left alone.
        - ``Usage(input_tokens, output_tokens, cache_read_tokens=cache_read_input_tokens or 0,
          cache_write_tokens=cache_creation_input_tokens or 0, requests=1)`` with ``cost=self._cost(...)``.
        - ``context_tokens`` = input + cache read + cache write + output; ``stop_reason`` as is.
        """
        content: list[Any] = []
        for block in message.content:
            if block.type == "text":
                content.append(TextBlock(block.text))
            elif block.type == "tool_use":
                content.append(ToolCall(block.name, dict(block.input), block.id))
            else:
                content.append(RawBlock("anthropic", block.model_dump(mode="json", exclude_none=True)))

        # If cut off at the output token limit, only the last block (where streaming stopped) is suspect.
        # Earlier tool_use blocks that arrived complete must stay usable.
        if message.stop_reason == "max_tokens" and content and isinstance(content[-1], ToolCall):
            content[-1] = ToolCall(content[-1].name, {INVALID_ARGS_KEY: TRUNCATED_ARGS_MESSAGE}, content[-1].id)

        reply_message = Message("assistant", tuple(content))

        raw_usage = Usage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            cache_read_tokens=message.usage.cache_read_input_tokens or 0,
            cache_write_tokens=message.usage.cache_creation_input_tokens or 0,
            requests=1,
        )
        usage = replace(raw_usage, cost=self._cost(raw_usage))
        context_tokens = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens + usage.output_tokens

        return Reply(
            message=reply_message,
            usage=usage,
            context_tokens=context_tokens,
            stop_reason=message.stop_reason,
        )

    def _client(self) -> Any:
        """Creates the SDK client only once (thread-safely). ``import anthropic`` happens here."""
        with self._client_lock:
            if self._sdk_client is None:
                import anthropic

                self._sdk_client = anthropic.Anthropic(**self._client_kwargs())
            return self._sdk_client

    def _async_client(self) -> Any:
        """The ``AsyncAnthropic`` client for the running event loop, created once per loop (its connection pool
        belongs to the loop that created it, so a client is not reused across ``asyncio.run`` calls)."""
        loop = asyncio.get_running_loop()
        with self._client_lock:
            client = self._async_clients.get(loop)
            if client is None:
                import anthropic

                client = anthropic.AsyncAnthropic(**self._client_kwargs())
                self._async_clients[loop] = client
            return client

    def _client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"max_retries": self.retries}
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        if self._base_url is not None:
            kwargs["base_url"] = self._base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        return kwargs
