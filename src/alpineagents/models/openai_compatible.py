"""OpenAI-compatible Chat Completions adapter (Ollama, vLLM, OpenRouter, etc.).

SDK: ``openai`` (the exact API is taken from the installed version's source). The SDK is imported on the first
request.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import threading
import uuid
import weakref
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from ..errors import AuthError, ContextTooLongError, ProviderError, RateLimitError
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

__all__ = ["OpenAICompatible"]

#: Text used when an assistant message has neither content nor tool_calls (OpenAI requires one of the
#: two).
_EMPTY_CONTENT_PLACEHOLDER = "(empty reply)"

#: Name pattern for OpenAI "reasoning" (o-series) models. These models reject ``max_tokens`` and
#: require ``max_completion_tokens``.
_REASONING_MODEL_NAME = re.compile(r"^o\d")


def _looks_like_context_too_long(error: Exception) -> bool:
    """Guesses whether a ``BadRequestError`` means a context overflow (many servers have no standard code)."""
    if getattr(error, "code", None) == "context_length_exceeded":
        return True
    message = str(error).lower()
    return any(marker in message for marker in ("context length", "maximum context", "context_length_exceeded"))


class _Collected:
    """The pieces of one streamed reply, gathered chunk by chunk (shared by ``respond`` and ``arespond``)."""

    def __init__(self) -> None:
        self.text_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        #: Tool calls by ``index``: id/name/arguments concatenated.
        self.calls: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.usage: Any = None

    def add(self, chunk: Any) -> str | None:
        """Takes one chunk and returns its text delta (``None`` if there is none)."""
        if chunk.usage is not None:
            self.usage = chunk.usage
        if not chunk.choices:
            return None
        choice = chunk.choices[0]
        delta = choice.delta
        if choice.finish_reason:
            self.finish_reason = choice.finish_reason
        if delta.content:
            self.text_parts.append(delta.content)
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            self.reasoning_parts.append(reasoning)
        for tc in delta.tool_calls or ():
            entry = self.calls.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
            if tc.id:
                entry["id"] = tc.id
            if tc.function is not None:
                if tc.function.name:
                    entry["name"] += tc.function.name
                if tc.function.arguments:
                    entry["arguments"] += tc.function.arguments
        return delta.content or None


class OpenAICompatible(Model):
    """A Model for servers that speak the OpenAI Chat Completions API: OpenAI, Ollama, vLLM, OpenRouter and others.

    Building it uses no network and needs no API key; the SDK client is created on the first request.

    Args:
        name: Model name as the server knows it, e.g. ``"llama3:8b"``.
        base_url: Server URL, e.g. ``"http://localhost:11434/v1"``. ``None`` uses ``OPENAI_BASE_URL`` or
            api.openai.com.
        api_key: API key. ``None`` reads ``OPENAI_API_KEY``. With a ``base_url`` and no key anywhere, a
            placeholder key is sent, which is what local servers expect.
        max_tokens: Maximum output tokens per reply. ``None`` leaves it to the server. For OpenAI reasoning
            models (``o1``, ``o3``, ...) it is sent as ``max_completion_tokens``.
        temperature: Sampling temperature. ``None`` leaves it to the server.
        timeout: Request timeout in seconds. ``None`` uses the SDK default.
        retries: How many times the SDK retries a failed request.
        price: Token prices for ``usage.cost``.
        context_window: Context window size in tokens.
        supports: Features the server supports. Empty by default, since it varies by server.

    Example:
        ```python
        model = OpenAICompatible("llama3:8b", base_url="http://localhost:11434/v1")
        ```
    """

    provider = "openai_compatible"
    """Always ``"openai_compatible"``."""
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
    base_url: str | None
    """Server URL, or ``None`` for the SDK default."""
    api_key: str | None
    """The API key passed to ``api_key=``, or ``None`` to read ``OPENAI_API_KEY``."""

    def __init__(
        self,
        name: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        retries: int = 2,
        price: Price | None = None,
        context_window: int = 128_000,
        supports: Iterable[str] | None = None,
    ) -> None:
        # Only stores settings: no network, no credentials. The client is created on the first respond.
        # api_key None + base_url set + no OPENAI_API_KEY -> "not-needed" is passed so the SDK does not fail at
        # client creation (_client_kwargs). A real key in the environment is still used.
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries
        self.price = price
        self._context_window = context_window
        if supports is not None:
            self.supports = frozenset(supports)
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

        Text chunks go to ``on_text`` as they arrive. Reasoning text some servers send (``reasoning_content``) is
        kept as a ``RawBlock`` and sent back on later requests. A tool call with arguments that are not a JSON
        object, or cut off by the output limit, is marked invalid so the tool is not run.

        Raises:
            RateLimitError: The server returned 429 after the SDK's retries.
            AuthError: The API key is missing or rejected.
            ContextTooLongError: The request is larger than the context window.
            ProviderError: Any other API error.
        """
        import openai

        try:
            client = self._client()
        except openai.OpenAIError as e:
            # openai.OpenAI(...) raises OpenAIError at construction when there are no credentials at all
            # (e.g. OPENAI_API_KEY not set, no base_url).
            raise AuthError(f"OpenAI-compatible API authentication failed (no credentials): {e}") from e

        collected = _Collected()
        try:
            stream = client.chat.completions.create(**self._request_kwargs(request))
            try:
                for chunk in stream:
                    text = collected.add(chunk)
                    if text and on_text is not None:
                        on_text(text)
            finally:
                # Close the SDK stream/HTTP response even if a consumer (on_text, etc.) raises or the loop
                # stops midway (openai.Stream has close(); test doubles/iterators without it are skipped).
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
        except openai.OpenAIError as e:
            raise self._wrap_error(e) from e

        return self._to_reply(collected)

    async def arespond(
        self, request: Request, on_text: OnText | None = None, on_event: OnEvent | None = None
    ) -> Reply:
        """The async version of ``respond``, with the same reply and errors. Cancelling it closes the stream."""
        # Uses AsyncOpenAI, one client per event loop (see _async_client).
        import openai

        try:
            client = self._async_client()
        except openai.OpenAIError as e:
            raise AuthError(f"OpenAI-compatible API authentication failed (no credentials): {e}") from e

        collected = _Collected()
        try:
            stream = await client.chat.completions.create(**self._request_kwargs(request))
            try:
                async for chunk in stream:
                    text = collected.add(chunk)
                    if text and on_text is not None:
                        on_text(text)
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    closing = close()
                    if inspect.isawaitable(closing):
                        await closing
        except openai.OpenAIError as e:
            raise self._wrap_error(e) from e

        return self._to_reply(collected)

    def _request_kwargs(self, request: Request) -> dict[str, Any]:
        """``client.chat.completions.create(..., stream=True, stream_options={"include_usage": True})`` arguments.

        - ``system`` -> first ``{"role": "system", "content": system}``
        - assistant ``Message`` -> ``{"role": "assistant", "content": text or None, "tool_calls": [{"id",
          "type": "function", "function": {"name", "arguments": json.dumps(args)}}]}``. The ``data`` of
          ``RawBlock(provider="openai_compatible")`` is merged into that dict as is (e.g.
          ``{"reasoning_content": ...}``). RawBlocks of other providers are dropped.
        - user ``Message``: one ``{"role": "tool", "tool_call_id", "content"}`` per ``ToolResultBlock`` in block
          order, then the remaining text as one ``{"role": "user", "content": text}``.
        - ``tools`` -> ``[{"type": "function", "function": {"name", "description", "parameters"}}]`` (omitted if
          empty). ``tool_choice="none"`` only when there are tools.
        - ``max_tokens`` and ``temperature`` only when not None. For o-series names, ``max_completion_tokens``
          instead of ``max_tokens`` (the old name those models reject).
        """
        kwargs: dict[str, Any] = {
            "model": self.name,
            "messages": self._to_openai_messages(request),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        tools = self._to_openai_tools(request)
        if tools:
            kwargs["tools"] = tools
            if request.tool_choice == "none":
                kwargs["tool_choice"] = "none"
        if self.max_tokens is not None:
            if self._uses_max_completion_tokens():
                kwargs["max_completion_tokens"] = self.max_tokens
            else:
                kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return kwargs

    @staticmethod
    def _wrap_error(e: Exception) -> Exception:
        """The common error type for an SDK exception.

        ``openai.RateLimitError`` -> ``RateLimitError``; ``AuthenticationError``/``PermissionDeniedError`` ->
        ``AuthError``; a ``BadRequestError`` with ``code == "context_length_exceeded"`` or a context overflow
        message -> ``ContextTooLongError``; any other ``OpenAIError`` -> ``ProviderError``. The caller raises
        ``from e``. The ``OpenAIError`` raised when the client is created without any credentials is wrapped in
        ``AuthError`` by ``respond``/``arespond`` directly.
        """
        import openai

        if isinstance(e, openai.RateLimitError):
            return RateLimitError(str(e))
        if isinstance(e, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return AuthError(str(e))
        if isinstance(e, openai.BadRequestError) and _looks_like_context_too_long(e):
            return ContextTooLongError(str(e))
        return ProviderError(str(e))

    def _to_reply(self, collected: _Collected) -> Reply:
        """The collected stream as a ``Reply``.

        Usage: ``prompt_tokens``, ``completion_tokens``, ``prompt_tokens_details.cached_tokens`` and
        ``.cache_write_tokens`` (0 if missing) -> ``Usage(input_tokens=prompt - cached - cache_write,
        output_tokens=completion, cache_read_tokens=cached, cache_write_tokens=cache_write, requests=1)`` with
        ``cost=self._cost(...)``. Zeros if no usage arrives. ``context_tokens`` = prompt + completion (None
        without usage). ``stop_reason`` = ``finish_reason``.
        """
        message = self._reply_message(
            collected.text_parts, collected.reasoning_parts, collected.calls, collected.finish_reason
        )
        usage_data = collected.usage
        if usage_data is not None:
            cached = 0
            cache_write = 0
            details = usage_data.prompt_tokens_details
            if details is not None:
                if details.cached_tokens is not None:
                    cached = details.cached_tokens
                if getattr(details, "cache_write_tokens", None) is not None:
                    cache_write = details.cache_write_tokens
            base_usage = Usage(
                input_tokens=usage_data.prompt_tokens - cached - cache_write,
                output_tokens=usage_data.completion_tokens,
                cache_read_tokens=cached,
                cache_write_tokens=cache_write,
                requests=1,
            )
            context_tokens: int | None = usage_data.prompt_tokens + usage_data.completion_tokens
        else:
            base_usage = Usage(requests=1)
            context_tokens = None
        usage = replace(base_usage, cost=self._cost(base_usage))

        return Reply(message=message, usage=usage, context_tokens=context_tokens, stop_reason=collected.finish_reason)

    def _uses_max_completion_tokens(self) -> bool:
        """``True`` for OpenAI reasoning (o-series) models (``max_completion_tokens`` instead of ``max_tokens``)."""
        return bool(_REASONING_MODEL_NAME.match(self.name))

    def _reply_message(
        self,
        text_parts: list[str],
        reasoning_parts: list[str],
        pending_calls: dict[int, dict[str, Any]],
        finish_reason: str | None = None,
    ) -> Message:
        """Builds the final assistant ``Message`` from the pieces collected from the stream.

        Block order: the reasoning ``RawBlock`` (if any), one ``TextBlock`` (if any), then the ``ToolCall``s in
        index order. Empty-string arguments become ``{}``; non-JSON or non-dict arguments become
        ``{INVALID_ARGS_KEY: raw text}``. A missing call id becomes ``call_{12 uuid hex chars}``: if servers that
        give none got ``call_0`` every turn, late results and new calls would get mixed up.
        ``finish_reason == "length"``: the last call may be incomplete even if its JSON parses, so its ``args``
        become ``{INVALID_ARGS_KEY: TRUNCATED_ARGS_MESSAGE}``. Earlier calls are left alone.
        """
        blocks: list[Any] = []
        text = "".join(text_parts)
        if text:
            blocks.append(TextBlock(text))
        indices = sorted(pending_calls)
        for index in indices:
            entry = pending_calls[index]
            call_id = entry["id"] or f"call_{uuid.uuid4().hex[:12]}"
            raw_args = entry["arguments"]
            if raw_args == "":
                args: dict[str, Any] = {}
            else:
                try:
                    parsed = json.loads(raw_args)
                except ValueError:
                    parsed = None
                args = parsed if isinstance(parsed, dict) else {INVALID_ARGS_KEY: raw_args}
            # If cut off by length, the last call (where streaming stopped) may be incomplete even if its
            # argument JSON happens to be valid -- overwrite it so the tool does not run. Earlier calls are
            # complete, so leave them alone.
            if finish_reason == "length" and index == indices[-1]:
                args = {INVALID_ARGS_KEY: TRUNCATED_ARGS_MESSAGE}
            blocks.append(ToolCall(entry["name"], args, call_id))
        reasoning_text = "".join(reasoning_parts)
        if reasoning_text:
            blocks.insert(0, RawBlock(self.provider, {"reasoning_content": reasoning_text}))
        return Message("assistant", tuple(blocks))

    def _to_openai_messages(self, request: Request) -> list[dict[str, Any]]:
        """Converts a ``Request`` into a Chat Completions ``messages`` list."""
        messages: list[dict[str, Any]] = []
        if request.system is not None:
            messages.append({"role": "system", "content": request.system})
        for message in self._merge_same_role(request.messages):
            if message.role == "assistant":
                messages.append(self._assistant_message(message))
            else:
                messages.extend(self._user_messages(message))
        return messages

    def _assistant_message(self, message: Message) -> dict[str, Any]:
        text = message.text
        calls = message.tool_calls
        if text:
            content: str | None = text
        elif calls:
            # With tool_calls, content may be absent (OpenAI rule).
            content = None
        else:
            # OpenAI rejects a message with neither content nor tool_calls -- add a placeholder.
            content = _EMPTY_CONTENT_PLACEHOLDER
        result: dict[str, Any] = {"role": "assistant", "content": content}
        for block in message.content:
            if isinstance(block, RawBlock) and block.provider == self.provider:
                result.update(block.data)
        if calls:
            result["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.args, ensure_ascii=False)},
                }
                for call in calls
            ]
        return result

    def _user_messages(self, message: Message) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for block in message.content:
            if isinstance(block, ToolResultBlock):
                out.append({"role": "tool", "tool_call_id": block.call_id, "content": block.content})
            elif isinstance(block, TextBlock):
                text_parts.append(block.text)
        text = "".join(text_parts)
        if text:
            out.append({"role": "user", "content": text})
        return out

    def _to_openai_tools(self, request: Request) -> list[dict[str, Any]] | None:
        if not request.tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.input_schema,
                },
            }
            for spec in request.tools
        ]

    def _client(self) -> Any:
        """Creates the SDK client only once (thread-safely). ``import openai`` happens here."""
        with self._client_lock:
            if self._sdk_client is None:
                import openai

                self._sdk_client = openai.OpenAI(**self._client_kwargs())
            return self._sdk_client

    def _async_client(self) -> Any:
        """The ``AsyncOpenAI`` client for the running event loop, created once per loop (its connection pool
        belongs to the loop that created it, so a client is not reused across ``asyncio.run`` calls)."""
        loop = asyncio.get_running_loop()
        with self._client_lock:
            client = self._async_clients.get(loop)
            if client is None:
                import openai

                client = openai.AsyncOpenAI(**self._client_kwargs())
                self._async_clients[loop] = client
            return client

    def _client_kwargs(self) -> dict[str, Any]:
        api_key = self.api_key
        if api_key is None and self.base_url is not None and not os.environ.get("OPENAI_API_KEY"):
            api_key = "not-needed"
        kwargs: dict[str, Any] = {"max_retries": self.retries}
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        if api_key is not None:
            kwargs["api_key"] = api_key
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        return kwargs
