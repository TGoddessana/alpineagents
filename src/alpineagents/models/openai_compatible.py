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
    """Adapter for OpenAI-compatible servers. ``provider = "openai_compatible"``.

    The default ``supports`` is empty (it varies by server). Declare the features the server supports with
    ``supports=``.
    """

    provider = "openai_compatible"

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
        """Only stores settings. Must be constructible without network or credentials (the client is created on
        the first ``respond``).

        - If ``base_url`` is None, the SDK default (``OPENAI_BASE_URL`` or api.openai.com).
        - If ``api_key`` is None, the SDK reads ``OPENAI_API_KEY``. If that is missing too and this is a local
          server given by ``base_url``, ``"not-needed"`` is passed so the SDK does not fail at construction (a
          real key in the environment is still used as is -- it is not ignored just because ``base_url`` is set).
        """
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
        return self._context_window

    def respond(
        self, request: Request, on_text: OnText | None = None, on_event: OnEvent | None = None
    ) -> Reply:
        """``client.chat.completions.create(..., stream=True, stream_options={"include_usage": True})``.

        Sending:

        - ``system`` → first ``{"role": "system", "content": system}``
        - assistant ``Message`` → ``{"role": "assistant", "content": text or None,
          "tool_calls": [{"id", "type": "function", "function": {"name", "arguments": json.dumps(args)}}]}``.
          The ``data`` of ``RawBlock(provider="openai_compatible")`` is merged into that message dict as is
          (e.g. ``{"reasoning_content": ...}``). RawBlocks of other providers are dropped.
        - user ``Message``: one ``{"role": "tool", "tool_call_id", "content"}`` per ``ToolResultBlock``, in block
          order, then the remaining text as a single ``{"role": "user", "content": text}``.
        - ``tools`` → ``[{"type": "function", "function": {"name", "description", "parameters": input_schema}}]``
          (omitted if empty). ``tool_choice="none"`` if ``tool_choice="none"`` and there are tools.
        - ``max_tokens`` and ``temperature`` are sent only when not None. But if the model name matches OpenAI's
          reasoning (o-series, e.g. ``o3``) pattern, ``max_completion_tokens`` is sent instead of ``max_tokens``
          (``max_tokens`` is the old name that o-series models reject).

        Receiving:

        - ``on_text(chunk)`` for each ``delta.content``. ``delta.tool_calls`` id/name/arguments are concatenated
          per ``index``.
        - Final block order: one ``TextBlock`` for the text (if any), then the ``ToolCall``s (in index order).
          Empty-string arguments become ``{}``; non-JSON or non-dict arguments become
          ``{INVALID_ARGS_KEY: raw text}``. A missing call id (servers that give none) becomes
          ``f"call_{12 uuid hex chars}"``. If the same id (``call_0``) repeated every turn, late results and new
          calls would easily get mixed up.
          ``finish_reason == "length"`` means the reply was cut off, so the last tool call (where streaming
          stopped) may be incomplete even if its arguments happen to be valid JSON: that call's ``args`` become
          ``{INVALID_ARGS_KEY: TRUNCATED_ARGS_MESSAGE}`` so the tool does not run (earlier, complete calls are
          left alone).
        - Reasoning text from the server (``delta.reasoning_content`` etc., an extra field on the SDK model), if
          any, goes first as ``RawBlock("openai_compatible", {"reasoning_content": concatenated value})``.
        - Usage: ``prompt_tokens``, ``completion_tokens``, ``prompt_tokens_details.cached_tokens`` and
          ``.cache_write_tokens`` (0 if missing) →
          ``Usage(input_tokens=prompt - cached - cache_write, output_tokens=completion,
          cache_read_tokens=cached, cache_write_tokens=cache_write, requests=1)``,
          ``cost=self._cost(...)``. Zeros if no usage arrives. ``context_tokens`` = prompt + completion
          (None if there is no usage). ``stop_reason`` = ``finish_reason``.

        Errors: ``openai.RateLimitError`` → ``RateLimitError``; ``AuthenticationError``/
        ``PermissionDeniedError``/the ``openai.OpenAIError`` raised when the client is created without
        credentials → ``AuthError``; a ``BadRequestError`` with ``code == "context_length_exceeded"``
        or a context overflow message (context length, maximum context, etc.) → ``ContextTooLongError``;
        any other ``openai.OpenAIError`` → ``ProviderError``. ``from e`` keeps the original.
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
        """The async version of ``respond`` with ``AsyncOpenAI``: the same request, reply and errors. A cancel
        closes the stream."""
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
        """The common error type for an SDK exception (see ``respond``)."""
        import openai

        if isinstance(e, openai.RateLimitError):
            return RateLimitError(str(e))
        if isinstance(e, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return AuthError(str(e))
        if isinstance(e, openai.BadRequestError) and _looks_like_context_too_long(e):
            return ContextTooLongError(str(e))
        return ProviderError(str(e))

    def _to_reply(self, collected: _Collected) -> Reply:
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
        """Builds the final assistant ``Message`` from the pieces collected from the stream."""
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
