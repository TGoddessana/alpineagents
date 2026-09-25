"""The Model role: talks to a provider.

The rules every adapter follows are in the ``Model`` docstring. Also: exceptions raised by ``on_text`` or
``on_event`` (a Reporter's, KeyboardInterrupt) propagate unwrapped, and the SDK is imported on the first request.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import replace

from .._async import Relay, run_in_thread
from .._tokens import context_tokens, estimate_overhead_tokens
from ..errors import fix_message
from ..types import Message, ModelEvent, Price, Reply, Request, Usage

__all__ = ["Model", "OnText", "OnEvent", "COMPACT_PROMPT"]

OnText = Callable[[str], None]
OnEvent = Callable[[ModelEvent], None]

#: The summary request appended by the default ``compact``.
COMPACT_PROMPT = (
    "Summarize the conversation so far. It must be possible to continue the work from this summary alone. "
    "Include, without omission, the goal, what has been done so far, facts learned, files changed, "
    "and remaining work. Do not call any tools."
)


class Model(ABC):
    """Base class for provider adapters. Subclass it to add a provider.

    Implement ``respond`` and ``context_window``. Everything else has a working default: ``arespond`` and
    ``acompact`` run the sync versions on a worker thread, ``compact`` sends a summary request through
    ``respond``, and ``mark_cache`` and ``count_tokens`` need no override.

    Rules for an adapter:

    - Only return a reply. Do not change the State; the Agent records the reply and its usage.
    - Use no network and no credentials in ``__init__``. Create the SDK client on the first request, so an Agent
      can be built without an API key.
    - Let the SDK do retries (``retries`` is passed as its retry count). Wrap only SDK exceptions that finally
      fail, as ``RateLimitError``, ``AuthError``, ``ContextTooLongError`` or ``ProviderError``
      (``raise ... from e``). Let every other exception propagate as is.
    - Print nothing. Report what happens outside the reply (a fallback, a retry) with ``on_event``.
    - Send back ``RawBlock``s whose ``provider`` matches ``self.provider`` unchanged, and drop the others.

    The settings below have class-level defaults, so a subclass works without calling ``super().__init__``.
    """

    name: str = ""
    """Model name without the provider prefix, e.g. ``"claude-sonnet-5"``."""
    provider: str = ""
    """Provider id, e.g. ``"anthropic"``. ``RawBlock``s with this provider are sent back to the model."""
    supports: frozenset[str] = frozenset()
    """Features this adapter supports: ``"thinking"``, ``"cache"``, ``"server_compact"``, ``"vision"``, etc."""
    max_tokens: int | None = None
    """Maximum output tokens per reply. ``None`` uses the adapter's default."""
    temperature: float | None = None
    """Sampling temperature. ``None`` leaves it to the provider."""
    timeout: float | None = None
    """Request timeout in seconds. ``None`` uses the SDK default."""
    retries: int = 2
    """How many times the SDK retries a failed request."""
    price: Price | None = None
    """Token prices used for ``usage.cost``. Without it, ``cost`` is ``None``."""

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Required. The context window size in tokens. ``state.context_used`` is measured against it."""

    @abstractmethod
    def respond(
        self, request: Request, on_text: OnText | None = None, on_event: OnEvent | None = None
    ) -> Reply:
        """Required. Sends one request and returns the reply.

        Args:
            request: The system prompt, messages and tool definitions to send.
            on_text: Called with each text chunk as it streams in. ``None`` when nobody is listening.
            on_event: Called with a ``ModelEvent`` for anything that happens outside the reply, such as a
                fallback to another model. A Model that wraps another Model passes both callbacks through.

        Returns:
            A ``Reply`` whose ``message`` is an assistant message with blocks in the order received,
            ``usage.requests == 1``, ``usage.cost`` from ``price`` (or ``None``), and ``context_tokens`` set to
            all input tokens (cached or not) plus output tokens.

        Raises:
            ProviderError: The provider request finally failed. Use a subclass such as ``RateLimitError`` when
                one fits.
        """
        # Exceptions raised by on_text or on_event propagate unwrapped.

    def count_tokens(self, request: Request) -> int:
        """Estimates the size of ``request`` in tokens.

        A utility for your own code. The framework never calls it, so overriding it does not change
        ``state.context_used`` or when ``compact_if_full`` fires.
        """
        # Default: the token count of the last reply (Message.tokens) plus an estimate for what came after it.
        # State uses the same rule (_tokens) directly.
        return context_tokens(request.messages, estimate_overhead_tokens(request.system, request.tools))

    def compact(
        self, request: Request, instructions: str | None = None, on_event: OnEvent | None = None
    ) -> Reply:
        """Optional. Summarizes the context and returns the summary in ``reply.text``.

        The default works with every model: it appends a summary request (``COMPACT_PROMPT`` plus
        ``instructions``) to the context and calls ``respond`` once with tool calls disabled. Override it for
        providers with server-side compaction.

        Args:
            request: The current context.
            instructions: What the summary must keep.
            on_event: As in ``respond``.
        """
        # Tool definitions are kept in the request: some providers require them when the context has tool_use.
        return self.respond(self._summary_request(request, instructions), None, on_event)

    async def arespond(
        self, request: Request, on_text: OnText | None = None, on_event: OnEvent | None = None
    ) -> Reply:
        """Optional. The async version of ``respond``, used by ``athink`` and ``aask``.

        The default runs ``respond`` on a worker thread and delivers ``on_text`` and ``on_event`` on the event
        loop. Cancelling it drops the reply, but the thread keeps running until its next callback. Override it
        with the provider's async SDK so a cancel also closes the request.
        """
        # The Relay runs callbacks on the loop thread and waits for them, so their exceptions still leave respond
        # as is. After cancel the relay is closed and the worker's next callback raises CancelledError.
        relay = Relay(asyncio.get_running_loop())
        try:
            return await run_in_thread(self.respond, request, relay.wrap(on_text), relay.wrap(on_event))
        finally:
            relay.close()

    async def acompact(
        self, request: Request, instructions: str | None = None, on_event: OnEvent | None = None
    ) -> Reply:
        """Optional. The async version of ``compact``, used by ``Agent.acompact``.

        The default sends the same summary request through ``arespond``. If a subclass overrides only
        ``compact``, that ``compact`` runs on a worker thread instead.
        """
        if type(self).compact is not Model.compact:
            relay = Relay(asyncio.get_running_loop())
            try:
                return await run_in_thread(self.compact, request, instructions, relay.wrap(on_event))
            finally:
                relay.close()
        return await self.arespond(self._summary_request(request, instructions), None, on_event)

    def mark_cache(self, request: Request) -> Request:
        """Optional. Returns the request with prompt cache markers placed. The default returns it unchanged.

        The Agent passes every request through this before ``respond``. ``Request`` has no marker field, so
        adapters that cache usually place markers inside ``respond`` while converting to the provider format.
        """
        return request

    # ------------------------------------------------------------ helpers for adapters

    @staticmethod
    def _summary_request(request: Request, instructions: str | None) -> Request:
        """The default summary request: ``COMPACT_PROMPT`` (+ what to keep) appended, ``tool_choice="none"``."""
        prompt = COMPACT_PROMPT
        if instructions:
            prompt += f"\n\nBe sure to keep: {instructions}"
        return replace(request, messages=request.messages + (Message.user(prompt),), tool_choice="none")

    def _cost(self, usage: Usage) -> float | None:
        """Computed from ``price`` if set, otherwise from ``_lookup_price()``. ``None`` if neither is available."""
        price = self.price if self.price is not None else self._lookup_price()
        return None if price is None else price.cost(usage)

    def _lookup_price(self) -> Price | None:
        """Extension point: look up the table from ``alpineagents[prices]`` (genai-prices) if installed. The MVP
        returns ``None``."""
        return None

    def _check_supported(self, feature: str, setting: str) -> None:
        """``ValueError`` at construction if a configured feature is not supported (mistake-proofing error).

        E.g. ``self._check_supported("thinking", "thinking=True")``.
        """
        if feature not in self.supports:
            listed = ", ".join(sorted(self.supports)) or "(none)"
            raise ValueError(
                fix_message(
                    f"{type(self).__name__}({self.name!r}) does not support {setting} "
                    f"(required feature: {feature!r})",
                    f"Features this adapter supports: {listed}. Remove the unsupported setting, or "
                    f"declare the features the server supports with supports=",
                )
            )

    @staticmethod
    def _merge_same_role(messages: Iterable[Message]) -> list[Message]:
        """Merges consecutive messages with the same role into one (blocks are concatenated). Shared by adapters."""
        merged: list[Message] = []
        for message in messages:
            if merged and merged[-1].role == message.role:
                last = merged[-1]
                merged[-1] = Message(last.role, last.content + message.content, message.tokens)
            else:
                merged.append(message)
        return merged

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"
