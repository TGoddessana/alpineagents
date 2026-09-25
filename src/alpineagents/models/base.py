"""The Model role: talks to a provider.

A new provider works once it implements the two required members (``respond``, ``context_window``). Everything
else has a default implementation. Shared settings have class-attribute defaults, so subclasses work even if they
do not call ``super().__init__``.

Rules shared by all adapters (ARCHITECTURE.md "Model adapter contract"):

- Do not change State. Only return a reply.
- Do not use the network or require credentials at construction. The SDK client and the SDK import are created on
  the first request.
- Wrap only SDK exceptions in the common types (``raise RateLimitError(...) from e``); let any other exception
  (e.g. one raised by the Reporter in ``on_text``, KeyboardInterrupt) propagate as is.
- Retries use the SDK's retries (``retries`` → SDK ``max_retries``).
- Things that happen outside the reply (falling back to another model, etc.) are not printed; report them with
  ``on_event(ModelEvent(...))``. The Agent does the history and Reporter notifications. Exceptions raised by
  ``on_event`` are not wrapped either; they propagate as is.
- Send back ``RawBlock``s of the same ``provider`` as is, and drop the others.
- ``arespond``/``acompact`` (the async API) work for every adapter by running the sync version on a worker thread.
  Adapters with an async SDK override ``arespond`` so a cancel also stops the request.
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
    """Base class for provider adapters.

    Shared settings (same names in every adapter): ``max_tokens``, ``temperature``, ``timeout``, ``retries``, ``price``.
    """

    #: Model name (without the provider prefix). E.g. ``"claude-sonnet-5"``
    name: str = ""
    #: Name used for ``RawBlock.provider``. E.g. ``"anthropic"``
    provider: str = ""
    #: Supported features: ``"thinking"``, ``"cache"``, ``"server_compact"``, ``"vision"``, etc.
    supports: frozenset[str] = frozenset()
    max_tokens: int | None = None
    temperature: float | None = None
    timeout: float | None = None
    retries: int = 2
    price: Price | None = None

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Required. Context window size (tokens). The basis for ``state.context_used``."""

    @abstractmethod
    def respond(
        self, request: Request, on_text: OnText | None = None, on_event: OnEvent | None = None
    ) -> Reply:
        """Required. Calls the API once (streaming) and calls ``on_text(chunk)`` as text arrives.

        Things that happen outside the reply are reported with ``on_event(ModelEvent(...))`` (no need to call it
        if there are none). Implementations that wrap another Model pass ``on_text`` and ``on_event`` through.

        The returned ``Reply``: ``message.role == "assistant"``, blocks in the order received,
        ``usage.requests == 1``, ``usage.cost`` is ``self._cost(usage)``, ``context_tokens`` is
        input (including cache) + output tokens.
        """

    def count_tokens(self, request: Request) -> int:
        """A utility for callers that want a size estimate of ``request``. The Agent and State never call it:
        ``state.context_tokens``/``context_used`` (and so ``compact_if_full``) always use ``_tokens``, so
        overriding this does not change them. Default: the usage of the last reply (``Message.tokens``) plus an
        estimate for what was added since."""
        return context_tokens(request.messages, estimate_overhead_tokens(request.system, request.tools))

    def compact(
        self, request: Request, instructions: str | None = None, on_event: OnEvent | None = None
    ) -> Reply:
        """Optional. Returns a reply that summarizes the context (summary text in ``reply.text``). The Agent
        records the usage.

        The default is a summary request that works with every model: append ``COMPACT_PROMPT`` (+ what to keep)
        to the end of the context as a user message, keep the tool definitions, and ``respond`` once with
        ``tool_choice="none"``. Adapters that support server-side compaction override it.
        """
        return self.respond(self._summary_request(request, instructions), None, on_event)

    async def arespond(
        self, request: Request, on_text: OnText | None = None, on_event: OnEvent | None = None
    ) -> Reply:
        """Optional. The async version of ``respond``, used by ``athink``/``aask``.

        The default runs ``respond`` on a daemon worker thread. ``on_text``/``on_event`` are relayed to the event
        loop thread (so Reporter notifications and State records happen there, and their exceptions propagate
        out of ``respond`` as is). When the caller is cancelled, the thread cannot be stopped: its next callback
        raises ``CancelledError`` (ending a stream early) and its reply is dropped. Adapters with an async SDK
        override this.
        """
        relay = Relay(asyncio.get_running_loop())
        try:
            return await run_in_thread(self.respond, request, relay.wrap(on_text), relay.wrap(on_event))
        finally:
            relay.close()

    async def acompact(
        self, request: Request, instructions: str | None = None, on_event: OnEvent | None = None
    ) -> Reply:
        """Optional. The async version of ``compact``, used by ``acompact``.

        The default sends the same summary request as ``compact`` with ``arespond``. An adapter that overrides
        only ``compact`` (server-side compaction) gets its ``compact`` run on a worker thread instead.
        """
        if type(self).compact is not Model.compact:
            relay = Relay(asyncio.get_running_loop())
            try:
                return await run_in_thread(self.compact, request, instructions, relay.wrap(on_event))
            finally:
                relay.close()
        return await self.arespond(self._summary_request(request, instructions), None, on_event)

    def mark_cache(self, request: Request) -> Request:
        """Optional. Decides where to place prompt cache markers. The default returns the request unchanged.

        Request has no cache marker field, so adapters that use caching usually mark it inside ``respond`` when
        converting to the provider format. The Agent always passes the request through this method before
        ``respond``.
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
