"""Test doubles that need no API: FakeModel, FakeHuman, tool_call.

::

    fake = FakeModel([tool_call("read_file", path="main.py"), "The bug is on line 3"])
    agent.copy(model=fake, reporter=None).run(State("Find the bug"))
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ._tokens import estimate_text_tokens
from .errors import ContextTooLongError
from .human import Human, parse_answer
from .models.base import Model, OnEvent, OnText
from .types import Message, Price, Reply, Request, TextBlock, ToolCall, Usage

if TYPE_CHECKING:
    from .state import State

__all__ = ["FakeModel", "FakeHuman", "tool_call"]


def tool_call(name: str, /, **args: Any) -> ToolCall:
    """Builds a tool call to put in ``FakeModel`` replies.

    Args:
        name: The tool name.
        **args: The tool arguments.

    Returns:
        A ``ToolCall`` with a new id each time (``"call_"`` + 12 hex digits).

    Example:
        ```python
        FakeModel([tool_call("read_file", path="main.py"), "Done"])
        ```
    """
    return ToolCall(name=name, args=dict(args), id=f"call_{uuid.uuid4().hex[:12]}")


class FakeModel(Model):
    """A Model that returns prepared replies in order, with no API key or network.

    Each request (``think``, one attempt of ``ask``, ``compact``) uses the next item:

    - ``str``: a text reply with no tool calls
    - ``ToolCall``: a reply with just that call (see ``tool_call``)
    - ``list`` or ``tuple`` of ``str``, ``ToolCall`` and ``RawBlock``: a reply with those blocks in order
    - ``Reply``: used as is
    - an exception instance: raised, for testing error paths
    - a callable: called with the ``Request``, and its result is used by the rules above

    Token usage is estimated from the text, so ``state.usage`` fills in as it would with a real model.
    Safe to use from several threads.

    Args:
        replies: The reply items, in order.
        name: Model name.
        context_window: Context window size in tokens. A request estimated above it raises
            ``ContextTooLongError``, so compaction can be tested.
        price: Token prices for ``usage.cost``.

    Example:
        ```python
        fake = FakeModel([tool_call("read_file", path="main.py"), "The bug is on line 3"])
        agent.copy(model=fake, reporter=None).run(State("Find the bug"))
        assert fake.remaining == 0
        ```
    """

    provider = "fake"
    """Always ``"fake"``."""
    name: str
    """Model name, ``"fake"`` by default."""
    price: Price | None
    """Token prices used for ``usage.cost``."""
    requests: list[Request]
    """Every request received, in order. Use it to assert what the agent sent."""

    def __init__(
        self,
        replies: Iterable[Any],
        *,
        name: str = "fake",
        context_window: int = 200_000,
        price: Price | None = None,
    ) -> None:
        # Items are taken under _lock, so concurrent runs share one reply list safely.
        self.name = name
        self.price = price
        self._context_window = context_window
        self._replies: list[Any] = list(replies)
        self._index = 0
        self._lock = threading.Lock()
        self.requests: list[Request] = []

    @property
    def context_window(self) -> int:
        """Context window size in tokens, as passed to ``context_window=``."""
        return self._context_window

    @property
    def remaining(self) -> int:
        """Number of replies not used yet."""
        with self._lock:
            return len(self._replies) - self._index

    def respond(
        self, request: Request, on_text: OnText | None = None, on_event: OnEvent | None = None
    ) -> Reply:
        """Records ``request`` in ``requests`` and returns the next prepared reply.

        Each text block is passed to ``on_text`` once.

        Raises:
            ContextTooLongError: The estimated request size is above ``context_window``. No item is used.
            RuntimeError: Every prepared reply has been used.
        """
        # Usage: input = count_tokens(request), output = estimate of the reply text, requests=1, cost from price.
        # context_tokens = input + output. stop_reason = "tool_use" with tool calls, otherwise "end_turn".
        with self._lock:
            self.requests.append(request)
            estimated = self.count_tokens(request)
            if estimated > self._context_window:
                raise ContextTooLongError(
                    f"FakeModel({self.name!r}): estimated input of {estimated} tokens is larger than "
                    f"context_window={self._context_window}"
                )
            if self._index >= len(self._replies):
                last_text = request.messages[-1].text if request.messages else ""
                raise RuntimeError(
                    f"FakeModel: ran out of prepared replies (all {len(self._replies)} used). "
                    f"Last message: {last_text!r}"
                )
            item = self._replies[self._index]
            self._index += 1

        return self._build_reply(item, request, on_text)

    def _build_reply(self, item: Any, request: Request, on_text: OnText | None) -> Reply:
        if callable(item):
            item = item(request)

        if isinstance(item, BaseException):
            raise item
        if isinstance(item, Reply):
            return item

        if isinstance(item, ToolCall):
            blocks: tuple[Any, ...] = (item,)
        elif isinstance(item, (list, tuple)):
            blocks = tuple(TextBlock(b) if isinstance(b, str) else b for b in item)
        elif isinstance(item, str):
            blocks = (TextBlock(item),)
        else:
            raise TypeError(f"FakeModel: unsupported reply item {item!r}")

        if on_text is not None:
            for block in blocks:
                if isinstance(block, TextBlock):
                    on_text(block.text)

        message = Message("assistant", blocks)
        input_tokens = self.count_tokens(request)
        output_tokens = sum(
            estimate_text_tokens(block.text) for block in blocks if isinstance(block, TextBlock)
        )
        usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens, requests=1)
        usage = replace(usage, cost=self._cost(usage))
        stop_reason = "tool_use" if any(isinstance(b, ToolCall) for b in blocks) else "end_turn"
        return Reply(
            message=message,
            usage=usage,
            context_tokens=input_tokens + output_tokens,
            stop_reason=stop_reason,
        )


class FakeHuman(Human):
    """A Human that returns prepared answers in order, for testing code that calls ``ask_human``.

    Each answer is converted to the requested ``returns`` type the same way a typed answer is. An answer that
    does not fit is skipped and the next one is used, as if the person answered again. An empty answer does not
    fit ``returns=str``. Safe to use from several threads; questions are answered one at a time.

    Args:
        answers: The answers, in order.

    Example:
        ```python
        human = FakeHuman(["yes", "always"])
        agent.copy(model=fake, human=human, reporter=None).run(state)
        assert human.remaining == 0
        ```
    """

    questions: list[str]
    """Every question asked, in order."""

    def __init__(self, answers: Iterable[Any]) -> None:
        self._answers: list[Any] = list(answers)
        self._index = 0
        self._lock = threading.Lock()
        self.questions: list[str] = []

    @property
    def remaining(self) -> int:
        """Number of answers not used yet."""
        with self._lock:
            return len(self._answers) - self._index

    def ask(self, state: State, prompt: str, returns: Any = str) -> Any:
        """Records ``prompt`` in ``questions`` and returns the next answer that fits ``returns``.

        Raises:
            RuntimeError: Every prepared answer has been used.
        """
        with self._lock:
            self.questions.append(prompt)
            while True:
                if self._index >= len(self._answers):
                    raise RuntimeError(
                        f"FakeHuman: ran out of prepared answers. Last question: {prompt!r}"
                    )
                answer = self._answers[self._index]
                self._index += 1
                try:
                    return parse_answer(answer, returns)
                except ValueError:
                    continue
