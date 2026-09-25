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
    """A tool call to put in FakeModel replies. ``id`` is new each time: ``"call_"`` + 12 hex digits (uuid4)."""
    return ToolCall(name=name, args=dict(args), id=f"call_{uuid.uuid4().hex[:12]}")


class FakeModel(Model):
    """A Model that returns prepared replies in order. ``provider = "fake"``, ``name`` defaults to ``"fake"``.

    Each reply item is used by one request (``think``, one attempt of ``ask``, ``compact``):

    - ``str`` → a text reply (no tool calls)
    - ``ToolCall`` → a reply with just that one call
    - ``list``/``tuple`` (mixing ``str``, ``ToolCall``, ``RawBlock``) → a reply with those blocks in order
    - ``Reply`` → used as is
    - a ``BaseException`` instance → that exception is raised (for testing error paths)
    - callable → calls ``item(request)`` and converts the result by the rules above
    """

    provider = "fake"

    def __init__(
        self,
        replies: Iterable[Any],
        *,
        name: str = "fake",
        context_window: int = 200_000,
        price: Price | None = None,
    ) -> None:
        """``requests``: a list of received Requests, in order (for assertions). Items are taken thread-safely."""
        self.name = name
        self.price = price
        self._context_window = context_window
        self._replies: list[Any] = list(replies)
        self._index = 0
        self._lock = threading.Lock()
        self.requests: list[Request] = []

    @property
    def context_window(self) -> int:
        return self._context_window

    @property
    def remaining(self) -> int:
        """Number of replies not used yet."""
        with self._lock:
            return len(self._replies) - self._index

    def respond(
        self, request: Request, on_text: OnText | None = None, on_event: OnEvent | None = None
    ) -> Reply:
        """Take the next reply item and build a Reply.

        1. Append ``request`` to ``self.requests``.
        2. If ``self.count_tokens(request)`` is larger than ``context_window``, raise ``ContextTooLongError``
           (no item is used).
        3. If no items are left, raise ``RuntimeError`` ("FakeModel: ran out of prepared replies (all N used)"
           plus the last message).
        4. Call ``on_text(text)`` once per text block (if ``on_text`` is given).
        5. ``Usage(input_tokens=count_tokens(request), output_tokens=<reply estimate>, requests=1,
           cost=self._cost(...))``, ``context_tokens`` = input + output, ``stop_reason`` = ``"tool_use"`` if there
           are tool calls, otherwise ``"end_turn"``.
        """
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
    """A Human that returns prepared answers in order (``FakeHuman(["yes", "always"])``).

    - ``questions``: a list of received questions (prompts), in order.
    - Answers go through ``parse_answer(answer, returns)``. If one does not fit (``ValueError``), the next answer is
      used, as if the person answered again. When answers run out, raises ``RuntimeError``
      ("FakeHuman: ran out of prepared answers", with the last question).
    - Thread-safe (one question at a time).
    """

    def __init__(self, answers: Iterable[Any]) -> None:
        self._answers: list[Any] = list(answers)
        self._index = 0
        self._lock = threading.Lock()
        self.questions: list[str] = []

    @property
    def remaining(self) -> int:
        with self._lock:
            return len(self._answers) - self._index

    def ask(self, state: State, prompt: str, returns: Any = str) -> Any:
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
