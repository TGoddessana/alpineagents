"""@loop and the default loop (ARCHITECTURE.md "Loop contract").

The loop contract is just ``Callable[[Agent, State], Any]`` (for ``arun``, one that returns an awaitable).
``@loop(until=..., limit=...)`` is a convenience that turns a one-turn function into that contract; an
``async def`` body makes an async loop. The header is closed: ``until`` and ``limit``, both required.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import warnings
from collections.abc import Callable
from types import MethodType
from typing import TYPE_CHECKING, Any

from ._async import ASYNC_RUN, is_async_callable
from .blocks import acompact_if_full, compact_if_full
from .errors import fix_message
from .state import State

if TYPE_CHECKING:
    from .agent import Agent

__all__ = ["loop", "Loop", "default_loop", "adefault_loop"]

Condition = Callable[[State], bool]

#: Reserved stopped_by names. Cannot be used as until function names.
_RESERVED_NAMES = ("finish", "limit")

_LOOP_EXAMPLE = "@loop(until=State.is_answered, limit=50)"

_ASYNC_LOOP_EXAMPLE = (
    "@loop(until=State.is_answered, limit=50)\n"
    "async def coding(agent: Agent, state: State):\n"
    "    await acompact_if_full(agent, state)\n"
    "    await agent.athink(state)\n"
    "    if state.wants_tools():\n"
    "        await agent.ause_tools(state)"
)


def _needs_until_and_limit() -> TypeError:
    return TypeError(
        fix_message(
            "@loop needs both until and limit",
            "set both",
            _LOOP_EXAMPLE,
        )
    )


class Loop:
    """A loop made by ``@loop``. Calling ``loop(agent, state)`` repeats the body until it stops.

    Attributes: ``body`` (the original function), ``until`` (tuple of condition functions), ``limit`` (int),
    ``is_async`` (the body is ``async def``, or an async Loop: calling the loop returns a coroutine).
    ``__name__``, ``__doc__`` and ``__wrapped__`` follow the body (functools.update_wrapper).
    Even when the body is another Loop, ``body``/``until``/``limit``/``is_async`` are this loop's own.
    """

    body: Callable[[Any, State], Any]
    until: tuple[Condition, ...]
    limit: int
    is_async: bool

    def __init__(self, body: Callable[[Any, State], Any], *, until: Any, limit: Any) -> None:
        """All checks happen here (mistake-proofing errors, when ``@loop`` is applied):

        - ``until`` or ``limit`` is ``None`` (missing) → ``TypeError``: both are required, example
          ``@loop(until=State.is_answered, limit=50)``
        - ``until`` is a bool (a call result, like ``until=state.is_answered()``) → ``TypeError``: pass
          ``State.is_answered`` without parentheses
        - ``until`` is a method bound to a State instance (``until=state.is_answered``) → ``TypeError``:
          pass ``State.is_answered``
        - ``until`` is one function or a list/tuple of functions. ``TypeError`` if any is not callable.
          If ``__name__`` is ``"<lambda>"``, ``warnings.warn`` (UserWarning): stopped_by needs a name, so use a
          named function. A function named ``"finish"``/``"limit"`` is a ``ValueError`` (reserved).
        - ``limit`` is an int ≥ 1 (not a bool). Otherwise ``TypeError``/``ValueError``.
        """
        if until is None or limit is None:
            raise _needs_until_and_limit()

        items = list(until) if isinstance(until, (list, tuple)) else [until]
        checked: list[Condition] = []
        for item in items:
            if isinstance(item, bool):
                raise TypeError(
                    fix_message(
                        "until got the result of a call (a bool)",
                        "pass the function itself, without parentheses",
                        _LOOP_EXAMPLE,
                    )
                )
            if isinstance(item, MethodType) and isinstance(item.__self__, State):
                raise TypeError(
                    fix_message(
                        "until got a method bound to a State instance",
                        "pass the unbound function itself",
                        _LOOP_EXAMPLE,
                    )
                )
            if not callable(item):
                raise TypeError(
                    fix_message(
                        f"until item {item!r} is not callable",
                        "pass a State -> bool function",
                        _LOOP_EXAMPLE,
                    )
                )
            name = _condition_name(item)
            if name == "<lambda>":
                warnings.warn(
                    "an unnamed function (lambda) in until leaves no name in stopped_by. "
                    "Use a named function.",
                    UserWarning,
                    stacklevel=3,
                )
            elif name in _RESERVED_NAMES:
                raise ValueError(
                    fix_message(
                        f"until function name {name!r} is reserved and cannot be used",
                        "use a function with a different name",
                    )
                )
            checked.append(item)

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError(
                fix_message(
                    f"limit must be an integer of 1 or more (got: {limit!r})",
                    "pass an integer, like limit=50",
                    _LOOP_EXAMPLE,
                )
            )
        if limit < 1:
            raise ValueError(
                fix_message(
                    f"limit must be 1 or more (got: {limit})",
                    "pass an integer of 1 or more, like limit=50",
                    _LOOP_EXAMPLE,
                )
            )

        # update_wrapper merges body.__dict__ into self. Set these after merging, so this loop's values are not
        # overwritten when the body is a Loop (a loop can be a block too) or has body/until/limit attributes.
        functools.update_wrapper(self, body)
        self.body = body
        self.until = tuple(checked)
        self.limit = limit
        self.is_async = is_async_callable(body)

    def __call__(self, agent: Agent, state: State) -> Any:
        """Each call counts turns from 0 again:

        ::

            count = 0
            while True:
                if state.is_finished():   stop("finish")
                for f in until:           if f(state): stop(f.__name__)
                if count >= limit:        stop("limit", limit=limit)
                result = body(agent, state); count += 1
                if result is not None:    TypeError (return value is ignored; set the answer with state.finish(answer))
            stop(reason) = state._set_stopped_by(reason, ...), then return state.answer

        An async loop returns a coroutine doing the same with ``await body(agent, state)`` (``until`` functions
        stay plain functions). Exceptions raised by the body or condition functions propagate as is.
        """
        if self.is_async:
            return self._acall(agent, state)
        count = 0
        while True:
            reason = self._stop_reason(state, count)
            if reason is not None:
                return self._stop(state, reason)
            result = self.body(agent, state)
            count += 1
            if inspect.isawaitable(result):
                _close(result)
                raise TypeError(
                    fix_message(
                        f"loop body {self._body_name()!r} returned an awaitable, but this loop is not async",
                        "write the body as async def to await async blocks and agent.a* methods, and run it "
                        "with await agent.arun(...)",
                        _ASYNC_LOOP_EXAMPLE,
                    )
                )
            self._check_result(result)

    async def _acall(self, agent: Agent, state: State) -> Any:
        token = ASYNC_RUN.set(asyncio.get_running_loop())
        try:
            count = 0
            while True:
                reason = self._stop_reason(state, count)
                if reason is not None:
                    return self._stop(state, reason)
                result = await self.body(agent, state)
                count += 1
                self._check_result(result)
        finally:
            ASYNC_RUN.reset(token)

    def _stop_reason(self, state: State, count: int) -> str | None:
        """Checked before every turn: ``"finish"``, the name of the first true ``until`` function, ``"limit"``."""
        if state.is_finished():
            return "finish"
        for condition in self.until:
            if condition(state):
                return _condition_name(condition)
        if count >= self.limit:
            return "limit"
        return None

    def _check_result(self, result: Any) -> None:
        if result is not None:
            raise TypeError(
                fix_message(
                    f"loop body {self._body_name()!r} returned a value ({result!r})",
                    "the return value is ignored. To set the answer, use state.finish(answer)",
                )
            )

    def _body_name(self) -> str:
        return getattr(self.body, "__name__", repr(self.body))

    def _stop(self, state: State, reason: str) -> Any:
        state._set_stopped_by(reason, limit=self.limit if reason == "limit" else None)
        return state.answer

    def copy(self, *, until: Any = ..., limit: Any = ...) -> Loop:
        """A new Loop with only ``until``/``limit`` changed. Omitted ones keep their value. The original is unchanged.

        New values go through the same checks as ``__init__``.
        """
        new_until = self.until if until is ... else until
        new_limit = self.limit if limit is ... else limit
        return Loop(self.body, until=new_until, limit=new_limit)

    def __repr__(self) -> str:
        names = [getattr(f, "__name__", repr(f)) for f in self.until]
        return f"Loop({self._body_name()}, until={names}, limit={self.limit})"


def loop(fn: Any = None, /, *, until: Any = None, limit: Any = None) -> Any:
    """``@loop(until=..., limit=...)``.

    - Used bare as ``@loop`` (``fn`` is a function) → ``TypeError``: until and limit are required, with an example.
    - ``@loop(until=..., limit=...)`` returns a decorator that takes the body and returns
      ``Loop(body, until=until, limit=limit)``. ``Loop.__init__`` does the checks.
    """
    if fn is not None:
        raise _needs_until_and_limit()

    def decorate(body: Callable[[Any, State], Any]) -> Loop:
        return Loop(body, until=until, limit=limit)

    return decorate


def _close(awaitable: Any) -> None:
    """Closes a coroutine that will never be awaited (so Python does not also warn about it)."""
    close = getattr(awaitable, "close", None)
    if callable(close):
        close()


def _condition_name(condition: Any) -> str:
    """The name recorded in stopped_by. A callable object without ``__name__`` uses its class name."""
    return getattr(condition, "__name__", None) or type(condition).__name__


@loop(until=State.is_answered, limit=50)
def default_loop(agent: Agent, state: State):
    """The default loop. Copy it with ``alpineagents add default_loop`` and edit it to start writing your own loop."""
    compact_if_full(agent, state)
    agent.think(state)
    if state.wants_tools():
        agent.use_tools(state)


@loop(until=State.is_answered, limit=50)
async def adefault_loop(agent: Agent, state: State):
    """The async default loop, used by ``agent.arun()`` when the Agent has the default loop. Same steps as
    ``default_loop``."""
    await acompact_if_full(agent, state)
    await agent.athink(state)
    if state.wants_tools():
        await agent.ause_tools(state)
