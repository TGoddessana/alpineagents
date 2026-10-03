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
from typing import TYPE_CHECKING, Any, cast

from ._async import ASYNC_RUN, is_async_callable
from .blocks import acompact_if_full, compact_if_full
from .errors import fix_message
from .state import State
from .types import Stopped, StoppedByFinish, StoppedByLimit, StoppedByPermission, StoppedByUntil

if TYPE_CHECKING:
    from .agent import Agent

__all__ = ["loop", "Loop", "default_loop", "adefault_loop"]

Condition = Callable[[State], bool]

#: Stops decided outside the loop's own checks (state.finish(), a permission's stop=True): every loop stops on
#: them, an outer loop too. An until/limit stop in state.stopped is a nested loop's, and ends only that loop.
_DECIDED_BY_THE_RUN = (StoppedByFinish, StoppedByPermission)

# An until function is the user's own: it takes the State and says whether to stop.
_UNTIL_EXAMPLE = (
    "def waiting_for_user(state: State) -> bool:\n"
    "    if state.pending_calls or not state.messages:\n"
    "        return False\n"
    "    last = state.messages[-1]\n"
    '    return last.role == "assistant" and not last.tool_calls\n'
    "\n"
)

_LOOP_EXAMPLE = _UNTIL_EXAMPLE + "@loop(until=waiting_for_user, limit=50)"

_ASYNC_LOOP_EXAMPLE = (
    "@loop(until=waiting_for_user, limit=50)\n"
    "async def coding(agent: Agent, state: State):\n"
    "    await acompact_if_full(agent, state)\n"
    "    await agent.athink(state)\n"
    "    if state.pending_calls:\n"
    "        await agent.ause_tools(state)"
)


def is_answered(state: State) -> bool:
    """The default loops' ``until``: the last message is the model's reply without tool calls, and nothing waits for
    a tool result. Not exported. Its name is what ``StoppedByUntil("is_answered")`` and ``str(state.stopped)`` show.

    Reads one snapshot, so the three checks see the same State even while another thread adds a message."""
    snap = state.snapshot()
    if snap.pending_calls or not snap.messages:
        return False
    last = snap.messages[-1]
    return last.role == "assistant" and not last.tool_calls


def _needs_until_and_limit() -> TypeError:
    return TypeError(
        fix_message(
            "@loop needs both until and limit",
            "set both",
            _LOOP_EXAMPLE,
        )
    )


class Loop:
    """A loop made by ``@loop``. Calling ``loop(agent, state)`` runs the body turn by turn and returns
    ``state.answer``.

    Before every turn it checks, in order:

    1. ``state.stopped`` already set by ``state.finish()`` (``StoppedByFinish``) or by a permission's
       ``Denied(..., stop=True)`` (``StoppedByPermission(call, ...)``): stops and keeps it
    2. each ``until`` function: stops with ``state.stopped == StoppedByUntil(name)``, the function's name
    3. the turn count against ``limit``: stops with ``state.stopped == StoppedByLimit(limit)``

    A stop the loop decides (2 and 3) is recorded in the history as a ``StopEntry``. Each call counts turns from
    zero. A nested loop's ``until``/``limit`` stop ends only that loop: the outer loop checks its own ``until`` and
    ``limit`` and, if it goes on, records ``StopEntry(None)``, so ``state.stopped`` goes back to ``None``.
    Exceptions from the body or the ``until`` functions propagate as is.
    An async loop (``async def`` body) returns a coroutine that does the same; its ``until`` functions stay
    plain functions.
    """

    # __name__, __doc__ and __wrapped__ follow the body (functools.update_wrapper). Even when the body is another
    # Loop, body/until/limit/is_async are this loop's own.
    body: Callable[[Any, State], Any]
    """The one-turn function."""
    until: tuple[Condition, ...]
    """The stop conditions, each a ``State -> bool`` function."""
    limit: int
    """The most turns one call runs."""
    is_async: bool
    """``True`` if the body is ``async def``. Calling the loop then returns a coroutine."""

    def __init__(self, body: Callable[[Any, State], Any], *, until: Any, limit: Any) -> None:
        """Usually created with ``@loop``. Every check happens here, when the decorator is applied.

        Args:
            body: The one-turn function ``(agent, state) -> None``, or an ``async def`` one.
            until: A ``State -> bool`` function or a list of them. Pass the function itself, e.g.
                ``waiting_for_user``. A lambda works but leaves no useful name in ``state.stopped``, so it warns.
            limit: The most turns, an integer of 1 or more.

        Raises:
            TypeError: ``until`` or ``limit`` is missing, ``until`` got a call result (``waiting_for_user(state)``),
                an ``until`` item is not callable, or ``limit`` is not an integer.
            ValueError: ``limit`` is less than 1.
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
            if not callable(item):
                raise TypeError(
                    fix_message(
                        f"until item {item!r} is not callable",
                        "pass a State -> bool function",
                        _LOOP_EXAMPLE,
                    )
                )
            if _condition_name(item) == "<lambda>":
                warnings.warn(
                    "an unnamed function (lambda) in until leaves no useful name in state.stopped "
                    '(StoppedByUntil("<lambda>")). Use a named function.',
                    UserWarning,
                    stacklevel=3,
                )
            # Checked above as far as it can be: the bool return is the caller's side of the contract.
            checked.append(cast(Condition, item))

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
        """Each call counts turns from 0 again::

            count = 0
            while True:
                if state.stopped is StoppedByFinish/StoppedByPermission (or state.finished):  stop(it)
                for f in until:           if f(state): stop(StoppedByUntil(f.__name__))
                if count >= limit:        stop(StoppedByLimit(limit))
                if state.stopped is set (a nested loop's until/limit): state._record_stop(None)
                result = body(agent, state); count += 1
                if result is not None:    TypeError (return value is ignored; set the answer with state.finish(answer))
            stop(stopped) = StopEntry(stopped) unless state.stopped already is it, then return state.answer

        An async loop returns a coroutine doing the same with ``await body(agent, state)`` (``until`` functions
        stay plain functions). Exceptions raised by the body or condition functions propagate as is.
        """
        if self.is_async:
            return self._acall(agent, state)
        count = 0
        while True:
            stopped = self._stop_reason(state, count)
            if stopped is not None:
                return self._stop(state, stopped)
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
                stopped = self._stop_reason(state, count)
                if stopped is not None:
                    return self._stop(state, stopped)
                result = await self.body(agent, state)
                count += 1
                self._check_result(result)
        finally:
            ASYNC_RUN.reset(token)

    def _stop_reason(self, state: State, count: int) -> Stopped | None:
        """Checked before every turn, in order: a stop already decided (``finish()`` or a permission's
        ``stop=True``, both record a ``StopEntry`` right away), the first true ``until`` function, the limit.
        Going on after a nested loop stopped on its own ``until``/limit clears that stale reason (``StopEntry(None)``)."""
        stopped = state.stopped
        if isinstance(stopped, _DECIDED_BY_THE_RUN):
            return stopped
        if state.finished:
            # finish() sets stopped, but a run that raised cleared it (the State stays finished).
            return StoppedByFinish()
        for condition in self.until:
            if condition(state):
                return StoppedByUntil(_condition_name(condition))
        if count >= self.limit:
            return StoppedByLimit(self.limit)
        if stopped is not None:
            state._record_stop(None)
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

    def _stop(self, state: State, stopped: Stopped) -> Any:
        # A stop the run decided is in state.stopped already; recording it again would only add an entry.
        if state.stopped != stopped:
            state._record_stop(stopped)
        return state.answer

    def copy(self, *, until: Any = ..., limit: Any = ...) -> Loop:
        """Returns a new Loop with ``until`` or ``limit`` changed. The original is unchanged.

        Example:
            ``quick = coding.copy(limit=5)``

        Args:
            until: New stop conditions. Omit to keep the current ones.
            limit: A new turn limit. Omit to keep the current one.

        Raises:
            TypeError: A new value fails the same checks as ``@loop``.
            ValueError: A new value fails the same checks as ``@loop``.
        """
        new_until = self.until if until is ... else until
        new_limit = self.limit if limit is ... else limit
        return Loop(self.body, until=new_until, limit=new_limit)

    def __repr__(self) -> str:
        names = [getattr(f, "__name__", repr(f)) for f in self.until]
        return f"Loop({self._body_name()}, until={names}, limit={self.limit})"


def loop(fn: Any = None, /, *, until: Any = None, limit: Any = None) -> Any:
    """Turns a one-turn function into a loop that takes ``(agent, state)`` and returns the answer.

    ``until`` and ``limit`` are both required, so how the loop stops is always written next to it. The body
    returns nothing; to set the answer yourself, call ``state.finish(answer)``. An ``async def`` body makes an
    async loop for ``agent.arun``.

    Example:
        ```python
        def waiting_for_user(state: State) -> bool:
            if state.pending_calls or not state.messages:
                return False
            last = state.messages[-1]
            return last.role == "assistant" and not last.tool_calls

        @loop(until=waiting_for_user, limit=50)
        def coding(agent: Agent, state: State):
            compact_if_full(agent, state)
            agent.think(state)
            if state.pending_calls:
                agent.use_tools(state)
        ```

    Args:
        until: A ``State -> bool`` function or a list of them, checked before every turn.
        limit: The most turns one run takes. Reaching it stops quietly with
            ``state.stopped == StoppedByLimit(limit)``.

    Returns:
        A decorator that turns the body into a ``Loop``.

    Raises:
        TypeError: ``@loop`` is used bare, without ``until`` and ``limit``. See ``Loop`` for the other checks.
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
    """The name recorded in ``StoppedByUntil``. A callable object without ``__name__`` uses its class name."""
    return getattr(condition, "__name__", None) or type(condition).__name__


@loop(until=is_answered, limit=50)
def default_loop(agent: Agent, state: State):
    """The loop ``agent.run`` uses when the Agent has no loop of its own.

    Each turn compacts the messages if the context is over 60% full, asks the model, and runs the tools it asked
    for. Stops when the model answers without tool calls (``stopped by is_answered``) or after 50 turns. Copy it as
    a starting point for your own loop.
    """
    compact_if_full(agent, state)
    agent.think(state)
    if state.pending_calls:
        agent.use_tools(state)


@loop(until=is_answered, limit=50)
async def adefault_loop(agent: Agent, state: State):
    """The async version of ``default_loop``, used by ``agent.arun`` when the Agent has no loop of its own."""
    await acompact_if_full(agent, state)
    await agent.athink(state)
    if state.pending_calls:
        await agent.ause_tools(state)
