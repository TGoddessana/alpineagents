"""Helpers the async API shares (ARCHITECTURE.md "Async API").

- ``run_in_thread``: runs a blocking function on a daemon thread and returns an awaitable (not
  ``asyncio.to_thread``: its pool is joined when ``asyncio.run`` ends, so a call that was cancelled and not waited
  for would hold the program until it finishes).
- ``Relay``: calls callbacks that a worker thread invokes (``on_text``/``on_event``) on the event loop thread and
  waits for them, so their exceptions reach the worker thread as is.
- ``ASYNC_RUN`` and ``check_sync_call``: a sync Agent method called on the event loop thread of an async run would
  block every other task, so it is reported with the async name to use instead.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import Future, wait
from typing import Any, TypeVar

from .errors import fix_message

__all__ = ["ASYNC_RUN", "Relay", "check_sync_call", "is_async_callable", "run_in_thread"]

T = TypeVar("T")

#: The event loop running the current async run (set by ``Agent.arun`` and async loops). ``None`` otherwise.
ASYNC_RUN: contextvars.ContextVar[asyncio.AbstractEventLoop | None] = contextvars.ContextVar(
    "alpineagents_async_run", default=None
)

#: How long (seconds) a worker thread waits at a time for a relayed callback, so it notices a cancel.
_POLL = 0.1


def run_in_thread(fn: Callable[..., T], /, *args: Any) -> Awaitable[T]:
    """Runs ``fn(*args)`` on a new daemon thread, with a copy of the current context. Cancelling the awaitable does
    not stop the thread; the result is dropped."""
    future: Future[T] = Future()
    context = contextvars.copy_context()

    def target() -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            result = context.run(fn, *args)
        except BaseException as e:
            future.set_exception(e)
        else:
            future.set_result(result)

    threading.Thread(target=target, name="alpineagents-worker", daemon=True).start()
    return asyncio.wrap_future(future)


class Relay:
    """Wraps callbacks so a worker thread runs them on ``loop``'s thread and gets their result or exception.

    After ``close()`` (the awaiting side finished or was cancelled), a wrapped callback raises
    ``CancelledError`` in the worker thread instead of running, which also ends a stream early.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._closed = False
        self._lock = threading.Lock()

    def wrap(self, callback: Callable[..., Any] | None) -> Callable[..., Any] | None:
        if callback is None:
            return None
        return lambda *args: self._call(callback, args)

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def _call(self, callback: Callable[..., Any], args: tuple[Any, ...]) -> Any:
        done: Future[Any] = Future()

        def run() -> None:
            # close() runs on this (the loop's) thread, so a callback queued before it is dropped here.
            if self._closed:
                done.cancel()
                return
            if not done.set_running_or_notify_cancel():
                return
            try:
                done.set_result(callback(*args))
            except BaseException as e:
                done.set_exception(e)

        with self._lock:
            if self._closed:
                raise asyncio.CancelledError()
            try:
                self._loop.call_soon_threadsafe(run)
            except RuntimeError:  # the loop is closed
                raise asyncio.CancelledError() from None
        # Poll with wait(), not result(timeout=): a TimeoutError raised by the callback must propagate as is.
        while not wait([done], timeout=_POLL).done:
            if self._closed:
                done.cancel()
                raise asyncio.CancelledError()
        if done.cancelled():
            raise asyncio.CancelledError()
        return done.result()


def check_sync_call(method: str, hint: str = "") -> None:
    """``TypeError`` if ``agent.{method}()`` (sync) is called on the event loop thread of an async run.

    The sync API still works where a loop is running but no async run is (Jupyter), and on worker threads.
    """
    run_loop = ASYNC_RUN.get()
    if run_loop is None:
        return
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        return
    if running is run_loop:
        raise TypeError(
            fix_message(
                f"agent.{method}() was called inside an async run, where it would block the event loop",
                f"Use the async version, await agent.a{method}(...){hint}",
                f"await agent.a{method}(state)" if method != "run" else "answer = await agent.arun(state)",
            )
        )


def is_async_callable(obj: Any) -> bool:
    """Whether calling ``obj`` returns an awaitable: an ``async def`` function (or a partial of one), an async
    ``@loop`` (``is_async``), or an object whose ``__call__`` is ``async def``."""
    if inspect.iscoroutinefunction(obj):
        return True
    if getattr(obj, "is_async", False) is True:
        return True
    return inspect.iscoroutinefunction(getattr(type(obj), "__call__", None))
