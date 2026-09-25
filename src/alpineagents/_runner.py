"""Runs one turn's tool calls (the body of ``Agent.use_tools`` and ``Agent.ause_tools``). Owned by the Agent module.

The contract is in ARCHITECTURE.md "Threads and tool execution" and "Async API".
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, wait
from functools import partial
from typing import TYPE_CHECKING, Any

from .errors import ToolInputError
from .state import closed_outcome, closed_result
from .types import ToolOutcome, format_call

if TYPE_CHECKING:
    from .reporter import Reporter
    from .state import State
    from .tool import Tool
    from .types import ToolCall

__all__ = ["run_calls", "arun_calls", "INPUT_ERROR", "UNKNOWN_TOOL", "LATE_EXCEPTION"]

#: Result when argument validation fails. ``INPUT_ERROR.format(message)``.
INPUT_ERROR = "(input error: {})"
#: Call to a tool that does not exist. ``UNKNOWN_TOOL.format(name, "a, b")``.
UNKNOWN_TOOL = "(input error: unknown tool {!r}. Available tools: {})"
#: Result of a call not waited for on interrupt that later raised. ``LATE_EXCEPTION.format(type_name, error)``.
#: Goes into the ``LATE_RESULT`` notice.
LATE_EXCEPTION = "(exception: {}: {})"

#: Maximum number of worker threads running at once in one turn.
_MAX_WORKERS = 32
#: How long (seconds) the main thread waits at a time. Kept short so Ctrl+C gets through right away.
_POLL = 0.1

_seq_lock = threading.Lock()
_seq = 0


def _next_seq() -> int:
    """Numbers worker threads in the order they finish (to pick the first exception by completion order)."""
    global _seq
    with _seq_lock:
        _seq += 1
        return _seq


class _Job:
    """Execution state of one call. The worker thread and the main thread share it under ``lock``."""

    __slots__ = (
        "call", "tool", "kwargs", "future", "waiter", "seq", "lock", "loop", "task", "cancel_requested",
        "announced", "ended", "cancelled_error", "is_error",
    )

    def __init__(self, call: ToolCall, tool: Tool, kwargs: dict[str, Any]) -> None:
        self.call = call
        self.tool = tool
        self.kwargs = kwargs
        #: The sync API's worker thread future. In the async API, the asyncio Task of an ``async def`` tool.
        self.future: Future[str] | asyncio.Task[str] | None = None
        #: Async API only: what the event loop awaits (the Task itself, or the wrapped thread future).
        self.waiter: asyncio.Future[str] | None = None
        self.seq: float = float("inf")
        self.lock = threading.Lock()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.task: asyncio.Future[Any] | None = None
        self.cancel_requested = False
        #: ``on_tool_start`` was called (so ``on_tool_end`` must also be called exactly once).
        self.announced = False
        #: The main thread has finished this call (recorded, ``on_tool_end``). Never done twice.
        self.ended = False
        #: The ``CancelledError`` standing for a Task the tool cancelled itself (one object, recorded once).
        self.cancelled_error: asyncio.CancelledError | None = None
        #: The tool returned an error result (``Tool.is_error_result``).
        self.is_error = False

    def request_cancel(self) -> None:
        """Cancels the running coroutine (if it has not started yet, it is cancelled as soon as it starts)."""
        with self.lock:
            self.cancel_requested = True
            if self.loop is not None and self.task is not None:
                try:
                    self.loop.call_soon_threadsafe(self.task.cancel)
                except RuntimeError:  # the loop just closed
                    pass


# ---------------------------------------------------------------- worker thread


def _run_awaitable(job: _Job, awaitable: Any) -> Any:
    """Creates a new event loop in the worker thread and awaits it as a Task (kept so it can be cancelled)."""
    loop = asyncio.new_event_loop()
    try:
        task = asyncio.ensure_future(awaitable, loop=loop)
        with job.lock:
            job.loop = loop
            job.task = task
            if job.cancel_requested:
                task.cancel()
        return loop.run_until_complete(task)
    finally:
        with job.lock:
            job.loop = None
            job.task = None
        try:
            remaining = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for t in remaining:
                t.cancel()
            if remaining:
                loop.run_until_complete(asyncio.gather(*remaining, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()


def _work(job: _Job, state: State) -> str:
    """Worker thread: calls the tool and builds the result string. Does no State recording or Reporter notification."""
    try:
        value = job.tool.invoke(job.kwargs, state)
        if inspect.isawaitable(value):
            value = _run_awaitable(job, value)
        text = job.tool.format_result(value)
        job.is_error = job.tool.is_error_result(value)
    except BaseException:
        job.seq = _next_seq()
        raise
    job.seq = _next_seq()
    return text


class _Executor:
    """A small executor that starts worker threads as **daemon** threads (instead of ``ThreadPoolExecutor``).

    ``ThreadPoolExecutor`` worker threads are not daemons, and ``concurrent.futures`` joins them all at
    interpreter exit. The program would then not exit until a sync tool that was closed without waiting on
    interrupt finishes (the framework closes calls without waiting for them). One daemon thread is started
    per call, and a semaphore caps how many run at once at ``max_workers``.

    ``submit`` sets ``job.future`` **before** starting the thread. So wherever Ctrl+C lands, the main thread
    knows the future of every call that has started. A future that has not started yet can be cancelled
    with ``cancel()`` (the worker thread does not call the tool if ``set_running_or_notify_cancel`` is false).
    """

    def __init__(self, max_workers: int) -> None:
        self._slots = threading.Semaphore(max_workers)

    def submit(self, job: _Job, state: State) -> None:
        future: Future[str] = Future()
        job.future = future
        # Tools see the caller's context variables (as with asyncio.to_thread).
        context = contextvars.copy_context()
        thread = threading.Thread(
            target=context.run, args=(self._run, job, future, state), name="alpineagents-tool", daemon=True
        )
        thread.start()

    def _run(self, job: _Job, future: Future[str], state: State) -> None:
        with self._slots:
            if not future.set_running_or_notify_cancel():
                return  # cancelled before it started
            try:
                text = _work(job, state)
            except BaseException as e:
                future.set_exception(e)
            else:
                future.set_result(text)


def _late(state: State, job: _Job, future: Future[str] | asyncio.Task[str]) -> None:
    """A call not waited for on interrupt finished later (the done callback of a worker thread's future, or of an
    abandoned ``async def`` tool Task)."""
    if future.cancelled():
        return
    error = future.exception()
    if error is None:
        text = future.result()
    else:
        if isinstance(error, asyncio.CancelledError) and job.cancel_requested:
            return  # a coroutine we cancelled; nothing new to report
        text = LATE_EXCEPTION.format(type(error).__name__, error)
    state._record_late_result(job.call, text, is_error=error is None and job.is_error)


# ---------------------------------------------------------------- main thread


def _is_pending(state: State, call: ToolCall) -> bool:
    return any(c.id == call.id for c in state.pending_calls)


def _record_if_pending(state: State, call: ToolCall, text: str, is_error: bool = False) -> bool:
    """If ``call`` is still a pending call, records the result and returns true. The check and the record
    happen under one ``state.lock`` (so a tool in the same turn cannot close it with ``state.deny`` in between)."""
    with state.lock:
        if not _is_pending(state, call):
            return False
        state._record_tool_result(call, text, is_error=is_error)
        return True


def _exception(job: _Job) -> BaseException | None:
    """The exception of a finished call. A Task the tool cancelled itself counts as ``CancelledError`` (always the
    same object, so ``_record_error`` records it only once)."""
    assert job.future is not None
    if job.future.cancelled():
        if job.cancelled_error is None:
            job.cancelled_error = asyncio.CancelledError()
        return job.cancelled_error
    return job.future.exception()


def _end(
    state: State, reporter: Reporter | None, job: _Job, text: str | None, outcome: ToolOutcome
) -> None:
    """Marks the call as finished and, if there is a result to report, calls ``on_tool_end``."""
    job.ended = True
    if text is not None and reporter is not None:
        reporter.on_tool_end(state, job.call, text, outcome)


def _finish(state: State, reporter: Reporter | None, job: _Job, failures: list[_Job]) -> None:
    """Records one finished call and calls ``on_tool_end`` (main thread).

    - Success: if it is still a pending call, records the result and calls ``on_tool_end(result, "done")``
      (``"error"`` and ``is_error`` for an error result, ``Tool.is_error_result``).
      If a tool in the same turn already closed it with ``state.deny``, nothing is recorded (``deny`` already
      called ``on_tool_end``).
    - Exception: records ``error``, adds the job to ``failures`` and calls
      ``on_tool_end(closed_result(e), closed_outcome(e))``.
      No result is recorded (the call stays in ``pending_calls`` and ``Agent.run`` closes it with the same text).
    """
    assert job.future is not None
    if job.ended:
        return
    error = _exception(job)
    if error is None:
        text = job.future.result()
        shown: str | None = text if _record_if_pending(state, job.call, text, job.is_error) else None
        outcome = ToolOutcome("error" if job.is_error else "done")
    else:
        state._record_error(error, job.call)
        if job not in failures:
            failures.append(job)
        shown = closed_result(error)
        outcome = closed_outcome(error)
    _end(state, reporter, job, shown if job.announced else None, outcome)


def _collect(
    state: State, reporter: Reporter | None, running: list[_Job], failures: list[_Job]
) -> None:
    """Waits until everything in ``running`` finishes, recording each as it finishes (completion order).

    A job is removed from ``running`` **after** it is recorded: if an interrupt lands in between,
    ``_abandon`` still sees that call.
    """
    while running:
        done, _ = wait([job.future for job in running], timeout=_POLL, return_when=FIRST_COMPLETED)
        if not done:
            continue
        finished = sorted((job for job in running if job.future in done), key=lambda job: job.seq)
        for job in finished:
            _finish(state, reporter, job, failures)
            running.remove(job)


def _abandon(
    state: State, reporter: Reporter | None, running: list[_Job], interrupt: BaseException
) -> None:
    """Interrupt: records what finished, cancels coroutines, and does not wait for sync calls.

    Unfinished calls stay in ``pending_calls`` (``Agent.run`` closes them). Calls that got ``on_tool_start``
    get ``on_tool_end(closed_result(interrupt), closed_outcome(interrupt))`` (to keep the pair).
    """
    closing = closed_result(interrupt)
    closing_outcome = closed_outcome(interrupt)
    for job in running:
        if job.ended:
            continue
        future = job.future
        if future is not None and not future.cancel():  # already running or done
            if not future.done():
                job.request_cancel()
                future.add_done_callback(partial(_late, state, job))
            else:
                _finish(state, reporter, job, [])
                continue
        _end(state, reporter, job, closing if job.announced else None, closing_outcome)
    running.clear()


def _record_now(state: State, reporter: Reporter | None, call: ToolCall, text: str) -> None:
    """A call whose result is decided right away without calling the tool (unknown tool, input error)."""
    if reporter is not None:
        reporter.on_tool_start(state, call)
    state._record_tool_result(call, text, is_error=True)
    if reporter is not None:
        reporter.on_tool_end(state, call, text, ToolOutcome("input_error"))


def _start(
    state: State, reporter: Reporter | None, executor: _Executor, job: _Job, running: list[_Job]
) -> None:
    """Adds to ``running`` first (so ``_abandon`` sees it wherever an interrupt lands), then ``on_tool_start``,
    then hands it to a worker thread."""
    running.append(job)
    job.announced = True
    if reporter is not None:
        reporter.on_tool_start(state, job.call)
    executor.submit(job, state)


def run_calls(
    state: State,
    calls: Sequence[ToolCall],
    tools: Mapping[str, Tool],
    reporter: Reporter | None,
) -> None:
    """Runs ``calls`` (a snapshot of ``state.pending_calls``) and records the results in State.

    All Reporter notifications and State records happen on **the thread that called this function (main)**.
    Worker threads only run tools.

    1. Prepare (in request order): if the tool does not exist the result is ``UNKNOWN_TOOL``; if
       ``tool.prepare(call.args)`` raises ``ToolInputError`` it is ``INPUT_ERROR``: ``on_tool_start`` →
       ``state._record_tool_result(call, result, is_error=True)`` → ``on_tool_end``. The tool is not called.
    2. Parallel group (``tool.parallel``): for each call, ``on_tool_start``, then hand it to ``_Executor``
       (daemon worker threads, at most ``_MAX_WORKERS`` at once).
       The worker thread runs ``value = tool.invoke(kwargs, state)``; if it is a coroutine, it creates a new
       event loop in that thread and runs it as a Task (keeping the loop and Task so it can be cancelled).
       The result is ``Tool.format_result(value)`` (on the worker thread).
    3. As they finish (completion order), on the main thread: success → ``state._record_tool_result(call, text)``
       → ``on_tool_end`` (check and record under one ``state.lock``; a call closed by ``state.deny`` in between
       is not recorded).
       Exception → ``state._record_error(e, call)`` →
       ``on_tool_end(state, call, closed_result(e), closed_outcome(e))``, and the first exception is remembered
       (no result is recorded, so the call stays in ``pending_calls``).
       Waiting repeats ``concurrent.futures.wait(..., timeout=0.1)`` so Ctrl+C gets through right away.
    4. After the whole parallel group finishes, if there was no exception, the serial group (``parallel=False``)
       runs one at a time in request order, the same way (one worker thread). If an exception happens during
       the serial group, the rest are not started.
    5. If there was an exception, the first one gets ``e.add_note(f"exception raised in tool {format_call(call)}")``
       and is re-raised as is (type and message unchanged). Calls that never started stay in ``pending_calls``.

    Interrupt (``KeyboardInterrupt``/``CancelledError`` raised on the main thread):

    - Running coroutine Tasks are cancelled with ``loop.call_soon_threadsafe(task.cancel)``.
    - Unfinished sync calls are not waited for. A done callback is attached to their future, and when they
      finish later, ``state._record_late_result(call, text)`` (if they ended in an exception,
      ``LATE_EXCEPTION``).
    - Futures that have not started are cancelled.
    - Calls that got ``on_tool_start`` but did not finish get
      ``on_tool_end(state, call, closed_result(interrupt_exception), closed_outcome(interrupt_exception))``.
    - Worker threads are daemons, so a sync tool that was not waited for does not hold up program exit.
    - Results of calls that already finished stay recorded, and the interrupt exception is re-raised as is
      (``Agent.run`` does the closing).
    """
    parallel, serial = _prepare(state, calls, tools, reporter)
    if not parallel and not serial:
        return

    executor = _Executor(max(1, min(len(parallel), _MAX_WORKERS)))
    running: list[_Job] = []
    failures: list[_Job] = []
    try:
        for job in parallel:
            _start(state, reporter, executor, job, running)
        _collect(state, reporter, running, failures)

        if not failures:
            for job in serial:
                _start(state, reporter, executor, job, running)
                _collect(state, reporter, running, failures)
                if failures:
                    break
    except BaseException as e:
        _abandon(state, reporter, running, e)
        raise

    _raise_first(failures)


def _prepare(
    state: State, calls: Sequence[ToolCall], tools: Mapping[str, Tool], reporter: Reporter | None
) -> tuple[list[_Job], list[_Job]]:
    """Step 1 of ``run_calls``: records unknown tools and input errors right away and splits the rest into the
    parallel and serial groups (request order)."""
    parallel: list[_Job] = []
    serial: list[_Job] = []
    for call in calls:
        tool = tools.get(call.name)
        if tool is None:
            available = ", ".join(tools) or "(none)"
            _record_now(state, reporter, call, UNKNOWN_TOOL.format(call.name, available))
            continue
        try:
            kwargs = tool.prepare(call.args)
        except ToolInputError as e:
            _record_now(state, reporter, call, INPUT_ERROR.format(e))
            continue
        (parallel if tool.parallel else serial).append(_Job(call, tool, kwargs))
    return parallel, serial


def _raise_first(failures: list[_Job]) -> None:
    """Step 5 of ``run_calls``: re-raises the first exception (completion order) with a note naming the call."""
    if not failures:
        return
    first = min(failures, key=lambda job: job.seq)
    error = _exception(first)
    assert error is not None
    error.add_note(f"exception raised in tool {format_call(first.call)}")
    raise error


# ---------------------------------------------------------------- async API (event loop thread)

#: Tasks of ``async def`` tools that were cancelled and not waited for. Kept here until they finish, so they are
#: not garbage collected midway (the event loop holds tasks only weakly).
_abandoned: set[asyncio.Task[str]] = set()


async def _awork(job: _Job, state: State) -> str:
    """An ``async def`` tool as a Task on the running event loop: calls it and builds the result string."""
    try:
        value = job.tool.invoke(job.kwargs, state)
        if inspect.isawaitable(value):
            value = await value
        text = job.tool.format_result(value)
        job.is_error = job.tool.is_error_result(value)
    except BaseException:
        job.seq = _next_seq()
        raise
    job.seq = _next_seq()
    return text


def _astart(
    state: State, reporter: Reporter | None, executor: _Executor, job: _Job, running: list[_Job]
) -> None:
    """Like ``_start``: an ``async def`` tool becomes a Task on this loop, any other tool goes to a worker thread."""
    running.append(job)
    job.announced = True
    if reporter is not None:
        reporter.on_tool_start(state, job.call)
    if job.tool.is_async:
        task = asyncio.get_running_loop().create_task(_awork(job, state))
        job.future = task
        job.waiter = task
    else:
        executor.submit(job, state)
        assert isinstance(job.future, Future)
        job.waiter = asyncio.wrap_future(job.future)
        # The result is read from job.future; mark the wrapper's exception as seen so asyncio does not log it.
        job.waiter.add_done_callback(_seen)


def _seen(future: asyncio.Future[str]) -> None:
    if not future.cancelled():
        future.exception()


async def _acollect(
    state: State, reporter: Reporter | None, running: list[_Job], failures: list[_Job]
) -> None:
    """Like ``_collect``, awaiting instead of polling."""
    while running:
        waiters = {job.waiter for job in running if job.waiter is not None}
        done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finished = sorted((job for job in running if job.waiter in done), key=lambda job: job.seq)
        for job in finished:
            _finish(state, reporter, job, failures)
            running.remove(job)


def _aabandon(
    state: State, reporter: Reporter | None, running: list[_Job], interrupt: BaseException
) -> None:
    """Like ``_abandon``: ``async def`` tool Tasks are cancelled and not waited for (if a Task still ends with a
    value or another exception, it is recorded as a late result); worker threads are treated as in ``_abandon``."""
    closing = closed_result(interrupt)
    closing_outcome = closed_outcome(interrupt)
    for job in running:
        if job.ended:
            continue
        future = job.future
        if isinstance(future, asyncio.Task):
            if future.done():
                _finish(state, reporter, job, [])
                continue
            job.cancel_requested = True
            future.cancel()
            _abandoned.add(future)
            future.add_done_callback(_abandoned.discard)
            future.add_done_callback(partial(_late, state, job))
        elif future is not None:
            if not future.cancel():  # already running or done
                if future.done():
                    _finish(state, reporter, job, [])
                    continue
                job.request_cancel()
                future.add_done_callback(partial(_late, state, job))
            if job.waiter is not None:
                job.waiter.cancel()  # nobody awaits it any more
        _end(state, reporter, job, closing if job.announced else None, closing_outcome)
    running.clear()


async def arun_calls(
    state: State,
    calls: Sequence[ToolCall],
    tools: Mapping[str, Tool],
    reporter: Reporter | None,
) -> None:
    """The async version of ``run_calls``, run on the event loop thread. Same steps, records and notifications.

    - ``async def`` tools run as Tasks on the running event loop (not on a worker thread's loop). Other tools run
      on daemon worker threads as in ``run_calls``; the loop awaits their futures.
    - Cancellation (``CancelledError`` of the awaiting task) is the interrupt: finished results stay recorded,
      Tasks are cancelled, sync threads are not waited for (late results), and the exception is re-raised as is.
    """
    parallel, serial = _prepare(state, calls, tools, reporter)
    if not parallel and not serial:
        return

    executor = _Executor(max(1, min(len(parallel), _MAX_WORKERS)))
    running: list[_Job] = []
    failures: list[_Job] = []
    try:
        for job in parallel:
            _astart(state, reporter, executor, job, running)
        await _acollect(state, reporter, running, failures)

        if not failures:
            for job in serial:
                _astart(state, reporter, executor, job, running)
                await _acollect(state, reporter, running, failures)
                if failures:
                    break
    except BaseException as e:
        _aabandon(state, reporter, running, e)
        raise

    _raise_first(failures)
