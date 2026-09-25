"""Regression tests for defects a review found in the Agent group (agent.py, _runner.py, _structured.py). No network."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path

import pytest

from alpineagents import Agent, Reporter, State, tool
from alpineagents import _runner
from alpineagents._structured import parse_reply
from alpineagents.state import INTERRUPTED, closed_result
from alpineagents.testing import FakeModel, tool_call
from alpineagents.types import ToolOutcome

SRC = str(Path(__file__).resolve().parents[1] / "src")


class _Recorder(Reporter):
    """A Reporter that records call order and arguments."""

    def __init__(self) -> None:
        self.events: list[tuple] = []

    def on_run_start(self, state):
        self.events.append(("on_run_start",))

    def on_run_end(self, state, error):
        self.events.append(("on_run_end", error))

    def on_think_start(self, state):
        self.events.append(("on_think_start",))

    def on_think_end(self, state, reply):
        self.events.append(("on_think_end", reply))

    def on_tool_start(self, state, call):
        self.events.append(("on_tool_start", call.name))

    def on_tool_end(self, state, call, result, outcome):
        self.events.append(("on_tool_end", call.name, result, outcome))

    def on_context_change(self, state, change):
        self.events.append(("on_context_change", change.kind))

    def names(self, kind: str) -> list[str]:
        return [e[1] for e in self.events if e[0] == kind]

    def end_result(self, name: str) -> str:
        (result,) = [e[2] for e in self.events if e[0] == "on_tool_end" and e[1] == name]
        return result

    def end_outcome(self, name: str) -> ToolOutcome:
        (outcome,) = [e[3] for e in self.events if e[0] == "on_tool_end" and e[1] == name]
        return outcome


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


# ================================================================ ask(returns=...): ``` fences inside JSON


@dataclass
class Review:
    approved: bool
    comments: str


_FENCED_COMMENT = "Change to:\n```python\nx = 1\n```"


def test_json_with_code_fence_in_string_value_parses():
    """A well-formed answer is accepted: a ``` fence inside a string value does not make the JSON get dropped."""
    text = json.dumps({"approved": False, "comments": _FENCED_COMMENT}, ensure_ascii=False)
    json.loads(text)  # valid JSON on its own
    assert parse_reply(text, Review) == Review(approved=False, comments=_FENCED_COMMENT)


def test_fenced_json_whose_string_value_contains_a_fence_parses():
    """JSON wrapped in a ```json fence is found even when a string value contains another ```."""
    body = json.dumps({"approved": True, "comments": _FENCED_COMMENT}, ensure_ascii=False)
    text = f"```json\n{body}\n```"
    assert parse_reply(text, Review) == Review(approved=True, comments=_FENCED_COMMENT)


def test_plain_fenced_json_and_prose_still_parse():
    """Stripping fences and finding JSON among chatter still work."""
    assert parse_reply('```json\n{"approved": true, "comments": "ok"}\n```', Review) == Review(True, "ok")
    assert parse_reply('Answer: {"approved": true, "comments": "ok"} end', Review) == Review(True, "ok")


def test_ask_accepts_json_answer_with_fence_in_string_value():
    """The reviewer example: reviewer.ask(state, "Review this", returns=Review) succeeds on the first try."""
    reply = json.dumps({"approved": False, "comments": _FENCED_COMMENT}, ensure_ascii=False)
    fake = FakeModel([reply])
    agent = Agent(model=fake, reporter=None)
    review = agent.ask(State("Task"), "Review this", returns=Review)
    assert review == Review(approved=False, comments=_FENCED_COMMENT)
    assert len(fake.requests) == 1


# ================================================================ after Ctrl+C the process does not wait for the tool


_CHILD = """
import os, signal, sys, threading, time
sys.path.insert(0, {src!r})
from alpineagents import Agent, State, tool
from alpineagents.testing import FakeModel, tool_call

t0 = time.monotonic()

@tool
def slow_tool() -> str:
    \"\"\"A slow sync tool.\"\"\"
    time.sleep(20)
    return "done"

def send_sigint():
    time.sleep(0.3)
    os.kill(os.getpid(), signal.SIGINT)

agent = Agent(model=FakeModel([tool_call("slow_tool")]), tools=[slow_tool], reporter=None)
threading.Thread(target=send_sigint, daemon=True).start()
try:
    agent.run(State("Task"))
except KeyboardInterrupt:
    print(f"RAISED_AT={{time.monotonic() - t0:.3f}}", flush=True)
    sys.exit(1)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="sends SIGINT to its own process")
def test_process_exits_promptly_after_ctrlc_during_blocked_sync_tool(tmp_path):
    """"For a sync tool ... the framework closes the call without waiting": the program also exits without
    waiting for that tool."""
    child = tmp_path / "child.py"
    child.write_text(_CHILD.format(src=SRC))

    t0 = time.monotonic()
    proc = subprocess.run([sys.executable, str(child)], capture_output=True, text=True, timeout=30)
    wall = time.monotonic() - t0

    assert "RAISED_AT" in proc.stdout, proc.stderr
    raised_at = float(proc.stdout.strip().split("=")[1])
    assert raised_at < 5.0
    assert wall < 10.0, (
        f"process took {wall:.1f}s to exit (run raised at {raised_at:.1f}s): it waited for the worker thread"
    )


# ================================================================ _finish: deny between the check and the record


def _two_pending_calls():
    call_a, call_b = tool_call("a"), tool_call("b")
    agent = Agent(model=FakeModel([[call_a, call_b]]), tools=[], reporter=None)
    state = State("task")
    agent.think(state)
    return state, call_a, call_b


def _done_job(call, result: str) -> _runner._Job:
    job = _runner._Job(call, tool=None, kwargs={})  # type: ignore[arg-type]
    job.future = Future()
    job.future.set_result(result)
    return job


def test_finish_does_not_raise_when_deny_races_the_check(monkeypatch):
    """"If a finished call is no longer a pending call by then (a tool in the same turn closed it with state.deny),
    its result is not recorded": another thread's deny cannot slip in between the check and the record."""
    state, call_a, call_b = _two_pending_calls()
    job = _done_job(call_b, "b's result")

    barrier = threading.Barrier(2)
    orig_is_pending = _runner._is_pending

    def widened_is_pending(state, call):
        result = orig_is_pending(state, call)
        barrier.wait(timeout=5)
        time.sleep(0.1)  # widen the gap between the check and the record
        return result

    monkeypatch.setattr(_runner, "_is_pending", widened_is_pending)
    deny_errors: list[BaseException] = []

    def deny_from_other_tool():
        barrier.wait(timeout=5)
        try:
            state.deny(call_b, "tool a denied b concurrently")
        except ValueError as e:  # if recorded first, there is no call to deny (deny's contract)
            deny_errors.append(e)

    denier = threading.Thread(target=deny_from_other_tool)
    denier.start()
    failures: list = []
    _runner._finish(state, None, job, failures)
    denier.join()

    assert failures == []
    assert [c.id for c in state.pending_calls] == [call_a.id]
    closings = [h for h in state.history if h.call is not None and h.call.id == call_b.id
                and h.kind in ("tool_result", "denied")]
    assert len(closings) == 1


def test_finish_skips_a_call_denied_before_it_finished():
    """If deny closed it first, the finished result is not recorded and nothing is raised."""
    state, call_a, call_b = _two_pending_calls()
    state.deny(call_b, "denied")
    _runner._finish(state, None, _done_job(call_b, "late"), [])
    assert [c.id for c in state.pending_calls] == [call_a.id]
    assert not any(h.kind == "tool_result" for h in state.history)


# ================================================================ on_tool_start / on_tool_end pairs


def test_failed_parallel_tool_call_gets_matching_on_tool_end():
    """Rule 6: on_tool_start and on_tool_end come in pairs. A tool that raised also gets on_tool_end."""

    @tool
    def ok() -> str:
        """A tool that finishes normally."""
        return "success"

    @tool
    def boom() -> str:
        """A tool that raises."""
        raise TimeoutError("too slow, failed")

    recorder = _Recorder()
    agent = Agent(model=FakeModel([[tool_call("boom"), tool_call("ok")]]), tools=[ok, boom], reporter=recorder)
    state = State("Task")
    with pytest.raises(TimeoutError):
        agent.run(state)

    assert sorted(recorder.names("on_tool_start")) == ["boom", "ok"]
    assert sorted(recorder.names("on_tool_end")) == ["boom", "ok"]
    assert recorder.end_result("boom") == "(aborted: TimeoutError)"
    assert recorder.end_result("ok") == "success"
    boom_outcome = recorder.end_outcome("boom")
    assert boom_outcome.kind == "aborted" and isinstance(boom_outcome.error, TimeoutError)
    assert recorder.end_outcome("ok") == ToolOutcome("done")
    # on_tool_end comes before on_run_end
    kinds = [e[0] for e in recorder.events]
    assert kinds.index("on_run_end") > max(i for i, k in enumerate(kinds) if k == "on_tool_end")


def test_serial_tool_not_started_after_failure_gets_no_tool_events():
    """A call that never started gets neither on_tool_start nor on_tool_end."""

    @tool
    def boom() -> str:
        """A tool that raises."""
        raise TimeoutError("failed")

    @tool(parallel=False)
    def seq() -> str:
        """A serial tool."""
        return "seq"

    recorder = _Recorder()
    agent = Agent(model=FakeModel([[tool_call("boom"), tool_call("seq")]]), tools=[boom, seq], reporter=recorder)
    with pytest.raises(TimeoutError):
        agent.run(State("Task"))
    assert recorder.names("on_tool_start") == ["boom"]
    assert recorder.names("on_tool_end") == ["boom"]


def test_interrupted_calls_get_matching_on_tool_end():
    """Calls closed by Ctrl+C (an unfinished sync tool, a cancelled async tool) also get on_tool_end."""
    import _thread

    started = threading.Event()

    @tool
    def slow_sync() -> str:
        """A sync tool that sends an interrupt and finishes well after it."""
        started.set()
        time.sleep(0.05)
        _thread.interrupt_main()
        time.sleep(0.3)
        return "exit 0"

    @tool
    async def slow_async() -> str:
        """An async tool that will be cancelled."""
        await asyncio.sleep(5)
        return "never"

    recorder = _Recorder()
    agent = Agent(
        model=FakeModel([[tool_call("slow_sync"), tool_call("slow_async")]]),
        tools=[slow_sync, slow_async],
        reporter=recorder,
    )
    with pytest.raises(KeyboardInterrupt):
        agent.run(State("Task"))

    assert sorted(recorder.names("on_tool_start")) == ["slow_async", "slow_sync"]
    assert sorted(recorder.names("on_tool_end")) == ["slow_async", "slow_sync"]
    assert recorder.end_result("slow_sync") == INTERRUPTED
    assert recorder.end_result("slow_async") == INTERRUPTED
    assert recorder.end_outcome("slow_sync").kind == "interrupted"
    assert recorder.end_outcome("slow_async").kind == "interrupted"
    assert closed_result(KeyboardInterrupt()) == INTERRUPTED


# ================================================================ Ctrl+C right after submit


def _interrupt_right_after_submit(monkeypatch, started: threading.Event):
    """The first submit really starts the tool, then, once the tool has started and before returning to run_calls,
    raises KeyboardInterrupt."""
    orig_submit = _runner._Executor.submit
    count = {"n": 0}

    def fake_submit(self, job, state):
        count["n"] += 1
        orig_submit(self, job, state)
        if count["n"] == 1:
            assert started.wait(timeout=2)
            raise KeyboardInterrupt()

    monkeypatch.setattr(_runner._Executor, "submit", fake_submit)


def test_late_result_after_submit_interrupt_is_not_lost(monkeypatch):
    """"If the result arrives later, it is kept in history and put into the context as [notice] ... right before the
    next think": an interrupt that comes right after submit does not lose that call either."""
    started = threading.Event()

    @tool
    def slow_tool() -> str:
        """A tool that starts right away and finishes a little later."""
        started.set()
        time.sleep(0.3)
        return "exit 0"

    agent = Agent(model=FakeModel([tool_call("slow_tool"), "Got it"]), tools=[slow_tool], reporter=None)
    state = State("run the slow tool")
    agent.think(state)

    _interrupt_right_after_submit(monkeypatch, started)
    with pytest.raises(KeyboardInterrupt):
        agent.use_tools(state)
    monkeypatch.undo()

    assert state.pending_calls, "right after the interrupt the call is still a pending call"
    state._close_pending(KeyboardInterrupt())  # what Agent.run's except does
    assert not state.pending_calls
    assert started.wait(timeout=2)

    assert _wait_until(lambda: any(h.kind == "tool_result" and h.late for h in state.history))
    late = [h for h in state.history if h.kind == "tool_result" and h.late]
    assert [h.content for h in late] == ["exit 0"]

    agent.think(state)
    assert any("finished later" in m.text and "exit 0" in m.text for m in state.context if m.role == "user")


def test_async_tool_is_cancelled_after_submit_interrupt(monkeypatch):
    """"async def tools are cancelled": they are cancelled on an interrupt right after submit too."""
    started = threading.Event()
    cancelled = threading.Event()

    @tool
    async def slow_async() -> str:
        """An async tool that waits a long time."""
        started.set()
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "never"

    agent = Agent(model=FakeModel([tool_call("slow_async")]), tools=[slow_async], reporter=None)
    state = State("Task")
    agent.think(state)

    _interrupt_right_after_submit(monkeypatch, started)
    with pytest.raises(KeyboardInterrupt):
        agent.use_tools(state)
    monkeypatch.undo()

    assert cancelled.wait(timeout=3), "the async tool was not cancelled"
    assert not any(h.kind == "tool_result" and h.late for h in state.history)
