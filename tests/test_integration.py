"""Tests for rules found in the integration review (ARCHITECTURE.md "Tool contract", "Loop contract"). No network.

Each test docstring quotes the rule it is based on.
"""

from __future__ import annotations

import threading
import time

import pytest

from alpineagents import Agent, Reporter, State, loop, tool
from alpineagents._tokens import context_tokens, estimate_overhead_tokens
from alpineagents.testing import FakeModel, tool_call
from alpineagents.types import Message, Reply, ToolResultBlock, Usage


def _result_texts(state: State) -> list[str]:
    return [b.content for m in state.context for b in m.content if isinstance(b, ToolResultBlock)]


# ================================================================ notices added by tools


def test_notice_added_by_tool_lands_after_all_results_of_the_turn():
    """"A notice added while a tool runs goes in after all results of that turn are recorded.\""""
    slow_started = threading.Event()

    @tool
    def quick(state: State) -> str:
        """Finishes first and leaves a notice."""
        slow_started.wait(2)
        state.add_notice("notice from quick")
        return "fast"

    @tool
    def slow() -> str:
        """Finishes late."""
        slow_started.set()
        time.sleep(0.1)
        return "slow"

    fake = FakeModel([[tool_call("quick"), tool_call("slow")], "Done"])
    state = State("Do both")
    Agent(fake, tools=[quick, slow], reporter=None).run(state)

    roles_and_texts = [
        (m.role, [getattr(b, "content", None) or getattr(b, "text", None) for b in m.content])
        for m in state.context
    ]
    # One result message (in request order), then the notice right after it, then the next answer.
    assert roles_and_texts[2] == ("user", ["fast", "slow"])
    assert roles_and_texts[3] == ("user", ["[notice] notice from quick"])
    assert roles_and_texts[4] == ("assistant", ["Done"])
    # The request the next think received has the same order.
    second_request = fake.requests[1]
    assert second_request.messages[-1].text == "[notice] notice from quick"


# ================================================================ tool exceptions


def test_tool_exception_waits_for_other_calls_and_keeps_type_and_message_with_note():
    """"The exception the tool raised, as is (e.g. TimeoutError). Waits until the other tools of the same turn
    finish, then raises the first exception." / "Keeps the exception's type and message and only adds the tool
    name and arguments with add_note().\""""
    finished: list[str] = []

    @tool
    def slow(n: int) -> str:
        """Finishes late."""
        time.sleep(0.15)
        finished.append("slow")
        return f"slow {n}"

    @tool
    def boom(path: str) -> str:
        """Fails right away."""
        raise TimeoutError("too slow, failed")

    fake = FakeModel([[tool_call("slow", n=1), tool_call("boom", path="a.txt")]])
    state = State("Do it")
    with pytest.raises(TimeoutError) as info:
        Agent(fake, tools=[slow, boom], reporter=None).run(state)

    assert str(info.value) == "too slow, failed"
    assert any('boom(path="a.txt")' in note for note in info.value.__notes__)
    # It waited for the other tool to finish, and that result stayed in the context.
    assert finished == ["slow"]
    assert _result_texts(state) == ["slow 1", "(aborted: TimeoutError)"]
    errors = [h for h in state.history if h.kind == "error"]
    assert len(errors) == 1 and errors[0].call is not None and errors[0].call.name == "boom"


def test_serial_tools_do_not_start_after_a_parallel_call_failed():
    """"parallel=False tools run one at a time after the rest finish." If the parallel group raises, the serial
    tools do not start and their calls stay in pending_calls (a loop that catches it can close them with
    state.deny)."""
    ran: list[str] = []

    @tool(parallel=False)
    def write_file(path: str) -> str:
        """Writes a file."""
        ran.append(path)
        return "written"

    @tool
    def boom() -> str:
        """Fails."""
        raise ValueError("broken")

    fake = FakeModel([[tool_call("write_file", path="a"), tool_call("boom")]])
    agent = Agent(fake, tools=[write_file, boom], reporter=None)
    state = State("Do it")
    agent.think(state)
    with pytest.raises(ValueError):
        agent.use_tools(state)
    assert ran == []
    assert [c.name for c in state.pending_calls] == ["write_file", "boom"]


def test_serial_tools_run_one_at_a_time_in_request_order():
    """"Mark tools that must not run together (e.g. writing files) with @tool(parallel=False). They then run
    one at a time after the rest finish.\""""
    active = 0
    max_active = 0
    order: list[str] = []
    lock = threading.Lock()

    @tool(parallel=False)
    def write_file(path: str) -> str:
        """Writes a file."""
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
            order.append(path)
        return path

    calls = [tool_call("write_file", path=p) for p in ("a", "b", "c")]
    fake = FakeModel([calls, "Done"])
    state = State("Write")
    Agent(fake, tools=[write_file], reporter=None).run(state)
    assert max_active == 1
    assert order == ["a", "b", "c"]
    assert _result_texts(state) == ["a", "b", "c"]


# ================================================================ think rollback


def test_exception_after_reply_recorded_rolls_back_reply_and_its_calls():
    """"think (during the model request) → roll back to just before that think." Even when on_think_end
    (Reporter) raises, it is still the think step, so the recorded reply and its pending calls are rolled back
    from the context. History is not erased."""

    class Broken(Reporter):
        def on_think_end(self, state, reply):
            raise RuntimeError("screen broken")

    @tool
    def read_file(path: str) -> str:
        """Reads a file."""
        return "contents"

    fake = FakeModel([tool_call("read_file", path="a")])
    state = State("Read it")
    with pytest.raises(RuntimeError, match="screen broken"):
        Agent(fake, tools=[read_file], reporter=Broken()).run(state)

    assert [m.role for m in state.context] == ["user"]
    assert state.pending_calls == ()
    assert state.turn == 0
    kinds = [h.kind for h in state.history]
    assert kinds == ["user", "reply", "error", "context_change"]
    # It was rolled back, so there is no call to close.
    assert not any(h.kind == "tool_result" for h in state.history)


# ================================================================ finish and stopped_by


def test_submit_tool_calling_finish_ends_the_run_after_that_turn():
    """"E.g. a submit(answer: str, state: State) tool, which the model calls when it finishes the task, calls
    state.finish(answer)." / "state.finish() was called → stop when that turn ends and return state.answer"
    (stopped_by="finish")."""

    @tool
    def submit(answer: str, state: State) -> None:
        """Submits the answer."""
        state.finish(answer)

    fake = FakeModel([tool_call("submit", answer="42")])
    state = State("Give the answer")
    result = Agent(fake, tools=[submit], reporter=None).run(state)
    assert result == "42"
    assert state.answer == "42"
    assert state.stopped_by == "finish"
    assert _result_texts(state) == ["(done)"]
    with pytest.raises(ValueError):
        Agent(fake, reporter=None).run(state)


def test_stopped_by_is_cleared_when_a_later_run_raises():
    """"stopped_by: why the last loop stopped." A run that ended with an exception has no stopped loop, so the
    reason from the previous run does not linger."""
    fake = FakeModel(["First answer", RuntimeError("API broken")])
    agent = Agent(fake, reporter=None)
    state = State("Question")
    agent.run(state)
    assert state.stopped_by == "is_answered"

    state.add_user_message("One more")
    with pytest.raises(RuntimeError):
        agent.run(state)
    assert state.stopped_by is None


def test_until_callable_object_without_name_uses_class_name_for_stopped_by():
    """"until takes one State -> bool function, or a list of functions." A callable object without a name must
    also be able to stop the loop, and stopped_by holds its class name."""

    class OverBudget:
        def __call__(self, state: State) -> bool:
            return state.turn >= 1

    @loop(until=[State.is_answered, OverBudget()], limit=10)
    def body(agent: Agent, state: State):
        agent.think(state)
        if state.wants_tools():
            agent.use_tools(state)

    @tool
    def noop() -> str:
        """Does nothing."""
        return "ok"

    fake = FakeModel([tool_call("noop"), "Done"])
    state = State("Do it")
    Agent(fake, tools=[noop], loop=body, reporter=None).run(state)
    assert state.stopped_by == "OverBudget"
    assert state.turn == 1


# ================================================================ context size estimate


def test_use_tools_keeps_the_overhead_of_the_tools_the_last_think_showed():
    """"think(state, tools=...) — given a list, it shows only those tools." / context_used is the context size of
    the next think. use_tools does not know which tools were shown, so it keeps the last think's estimate."""

    @tool
    def small() -> str:
        """Small."""
        return "ok"

    @tool(description="A tool with a very long description. " + "description " * 200)
    def big(a: str, b: str, c: str, d: str, e: str, f: str) -> str:
        return "ok"

    call = tool_call("small")
    # A reply with unknown usage (context_tokens=None): the context estimate relies on the overhead.
    reply = Reply(Message("assistant", (call,)), Usage(requests=1), context_tokens=None)
    fake = FakeModel([reply])
    agent = Agent(fake, system="System", tools=[small, big], reporter=None)
    state = State("Do it")
    agent.think(state, tools=[small])
    overhead_small = estimate_overhead_tokens("System", [small.spec])
    assert state.context_tokens == context_tokens(state.context, overhead_small)

    agent.use_tools(state)
    assert state.context_tokens == context_tokens(state.context, overhead_small)


# ================================================================ mistake-proofing errors


def test_putting_a_class_instead_of_an_object_in_tools_says_how_to_fix():
    """"Put an object in tools= and all of its @tool methods become tools." Passing the class by mistake is
    reported, with how to fix it, when the Agent is created."""

    class FileSystem:
        @tool
        def read_file(self, path: str) -> str:
            """Reads a file."""
            return path

    with pytest.raises(TypeError, match=r"FileSystem\(\)"):
        Agent(FakeModel([]), tools=[FileSystem], reporter=None)


def test_ask_by_another_agent_does_not_take_ownership_and_owner_keeps_thinking():
    """"The first Agent to think becomes the owner; another Agent that thinks gets an error. Other Agents only
    read through ask.\""""
    coder = Agent(FakeModel(["Wrote the code", "Fixed"]), reporter=None)
    reviewer = Agent(FakeModel(["Looks good"]), reporter=None)
    state = State("Write the code")
    coder.think(state)
    assert reviewer.ask(state, "Review this") == "Looks good"
    with pytest.raises(ValueError, match="ask"):
        reviewer.think(state)
    with pytest.raises(ValueError):
        reviewer.use_tools(state)
    with pytest.raises(ValueError):
        reviewer.compact(state)
    state.add_user_message("Fix it")
    coder.think(state)
    assert state.answer == "Fixed"
    assert [h.kind for h in state.history].count("ask") == 1


def test_interrupt_caught_inside_the_loop_keeps_calls_pending_for_deny():
    """"Calls are closed only when an exception leaves run(). If the loop catches the exception, the calls stay
    in pending_calls." An interrupt (KeyboardInterrupt) follows the same rule."""

    @tool
    async def long_job() -> str:
        """A slow async tool."""
        import asyncio

        await asyncio.sleep(5)
        return "finished"

    @loop(until=State.is_answered, limit=3)
    def careful(agent: Agent, state: State):
        agent.think(state)
        if state.wants_tools():
            try:
                agent.use_tools(state)
            except KeyboardInterrupt:
                assert [c.name for c in state.pending_calls] == ["long_job"]
                for call in state.pending_calls:
                    state.deny(call, "stopped by the user")

    class InterruptOnStart(Reporter):
        def on_tool_start(self, state, call):
            def fire():
                time.sleep(0.05)
                import _thread

                _thread.interrupt_main()

            threading.Thread(target=fire, daemon=True).start()

    fake = FakeModel([tool_call("long_job"), "Got it"])
    state = State("A long job")
    answer = Agent(fake, tools=[long_job], loop=careful, reporter=InterruptOnStart()).run(state)
    assert answer == "Got it"
    assert _result_texts(state) == ["stopped by the user"]
    assert not any(h.kind == "tool_result" for h in state.history)
