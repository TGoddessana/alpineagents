"""Tests for rules found in the integration review (ARCHITECTURE.md "Tool contract", "Loop contract"). No network.

Each test docstring quotes the rule it is based on.
"""

from __future__ import annotations

import threading
import time

import pytest

from alpineagents import Agent, Reporter, State, StopEntry, StoppedByFinish, StoppedByUntil, loop, tool
from alpineagents._tokens import context_tokens, estimate_overhead_tokens
from alpineagents.testing import FakeModel, tool_call
from alpineagents.types import Message, Reply, ToolOutcomeKind, ToolResultBlock, Usage


def waiting_for_user(state: State) -> bool:
    """The user-written until function of SPEC section 6 (State.is_answered is gone)."""
    if state.pending_calls or not state.messages:
        return False
    last = state.messages[-1]
    return last.role == "assistant" and not last.tool_calls


def _result_texts(state: State) -> list[str]:
    return [b.content for m in state.messages for b in m.content if isinstance(b, ToolResultBlock)]


# ================================================================ notices added by tools


def test_notice_added_by_tool_lands_after_all_results_of_the_turn():
    """ "A notice added while a tool runs goes in after all results of that turn are recorded.\""""
    slow_started = threading.Event()

    @tool
    def quick(state: State) -> str:
        """Finishes first and leaves a notice."""
        slow_started.wait(2)
        state.add_message(Message.notice("notice from quick"))
        return "fast"

    @tool
    def slow() -> str:
        """Finishes late."""
        slow_started.set()
        time.sleep(0.1)
        return "slow"

    fake = FakeModel([[tool_call("quick"), tool_call("slow")], "Done"])
    state = State(messages=[Message.user("Do both")])
    Agent(fake, tools=[quick, slow], reporter=None).run(state)

    roles_and_texts = [
        (m.role, [getattr(b, "content", None) or getattr(b, "text", None) for b in m.content]) for m in state.messages
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
    """ "The exception the tool raised, as is (e.g. TimeoutError). Waits until the other tools of the same turn
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
    state = State(messages=[Message.user("Do it")])
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
    """ "parallel=False tools run one at a time after the rest finish." If the parallel group raises, the serial
    tools do not start and their calls stay in pending_calls (a loop that catches it can run them again with
    use_tools)."""
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
    state = State(messages=[Message.user("Do it")])
    agent.think(state)
    with pytest.raises(ValueError):
        agent.use_tools(state)
    assert ran == []
    assert [c.name for c in state.pending_calls] == ["write_file", "boom"]


def test_serial_tools_run_one_at_a_time_in_request_order():
    """ "Mark tools that must not run together (e.g. writing files) with @tool(parallel=False). They then run
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
    state = State(messages=[Message.user("Write")])
    Agent(fake, tools=[write_file], reporter=None).run(state)
    assert max_active == 1
    assert order == ["a", "b", "c"]
    assert _result_texts(state) == ["a", "b", "c"]


# ================================================================ think rollback


def test_exception_after_reply_recorded_does_not_roll_back_the_reply():
    """ "An exception after the reply entry (e.g. raised by Reporter.on_think_end) does NOT undo the reply."
    Only a request that got no reply is taken back. The reply and its calls stay, and because the exception
    leaves run(), the calls are closed as aborted (the State stays saveable and runnable)."""

    class Broken(Reporter):
        def on_think_end(self, state, reply):
            raise RuntimeError("screen broken")

    @tool
    def read_file(path: str) -> str:
        """Reads a file."""
        return "contents"

    fake = FakeModel([tool_call("read_file", path="a")])
    state = State(messages=[Message.user("Read it")])
    with pytest.raises(RuntimeError, match="screen broken"):
        Agent(fake, tools=[read_file], reporter=Broken()).run(state)

    assert [m.role for m in state.messages] == ["user", "assistant", "user"]
    assert state.messages[1].tool_calls and state.pending_calls == ()
    assert state.turn == 1
    kinds = [h.kind for h in state.history]
    assert kinds == [
        "context_change",
        "run_start",
        "model_request",
        "model_reply",
        "error",
        "tool_result",
    ]
    # The call was not run: it is closed with the aborted marker.
    assert _result_texts(state) == ["(aborted: RuntimeError)"]
    assert state.history[-1].outcome == ToolOutcomeKind.ABORTED


# ================================================================ finish and stopped


def test_submit_tool_calling_finish_ends_the_run_after_that_turn():
    """ "E.g. a submit(answer: str, state: State) tool, which the model calls when it finishes the task, calls
    state.finish(answer)." / "state.finish() was called → stop when that turn ends and return state.answer"
    (stopped == StoppedByFinish(answer))."""

    @tool
    def submit(answer: str, state: State) -> None:
        """Submits the answer."""
        state.finish(answer)

    fake = FakeModel([tool_call("submit", answer="42")])
    state = State(messages=[Message.user("Give the answer")])
    result = Agent(fake, tools=[submit], reporter=None).run(state)
    assert result == "42"
    assert state.answer == "42"
    assert state.stopped == StoppedByFinish("42")
    assert [h.content for h in state.history if isinstance(h, StopEntry)] == [StoppedByFinish("42")]
    assert _result_texts(state) == ["(done)"]
    with pytest.raises(ValueError):
        Agent(fake, reporter=None).run(state)


def test_stopped_is_cleared_when_a_later_run_raises():
    """ "stopped: why the last loop stopped." A run that ended with an exception has no stopped loop, so the
    reason from the previous run does not linger."""
    fake = FakeModel(["First answer", RuntimeError("API broken")])
    agent = Agent(fake, reporter=None)
    state = State(messages=[Message.user("Question")])
    agent.run(state)
    assert state.stopped == StoppedByUntil("waiting_for_user")

    state.add_message(Message.user("One more"))
    with pytest.raises(RuntimeError):
        agent.run(state)
    assert state.stopped is None


def test_until_callable_object_without_name_uses_class_name_for_stopped():
    """ "until takes one State -> bool function, or a list of functions." A callable object without a name must
    also be able to stop the loop, and stopped holds its class name."""

    class OverBudget:
        def __call__(self, state: State) -> bool:
            return state.turn >= 1

    @loop(until=[waiting_for_user, OverBudget()], limit=10)
    def body(agent: Agent, state: State):
        agent.think(state)
        if state.pending_calls:
            agent.use_tools(state)

    @tool
    def noop() -> str:
        """Does nothing."""
        return "ok"

    fake = FakeModel([tool_call("noop"), "Done"])
    state = State(messages=[Message.user("Do it")])
    Agent(fake, tools=[noop], loop=body, reporter=None).run(state)
    assert state.stopped == StoppedByUntil("OverBudget")
    assert state.turn == 1


# ================================================================ context size estimate


def test_context_tokens_counts_the_overhead_of_all_the_agents_tools_whatever_think_showed():
    """ "agent.context_tokens(state): estimate with this Agent's system/tool overhead." The State no longer
    remembers which tools the last think(tools=...) showed, so the estimate always uses the Agent's whole tool
    list (the 0.4 "keeps the last think's overhead" rule is gone with the State's own window)."""

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
    state = State(messages=[Message.user("Do it")])
    agent.think(state, tools=[small])
    overhead_all = estimate_overhead_tokens("System", [small.spec, big.spec])
    overhead_small = estimate_overhead_tokens("System", [small.spec])
    assert overhead_all > overhead_small
    assert agent.context_tokens(state) == context_tokens(state.messages, overhead_all)

    agent.use_tools(state)
    assert agent.context_tokens(state) == context_tokens(state.messages, overhead_all)


# ================================================================ mistake-proofing errors


def test_putting_a_class_instead_of_an_object_in_tools_says_how_to_fix():
    """ "Put an object in tools= and all of its @tool methods become tools." Passing the class by mistake is
    reported, with how to fix it, when the Agent is created."""

    class FileSystem:
        @tool
        def read_file(self, path: str) -> str:
            """Reads a file."""
            return path

    with pytest.raises(TypeError, match=r"FileSystem\(\)"):
        Agent(FakeModel([]), tools=[FileSystem], reporter=None)


def test_another_agent_can_ask_think_and_compact_and_the_first_one_continues():
    """The 0.4 owner rule is gone: any Agent can ask, think, use_tools and compact a State. An ask changes
    neither the messages nor the turn; a think by the second Agent is an ordinary turn."""
    coder = Agent(FakeModel(["Wrote the code", "Fixed"]), reporter=None)
    reviewer = Agent(FakeModel(["Looks good", "Reviewed", "Summary of the work"]), reporter=None)
    state = State(messages=[Message.user("Write the code")])
    coder.think(state)
    before = state.messages
    assert reviewer.ask(state, "Review this") == "Looks good"
    assert state.messages == before and state.turn == 1

    reviewer.think(state)  # does not raise
    assert state.answer == "Reviewed" and state.turn == 2
    reviewer.use_tools(state)  # nothing pending: does nothing
    reviewer.compact(state)
    assert state.messages[-1].text.endswith("Summary of the work")

    state.add_message(Message.user("Fix it"))
    coder.think(state)
    assert state.answer == "Fixed"
    assert [h.kind for h in state.history].count("ask") == 1


def test_interrupt_caught_inside_the_loop_keeps_calls_pending_for_deny():
    """ "Calls are closed only when an exception leaves run(). If the loop catches the exception, the calls stay
    in pending_calls." An interrupt (KeyboardInterrupt) follows the same rule."""

    @tool
    async def long_job() -> str:
        """A slow async tool."""
        import asyncio

        await asyncio.sleep(5)
        return "finished"

    @loop(until=waiting_for_user, limit=3)
    def careful(agent: Agent, state: State):
        agent.think(state)
        if state.pending_calls:
            try:
                agent.use_tools(state)
            except KeyboardInterrupt:
                assert [c.name for c in state.pending_calls] == ["long_job"]
                for call in state.pending_calls:
                    # what a permission's denial records
                    state._tool_result(call, "stopped by the user", ToolOutcomeKind.DENIED)

    class InterruptOnStart(Reporter):
        def on_tool_start(self, state, call):
            def fire():
                time.sleep(0.05)
                import _thread

                _thread.interrupt_main()

            threading.Thread(target=fire, daemon=True).start()

    fake = FakeModel([tool_call("long_job"), "Got it"])
    state = State(messages=[Message.user("A long job")])
    answer = Agent(fake, tools=[long_job], loop=careful, reporter=InterruptOnStart()).run(state)
    assert answer == "Got it"
    assert _result_texts(state) == ["stopped by the user"]
    # the denial is a tool_result entry with outcome "denied" (there is no separate NotRun entry any more)
    assert [h.outcome for h in state.history if h.kind == "tool_result"] == [ToolOutcomeKind.DENIED]
