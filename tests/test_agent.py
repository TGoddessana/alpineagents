"""Agent black-box tests (ARCHITECTURE.md "Agent contract"). No network.

The rules come from the written contract, not the implementation, with one test per rule. Each test's
docstring quotes its rule. Only ``FakeModel``/``FakeHuman`` from ``alpineagents.testing`` are used, and
``reporter=None`` turns off screen output (where it matters, a hand-made Reporter checks the calls).
"""

from __future__ import annotations

import _thread
import json
import threading
import time
import warnings
from dataclasses import dataclass

import pytest

from alpineagents import (
    Agent,
    Message,
    ModelReplyEntry,
    ModelRequestEntry,
    NoHumanError,
    OutputError,
    Reporter,
    RunStartEntry,
    State,
    StopEntry,
    StoppedByFinish,
    tool,
)
from alpineagents.state import INTERRUPTED, closed_result
from alpineagents.testing import FakeHuman, FakeModel, tool_call
from alpineagents.types import ToolResultBlock


class _Recorder(Reporter):
    """A Reporter that records call order and arguments as is (for checks)."""

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
        self.events.append(("on_tool_end", call.name, result))

    def on_context_change(self, state, change):
        self.events.append(("on_context_change", change.kind))


def _result_texts(state: State) -> list[str]:
    """The content of every ``ToolResultBlock`` in the context."""
    return [b.content for m in state.messages for b in m.content if isinstance(b, ToolResultBlock)]


def _agent_with_ok_and_boom_tools() -> Agent:
    """An Agent with two tools: ``ok_tool`` succeeds and ``boom_tool`` raises ``TimeoutError``."""

    @tool
    def ok_tool() -> str:
        """A tool that finishes normally."""
        return "success"

    @tool
    def boom_tool() -> str:
        """A tool that raises."""
        raise TimeoutError("too slow, failed")

    fake = FakeModel([[tool_call("ok_tool"), tool_call("boom_tool")]])
    return Agent(model=fake, tools=[ok_tool, boom_tool], reporter=None)


@dataclass
class _Plan:
    steps: list[str]


# ================================================================ Agent.run


def test_run_with_string_creates_new_state_and_returns_loop_result():
    """ "A string creates a new State. Calls the loop and returns what the loop returned." """
    agent = Agent(model=FakeModel(["Final answer"]), reporter=None)
    assert agent.run("Find the bug") == "Final answer"


def test_run_with_existing_state_uses_it_directly():
    """ "State → used as is" (``run(task or state) -> answer``)"""
    agent = Agent(model=FakeModel(["Final answer"]), reporter=None)
    state = State(messages=[Message.user("Find the bug")])
    assert agent.run(state) == "Final answer"
    assert state.turn == 1


def test_run_rejects_task_that_is_neither_string_nor_state():
    """ "run(task or state) -> answer": any other value is a misuse."""
    agent = Agent(model=FakeModel([]), reporter=None)
    with pytest.raises(TypeError):
        agent.run(12345)


def test_run_notifies_reporter_on_run_start_then_on_run_end():
    """ "on_run_start / on_run_end(state, error) | Agent.run | before the loop / finally" """
    recorder = _Recorder()
    agent = Agent(model=FakeModel(["Answer"]), reporter=recorder)
    agent.run("Task")
    kinds = [event[0] for event in recorder.events]
    assert kinds[0] == "on_run_start"
    assert kinds[-1] == "on_run_end"
    assert recorder.events[-1][1] is None  # error is None on a normal finish


def test_run_end_still_fires_and_reports_the_exception_that_escaped():
    """ "Exceptions are not swallowed; they are re-raised as is."" and ""on_run_end is always called in finally."""
    recorder = _Recorder()
    agent = Agent(model=FakeModel([RuntimeError("something went wrong")]), reporter=recorder)
    with pytest.raises(RuntimeError):
        agent.run("Task")
    assert recorder.events[-1][0] == "on_run_end"
    assert isinstance(recorder.events[-1][1], RuntimeError)


# ================================================================ Agent.think


def test_think_records_model_reply_and_advances_turn():
    """ "think(state, tools=...) | Sends one request to the Model and records the reply."" and ""turn: how many
    times think ran so far. Counts from 1."""
    agent = Agent(model=FakeModel(["First thought"]), reporter=None)
    state = State(messages=[Message.user("Task")])
    assert state.turn == 0
    agent.think(state)
    assert state.turn == 1
    assert state.messages[-1].text == "First thought"


def test_think_with_empty_tools_list_hides_all_tool_definitions():
    """ "tools=[] shows no tools; a list shows only those tools." (the ``[]`` case)"""

    @tool
    def only_tool() -> str:
        """A single tool."""
        return "ok"

    fake = FakeModel(["Reply"])
    agent = Agent(model=fake, tools=[only_tool], reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.think(state, tools=[])
    assert fake.requests[-1].tools == ()


def test_think_with_tool_subset_shows_only_that_tool():
    """ "tools=[] shows no tools; a list shows only those tools." (the list case)"""

    @tool
    def a_tool() -> str:
        """Tool A."""
        return "a"

    @tool
    def b_tool() -> str:
        """Tool B."""
        return "b"

    fake = FakeModel(["Reply"])
    agent = Agent(model=fake, tools=[a_tool, b_tool], reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.think(state, tools=[a_tool])
    assert [spec.name for spec in fake.requests[-1].tools] == ["a_tool"]


# ================================================================ Agent.use_tools


def test_use_tools_executes_pending_calls_and_records_results():
    """ "use_tools(state) | Runs every pending call (state.pending_calls) and records the results."""

    @tool
    def echo(msg: str) -> str:
        """Returns its input."""
        return msg

    fake = FakeModel([tool_call("echo", msg="hello"), "Next"])
    agent = Agent(model=fake, tools=[echo], reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.think(state)
    assert state.pending_calls
    agent.use_tools(state)
    assert not state.pending_calls
    assert "hello" in _result_texts(state)


def test_use_tools_does_nothing_when_no_pending_calls():
    """With no calls (``pending_calls`` is empty), ``use_tools`` changes neither the context nor the history."""
    agent = Agent(model=FakeModel(["An answer with no tool calls"]), reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.think(state)
    before_context = state.messages
    before_history_len = len(state.history)
    agent.use_tools(state)
    assert state.messages == before_context
    assert len(state.history) == before_history_len


# ================================================================ Agent.ask, and how think and ask differ


def test_ask_does_not_touch_context_but_adds_to_history_and_usage():
    """ "ask(...) | Changes neither the context nor the answer."" ""ask ... is kept in history but not in the
    context."" ""Usage is added to state.usage."""
    fake = FakeModel(["A tidy plan"])
    agent = Agent(model=fake, reporter=None)
    state = State(messages=[Message.user("Task")])
    before_context = state.messages
    before_answer = state.answer
    before_requests = state.usage.requests

    value = agent.ask(state, "Summarize the work so far")

    assert value == "A tidy plan"
    assert state.messages == before_context
    assert state.answer == before_answer
    assert state.usage.requests == before_requests + 1
    assert any(h.kind == "ask" for h in state.history)


def test_ask_sends_tool_definitions_but_sets_tool_choice_none():
    """ ""tool_choice="none": send the tool definitions but do not allow calls (used by ask and compact)."""

    @tool
    def some_tool() -> str:
        """A tool."""
        return "x"

    fake = FakeModel(["Answer"])
    agent = Agent(model=fake, tools=[some_tool], reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.ask(state, "Question")
    request = fake.requests[-1]
    assert request.tool_choice == "none"
    assert [spec.name for spec in request.tools] == ["some_tool"]


def test_think_advances_context_while_ask_leaves_it_unchanged():
    """ ""think is a command that moves the conversation forward; ask is a query that leaves it alone
    (command-query separation)."""
    fake = FakeModel(["Thought answer", "Asked answer"])
    agent = Agent(model=fake, reporter=None)
    state = State(messages=[Message.user("Task")])

    agent.think(state)
    after_think = len(state.messages)

    agent.ask(state, "Check this")
    assert len(state.messages) == after_think


def test_ask_retries_on_invalid_format_then_succeeds():
    """ "If the answer does not match the format, the validation error is shown to the model and it asks again,
    up to two more times (set with retries=)."""
    fake = FakeModel(["This is not JSON", '{"steps": ["First step"]}'])
    agent = Agent(model=fake, reporter=None)
    state = State(messages=[Message.user("Task")])
    plan = agent.ask(state, "Make a plan", returns=_Plan)
    assert plan == _Plan(steps=["First step"])
    assert fake.remaining == 0


def test_ask_raises_outputerror_after_retries_exhausted():
    """ "If it still does not match, alpineagents.OutputError is raised."""
    fake = FakeModel(["Nope", "Still nope"])
    agent = Agent(model=fake, reporter=None)
    state = State(messages=[Message.user("Task")])
    with pytest.raises(OutputError):
        agent.ask(state, "Make a plan", returns=_Plan, retries=1)
    assert fake.remaining == 0
    last_ask = [h for h in state.history if h.kind == "ask"][-1]
    assert last_ask.content.answer is None


def test_ask_retries_zero_means_a_single_attempt():
    """ "Set with retries=": ``retries=0`` tries once and never asks again."""
    fake = FakeModel(["Wrong format"])
    agent = Agent(model=fake, reporter=None)
    state = State(messages=[Message.user("Task")])
    with pytest.raises(OutputError):
        agent.ask(state, "Make a plan", returns=_Plan, retries=0)
    assert len(fake.requests) == 1


# ================================================================ Agent.ask_human


def test_ask_human_returns_value_and_records_history_without_touching_context():
    """ "ask_human(state, prompt, returns) -> value ... Leaves the context alone and is kept in history as human."""
    human = FakeHuman(["yes"])
    agent = Agent(model=FakeModel([]), human=human, reporter=None)
    state = State(messages=[Message.user("Task")])
    before_context = state.messages

    value = agent.ask_human(state, "Continue?", returns=bool)

    assert value is True
    assert state.messages == before_context
    assert human.questions == ["Continue?"]
    assert any(h.kind == "human" for h in state.history)


# ================================================================ Agent.compact


def test_compact_replaces_context_with_task_plus_summary():
    """ "compact(state, instructions=None) | Gets a summary from the Model and replaces the context with it."" and
    ""compact(summary) replaces the context with two messages: the original task + the summary." (compact
    follows the same rule)"""
    fake = FakeModel(["Files read so far: main.py"])
    agent = Agent(model=fake, reporter=None)
    state = State(messages=[Message.user("Find the bug")])
    agent.compact(state)
    assert len(state.messages) == 2
    assert state.messages[0].text == "Find the bug"
    assert "Files read so far: main.py" in state.messages[1].text


def test_compact_sends_tool_definitions_but_sets_tool_choice_none():
    """ ""tool_choice="none": send the tool definitions but do not allow calls (used by ask and compact)."""

    @tool
    def some_tool() -> str:
        """A tool."""
        return "x"

    fake = FakeModel(["Summary"])
    agent = Agent(model=fake, tools=[some_tool], reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.compact(state)
    request = fake.requests[-1]
    assert request.tool_choice == "none"
    assert [spec.name for spec in request.tools] == ["some_tool"]


def test_compact_with_empty_summary_raises_outputerror_and_keeps_context():
    """ "If the summary is empty, the context is left alone and OutputError is raised (so no work is lost)."""
    fake = FakeModel([""])
    agent = Agent(model=fake, reporter=None)
    state = State(messages=[Message.user("Find the bug")])
    before = state.messages
    with pytest.raises(OutputError):
        agent.compact(state)
    assert state.messages == before


# ================================================================ Agent.copy


def test_copy_overrides_only_given_settings_and_leaves_original_untouched():
    """ "copy(**settings to change) -> Agent | A new Agent with only some settings changed. The original is
    untouched."""
    original = Agent(model=FakeModel([]), system="Original", reporter=None)
    copied = original.copy(system="New system")
    assert copied.system == "New system"
    assert original.system == "Original"
    assert copied is not original


# ================================================================ mistake-proofing errors (Agent/State)


def test_agent_creation_with_duplicate_tool_names_raises_valueerror():
    """ "Duplicate tool names ... | when Agent(...) is created | ValueError" """

    @tool(name="dup")
    def a() -> str:
        """A."""
        return "a"

    @tool(name="dup")
    def b() -> str:
        """B."""
        return "b"

    with pytest.raises(ValueError):
        Agent(model=FakeModel([]), tools=[a, b], reporter=None)


def test_agent_creation_subagent_missing_name_and_description_raises_valueerror():
    """ "An Agent without name or description in tools= | when Agent(...) is created | ValueError" """
    incomplete = Agent(model=FakeModel([]), reporter=None)  # no name and no description
    with pytest.raises(ValueError):
        Agent(model=FakeModel([]), tools=[incomplete], reporter=None)


def test_finish_then_think_raises_valueerror():
    """ "think, use_tools or ask called after finish() | on call | ValueError" (think)"""
    agent = Agent(model=FakeModel([]), reporter=None)
    state = State(messages=[Message.user("Task")])
    state.finish()
    with pytest.raises(ValueError):
        agent.think(state)


def test_finish_then_use_tools_raises_valueerror():
    """ "think, use_tools or ask called after finish() | on call | ValueError" (use_tools)"""
    agent = Agent(model=FakeModel([]), reporter=None)
    state = State(messages=[Message.user("Task")])
    state.finish()
    with pytest.raises(ValueError):
        agent.use_tools(state)


def test_finish_then_ask_raises_valueerror():
    """ "think, use_tools or ask called after finish() | on call | ValueError" (ask)"""
    agent = Agent(model=FakeModel([]), reporter=None)
    state = State(messages=[Message.user("Task")])
    state.finish()
    with pytest.raises(ValueError):
        agent.ask(state, "Question")


def test_think_with_pending_calls_raises_valueerror():
    """ "think, compact or save while there are pending calls | on call | ValueError" (think)"""

    @tool
    def noop() -> str:
        """Does nothing."""
        return "ok"

    fake = FakeModel([tool_call("noop"), "Next line"])
    agent = Agent(model=fake, tools=[noop], reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.think(state)
    assert state.pending_calls
    with pytest.raises(ValueError):
        agent.think(state)
    assert fake.remaining == 1  # the second reply is not used yet


def test_compact_with_pending_calls_raises_valueerror():
    """ "think, compact or save while there are pending calls | on call | ValueError" (compact)"""

    @tool
    def noop() -> str:
        """Does nothing."""
        return "ok"

    fake = FakeModel([tool_call("noop")])
    agent = Agent(model=fake, tools=[noop], reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.think(state)
    with pytest.raises(ValueError):
        agent.compact(state)


def test_any_agent_can_think_on_a_state_another_agent_thought_on():
    """ "Any Agent can think, use_tools and compact a State" (the 0.4 owner rule is gone)."""
    state = State(messages=[Message.user("Task")])
    first = Agent(model=FakeModel(["First answer"]), reporter=None)
    first.think(state)
    second = Agent(model=FakeModel(["Second answer"]), reporter=None)
    second.think(state)  # does not raise
    assert state.answer == "Second answer"
    assert state.turn == 2


def test_run_on_already_finished_state_raises_valueerror():
    """ "run again on a finish()ed State | on call | ValueError" """
    agent = Agent(model=FakeModel([]), reporter=None)
    state = State(messages=[Message.user("Task")])
    state.finish("Already done")
    with pytest.raises(ValueError):
        agent.run(state)


def test_ask_human_without_human_raises_nohumanerror():
    """ "ask_human called with human=None | on call | NoHumanError" """
    agent = Agent(model=FakeModel([]), human=None, reporter=None)
    state = State(messages=[Message.user("Task")])
    with pytest.raises(NoHumanError):
        agent.ask_human(state, "Run it?")


def test_think_tools_argument_rejects_tool_not_registered_on_agent():
    """ "Something that is not this Agent's tool in think(tools=[...])" (a tool that is not registered cannot be
    passed to think(tools=...)) — raised as ValueError with how to fix it."""

    @tool
    def allowed() -> str:
        """An allowed tool."""
        return "ok"

    @tool
    def outsider() -> str:
        """A tool not registered on this Agent."""
        return "ok"

    agent = Agent(model=FakeModel([]), tools=[allowed], reporter=None)
    state = State(messages=[Message.user("Task")])
    with pytest.raises(ValueError):
        agent.think(state, tools=[outsider])


def test_copy_with_unknown_setting_raises_typeerror():
    """ "Unknown setting name in copy() | TypeError" """
    agent = Agent(model=FakeModel([]), reporter=None)
    with pytest.raises(TypeError):
        agent.copy(unknown_option=True)


def test_ask_unsupported_returns_raises_typeerror():
    """ "Unsupported format in ask(returns=...) | TypeError" """
    agent = Agent(model=FakeModel([]), reporter=None)
    state = State(messages=[Message.user("Task")])
    with pytest.raises(TypeError):
        agent.ask(state, "Question", returns=None)


def test_ask_human_unsupported_returns_raises_typeerror():
    """ "Unsupported format in ask_human(returns=...) | TypeError" """
    agent = Agent(model=FakeModel([]), human=FakeHuman(["1"]), reporter=None)
    state = State(messages=[Message.user("Task")])
    with pytest.raises(TypeError):
        agent.ask_human(state, "A number?", returns=int)


def test_state_has_no_public_deny():
    """Permissions refuse calls now; state.deny is gone from the public API."""
    assert not hasattr(State(messages=[Message.user("Task")]), "deny")


# ================================================================ what the Agent "does not do"


def test_agent_prints_nothing_and_never_reads_input(monkeypatch, capsys):
    """ "It does not print to the screen or read human input. It only notifies the Reporter and asks the Human."""

    def fail_input(*args, **kwargs):
        raise AssertionError("Agent called input() directly")

    monkeypatch.setattr("builtins.input", fail_input)
    agent = Agent(model=FakeModel(["Answer"]), reporter=None, human=None)
    assert agent.run("Task") == "Answer"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_agent_keeps_no_conversation_state_of_its_own():
    """ "It keeps no conversation history of its own. History belongs to the State."""
    agent = Agent(model=FakeModel(["First answer", "Second answer"]), reporter=None)
    first = State(messages=[Message.user("Task 1")])
    second = State(messages=[Message.user("Task 2")])

    assert agent.run(first) == "First answer"
    assert agent.run(second) == "Second answer"
    # The task is in each State's own messages (the first history entry is the import of them).
    assert first.messages[0].text == "Task 1"
    assert second.messages[0].text == "Task 2"
    assert first.history[0].content.messages[0].text == "Task 1"
    assert second.history[0].content.messages[0].text == "Task 2"


# ================================================================ the exception table / the State after an exception


def test_closed_result_text_distinguishes_exception_from_interrupt():
    """ "Closed by an exception | (aborted: TimeoutError)" ""Closed by an interrupt | (interrupted by user)"
    (state.closed_result)"""
    assert closed_result(TimeoutError()) == "(aborted: TimeoutError)"
    assert closed_result(KeyboardInterrupt()) == INTERRUPTED == "(interrupted by user)"


def test_think_exception_rolls_back_to_state_right_before_that_think():
    """ "think (during the model request) | roll back to just before that think | leaves no trace in the outside
    world. Just request again"""
    fake = FakeModel([RuntimeError("model error"), "Worked this time"])
    agent = Agent(model=fake, reporter=None)
    state = State(messages=[Message.user("Task")])
    before_context = state.messages
    before_turn = state.turn

    with pytest.raises(RuntimeError):
        agent.think(state)

    assert state.messages == before_context
    assert state.turn == before_turn
    assert any(h.kind == "error" for h in state.history)

    # It was rolled back, so it can think again.
    agent.think(state)
    assert state.answer == "Worked this time"


def test_use_tools_exception_leaves_failed_call_pending_but_records_finished_ones():
    """ "use_tools (while tools run) | Results of finished calls are recorded. Failed, unfinished and unstarted
    calls stay in pending_calls and the exception is raised"""
    agent = _agent_with_ok_and_boom_tools()
    state = State(messages=[Message.user("Task")])
    agent.think(state)

    with pytest.raises(TimeoutError):
        agent.use_tools(state)

    pending_names = {c.name for c in state.pending_calls}
    assert pending_names == {"boom_tool"}
    assert any(h.kind == "tool_result" and h.call and h.call.name == "ok_tool" for h in state.history)


def test_run_closes_unfinished_tool_call_with_closed_marker_when_exception_escapes():
    """ "When it leaves run (wherever it came from) | after _record_error(e), close the remaining pending calls
    with closed_result(e)"" (e.g. "(aborted: TimeoutError)")"""
    agent = _agent_with_ok_and_boom_tools()
    state = State(messages=[Message.user("Task")])

    with pytest.raises(TimeoutError):
        agent.run(state)

    assert state.pending_calls == ()
    texts = _result_texts(state)
    assert "(aborted: TimeoutError)" in texts
    assert "success" in texts  # results of finished calls stay


def test_error_recorded_only_once_for_the_same_exception_instance():
    """ "In every case an error entry is recorded. The same exception object is recorded only once
    (_record_error checks with is)."""
    agent = _agent_with_ok_and_boom_tools()
    state = State(messages=[Message.user("Task")])
    with pytest.raises(TimeoutError):
        agent.run(state)
    error_entries = [h for h in state.history if h.kind == "error" and isinstance(h.error, TimeoutError)]
    assert len(error_entries) == 1


def test_loop_body_exception_outside_think_and_use_tools_keeps_context_and_closes_pending():
    """ "Loop body or block (outside the two above) | left as is. Pending calls left over are closed as above |
    keeps the state saveable" """

    @tool
    def ok_tool() -> str:
        """A normal tool."""
        return "success"

    def flaky_loop(agent, state):
        agent.think(state)  # gets a reply with a tool call (use_tools is not called)
        raise ValueError("exception in the loop body")

    fake = FakeModel([tool_call("ok_tool")])
    agent = Agent(model=fake, tools=[ok_tool], loop=flaky_loop, reporter=None)
    state = State(messages=[Message.user("Task")])

    with pytest.raises(ValueError):
        agent.run(state)

    # The reply with the tool call that think recorded is not rolled back; it stays.
    assert any(m.role == "assistant" and m.tool_calls for m in state.messages)
    # run() closes the call that was left pending.
    assert state.pending_calls == ()
    assert "(aborted: ValueError)" in _result_texts(state)


def test_loop_catching_use_tools_exception_and_retrying_keeps_run_from_raising():
    """Example: if the loop catches a use_tools exception, the calls are not closed and stay in pending_calls,
    and use_tools runs them again. ("Calls are closed only when an exception leaves run().")"""
    attempts: list[int] = []

    @tool
    def flaky_tool() -> str:
        """A tool that fails the first time."""
        attempts.append(1)
        if len(attempts) == 1:
            raise TimeoutError("too slow, failed")
        return "ok"

    def retry_on_timeout_loop(agent, state):
        agent.think(state)
        if state.pending_calls:
            try:
                agent.use_tools(state)
            except TimeoutError:
                assert [c.name for c in state.pending_calls] == ["flaky_tool"]
                agent.use_tools(state)
        state.finish("Cleaned up")
        return state.answer  # a raw callable was passed as loop=, so its return value is run()'s result

    fake = FakeModel([tool_call("flaky_tool")])
    agent = Agent(model=fake, tools=[flaky_tool], loop=retry_on_timeout_loop, reporter=None)
    state = State(messages=[Message.user("Task")])

    result = agent.run(state)  # the loop caught the exception, so run() finishes normally

    assert result == "Cleaned up"
    assert state.pending_calls == ()
    assert _result_texts(state) == ["ok"]
    assert not any("(aborted:" in text for text in _result_texts(state))


# ================================================================ interrupts (KeyboardInterrupt)


def test_keyboard_interrupt_during_think_rolls_back_like_any_other_exception():
    """ "Ctrl+C (KeyboardInterrupt) ... during a model reply, it rolls back" """
    fake = FakeModel([KeyboardInterrupt(), "success"])
    agent = Agent(model=fake, reporter=None)
    state = State(messages=[Message.user("Task")])
    before_context = state.messages
    before_turn = state.turn

    with pytest.raises(KeyboardInterrupt):
        agent.think(state)

    assert state.messages == before_context
    assert state.turn == before_turn

    agent.think(state)
    assert state.answer == "success"


def test_keyboard_interrupt_during_tool_execution_closes_unfinished_calls_as_interrupted():
    """ ""While tools run, unfinished calls are closed as (interrupted by user)." ("Unfinished" here covers both
    calls cut off by an exception midway and serial calls that never started because the parallel group
    failed)"""

    @tool
    def tool_kb() -> str:
        """Stops right away, imitating Ctrl+C while it runs."""
        raise KeyboardInterrupt()

    @tool(parallel=False)
    def tool_seq() -> str:
        """A serial tool that never starts if the parallel group fails."""
        return "completed"

    fake = FakeModel([[tool_call("tool_kb"), tool_call("tool_seq")]])
    agent = Agent(model=fake, tools=[tool_kb, tool_seq], reporter=None)
    state = State(messages=[Message.user("Task")])

    with pytest.raises(KeyboardInterrupt):
        agent.run(state)

    assert state.pending_calls == ()
    blocks = [b for m in state.messages for b in m.content if isinstance(b, ToolResultBlock)]
    interrupted = {b.name for b in blocks if b.content == INTERRUPTED}
    assert interrupted == {"tool_kb", "tool_seq"}


def test_late_sync_tool_result_is_recorded_and_announced_before_next_think():
    """ "For a sync tool, Python cannot stop the thread. The framework closes the call without waiting.
    If the result arrives later, it is kept in history and put into the context right before the next think
    as a notice like [notice] interrupted ... finished later: ..."""

    @tool
    def slow_sync() -> str:
        """A sync tool that interrupts the main thread like a real Ctrl+C, then finishes well after."""
        time.sleep(0.05)  # interrupt only once the main thread is surely inside use_tools' wait loop
        _thread.interrupt_main()
        time.sleep(0.3)
        return "exit 0"

    fake = FakeModel([tool_call("slow_sync"), "Got it"])
    agent = Agent(model=fake, tools=[slow_sync], reporter=None)
    state = State(messages=[Message.user("Build it")])

    with pytest.raises(KeyboardInterrupt):
        agent.run(state)

    # run() does not wait; it closes the call right away as "(interrupted by user)".
    assert state.pending_calls == ()
    assert INTERRUPTED in _result_texts(state)

    # Wait until slow_sync really finishes (the worker thread's done callback records the late result).
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not any(h.kind == "tool_result" and h.late for h in state.history):
        time.sleep(0.02)
    late_entries = [h for h in state.history if h.kind == "tool_result" and h.late]
    assert len(late_entries) == 1
    assert late_entries[0].content == "exit 0"

    # When an Agent thinks again, the notice goes into the context right before it.
    agent.think(state)
    notices = [m.text for m in state.messages if m.role == "user" and "finished later" in m.text]
    assert any("slow_sync" in n and "exit 0" in n for n in notices)


# ================================================================ parallel tool execution and parallel=False


def test_calls_in_one_turn_run_at_the_same_time():
    """ "Calls in one turn run at the same time."""
    barrier = threading.Barrier(2, timeout=2)

    @tool
    def wait_a() -> str:
        """Waits for the other at the barrier."""
        barrier.wait()
        return "a"

    @tool
    def wait_b() -> str:
        """Waits for the other at the barrier."""
        barrier.wait()
        return "b"

    fake = FakeModel([[tool_call("wait_a"), tool_call("wait_b")]])
    agent = Agent(model=fake, tools=[wait_a, wait_b], reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.think(state)

    # If they do not run at the same time, the first one times out at the barrier and raises.
    agent.use_tools(state)

    assert set(_result_texts(state)) == {"a", "b"}


def test_parallel_false_tool_runs_only_after_the_parallel_batch_finishes():
    """ "Mark tools that must not run together (e.g. writing files) with @tool(parallel=False). They then run
    one at a time after the rest finish."""
    log: list[str] = []
    lock = threading.Lock()

    @tool
    def slow_parallel() -> str:
        """A slow tool that runs in parallel."""
        with lock:
            log.append("parallel start")
        time.sleep(0.2)
        with lock:
            log.append("parallel end")
        return "parallel done"

    @tool(parallel=False)
    def exclusive() -> str:
        """A tool that runs alone."""
        with lock:
            log.append("exclusive run")
        return "exclusive done"

    fake = FakeModel([[tool_call("slow_parallel"), tool_call("exclusive")]])
    agent = Agent(model=fake, tools=[slow_parallel, exclusive], reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.think(state)
    agent.use_tools(state)

    assert log == ["parallel start", "parallel end", "exclusive run"]


def test_tool_results_are_kept_in_the_order_the_model_requested_them():
    """ "The tool results of one turn go into a single user message, as ToolResultBlocks in the order the model
    requested them."""

    @tool
    def slow() -> str:
        """Finishes late."""
        time.sleep(0.2)
        return "slow"

    @tool
    def fast() -> str:
        """Finishes fast."""
        return "fast"

    # The model asked for slow first, but fast finishes first.
    fake = FakeModel([[tool_call("slow"), tool_call("fast")]])
    agent = Agent(model=fake, tools=[slow, fast], reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.think(state)
    agent.use_tools(state)

    last_message = state.messages[-1]
    names = [b.name for b in last_message.content if isinstance(b, ToolResultBlock)]
    assert names == ["slow", "fast"]


# ================================================================ run(prompt_or_state, /)


def test_run_with_string_starts_a_state_with_that_one_user_message_and_a_run_start_entry():
    """ "A string creates a new State: State(messages=[Message.user(text)])." The State is not returned, so the
    request the model got shows it."""
    fake = FakeModel(["Final answer"], name="m1")
    Agent(model=fake, reporter=None).run("Find the bug")
    request = fake.requests[0]
    assert [(m.role, m.text) for m in request.messages] == [("user", "Find the bug")]


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t"])
def test_run_with_an_empty_or_blank_string_raises_valueerror_before_any_model_call(prompt):
    """A provider rejects an empty message, so ``run("")`` is refused up front (ValueError, with how to fix it)."""
    fake = FakeModel(["never used"])
    with pytest.raises(ValueError, match="empty prompt"):
        Agent(model=fake, reporter=None).run(prompt)
    assert fake.requests == []


def test_run_takes_its_argument_positionally_only():
    """``run(prompt_or_state, /)``: the argument has no keyword name."""
    agent = Agent(model=FakeModel(["x"]), reporter=None)
    with pytest.raises(TypeError):
        agent.run(prompt_or_state="Task")  # type: ignore[call-arg]


def test_run_records_one_run_start_entry_holding_the_agents_info_before_the_first_think():
    """ "Record RunStartEntry(AgentInfo) at run start." It comes after the messages and before the model request."""

    @tool
    def helper() -> str:
        """A tool."""
        return "x"

    agent = Agent(
        model=FakeModel(["Answer"], name="m1"), system="Be brief", tools=[helper], name="coder", reporter=None
    )
    state = State(messages=[Message.user("Task")], extra_data={"repo": "api"})
    agent.run(state)

    kinds = [h.kind for h in state.history]
    assert kinds == [
        "context_change",  # import of the starting messages
        "extra_data",
        "run_start",
        "model_request",
        "model_reply",
        "stop",
    ]
    (start,) = [h for h in state.history if isinstance(h, RunStartEntry)]
    assert start.content.name == "coder"
    assert start.content.model == "fake/m1"
    assert start.content.tools == ("helper",)
    assert start.content.system_sha256 is not None
    assert start.turn == 0


def test_every_run_records_its_own_run_start_entry():
    """One ``RunStartEntry`` per ``run`` call on the same State."""
    agent = Agent(model=FakeModel(["First", "Second"]), reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.run(state)
    state.add_message(Message.user("Go on"))
    agent.run(state)
    starts = [h for h in state.history if isinstance(h, RunStartEntry)]
    assert len(starts) == 2
    assert starts[0].turn == 0 and starts[1].turn == 1


def test_run_start_clears_the_previous_runs_stop_reason():
    """``RunStartEntry`` resets ``state.stopped`` to None until the new run records a stop."""
    seen: list = []

    class _Peek(Reporter):
        def on_run_start(self, state):
            seen.append(state.stopped)

    agent = Agent(model=FakeModel(["First", "Second"]), reporter=_Peek())
    state = State(messages=[Message.user("Task")])
    agent.run(state)
    assert state.stopped is not None
    state.add_message(Message.user("Go on"))
    agent.run(state)
    assert seen == [None, None]


# ================================================================ one run at a time


def test_a_state_being_run_by_a_thread_refuses_a_second_run_from_any_agent():
    """ "Refuse a State that is already being run by any Agent (ValueError)." The first run is not disturbed."""
    started = threading.Event()
    release = threading.Event()

    @tool
    def block() -> str:
        """Waits until the test lets it go."""
        started.set()
        assert release.wait(5)
        return "released"

    first = Agent(model=FakeModel([tool_call("block"), "Done"]), tools=[block], reporter=None)
    rival = Agent(model=FakeModel(["never used"]), reporter=None)
    state = State(messages=[Message.user("Task")])
    results: list = []
    worker = threading.Thread(target=lambda: results.append(first.run(state)))
    worker.start()
    try:
        assert started.wait(5)
        with pytest.raises(ValueError, match="already being run"):
            first.run(state)
        with pytest.raises(ValueError, match="already being run"):
            rival.run(state)
    finally:
        release.set()
        worker.join(5)
    assert results == ["Done"]
    assert len([h for h in state.history if isinstance(h, RunStartEntry)]) == 1  # the refused runs left no trace


def test_a_refused_second_run_records_nothing_in_the_state():
    """The guard raises before ``RunStartEntry``: the history is exactly what it was."""
    started = threading.Event()
    release = threading.Event()

    @tool
    def block() -> str:
        """Waits until the test lets it go."""
        started.set()
        assert release.wait(5)
        return "ok"

    agent = Agent(model=FakeModel([tool_call("block"), "Done"]), tools=[block], reporter=None)
    state = State(messages=[Message.user("Task")])
    worker = threading.Thread(target=lambda: agent.run(state))
    worker.start()
    try:
        assert started.wait(5)
        before = state.history
        with pytest.raises(ValueError):
            Agent(model=FakeModel([]), reporter=None).run(state)
        assert state.history == before
    finally:
        release.set()
        worker.join(5)


def test_the_run_guard_is_released_when_the_run_ends_normally_or_by_exception():
    """After a run returns, or an exception leaves it, the State can be run again."""
    state = State(messages=[Message.user("Task")])
    failing = Agent(model=FakeModel([RuntimeError("boom")]), reporter=None)
    with pytest.raises(RuntimeError):
        failing.run(state)
    ok = Agent(model=FakeModel(["Recovered"]), reporter=None)
    assert ok.run(state) == "Recovered"


def test_think_and_use_tools_are_not_blocked_by_the_run_guard_outside_a_run():
    """Only ``run``/``arun`` take the guard; calling the steps by hand on an idle State is always fine."""
    state = State(messages=[Message.user("Task")])
    agent = Agent(model=FakeModel(["a", "b"]), reporter=None)
    agent.think(state)
    agent.think(state)
    assert state.turn == 2


# ================================================================ any Agent continues a State


def test_agent_copy_with_another_model_continues_the_same_state():
    """ "Any Agent (e.g. agent.copy(model=...)) may run a State." Each run has its own RunStartEntry, and the
    ModelRequestEntry of each names the model that was asked."""
    cheap = Agent(model=FakeModel(["First answer"], name="cheap"), reporter=None)
    strong = cheap.copy(model=FakeModel(["Second answer"], name="strong"))
    state = State(messages=[Message.user("Task")])

    assert cheap.run(state) == "First answer"
    state.add_message(Message.user("Check it again"))
    assert strong.run(state) == "Second answer"

    starts = [h.content.model for h in state.history if isinstance(h, RunStartEntry)]
    asked = [h.content for h in state.history if isinstance(h, ModelRequestEntry)]
    assert starts == ["fake/cheap", "fake/strong"]
    assert asked == ["fake/cheap", "fake/strong"]
    assert state.turn == 2
    assert [m.role for m in state.messages] == ["user", "assistant", "user", "assistant"]


def test_switching_models_does_not_warn():
    """``ResumeWarning`` is only for the first run after ``store.load``, never for ordinary model switching."""
    state = State(messages=[Message.user("Task")])
    first = Agent(model=FakeModel(["A"], name="one"), reporter=None)
    first.run(state)
    state.add_message(Message.user("More"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first.copy(model=FakeModel(["B"], name="two"), system="Different").run(state)


def test_a_second_agent_can_use_tools_and_compact_a_state_the_first_one_thought_on():
    """The owner rule is gone for ``use_tools`` and ``compact`` too."""

    @tool
    def echo(msg: str) -> str:
        """Returns its input."""
        return msg

    state = State(messages=[Message.user("Task")])
    Agent(model=FakeModel([tool_call("echo", msg="hi")]), tools=[echo], reporter=None).think(state)
    other = Agent(model=FakeModel(["Summary text"]), tools=[echo], reporter=None)
    other.use_tools(state)
    assert "hi" in _result_texts(state)
    other.compact(state)
    assert state.messages[-1].text.endswith("Summary text")


# ================================================================ ModelRequestEntry / ModelReplyEntry


def test_think_records_a_request_entry_then_a_reply_entry_for_each_turn():
    """ "think records ModelRequestEntry(model name) before calling the model, then ModelReplyEntry." The request
    carries the turn it starts."""

    @tool
    def echo(msg: str) -> str:
        """Returns its input."""
        return msg

    agent = Agent(model=FakeModel([tool_call("echo", msg="x"), "Done"], name="m1"), tools=[echo], reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.think(state)
    agent.use_tools(state)
    agent.think(state)

    pairs = [(type(h), h.turn) for h in state.history if isinstance(h, (ModelRequestEntry, ModelReplyEntry))]
    assert pairs == [
        (ModelRequestEntry, 1),
        (ModelReplyEntry, 1),
        (ModelRequestEntry, 2),
        (ModelReplyEntry, 2),
    ]
    request = next(h for h in state.history if isinstance(h, ModelRequestEntry))
    assert request.kind == "model_request"
    assert request.content == "fake/m1"


def test_the_model_request_is_on_record_while_the_model_is_still_working():
    """The request entry is recorded before the model is called (so a process that dies while waiting leaves it)."""
    state = State(messages=[Message.user("Task")])
    seen: list = []

    def peek(request):
        seen.append(([h.kind for h in state.history][-1], state.turn))
        return "Reply"

    Agent(model=FakeModel([peek]), reporter=None).think(state)
    assert seen == [("model_request", 1)]
    assert state.history[-1].kind == "model_reply"


def test_failed_think_records_request_and_error_and_the_turn_reverts():
    """ "Request then error, no reply: a rollback." Messages and turn are as before; the error stays in history."""
    fake = FakeModel([RuntimeError("model error"), "Worked this time"])
    agent = Agent(model=fake, reporter=None)
    state = State(messages=[Message.user("Task")])
    before = state.snapshot()

    with pytest.raises(RuntimeError):
        agent.think(state)

    assert state.messages == before.messages
    assert state.turn == before.turn == 0
    assert state.usage == before.usage
    assert [h.kind for h in state.history[len(before.history) :]] == ["model_request", "error"]
    assert isinstance(state.history[-1].error, RuntimeError)

    agent.think(state)  # the same turn number again, no gap
    assert state.turn == 1
    assert state.history[-2].turn == 1  # the request of the retry
    assert state.answer == "Worked this time"


def test_failed_think_flushes_messages_added_while_the_model_was_working():
    """A message added while waiting is deferred. When the request fails it joins ``messages`` (after the
    rollback), so nothing the user added is lost."""
    state = State(messages=[Message.user("Task")])

    def fail_after_a_message(request):
        state.add_message(Message.user("Added while waiting"))
        assert [m.text for m in state.messages] == ["Task"]  # deferred
        raise RuntimeError("model error")

    agent = Agent(model=FakeModel([fail_after_a_message]), reporter=None)
    with pytest.raises(RuntimeError):
        agent.think(state)

    assert [m.text for m in state.messages] == ["Task", "Added while waiting"]
    assert state.turn == 0


def test_successful_think_puts_messages_added_while_waiting_after_the_reply():
    """With a reply the deferred messages follow the assistant message."""
    state = State(messages=[Message.user("Task")])

    def reply_after_a_message(request):
        state.add_message(Message.user("Added while waiting"))
        return "Reply"

    Agent(model=FakeModel([reply_after_a_message]), reporter=None).think(state)
    assert [(m.role, m.text) for m in state.messages] == [
        ("user", "Task"),
        ("assistant", "Reply"),
        ("user", "Added while waiting"),
    ]


def test_an_exception_after_the_reply_is_recorded_does_not_undo_the_reply():
    """ "An exception after the reply entry (e.g. raised by Reporter.on_think_end) does NOT undo the reply." """

    class _Failing(Reporter):
        def on_think_end(self, state, reply):
            raise RuntimeError("reporter failed")

    agent = Agent(model=FakeModel(["Kept reply"]), reporter=_Failing())
    state = State(messages=[Message.user("Task")])
    with pytest.raises(RuntimeError, match="reporter failed"):
        agent.think(state)
    assert state.turn == 1
    assert state.messages[-1].text == "Kept reply"
    assert [h.kind for h in state.history][-3:] == ["model_request", "model_reply", "error"]


def test_think_on_an_empty_state_raises_valueerror_and_records_nothing():
    """ "agent.think on a State with no messages raises ValueError." The model is not called."""
    fake = FakeModel(["never used"])
    agent = Agent(model=fake, reporter=None)
    state = State()
    with pytest.raises(ValueError, match="no messages"):
        agent.think(state)
    assert fake.requests == []
    assert state.history == ()
    assert state.turn == 0


def test_think_works_once_a_message_is_added_to_an_empty_state():
    """A State may start empty; its first message can be added later."""
    agent = Agent(model=FakeModel(["Hello"]), reporter=None)
    state = State()
    state.add_message(Message.user("Hi"))
    agent.think(state)
    assert [m.text for m in state.messages] == ["Hi", "Hello"]


# ================================================================ agent.context_tokens / context_used


def test_context_tokens_counts_messages_plus_this_agents_system_and_tools():
    """ "Estimate with this Agent's system/tool overhead." The same State costs more on an Agent with a system
    prompt and tools."""

    @tool
    def lookup(query: str) -> str:
        """Looks something up."""
        return query

    state = State(messages=[Message.user("Find the bug in calc.py " * 10)])
    plain = Agent(model=FakeModel([]), reporter=None)
    rich = Agent(model=FakeModel([]), system="You are a careful engineer. " * 5, tools=[lookup], reporter=None)
    assert plain.context_tokens(state) > 0
    assert rich.context_tokens(state) > plain.context_tokens(state)


def test_context_used_is_the_fraction_of_the_agents_models_context_window():
    """``context_used(state) == context_tokens(state) / model.context_window``, per Agent."""
    state = State(messages=[Message.user("word " * 1_000)])
    small = Agent(model=FakeModel([], context_window=1_000), reporter=None)
    large = Agent(model=FakeModel([], context_window=100_000), reporter=None)
    assert small.context_used(state) == pytest.approx(small.context_tokens(state) / 1_000)
    assert large.context_used(state) == pytest.approx(large.context_tokens(state) / 100_000)
    assert small.context_used(state) > 1.0 > large.context_used(state)


def test_context_tokens_follows_the_state_as_it_grows_and_compacts():
    """Sizes come from the State's messages now, so a compaction makes them smaller."""
    agent = Agent(model=FakeModel(["Short summary"]), reporter=None)
    state = State(messages=[Message.user("Task"), Message.assistant("word " * 2_000)])
    big = agent.context_tokens(state)
    agent.compact(state)
    assert agent.context_tokens(state) < big


def test_the_state_no_longer_has_context_tokens_or_context_used():
    """They moved to the Agent (the State does not know a model)."""
    state = State(messages=[Message.user("Task")])
    assert not hasattr(state, "context_tokens")
    assert not hasattr(state, "context_used")


# ================================================================ agent.compact records kept and usage


def test_compact_records_a_compact_context_change_with_the_summary_and_the_usage_of_the_request():
    """ "agent.compact ... same internal command as state.compact(summary) with usage=reply.usage"."""
    agent = Agent(model=FakeModel(["Files read so far: main.py"]), reporter=None)
    state = State(messages=[Message.user("Find the bug")])
    usage_before = state.usage
    agent.compact(state)

    change = state.history[-1].content
    assert state.history[-1].kind == "context_change"
    assert change.kind == "compact"
    assert change.summary == "Files read so far: main.py"
    assert change.kept == 0
    assert change.usage is not None and change.usage.requests == 1
    assert state.usage == usage_before + change.usage  # the cost of the summary is counted
    assert change.before_tokens > 0 and change.after_tokens > 0


def test_manual_compact_records_a_compact_change_without_usage():
    """``state.compact(summary)`` is the same entry with no model behind it."""
    state = State(messages=[Message.user("Find the bug"), Message.assistant("Looking"), Message.user("Go on")])
    state.compact("I wrote this")
    change = state.history[-1].content
    assert (change.kind, change.summary, change.kept, change.usage) == ("compact", "I wrote this", 0, None)
    assert [m.text for m in state.messages][0] == "Find the bug"
    assert state.messages[-1].text.endswith("I wrote this")
    assert len(state.messages) == 2
    assert state.usage.requests == 0


def test_compact_keeps_messages_added_while_the_model_wrote_the_summary():
    """ "kept= the number of messages added while the model was summarizing": they stay after the summary."""
    state = State(messages=[Message.user("Find the bug")])

    def summarize_while_user_types(request):
        state.add_message(Message.user("One more thing"))
        return "The summary"

    agent = Agent(model=FakeModel([summarize_while_user_types]), reporter=None)
    agent.compact(state)

    change = state.history[-1].content
    assert change.kept == 1
    assert [m.text for m in state.messages][0] == "Find the bug"
    assert state.messages[1].text.endswith("The summary")
    assert state.messages[-1].text == "One more thing"
    assert len(state.messages) == 3


def test_failed_compact_changes_nothing_and_a_later_one_still_works():
    """Nothing is recorded when the model fails, and the compaction window is closed again."""
    fake = FakeModel([RuntimeError("model error"), "Summary"])
    agent = Agent(model=fake, reporter=None)
    state = State(messages=[Message.user("Task")])
    before = state.history
    with pytest.raises(RuntimeError):
        agent.compact(state)
    assert state.history == before
    agent.compact(state)
    assert state.history[-1].content.kind == "compact"


# ================================================================ agent.ask records usage and JSON answers


def test_ask_records_an_ask_entry_with_its_usage_and_that_usage_reaches_the_state():
    """ "agent.ask: ExchangeEntry("ask", Exchange(question, json_answer), usage=reply.usage)"."""
    agent = Agent(model=FakeModel(["A tidy plan"]), reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.ask(state, "Summarize the work so far")

    entry = state.history[-1]
    assert entry.kind == "ask"
    assert entry.content.question == "Summarize the work so far"
    assert entry.content.answer == "A tidy plan"
    assert entry.usage is not None and entry.usage.requests == 1
    assert state.usage == entry.usage


def test_ask_returns_the_real_object_but_records_json():
    """ "returns the real object to the caller" while the history keeps JSON (a dict for a dataclass)."""
    agent = Agent(model=FakeModel(['{"steps": ["First", "Second"]}']), reporter=None)
    state = State(messages=[Message.user("Task")])
    plan = agent.ask(state, "Make a plan", returns=_Plan)

    assert plan == _Plan(steps=["First", "Second"])  # the object itself
    answer = state.history[-1].content.answer
    assert answer == {"steps": ["First", "Second"]}
    assert type(answer) is not _Plan
    json.dumps(answer)  # JSON all the way down


def test_ask_with_retries_records_one_entry_with_the_usage_of_every_try():
    """A retry is part of the same ask: one entry, usage summed."""
    fake = FakeModel(["This is not JSON", '{"steps": ["First step"]}'])
    agent = Agent(model=fake, reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.ask(state, "Make a plan", returns=_Plan)

    asks = [h for h in state.history if h.kind == "ask"]
    assert len(asks) == 1
    assert asks[0].usage.requests == 2
    assert state.usage.requests == 2


def test_ask_human_records_a_human_entry_without_usage():
    """ "ask_human: ExchangeEntry("human", ..., usage=None)"."""
    agent = Agent(model=FakeModel([]), human=FakeHuman(["yes"]), reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.ask_human(state, "Continue?", returns=bool)
    entry = state.history[-1]
    assert entry.kind == "human"
    assert entry.usage is None
    assert entry.content.answer is True
    assert state.usage.requests == 0


# ================================================================ finish is a stop


def test_finish_records_a_stop_entry_with_the_answer():
    """ "finish records StopEntry(StoppedByFinish(answer))" and ``finished`` / ``answer`` follow it."""
    state = State(messages=[Message.user("Task")])
    state.finish({"summary": "done"})
    entry = state.history[-1]
    assert isinstance(entry, StopEntry)
    assert entry.content == StoppedByFinish(answer={"summary": "done"})
    assert state.finished is True
    assert state.answer == {"summary": "done"}
    assert state.stopped == entry.content


def test_finish_inside_a_run_ends_it_and_is_recorded_as_a_stop():
    """A tool-less loop that finishes: the run's stop is the finish, not the default until."""

    def finishing_loop(agent, state):
        agent.think(state)
        state.finish("All done")
        return state.answer

    agent = Agent(model=FakeModel(["thinking"]), loop=finishing_loop, reporter=None)
    state = State(messages=[Message.user("Task")])
    assert agent.run(state) == "All done"
    stops = [h.content for h in state.history if isinstance(h, StopEntry)]
    assert stops == [StoppedByFinish(answer="All done")]


# ================================================================ reporter= check


def test_duck_typed_reporter_must_have_every_reporter_hook():
    """The check follows the Reporter class: an object missing any on_* hook (here on_model_event) fails at
    construction, not at the first model event."""
    hooks = [name for name in vars(Reporter) if name.startswith("on_")]
    assert "on_model_event" in hooks

    class Partial:
        pass

    for name in hooks:
        if name != "on_model_event":
            setattr(Partial, name, lambda self, *args: None)

    with pytest.raises(TypeError, match="missing methods: on_model_event"):
        Agent(model=FakeModel([]), reporter=Partial(), human=None)
