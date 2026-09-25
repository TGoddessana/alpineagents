"""Black-box tests for State (ARCHITECTURE.md "State internal contract").

Tests follow the spec sentences (derived from the spec, not the implementation). Covered:

- All of State: the exact meaning of questions and values, history content per kind, "what it does not do"
- Context management: start_from, clear_tool_results, the three stages of compaction
- Lock rules for concurrency

Excluded: save/load, subagent (parent/root/depth) behavior.

No network. Uses ``alpineagents.testing.FakeModel``/``FakeHuman`` and ``reporter=None``.
"""

import threading
import time

import pytest

from alpineagents import Agent, Reporter, State, tool
from alpineagents.testing import FakeHuman, FakeModel, tool_call
from alpineagents.types import HistoryEntry, RawBlock, ToolOutcome, Usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SpyReporter(Reporter):
    """Observer that records only the two notifications State sends (on_context_change, on_tool_end)."""

    def __init__(self):
        self.context_changes = []
        self.tool_ends = []

    def on_context_change(self, state, change):
        self.context_changes.append(change)

    def on_tool_end(self, state, call, result, outcome):
        self.tool_ends.append((call, result, outcome))


def make_agent(replies, *, tools=(), human=None, reporter=None):
    return Agent(model=FakeModel(list(replies)), tools=list(tools), human=human, reporter=reporter)


# ---------------------------------------------------------------------------
# State(task): values at construction
# ---------------------------------------------------------------------------


def test_task_must_be_a_string():
    # "state = State(...)" takes a single sentence (str).
    with pytest.raises(TypeError):
        State(123)


def test_initial_values_match_construction_contract():
    # State(task) -> history=[user(task)], context=[user(task)], turn=0, usage=Usage(),
    # pending_calls=(), stopped_by=None.
    state = State("Find the bug")

    assert state.history == (HistoryEntry("user", "Find the bug", turn=0),)
    assert [m.text for m in state.context] == ["Find the bug"]
    assert state.turn == 0
    assert state.usage == Usage()
    assert state.pending_calls == ()
    assert state.stopped_by is None
    assert state.answer is None
    assert state.is_finished() is False
    assert state.wants_tools() is False


# ---------------------------------------------------------------------------
# The exact meaning of questions and values
# ---------------------------------------------------------------------------


def test_is_answered_true_when_last_message_is_toolcall_free_reply():
    # "is_answered(): the last item in the context is a model reply without tool calls."
    agent = make_agent(["The bug is on line 3"])
    state = State("Find the bug")
    agent.think(state)
    assert state.is_answered() is True


def test_is_answered_false_after_appending_anything_after_answer():
    # "Adding anything after it makes it false."
    agent = make_agent(["The bug is on line 3"])
    state = State("Find the bug")
    agent.think(state)
    assert state.is_answered() is True
    state.add_notice("Extra instruction")
    assert state.is_answered() is False


def test_is_answered_false_while_pending_calls_exist():
    # is_answered() is false while there are pending calls ("false if there are pending calls").
    call = tool_call("read_file", path="main.py")
    agent = make_agent([call])
    state = State("Find the bug")
    agent.think(state)
    assert state.is_answered() is False


def test_wants_tools_true_iff_pending_calls_not_empty():
    # "wants_tools(): there are pending calls (pending_calls is not empty)"
    call = tool_call("read_file", path="main.py")

    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return "contents"

    agent = make_agent([call, "done"], tools=[read_file])
    state = State("Find the bug")
    agent.think(state)
    assert state.wants_tools() is True
    assert len(state.pending_calls) == 1

    agent.use_tools(state)
    assert state.wants_tools() is False
    assert state.pending_calls == ()


def test_is_finished_reflects_finish_call():
    # "is_finished(): finish() was called"
    state = State("Task")
    assert state.is_finished() is False
    state.finish()
    assert state.is_finished() is True


def test_answer_is_none_before_any_reply():
    # "answer: ... None if there is none yet"
    assert State("Task").answer is None


def test_answer_is_last_toolcall_free_reply_text():
    # "answer: the text of the most recent model reply without tool calls"
    agent = make_agent(["This is the final answer"])
    state = State("Task")
    agent.think(state)
    assert state.answer == "This is the final answer"


def test_finish_with_answer_overrides_last_reply_text():
    # "answer: ... or the value given to finish(answer)"
    agent = make_agent(["the model's answer"])
    state = State("Task")
    agent.think(state)
    state.finish("the answer a human chose")
    assert state.answer == "the answer a human chose"


def test_finish_without_answer_keeps_previous_answer():
    # finish(answer=None) does not change answer (pairs with the "last value wins" rule).
    agent = make_agent(["the model's answer"])
    state = State("Task")
    agent.think(state)
    state.finish()
    assert state.answer == "the model's answer"


def test_finish_called_twice_last_given_value_wins():
    # "Calling it twice is not an error, and if answer is given, the last value wins."
    state = State("Task")
    state.finish("first")
    state.finish("second")
    assert state.answer == "second"
    state.finish()  # calling again without answer keeps the previous value
    assert state.answer == "second"


def test_pending_calls_expose_name_args_and_id():
    # "pending_calls: calls the model requested this turn that have no result yet. Each has name, args and id"
    call = tool_call("search", query="alpineagents")
    agent = make_agent([call])
    state = State("Task")
    agent.think(state)

    [pending] = state.pending_calls
    assert pending.name == "search"
    assert pending.args == {"query": "alpineagents"}
    assert pending.id == call.id


def test_turn_starts_at_zero_and_counts_thinks():
    # "turn: number of thinks so far. Counts from 1" (value after the call)
    agent = make_agent(["one", "two"])
    state = State("Task")
    assert state.turn == 0
    agent.think(state)
    assert state.turn == 1
    agent.think(state)
    assert state.turn == 2


def test_usage_accumulates_think_and_ask():
    # "usage: ... includes ask and subagents"
    agent = make_agent(["thought answer", "summarized answer"])
    state = State("Task")
    agent.think(state)
    agent.ask(state, "Summarize what you have so far")
    assert state.usage.requests == 2
    assert state.usage.input_tokens > 0
    assert state.usage.output_tokens > 0


def test_context_used_is_zero_before_first_claim():
    # "before _claim, overhead 0 and context_used 0.0" (ARCHITECTURE.md "Turns and pending calls")
    assert State("Task").context_used == 0.0


def test_stopped_by_is_until_function_name_when_condition_true():
    # "stopped_by: ... the name of the until condition function"
    agent = make_agent(["final answer"])
    state = State("Task")
    answer = agent.run(state)
    assert answer == "final answer"
    assert state.stopped_by == "is_answered"


def test_stopped_by_is_limit_when_turn_limit_reached():
    # "limit reached: stop without an exception and return state.answer (None if no answer yet)"
    from alpineagents import loop as loop_decorator

    @tool
    def noop(x: int) -> str:
        """Do nothing."""
        return "ok"

    @loop_decorator(until=State.is_answered, limit=2)
    def never_answers(agent, state):
        agent.think(state)
        if state.wants_tools():
            agent.use_tools(state)

    replies = [tool_call("noop", x=1), tool_call("noop", x=2)]
    agent = Agent(model=FakeModel(replies), tools=[noop], loop=never_answers, reporter=None, human=None)
    state = State("Task")
    answer = agent.run(state)

    assert answer is None
    assert state.stopped_by == "limit"
    assert state.turn == 2


# ---------------------------------------------------------------------------
# history content per kind
# ---------------------------------------------------------------------------


def test_history_first_entry_is_user_kind_with_task():
    entry = State("Find the bug").history[0]
    assert entry.kind == "user"
    assert entry.content == "Find the bug"
    assert entry.turn == 0


def test_add_user_message_appends_user_kind_entry():
    # "add_user_message(text): what the human says"
    state = State("Task")
    state.add_user_message("Move on to the next step")
    assert state.history[-1].kind == "user"
    assert state.history[-1].content == "Move on to the next step"


def test_history_reply_kind_holds_reply_object():
    # "reply: Reply"
    agent = make_agent(["text answer"])
    state = State("Task")
    agent.think(state)
    entry = state.history[-1]
    assert entry.kind == "reply"
    assert entry.content.text == "text answer"


def test_history_tool_result_kind_after_use_tools():
    # "tool_result: str (the result sent to the model). Has call"
    call = tool_call("read_file", path="main.py")

    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return "file contents"

    agent = make_agent([call], tools=[read_file])
    state = State("Task")
    agent.think(state)
    agent.use_tools(state)

    entry = state.history[-1]
    assert entry.kind == "tool_result"
    assert entry.content == "file contents"
    assert entry.call.id == call.id


def test_add_notice_prefixes_and_records_notice_kind():
    # "notice: str (starts with [notice] ). Added if missing."
    state = State("Task")
    state.add_notice("The tests failed")
    entry = state.history[-1]
    assert entry.kind == "notice"
    assert entry.content == "[notice] The tests failed"


def test_add_notice_does_not_double_prefix_already_prefixed_text():
    state = State("Task")
    state.add_notice("[notice] already prefixed")
    assert state.history[-1].content == "[notice] already prefixed"


def test_history_denied_kind_on_deny():
    # "denied: str (the reason for denial). Has call"
    call = tool_call("bash", command="rm -rf /")
    agent = make_agent([call])
    state = State("Task")
    agent.think(state)

    state.deny(state.pending_calls[0], "This is a dangerous command")

    entry = state.history[-1]
    assert entry.kind == "denied"
    assert entry.content == "This is a dangerous command"
    assert entry.call.id == call.id


def test_history_ask_kind_holds_question_and_answer_exchange():
    # "ask: Exchange"
    agent = make_agent(["a tidy summary"])
    state = State("Task")
    value = agent.ask(state, "Summarize what you have so far")

    entry = state.history[-1]
    assert entry.kind == "ask"
    assert entry.content.question == "Summarize what you have so far"
    assert entry.content.answer == value == "a tidy summary"


def test_history_human_kind_holds_question_and_answer_exchange():
    # "human: Exchange" (the question and answer of ask_human)
    agent = Agent(model=FakeModel([]), human=FakeHuman(["yes"]), reporter=None)
    state = State("Task")
    value = agent.ask_human(state, "Proceed?")

    entry = state.history[-1]
    assert entry.kind == "human"
    assert entry.content.question == "Proceed?"
    assert entry.content.answer == value == "yes"


def test_history_context_change_kind_on_start_from():
    # "context_change: ContextChange"
    state = State("Task")
    state.start_from("summary so far")
    entry = state.history[-1]
    assert entry.kind == "context_change"
    assert entry.content.kind == "start_from"
    assert entry.content.summary == "summary so far"


def test_history_error_kind_recorded_on_model_exception():
    # "error: str (\"TimeoutError: ...\"). The exception object in error"
    agent = make_agent([TimeoutError("timed out")])
    state = State("Task")
    with pytest.raises(TimeoutError):
        agent.think(state)

    entry = state.history[-1]
    assert entry.kind == "error"
    assert entry.content == "TimeoutError: timed out"
    assert isinstance(entry.error, TimeoutError)


# ---------------------------------------------------------------------------
# "What it does not do"
# ---------------------------------------------------------------------------


def test_history_never_shrinks_when_context_is_cleared():
    # "It never edits or deletes history. Shrinking or rolling back the context leaves history unchanged."
    call = tool_call("read_file", path="a.py")

    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return "original contents"

    agent = make_agent([call], tools=[read_file])
    state = State("Task")
    agent.think(state)
    agent.use_tools(state)
    history_len_before = len(state.history)

    state.clear_tool_results(keep_last=0)

    assert len(state.history) > history_len_before
    tool_result_entries = [e for e in state.history if e.kind == "tool_result"]
    assert tool_result_entries[0].content == "original contents"  # cleared only in the context; history is unchanged


def test_raw_blocks_pass_through_unmodified():
    # "It does not transform received messages. ... keeps the original and sends it back as is."
    raw = RawBlock(provider="fake", data={"thinking": "inner thoughts"})
    agent = make_agent([[raw, "answer"]])
    state = State("Task")
    agent.think(state)

    assert raw in state.context[-1].content


def test_second_agent_thinking_on_same_state_raises_valueerror():
    # "Two Agents' conversations never mix. The first Agent to think becomes the owner, and another Agent
    # thinking raises an error."
    agent1 = make_agent(["A's thought"])
    agent2 = make_agent(["B's thought"])
    state = State("Task")
    agent1.think(state)

    with pytest.raises(ValueError):
        agent2.think(state)


def test_other_agent_can_read_via_ask_without_owning_state():
    # "Other Agents only read it with ask."
    agent1 = make_agent(["A's thought"])
    agent2 = make_agent(["B's review"])
    state = State("Task")
    agent1.think(state)

    review = agent2.ask(state, "Review the changes so far")
    assert review == "B's review"


def test_state_data_is_not_visible_in_context():
    # "state.data: ... not visible to the model."
    state = State("Task")
    state.data["secret"] = "a value the model must not see"
    assert not any("a value the model must not see" in m.text for m in state.context)


# ---------------------------------------------------------------------------
# Turns and pending calls (ARCHITECTURE.md "Turns and pending calls")
# ---------------------------------------------------------------------------


def _agent_with_pending_call(call_name="bash", **args):
    call = tool_call(call_name, **args)

    @tool
    def bash(command: str) -> str:
        """Run a command."""
        return "ran"

    agent = make_agent([call, "finished answer"], tools=[bash])
    state = State("Task")
    agent.think(state)
    assert state.wants_tools() is True
    return agent, state


def test_think_raises_valueerror_when_pending_calls_remain():
    # "think ... while there are pending calls is a ValueError."
    agent, state = _agent_with_pending_call(command="ls")
    with pytest.raises(ValueError):
        agent.think(state)


def test_compact_raises_valueerror_when_pending_calls_remain():
    agent, state = _agent_with_pending_call(command="ls")
    with pytest.raises(ValueError):
        agent.compact(state)


def test_start_from_raises_valueerror_when_pending_calls_remain():
    _, state = _agent_with_pending_call(command="ls")
    with pytest.raises(ValueError):
        state.start_from("summary")


def test_clear_tool_results_raises_valueerror_when_pending_calls_remain():
    _, state = _agent_with_pending_call(command="ls")
    with pytest.raises(ValueError):
        state.clear_tool_results()


def test_add_notice_defers_instead_of_raising_when_pending():
    # "add_notice/add_user_message defer instead of raising."
    agent, state = _agent_with_pending_call(command="ls")
    state.add_notice("hold on")
    assert not any("hold on" in m.text for m in state.context)  # not in the context yet

    agent.use_tools(state)
    assert "hold on" in state.context[-1].text  # goes in after the last call is closed


def test_add_user_message_defers_instead_of_raising_when_pending():
    agent, state = _agent_with_pending_call(command="ls")
    state.add_user_message("earlier message")
    assert not any("earlier message" in m.text for m in state.context)

    agent.use_tools(state)
    assert "earlier message" in state.context[-1].text


def test_deny_ask_and_ask_human_allowed_while_pending_calls_remain():
    # "finish, deny, ask and ask_human are allowed." (here: deny/ask/ask_human)
    _, state = _agent_with_pending_call(command="ls")

    # ask works even with pending calls (it does not change the context)
    agent2 = make_agent(["separate review"])
    before = state.context
    review = agent2.ask(state, "Review it")
    assert review == "separate review"
    assert state.context == before

    human_agent = Agent(model=FakeModel([]), human=FakeHuman(["yes"]), reporter=None)
    assert human_agent.ask_human(state, "Proceed?") == "yes"

    state.deny(state.pending_calls[0], "the user denied it")  # no error
    assert state.pending_calls == ()


def test_finish_allowed_while_pending_calls_remain():
    # "finish, deny, ask and ask_human are allowed." (here: finish, as called by a submit tool)
    _, state = _agent_with_pending_call(command="ls")
    state.finish("finished inside a tool")  # no error
    assert state.is_finished()
    assert state.answer == "finished inside a tool"


def test_deny_unknown_call_raises_valueerror():
    # "deny of a call not in pending_calls" (mistake-proofing error)
    state = State("Task")
    unknown = tool_call("nonexistent")
    with pytest.raises(ValueError):
        state.deny(unknown, "reason")


def test_ask_uses_not_run_yet_placeholder_for_unresolved_call():
    # "placeholder result for ask | (not run yet)" - ask puts this text in place of pending calls.
    agent, state = _agent_with_pending_call(command="ls")
    fake = agent.model
    agent.ask(state, "Summarize the current situation")

    sent = fake.requests[-1].messages
    placeholder_message = sent[-2]  # right before the question message
    assert placeholder_message.content[0].content == "(not run yet)"


def test_tool_results_enter_context_in_request_order_not_completion_order():
    # "Results collect in this turn's buffer and, the moment the last call closes, enter the context as one
    # user message in request order." Even if fast finishes first, request order (slow, fast) is kept.

    @tool
    def slow(x: int) -> str:
        """A tool that finishes slowly."""
        time.sleep(0.05)
        return "slow"

    @tool
    def fast(x: int) -> str:
        """A tool that finishes quickly."""
        return "fast"

    call_slow = tool_call("slow", x=1)
    call_fast = tool_call("fast", x=2)
    agent = make_agent([[call_slow, call_fast]], tools=[slow, fast])
    state = State("Task")
    agent.think(state)
    agent.use_tools(state)

    result_message = state.context[-1]
    contents = [block.content for block in result_message.content]
    assert contents == ["slow", "fast"]


def test_deferred_messages_enter_context_in_arrival_order_after_tool_results():
    # "After that, deferred user messages and notices go in, in arrival order."
    agent, state = _agent_with_pending_call(command="ls")
    state.add_user_message("earlier message")
    state.add_notice("later notice")
    agent.use_tools(state)

    tail = state.context[-3:]
    assert tail[0].content[0].content == "ran"  # the tool result message
    assert tail[1].text == "earlier message"
    assert "later notice" in tail[2].text


# ---------------------------------------------------------------------------
# Context rules after exceptions and interrupts (ARCHITECTURE.md "run order and the context rules after exceptions")
# ---------------------------------------------------------------------------


def test_think_exception_rolls_context_back_to_before_that_think():
    # "think (during the model request): roll back to just before that think. No trace in the outside world."
    agent = make_agent([TimeoutError("timed out")])
    state = State("Task")
    with pytest.raises(TimeoutError):
        agent.think(state)

    assert state.turn == 0
    assert [m.text for m in state.context] == ["Task"]


def test_use_tools_exception_keeps_unfinished_call_pending_but_records_finished_one():
    # "use_tools (during tool execution): results of finished calls are recorded. Failed, unfinished and
    # unstarted calls stay in pending_calls, and the exception is raised."

    @tool
    def boom(x: int) -> str:
        """A tool that raises."""
        raise TimeoutError("too slow, failed")

    @tool
    def ok(x: int) -> str:
        """A normal tool."""
        return "success"

    call_ok = tool_call("ok", x=1)
    call_boom = tool_call("boom", x=2)
    agent = make_agent([[call_ok, call_boom]], tools=[ok, boom])
    state = State("Task")
    agent.think(state)

    with pytest.raises(TimeoutError):
        agent.use_tools(state)

    assert {c.name for c in state.pending_calls} == {"boom"}
    tool_result_contents = [e.content for e in state.history if e.kind == "tool_result"]
    assert "success" in tool_result_contents


def test_run_closes_remaining_pending_call_with_interrupted_marker_on_exception():
    # Result text table: "closed by an exception | (aborted: TimeoutError)"
    @tool
    def boom(x: int) -> str:
        """A tool that raises."""
        raise TimeoutError("failed")

    agent = make_agent([tool_call("boom", x=1)], tools=[boom])
    state = State("Task")
    with pytest.raises(TimeoutError):
        agent.run(state)

    assert state.pending_calls == ()
    last_result = [e for e in state.history if e.kind == "tool_result"][-1]
    assert last_result.content == "(aborted: TimeoutError)"


def test_run_closes_remaining_pending_call_with_user_interrupted_marker_on_keyboardinterrupt():
    # Result text table: "closed by an interrupt | (interrupted by user)"
    @tool
    def boom(x: int) -> str:
        """A tool that simulates an interrupt."""
        raise KeyboardInterrupt()

    agent = make_agent([tool_call("boom", x=1)], tools=[boom])
    state = State("Task")
    with pytest.raises(KeyboardInterrupt):
        agent.run(state)

    last_result = [e for e in state.history if e.kind == "tool_result"][-1]
    assert last_result.content == "(interrupted by user)"


def test_think_after_finish_raises_valueerror():
    # "calling think, use_tools or ask after finish()"
    agent = make_agent(["text"])
    state = State("Task")
    state.finish()
    with pytest.raises(ValueError):
        agent.think(state)


def test_use_tools_and_ask_after_finish_also_raise_valueerror():
    agent = make_agent(["text"])
    state = State("Task")
    state.finish()
    with pytest.raises(ValueError):
        agent.use_tools(state)
    with pytest.raises(ValueError):
        agent.ask(state, "question")


# ---------------------------------------------------------------------------
# Context management: the three stages of compaction
# ---------------------------------------------------------------------------


def test_compact_stage1_plain_call_replaces_context_with_summary():
    # "Stage 1: just call it -- agent.compact(state)"
    agent = make_agent(["summary of the work so far"])
    state = State("Task")
    agent.compact(state)

    assert [m.text for m in state.context] == [
        "Task",
        "[notice] Summary so far:\nsummary of the work so far",
    ]


def test_compact_stage2_instructions_are_forwarded_to_the_summary_request():
    # "Stage 2: say what to keep -- agent.compact(state, instructions=...)"
    agent = make_agent(["summary"])
    state = State("Task")
    agent.compact(state, instructions="Keep architecture decisions, remaining bugs and changed files")

    sent_text = agent.model.requests[-1].messages[-1].text
    assert "Be sure to keep: Keep architecture decisions, remaining bugs and changed files" in sent_text


def test_compact_stage3_ask_then_start_from_leaves_context_untouched_until_start_from():
    # "Stage 3: make the summary yourself (ask does not change the context or call tools)"
    agent = make_agent(["a summary of the work done so far"])
    state = State("Task")
    before = state.context

    summary = agent.ask(state, "Summarize the work done so far")
    assert state.context == before  # ask does not change the context

    state.start_from(summary)
    assert [m.text for m in state.context] == [
        "Task",
        "[notice] Summary so far:\na summary of the work done so far",
    ]


def test_start_from_reduces_context_to_task_and_summary_only():
    # "start_from(summary) replaces the context with two messages: the original task + the summary. Old turns
    # are not sent to the model again."
    call = tool_call("read_file", path="a.py")

    @tool
    def read_file(path: str) -> str:
        """Read."""
        return "long file contents"

    agent = make_agent([call, "intermediate answer"], tools=[read_file])
    state = State("Task")
    agent.think(state)
    agent.use_tools(state)
    agent.think(state)
    assert len(state.context) > 2  # several turns have piled up

    state.start_from("the summary")
    assert len(state.context) == 2
    assert state.context[0].text == "Task"
    assert state.context[1].text == "[notice] Summary so far:\nthe summary"


def test_context_change_notifies_reporter_on_start_from():
    # "The Reporter shows this with on_context_change."
    spy = SpyReporter()
    agent = Agent(model=FakeModel(["first answer"]), reporter=spy, human=None)
    state = State("Task")
    agent.think(state)  # becomes this State's owner, so spy is the Reporter to notify

    state.start_from("summary")

    assert len(spy.context_changes) == 1
    assert spy.context_changes[0].kind == "start_from"


def test_deny_notifies_reporter_on_tool_end_with_reason_as_result():
    # "on_tool_end (denied) | State.deny | right after recording (outside the lock)"
    spy = SpyReporter()
    call = tool_call("bash", command="rm -rf /")
    agent = Agent(model=FakeModel([call]), reporter=spy, human=None)
    state = State("Task")
    agent.think(state)

    state.deny(state.pending_calls[0], "dangerous command")

    assert len(spy.tool_ends) == 1
    denied_call, result, outcome = spy.tool_ends[0]
    assert denied_call.id == call.id
    assert result == "dangerous command"
    assert outcome == ToolOutcome("denied")


# ---------------------------------------------------------------------------
# Context management: clearing tool results
# ---------------------------------------------------------------------------


def test_clear_tool_results_clears_older_results_and_keeps_last_n():
    # "Old tool results are cleared only from the context with state.clear_tool_results(keep_last=5).
    # (cleared: kept in history) is left in their place."

    @tool
    def echo(x: int) -> str:
        """Return it unchanged."""
        return f"result{x}"

    agent = make_agent(
        [tool_call("echo", x=1), tool_call("echo", x=2), tool_call("echo", x=3)],
        tools=[echo],
    )
    state = State("Task")
    for _ in range(3):
        agent.think(state)
        agent.use_tools(state)

    state.clear_tool_results(keep_last=1)

    results = [
        block.content
        for message in state.context
        for block in message.content
        if hasattr(block, "call_id")
    ]
    assert results == ["(cleared: kept in history)", "(cleared: kept in history)", "result3"]


def test_clear_tool_results_does_not_change_history():
    # "It differs from compaction in that it clears instead of summarizing" -- history keeps the original.
    @tool
    def echo(x: int) -> str:
        """Return it unchanged."""
        return f"result{x}"

    agent = make_agent([tool_call("echo", x=1), tool_call("echo", x=2)], tools=[echo])
    state = State("Task")
    for _ in range(2):
        agent.think(state)
        agent.use_tools(state)

    state.clear_tool_results(keep_last=0)

    tool_result_contents = [e.content for e in state.history if e.kind == "tool_result"]
    assert tool_result_contents == ["result1", "result2"]


# ---------------------------------------------------------------------------
# Lock rules for concurrency (ARCHITECTURE.md "Locking")
# ---------------------------------------------------------------------------


def test_state_lock_is_reentrant():
    # "State methods are guarded by the reentrant lock state.lock" -- the same thread taking it twice
    # (and calling State methods inside) does not block.
    state = State("Task")
    with state.lock:
        with state.lock:
            state.finish("done")  # State methods take the same lock again internally
    assert state.is_finished()


def test_state_lock_protects_multi_step_data_mutation_across_threads():
    # "Multi-step changes (data.setdefault(...).append(...)) go inside with state.lock:."
    # No value may be lost even when several threads read->compute->write at the same time.
    state = State("Task")

    def worker():
        for _ in range(200):
            with state.lock:
                current = state.data.get("count", 0)
                state.data["count"] = current + 1

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert state.data["count"] == 1600


def test_str_summarizes_history_one_line_per_entry():
    @tool
    def read_file(path: str) -> str:
        """Read a file"""
        return "x" * 1300

    fake = FakeModel([tool_call("read_file", path="main.py"), "The bug is on line 3"])
    state = State("Find the bug")
    Agent(model=fake, tools=[read_file], reporter=None, human=None).run(state)

    text = str(state)
    lines = text.splitlines()
    assert len(lines) == len(state.history) + 1
    assert "Find the bug" in lines[0]
    assert "read_file" in text
    assert "The bug is on line 3" in text
    assert lines[-1] == "done: is_answered (2 turns)"
