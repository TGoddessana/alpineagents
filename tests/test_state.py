"""Black-box tests for State (SPEC-0.5 sections 2 and 4).

Tests follow the spec sentences (derived from the spec, not the implementation). Covered:

- All of State: the exact meaning of values, history content per kind, "what it does not do"
- Context management: compact, clear_tool_results, the three stages of compaction
- Concurrency rules (edit_extra_data, thread-safe commands)

Excluded: save/load, subagent (parent/root/depth) behavior. The snapshot value, the constructor and the
``history`` invariant are in test_state_snapshot.py.

No network. Uses ``alpineagents.testing.FakeModel``/``FakeHuman`` and ``reporter=None``.
"""

import threading
import typing
import time

import pytest

from alpineagents import Agent, Message, Reporter, State, StoppedByLimit, StoppedByUntil, tool
from alpineagents.loop import is_answered
from alpineagents.permissions import DenyByName
from alpineagents.testing import FakeHuman, FakeModel, tool_call
from alpineagents import types
from alpineagents.types import (
    ContextChangeEntry,
    ErrorEntry,
    ModelReplyEntry,
    ModelRequestEntry,
    RawBlock,
    ToolOutcome,
    ToolOutcomeKind,
    ToolResultEntry,
    Usage,
)

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


def task(text="Task", **kwargs):
    """A State that starts with one user message (the 0.4 ``State(text)``)."""
    return State(messages=[Message.user(text)], **kwargs)


# ---------------------------------------------------------------------------
# State(messages=...): values at construction
# ---------------------------------------------------------------------------


def test_state_takes_no_positional_argument():
    # 0.4's State("text") is now State(messages=[Message.user("text")]); the error shows that form.
    with pytest.raises(TypeError, match=r"State\(messages="):
        State("Find the bug")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        State(123)  # type: ignore[call-arg]


def test_messages_must_be_message_objects():
    with pytest.raises(TypeError):
        State(messages=["Find the bug"])  # type: ignore[list-item]


def test_initial_values_match_construction_contract():
    # State(messages=[user]) -> one import entry at turn 0, messages=[user], turn=0, usage=Usage(),
    # pending_calls=(), stopped=None.
    state = State(messages=[Message.user("Find the bug")])

    [entry] = state.history
    assert isinstance(entry, ContextChangeEntry)
    assert entry.content.kind == "import"
    assert entry.turn == 0
    assert [m.text for m in state.messages] == ["Find the bug"]
    assert state.turn == 0
    assert state.usage == Usage()
    assert state.pending_calls == ()
    assert state.stopped is None
    assert state.answer is None
    assert state.finished is False
    assert dict(state.extra_data) == {}


def test_an_empty_state_has_no_history_and_no_times():
    state = State()
    assert state.history == ()
    assert state.messages == ()
    assert state.created_at is None and state.updated_at is None


# ---------------------------------------------------------------------------
# The exact meaning of values
# ---------------------------------------------------------------------------


def test_default_until_is_true_when_last_message_is_toolcall_free_reply():
    # The default loops' until (loop.is_answered): the last message is a model reply without tool calls.
    agent = make_agent(["The bug is on line 3"])
    state = task("Find the bug")
    assert is_answered(state) is False
    agent.think(state)
    assert is_answered(state) is True


def test_default_until_is_false_after_adding_anything_after_the_answer():
    # "Adding anything after it makes it false."
    agent = make_agent(["The bug is on line 3"])
    state = task("Find the bug")
    agent.think(state)
    assert is_answered(state) is True
    state.add_message(Message.notice("Extra instruction"))
    assert is_answered(state) is False


def test_default_until_is_false_while_pending_calls_exist():
    call = tool_call("read_file", path="main.py")
    agent = make_agent([call])
    state = task("Find the bug")
    agent.think(state)
    assert is_answered(state) is False


def test_pending_calls_are_non_empty_iff_the_model_asked_for_tools():
    call = tool_call("read_file", path="main.py")

    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return "contents"

    agent = make_agent([call, "done"], tools=[read_file])
    state = task("Find the bug")
    assert state.pending_calls == ()
    agent.think(state)
    assert bool(state.pending_calls) is True
    assert len(state.pending_calls) == 1

    agent.use_tools(state)
    assert bool(state.pending_calls) is False
    assert state.pending_calls == ()


def test_finished_reflects_finish_call():
    state = task()
    assert state.finished is False
    state.finish()
    assert state.finished is True


def test_answer_is_none_before_any_reply():
    assert task().answer is None


def test_answer_is_last_toolcall_free_reply_text():
    agent = make_agent(["This is the final answer"])
    state = task()
    agent.think(state)
    assert state.answer == "This is the final answer"


def test_finish_with_answer_overrides_last_reply_text():
    agent = make_agent(["the model's answer"])
    state = task()
    agent.think(state)
    state.finish("the answer a human chose")
    assert state.answer == "the answer a human chose"


def test_finish_without_answer_keeps_previous_answer():
    agent = make_agent(["the model's answer"])
    state = task()
    agent.think(state)
    state.finish()
    assert state.answer == "the model's answer"


def test_finish_called_twice_last_given_value_wins():
    state = task()
    state.finish("first")
    state.finish("second")
    assert state.answer == "second"
    state.finish()  # calling again without answer keeps the previous value
    assert state.answer == "second"


def test_finish_records_one_stop_entry_per_call():
    state = task()
    state.finish("first")
    state.finish("second")
    stops = [e for e in state.history if e.kind == "stop"]
    assert [e.content.answer for e in stops] == ["first", "second"]


def test_pending_calls_expose_name_args_and_id():
    call = tool_call("search", query="alpineagents")
    agent = make_agent([call])
    state = task()
    agent.think(state)

    [pending] = state.pending_calls
    assert pending.name == "search"
    assert pending.args == {"query": "alpineagents"}
    assert pending.id == call.id


def test_turn_starts_at_zero_and_counts_thinks():
    agent = make_agent(["one", "two"])
    state = task()
    assert state.turn == 0
    agent.think(state)
    assert state.turn == 1
    agent.think(state)
    assert state.turn == 2


def test_usage_accumulates_think_and_ask():
    agent = make_agent(["thought answer", "summarized answer"])
    state = task()
    agent.think(state)
    agent.ask(state, "Summarize what you have so far")
    assert state.usage.requests == 2
    assert state.usage.input_tokens > 0
    assert state.usage.output_tokens > 0


def test_context_used_and_tokens_are_asked_of_the_agent():
    # context_used / context_tokens moved from State to Agent: they use that Agent's window and overhead.
    agent = make_agent(["x"])
    assert agent.context_tokens(State()) == 0
    assert agent.context_used(State()) == 0.0

    state = task("x" * 4000)
    assert agent.context_tokens(state) > 0
    assert 0.0 < agent.context_used(state) < 1.0
    assert agent.context_used(state) == pytest.approx(agent.context_tokens(state) / agent.model.context_window)
    # A smaller window of another Agent gives a bigger fraction for the same State.
    small = Agent(model=FakeModel([], context_window=2000), reporter=None, human=None)
    assert small.context_used(state) > agent.context_used(state)


def test_stopped_is_until_function_name_when_condition_true():
    agent = make_agent(["final answer"])
    state = task()
    answer = agent.run(state)
    assert answer == "final answer"
    assert state.stopped == StoppedByUntil("is_answered")


def test_stopped_is_limit_when_turn_limit_reached():
    # "limit reached: stop without an exception and return state.answer (None if no answer yet)"
    from alpineagents import loop as loop_decorator

    @tool
    def noop(x: int) -> str:
        """Do nothing."""
        return "ok"

    @loop_decorator(until=is_answered, limit=2)
    def never_answers(agent, state):
        agent.think(state)
        if state.pending_calls:
            agent.use_tools(state)

    replies = [tool_call("noop", x=1), tool_call("noop", x=2)]
    agent = Agent(model=FakeModel(replies), tools=[noop], loop=never_answers, reporter=None, human=None)
    state = task()
    answer = agent.run(state)

    assert answer is None
    assert state.stopped == StoppedByLimit(2)
    assert state.turn == 2


# ---------------------------------------------------------------------------
# history content per kind
# ---------------------------------------------------------------------------


def test_add_message_appends_user_kind_entry():
    state = task()
    state.add_message(Message.user("Move on to the next step"))
    assert state.history[-1].kind == "user"
    assert state.history[-1].content == "Move on to the next step"


def test_history_model_request_then_model_reply_per_think():
    # think records the request before calling the model and the reply after it.
    agent = make_agent(["text answer"])
    state = task()
    agent.think(state)

    request, reply = state.history[-2:]
    assert isinstance(request, ModelRequestEntry)
    assert request.kind == "model_request"
    assert request.content == "fake/fake"
    assert request.turn == 1
    assert isinstance(reply, ModelReplyEntry)
    assert reply.kind == "model_reply"
    assert reply.content.text == "text answer"


def test_history_tool_result_kind_after_use_tools():
    call = tool_call("read_file", path="main.py")

    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return "file contents"

    agent = make_agent([call], tools=[read_file])
    state = task()
    agent.think(state)
    agent.use_tools(state)

    entry = state.history[-1]
    assert isinstance(entry, ToolResultEntry)
    assert entry.kind == "tool_result"
    assert entry.content == "file contents"
    assert entry.call.id == call.id
    assert entry.outcome == ToolOutcomeKind.DONE
    assert entry.is_error is False


def test_add_notice_message_prefixes_and_records_notice_kind():
    state = task()
    state.add_message(Message.notice("The tests failed"))
    entry = state.history[-1]
    assert entry.kind == "notice"
    assert entry.content == "[notice] The tests failed"


def test_notice_does_not_double_prefix_already_prefixed_text():
    state = task()
    state.add_message(Message.notice("[notice] already prefixed"))
    assert state.history[-1].content == "[notice] already prefixed"
    # A user message that happens to start with the prefix is a notice too (the kind follows the text).
    state.add_message(Message.user("[notice] typed by hand"))
    assert state.history[-1].kind == "notice"


def test_history_denied_outcome_on_deny():
    # NotRunEntry is gone: a denied call is a ToolResultEntry with outcome DENIED.
    call = tool_call("bash", command="rm -rf /")
    agent = make_agent([call])
    state = task()
    agent.think(state)

    state._tool_result(state.pending_calls[0], "This is a dangerous command", ToolOutcomeKind.DENIED)

    entry = state.history[-1]
    assert entry.kind == "tool_result"
    assert entry.outcome == ToolOutcomeKind.DENIED
    assert entry.is_error is True
    assert entry.content == "This is a dangerous command"
    assert entry.call.id == call.id


def test_history_ask_kind_holds_question_and_answer_exchange():
    agent = make_agent(["a tidy summary"])
    state = task()
    value = agent.ask(state, "Summarize what you have so far")

    entry = state.history[-1]
    assert entry.kind == "ask"
    assert entry.content.question == "Summarize what you have so far"
    assert entry.content.answer == value == "a tidy summary"
    assert entry.usage is not None and entry.usage.requests == 1


def test_history_human_kind_holds_question_and_answer_exchange():
    agent = Agent(model=FakeModel([]), human=FakeHuman(["yes"]), reporter=None)
    state = task()
    value = agent.ask_human(state, "Proceed?")

    entry = state.history[-1]
    assert entry.kind == "human"
    assert entry.content.question == "Proceed?"
    assert entry.content.answer == value == "yes"
    assert entry.usage is None


def test_history_context_change_kind_on_compact():
    state = task()
    state.compact("summary so far")
    entry = state.history[-1]
    assert entry.kind == "context_change"
    assert entry.content.kind == "compact"
    assert entry.content.summary == "summary so far"
    assert entry.content.kept == 0


def test_history_error_kind_recorded_on_model_exception():
    agent = make_agent([TimeoutError("timed out")])
    state = task()
    with pytest.raises(TimeoutError):
        agent.think(state)

    entry = state.history[-1]
    assert isinstance(entry, ErrorEntry)
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
    state = task()
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
    state = task()
    agent.think(state)

    assert raw in state.messages[-1].content


def test_any_agent_can_think_on_the_same_state():
    # 0.4's owner rule is gone: any Agent (a copy with another model, say) may think on a State.
    agent1 = make_agent(["A's thought"])
    agent2 = make_agent(["B's thought"])
    state = task()
    agent1.think(state)
    agent2.think(state)

    assert [m.text for m in state.messages if m.role == "assistant"] == ["A's thought", "B's thought"]
    models = [e for e in state.history if e.kind == "model_request"]
    assert len(models) == 2


def test_other_agent_can_read_via_ask_without_changing_state():
    agent1 = make_agent(["A's thought"])
    agent2 = make_agent(["B's review"])
    state = task()
    agent1.think(state)

    before = state.messages
    review = agent2.ask(state, "Review the changes so far")
    assert review == "B's review"
    assert state.messages == before


def test_extra_data_is_not_visible_in_messages():
    state = task()
    with state.edit_extra_data() as data:
        data["secret"] = "a value the model must not see"
    assert state.extra_data["secret"] == "a value the model must not see"
    assert not any("a value the model must not see" in m.text for m in state.messages)

    agent = make_agent(["ok"])
    agent.think(state)
    assert "a value the model must not see" not in str(agent.model.requests[-1].messages)


def test_extra_data_is_read_only():
    state = task(extra_data={"notes": ["first"]})
    with pytest.raises(TypeError, match="edit_extra_data"):
        state.extra_data["x"] = 1
    with pytest.raises(TypeError):
        state.extra_data["notes"].append("second")
    assert state.extra_data == {"notes": ["first"]}


# ---------------------------------------------------------------------------
# Turns and pending calls
# ---------------------------------------------------------------------------


def _agent_with_pending_call(call_name="bash", **args):
    call = tool_call(call_name, **args)

    @tool
    def bash(command: str) -> str:
        """Run a command."""
        return "ran"

    agent = make_agent([call, "finished answer"], tools=[bash])
    state = task()
    agent.think(state)
    assert state.pending_calls
    return agent, state


def test_think_raises_valueerror_when_pending_calls_remain():
    agent, state = _agent_with_pending_call(command="ls")
    with pytest.raises(ValueError):
        agent.think(state)


def test_compact_raises_valueerror_when_pending_calls_remain():
    agent, state = _agent_with_pending_call(command="ls")
    with pytest.raises(ValueError):
        agent.compact(state)


def test_state_compact_raises_valueerror_when_pending_calls_remain():
    _, state = _agent_with_pending_call(command="ls")
    with pytest.raises(ValueError):
        state.compact("summary")


def test_clear_tool_results_raises_valueerror_when_pending_calls_remain():
    _, state = _agent_with_pending_call(command="ls")
    with pytest.raises(ValueError):
        state.clear_tool_results()


def test_add_notice_defers_instead_of_raising_when_pending():
    # "add_message defers instead of raising."
    agent, state = _agent_with_pending_call(command="ls")
    state.add_message(Message.notice("hold on"))
    assert not any("hold on" in m.text for m in state.messages)  # not in the messages yet

    agent.use_tools(state)
    assert "hold on" in state.messages[-1].text  # goes in after the last call is closed


def test_add_user_message_defers_instead_of_raising_when_pending():
    agent, state = _agent_with_pending_call(command="ls")
    state.add_message(Message.user("earlier message"))
    assert not any("earlier message" in m.text for m in state.messages)

    agent.use_tools(state)
    assert "earlier message" in state.messages[-1].text


def test_deny_ask_and_ask_human_allowed_while_pending_calls_remain():
    _, state = _agent_with_pending_call(command="ls")

    # ask works even with pending calls (it does not change the messages)
    agent2 = make_agent(["separate review"])
    before = state.messages
    review = agent2.ask(state, "Review it")
    assert review == "separate review"
    assert state.messages == before

    human_agent = Agent(model=FakeModel([]), human=FakeHuman(["yes"]), reporter=None)
    assert human_agent.ask_human(state, "Proceed?") == "yes"

    state._tool_result(state.pending_calls[0], "the user denied it", ToolOutcomeKind.DENIED)  # no error
    assert state.pending_calls == ()


def test_finish_allowed_while_pending_calls_remain():
    # here: finish, as called by a submit tool
    _, state = _agent_with_pending_call(command="ls")
    state.finish("finished inside a tool")  # no error
    assert state.finished
    assert state.answer == "finished inside a tool"


def test_result_for_unknown_call_raises_valueerror():
    state = task()
    unknown = tool_call("nonexistent")
    with pytest.raises(ValueError):
        state._tool_result(unknown, "reason", ToolOutcomeKind.DENIED)
    assert len(state.history) == 1  # nothing was recorded


def test_ask_uses_not_run_yet_placeholder_for_unresolved_call():
    # "placeholder result for ask | (not run yet)" - ask puts this text in place of pending calls.
    agent, state = _agent_with_pending_call(command="ls")
    fake = agent.model
    agent.ask(state, "Summarize the current situation")

    sent = fake.requests[-1].messages
    placeholder_message = sent[-2]  # right before the question message
    assert placeholder_message.content[0].content == "(not run yet)"


def test_tool_results_enter_messages_in_request_order_not_completion_order():
    # Results collect in this turn's buffer and, the moment the last call closes, enter the messages as one
    # user message in request order. Even if fast finishes first, request order (slow, fast) is kept.

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
    state = task()
    agent.think(state)
    agent.use_tools(state)

    result_message = state.messages[-1]
    contents = [block.content for block in result_message.content]
    assert contents == ["slow", "fast"]


def test_deferred_messages_enter_messages_in_arrival_order_after_tool_results():
    agent, state = _agent_with_pending_call(command="ls")
    state.add_message(Message.user("earlier message"))
    state.add_message(Message.notice("later notice"))
    agent.use_tools(state)

    tail = state.messages[-3:]
    assert tail[0].content[0].content == "ran"  # the tool result message
    assert tail[1].text == "earlier message"
    assert "later notice" in tail[2].text


# ---------------------------------------------------------------------------
# Rules after exceptions and interrupts
# ---------------------------------------------------------------------------


def test_think_exception_takes_the_turn_back():
    # "think (during the model request): back to just before that think." History keeps the request and the error.
    agent = make_agent([TimeoutError("timed out")])
    state = task()
    with pytest.raises(TimeoutError):
        agent.think(state)

    assert state.turn == 0
    assert [m.text for m in state.messages] == ["Task"]
    assert [e.kind for e in state.history[-2:]] == ["model_request", "error"]


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
    state = task()
    agent.think(state)

    with pytest.raises(TimeoutError):
        agent.use_tools(state)

    assert {c.name for c in state.pending_calls} == {"boom"}
    tool_result_contents = [e.content for e in state.history if e.kind == "tool_result"]
    assert "success" in tool_result_contents


def test_run_closes_remaining_pending_call_with_aborted_marker_on_exception():
    # Result text table: "closed by an exception | (aborted: TimeoutError)"
    @tool
    def boom(x: int) -> str:
        """A tool that raises."""
        raise TimeoutError("failed")

    agent = make_agent([tool_call("boom", x=1)], tools=[boom])
    state = task()
    with pytest.raises(TimeoutError):
        agent.run(state)

    assert state.pending_calls == ()
    last_result = [e for e in state.history if e.kind == "tool_result"][-1]
    assert last_result.content == "(aborted: TimeoutError)"
    assert last_result.outcome == ToolOutcomeKind.ABORTED
    assert isinstance(last_result.error, TimeoutError)


def test_run_closes_remaining_pending_call_with_user_interrupted_marker_on_keyboardinterrupt():
    # Result text table: "closed by an interrupt | (interrupted by user)"
    @tool
    def boom(x: int) -> str:
        """A tool that simulates an interrupt."""
        raise KeyboardInterrupt()

    agent = make_agent([tool_call("boom", x=1)], tools=[boom])
    state = task()
    with pytest.raises(KeyboardInterrupt):
        agent.run(state)

    last_result = [e for e in state.history if e.kind == "tool_result"][-1]
    assert last_result.content == "(interrupted by user)"
    assert last_result.outcome == ToolOutcomeKind.INTERRUPTED


def test_think_after_finish_raises_valueerror():
    agent = make_agent(["text"])
    state = task()
    state.finish()
    with pytest.raises(ValueError):
        agent.think(state)


def test_use_tools_and_ask_after_finish_also_raise_valueerror():
    agent = make_agent(["text"])
    state = task()
    state.finish()
    with pytest.raises(ValueError):
        agent.use_tools(state)
    with pytest.raises(ValueError):
        agent.ask(state, "question")


def test_think_on_a_state_without_messages_raises_valueerror():
    agent = make_agent(["text"])
    state = State()
    with pytest.raises(ValueError, match="no messages"):
        agent.think(state)
    assert state.history == ()  # nothing was recorded
    state.add_message(Message.user("Now there is one"))
    agent.think(state)
    assert state.answer == "text"


# ---------------------------------------------------------------------------
# Context management: the three stages of compaction
# ---------------------------------------------------------------------------


def test_compact_stage1_plain_call_replaces_messages_with_summary():
    # "Stage 1: just call it -- agent.compact(state)"
    agent = make_agent(["summary of the work so far"])
    state = task()
    agent.compact(state)

    assert [m.text for m in state.messages] == [
        "Task",
        "[notice] Summary so far:\nsummary of the work so far",
    ]
    change = state.history[-1].content
    assert change.kind == "compact" and change.kept == 0
    assert change.usage is not None and change.usage.requests == 1
    assert state.usage.requests == 1  # the compaction request is counted


def test_compact_stage2_instructions_are_forwarded_to_the_summary_request():
    # "Stage 2: say what to keep -- agent.compact(state, instructions=...)"
    agent = make_agent(["summary"])
    state = task()
    agent.compact(state, instructions="Keep architecture decisions, remaining bugs and changed files")

    sent_text = agent.model.requests[-1].messages[-1].text
    assert "Be sure to keep: Keep architecture decisions, remaining bugs and changed files" in sent_text


def test_compact_stage3_ask_then_compact_leaves_messages_untouched_until_compact():
    # "Stage 3: make the summary yourself (ask does not change the messages or call tools)"
    agent = make_agent(["a summary of the work done so far"])
    state = task()
    before = state.messages

    summary = agent.ask(state, "Summarize the work done so far")
    assert state.messages == before  # ask does not change the messages

    state.compact(summary)
    assert [m.text for m in state.messages] == [
        "Task",
        "[notice] Summary so far:\na summary of the work done so far",
    ]


def test_compact_reduces_messages_to_first_message_and_summary_only():
    # "compact(summary) replaces the messages with two: the first user message + the summary. Old turns are not
    # sent to the model again."
    call = tool_call("read_file", path="a.py")

    @tool
    def read_file(path: str) -> str:
        """Read."""
        return "long file contents"

    agent = make_agent([call, "intermediate answer"], tools=[read_file])
    state = task()
    agent.think(state)
    agent.use_tools(state)
    agent.think(state)
    assert len(state.messages) > 2  # several turns have piled up

    state.compact("the summary")
    assert len(state.messages) == 2
    assert state.messages[0].text == "Task"
    assert state.messages[1].text == "[notice] Summary so far:\nthe summary"


def test_compact_without_a_first_user_message_keeps_only_the_summary():
    state = State()
    state.compact("only the summary")
    assert [m.text for m in state.messages] == ["[notice] Summary so far:\nonly the summary"]


def test_context_change_notifies_reporter_on_compact():
    # "The Reporter shows this with on_context_change."
    spy = SpyReporter()
    agent = Agent(model=FakeModel(["first answer"]), reporter=spy, human=None)
    state = task()
    agent.think(state)  # links this Agent to the State, so spy is the Reporter to notify

    state.compact("summary")

    assert len(spy.context_changes) == 1
    assert spy.context_changes[0].kind == "compact"


def test_import_does_not_notify_the_reporter():
    spy = SpyReporter()
    agent = Agent(model=FakeModel(["first answer"]), reporter=spy, human=None)
    state = task()
    agent.think(state)
    assert spy.context_changes == []


def test_deny_notifies_reporter_on_tool_end_with_reason_as_result():
    # "on_tool_end (denied)": the runner notifies after the denied result was recorded (outside the lock)
    @tool
    def bash(command: str) -> str:
        """Run a command"""
        return "ran"

    spy = SpyReporter()
    call = tool_call("bash", command="rm -rf /")
    agent = Agent(
        model=FakeModel([call]),
        tools=[bash],
        reporter=spy,
        human=None,
        permissions=[DenyByName(["bash"], "dangerous command")],
    )
    state = task()
    agent.think(state)

    agent.use_tools(state)

    assert len(spy.tool_ends) == 1
    denied_call, result, outcome = spy.tool_ends[0]
    assert denied_call.id == call.id
    assert result == "dangerous command"
    assert outcome == ToolOutcome("denied", decided_by="DenyByName(['bash'], reason='dangerous command')")
    entry = state.history[-1]
    assert entry.kind == "tool_result" and entry.outcome == ToolOutcomeKind.DENIED


# ---------------------------------------------------------------------------
# Context management: clearing tool results
# ---------------------------------------------------------------------------


def _echo_state(count):
    @tool
    def echo(x: int) -> str:
        """Return it unchanged."""
        return f"result{x}"

    calls = [tool_call("echo", x=i) for i in range(1, count + 1)]
    agent = make_agent(calls, tools=[echo])
    state = task()
    for _ in range(count):
        agent.think(state)
        agent.use_tools(state)
    return state, calls


def test_clear_tool_results_clears_older_results_and_keeps_last_n():
    # "Old tool results are cleared only from the messages with state.clear_tool_results(keep_last=5).
    # (cleared: kept in history) is left in their place."
    state, _ = _echo_state(3)

    state.clear_tool_results(keep_last=1)

    results = [
        block.content
        for message in state.messages
        for block in message.content
        if hasattr(block, "call_id")
    ]
    assert results == ["(cleared: kept in history)", "(cleared: kept in history)", "result3"]


def test_clear_tool_results_records_the_cleared_call_ids():
    state, calls = _echo_state(3)

    state.clear_tool_results(keep_last=1)

    change = state.history[-1].content
    assert change.kind == "clear_tool_results"
    assert change.cleared == (calls[0].id, calls[1].id)


def test_clear_tool_results_records_nothing_when_nothing_changes():
    state, _ = _echo_state(2)
    length = len(state.history)

    state.clear_tool_results(keep_last=5)  # fewer results than keep_last
    assert len(state.history) == length

    state.clear_tool_results(keep_last=0)
    assert len(state.history) == length + 1
    state.clear_tool_results(keep_last=0)  # already cleared
    assert len(state.history) == length + 1


def test_clear_tool_results_does_not_change_history():
    # "It differs from compaction in that it clears instead of summarizing" -- history keeps the original.
    state, _ = _echo_state(2)

    state.clear_tool_results(keep_last=0)

    tool_result_contents = [e.content for e in state.history if e.kind == "tool_result"]
    assert tool_result_contents == ["result1", "result2"]


# ---------------------------------------------------------------------------
# Concurrency (SPEC 2.1 edit_extra_data, 4 thread safety)
# ---------------------------------------------------------------------------


def test_commands_can_be_called_inside_an_extra_data_edit():
    # The edit lock is not the State lock: other commands work inside the block (a tool may do it).
    state = task()
    with state.edit_extra_data() as data:
        data["step"] = 1
        state.finish("done")  # a State command inside the block takes the State lock, no deadlock
        state.add_message(Message.notice("inside"))
    assert state.finished
    assert state.extra_data["step"] == 1


def test_edit_extra_data_protects_multi_step_mutation_across_threads():
    # "Multi-step changes go inside with state.edit_extra_data()." No value may be lost even when several threads
    # read->compute->write at the same time.
    state = task()

    def worker():
        for _ in range(200):
            with state.edit_extra_data() as data:
                data["count"] = data.get("count", 0) + 1

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert state.extra_data["count"] == 1600
    assert State(history=state.history).snapshot() == state.snapshot()


def test_str_summarizes_history_one_line_per_entry():
    @tool
    def read_file(path: str) -> str:
        """Read a file"""
        return "x" * 1300

    fake = FakeModel([tool_call("read_file", path="main.py"), "The bug is on line 3"])
    state = State()
    state.add_message(Message.user("Find the bug"))
    Agent(model=fake, tools=[read_file], reporter=None, human=None).run(state)

    text = str(state)
    lines = text.splitlines()
    assert len(lines) == len(state.history) + 1
    assert "Find the bug" in lines[0]
    assert "read_file" in text
    assert "The bug is on line 3" in text
    assert lines[-1] == "done: stopped by is_answered (2 turns)"


def test_repr_shows_the_snapshot_values():
    state = task(id="bug-1")
    assert repr(state) == "<State id='bug-1' turn=0 history=1 messages=1 pending=0>"
    state.finish()
    assert repr(state).endswith("pending=0 finished>")


def test_state_equality_is_identity():
    a, b = task(id="same"), task(id="same")
    assert a == a
    assert a != b
    assert State(history=a.history).snapshot() == a.snapshot()  # the value follows the history, not the identity


def test_every_history_kind_has_exactly_one_entry_class():
    kinds_by_class = {cls: typing.get_args(typing.get_type_hints(cls)["kind"]) for cls in typing.get_args(types.HistoryEntry)}
    all_kinds = [kind for kinds in kinds_by_class.values() for kind in kinds]
    assert sorted(all_kinds) == sorted(typing.get_args(types.HistoryKind))
    assert len(kinds_by_class) == 11 and len(all_kinds) == 13
