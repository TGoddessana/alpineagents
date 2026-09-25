"""Black-box tests for ``@loop``/``Loop``/``default_loop``.

Turns the rules of ARCHITECTURE.md "Loop contract" (stopping, blocks such as ``compact_if_full``/``CompactIfFull``)
and "Mistake-proofing errors" (the ``@loop`` items) into checks, one to one.
The tests come from the spec sentences, not from the implementation. No network (only ``FakeModel``).

The ``Loop`` contract is duck typing (``Callable[[Agent, State], Any]``), so the pure loop mechanics
(``until``/``limit`` checks, stop order, ``copy``) are checked with a tiny fake State/Agent.
Rules that need a real State and Model, like ``is_answered()``/``add_user_message``/``context_used``,
are checked with a real ``State`` + ``Agent`` + ``FakeModel`` (``reporter=None``).
"""

from __future__ import annotations

import inspect

import pytest

from alpineagents import (
    Agent,
    CompactIfFull,
    ContextTooLongError,
    Loop,
    State,
    compact_if_full,
    default_loop,
    loop,
    tool,
)
from alpineagents.testing import FakeModel, tool_call


class FakeState:
    """Minimal fake State that only checks loop execution (for pure loop mechanics that need no real State)."""

    def __init__(self):
        self._finished = False
        self._answer = None
        self.stopped_by = None
        self.stopped_limit = None

    def is_finished(self):
        return self._finished

    def is_answered(self):
        return self._answer is not None

    @property
    def answer(self):
        return self._answer

    def finish(self, answer=None):
        self._finished = True
        if answer is not None:
            self._answer = answer

    def _set_stopped_by(self, reason, *, limit=None):
        self.stopped_by = reason
        self.stopped_limit = limit


class FakeAgent:
    """Placeholder that is passed to the loop but never used."""


# ============================================================ @loop rules 1-4, until/limit shape


def test_loop_without_parens_raises_typeerror():
    # "The header holds only until and limit, the loop's own properties, and both are required"
    with pytest.raises(TypeError):

        @loop
        def body(agent, state):
            pass


def test_loop_missing_until_raises_typeerror():
    # "@loop without until or limit" -> TypeError, both are required
    with pytest.raises(TypeError):
        Loop(lambda a, s: None, until=None, limit=10)


def test_loop_missing_limit_raises_typeerror():
    with pytest.raises(TypeError):
        Loop(lambda a, s: None, until=FakeState.is_answered, limit=None)


def test_until_called_result_raises_typeerror():
    # "Passing a call result like until=state.is_answered() -> pass State.is_answered without parentheses"
    state = FakeState()
    with pytest.raises(TypeError):
        Loop(lambda a, s: None, until=state.is_finished(), limit=10)


def test_until_bound_to_state_instance_raises_typeerror():
    # "until is a method bound to a State instance (until=state.is_answered) -> TypeError"
    real_state = State("Task")
    with pytest.raises(TypeError):
        Loop(lambda a, s: None, until=real_state.is_answered, limit=10)


def test_until_lambda_warns():
    # "A lambda gives a warning" (its name does not show in stopped_by)
    with pytest.warns(UserWarning):
        Loop(lambda a, s: None, until=lambda s: True, limit=5)


def test_until_reserved_names_rejected():
    # "'finish' and 'limit' are reserved names"
    def finish(s):
        return True

    def limit(s):
        return True

    with pytest.raises(ValueError):
        Loop(lambda a, s: None, until=finish, limit=5)
    with pytest.raises(ValueError):
        Loop(lambda a, s: None, until=limit, limit=5)


def test_until_uncallable_item_rejected():
    # "until takes one function or a list of functions" - the list items must be callable
    with pytest.raises(TypeError):
        Loop(lambda a, s: None, until=[FakeState.is_answered, "not callable"], limit=5)


def test_until_accepts_tuple_as_well_as_list():
    # "until takes one State -> bool function or a list of functions" - a tuple works in the same place
    def cond_a(s):
        return False

    def cond_b(s):
        return True

    body_loop = Loop(lambda a, s: None, until=(cond_a, cond_b), limit=10)
    state = FakeState()
    body_loop(FakeAgent(), state)

    assert state.stopped_by == "cond_b"


@pytest.mark.parametrize("bad_limit", [True, 0, -1, 1.5, "50"])
def test_limit_validation(bad_limit):
    def cond(s):
        return False

    with pytest.raises((TypeError, ValueError)):
        Loop(lambda a, s: None, until=cond, limit=bad_limit)


def test_body_returning_value_raises_typeerror():
    # "A body that returns a value other than None is an error, because the value would be ignored"
    def bad_body(agent, state):
        return "oops"

    body_loop = Loop(bad_body, until=FakeState.is_answered, limit=5)
    with pytest.raises(TypeError):
        body_loop(FakeAgent(), FakeState())


def test_stops_when_until_condition_true():
    # Rule 1: "Before a turn starts, stop if any until condition is true" / Rule 4: "return state.answer"
    calls = {"n": 0}

    def body(agent, state):
        calls["n"] += 1
        if calls["n"] == 2:
            state._answer = "answer text"

    body_loop = Loop(body, until=FakeState.is_answered, limit=10)
    state = FakeState()
    result = body_loop(FakeAgent(), state)

    assert result == "answer text"
    assert state.stopped_by == "is_answered"
    assert calls["n"] == 2


def test_stops_on_finish():
    # "state.finish() is called -> stop when that turn ends and return state.answer" stopped_by="finish"
    def body(agent, state):
        state.finish("done!")

    body_loop = Loop(body, until=FakeState.is_answered, limit=10)
    state = FakeState()
    result = body_loop(FakeAgent(), state)

    assert result == "done!"
    assert state.stopped_by == "finish"


def test_zero_turn_stop_when_already_finished():
    # Rule 1: if it is already finished "before" the turn starts, the body never runs
    calls = {"n": 0}

    def body(agent, state):
        calls["n"] += 1

    already_finished = FakeState()
    already_finished.finish("pre")

    body_loop = Loop(body, until=FakeState.is_answered, limit=10)
    result = body_loop(FakeAgent(), already_finished)

    assert calls["n"] == 0
    assert result == "pre"
    assert already_finished.stopped_by == "finish"


def test_stops_on_limit_with_none_answer():
    # "After limit iterations... stop quietly with stopped_by='limit'" / "None if there is no answer yet"
    calls = {"n": 0}

    def body(agent, state):
        calls["n"] += 1

    body_loop = Loop(body, until=FakeState.is_answered, limit=3)
    state = FakeState()
    result = body_loop(FakeAgent(), state)

    assert result is None
    assert state.stopped_by == "limit"
    assert state.stopped_limit == 3
    assert calls["n"] == 3


def test_limit_checks_condition_once_more_before_stopping():
    # Rule 3: "after limit iterations, check the conditions once more, and if still false" stop -
    # if a condition becomes true right after the limit-th body, it must stop with the condition name, not limit
    calls = {"n": 0}

    def done(state):
        return calls["n"] >= 3

    def body(agent, state):
        calls["n"] += 1

    body_loop = Loop(body, until=done, limit=3)
    state = FakeState()
    body_loop(FakeAgent(), state)

    assert calls["n"] == 3
    assert state.stopped_by == "done"


def test_multiple_until_conditions_checked_in_order():
    # "until takes... a list of functions" - conditions are checked in order, and the true one's name is stopped_by
    def cond_a(s):
        return False

    def cond_b(s):
        return True

    body_loop = Loop(lambda a, s: None, until=[cond_a, cond_b], limit=10)
    state = FakeState()
    body_loop(FakeAgent(), state)

    assert state.stopped_by == "cond_b"


def test_copy_overrides_only_given_fields():
    # "A loop made by @loop has copy(until=, limit=)... the original is unchanged"
    def body(agent, state):
        pass

    original = Loop(body, until=FakeState.is_answered, limit=5)
    copied = original.copy(limit=99)

    assert copied.limit == 99
    assert copied.until == original.until
    assert copied.body is original.body
    assert original.limit == 5


def test_turn_count_resets_per_call():
    # "limit is counted afresh on every loop call"
    calls = {"n": 0}

    def body(agent, state):
        calls["n"] += 1

    body_loop = Loop(body, until=FakeState.is_answered, limit=2)
    body_loop(FakeAgent(), FakeState())
    assert calls["n"] == 2
    body_loop(FakeAgent(), FakeState())
    assert calls["n"] == 4


def test_finish_ends_whole_run_so_later_loop_does_not_run():
    # "finish() ends the whole run, so later loops do not run either"
    def never(state):
        return False

    def finishes(agent, state):
        state.finish("early")

    calls = {"n": 0}

    def counts(agent, state):
        calls["n"] += 1

    loop_a = Loop(finishes, until=never, limit=5)
    loop_b = Loop(counts, until=never, limit=5)

    state = FakeState()
    result_a = loop_a(FakeAgent(), state)
    result_b = loop_b(FakeAgent(), state)  # called next with the same State

    assert result_a == "early"
    assert result_b == "early"  # the answer set by finish, unchanged
    assert calls["n"] == 0  # loop_b's body never ran
    assert state.stopped_by == "finish"


def test_loop_can_be_used_as_block_inside_another_loop():
    # "A loop can also be a block of another loop" - a Loop is a callable taking (agent, state),
    # so there is one contract
    counts = {"inner": 0}

    def inner_done(state):
        return counts["inner"] >= 2

    def inner_body(agent, state):
        counts["inner"] += 1

    inner = Loop(inner_body, until=inner_done, limit=10)

    def outer_body(agent, state):
        inner(agent, state)  # use the loop like a block
        state.finish("outer done")

    outer = Loop(outer_body, until=FakeState.is_answered, limit=5)
    state = FakeState()
    result = outer(FakeAgent(), state)

    assert counts["inner"] == 2
    assert result == "outer done"


def test_block_can_be_callable_object_not_just_function():
    # "A block is a function or __call__ object taking (agent, state), the same shape as a loop"
    class FinishingBlock:
        def __call__(self, agent, state):
            state.finish("via block object")

    body_loop = Loop(FinishingBlock(), until=FakeState.is_answered, limit=5)
    state = FakeState()
    result = body_loop(FakeAgent(), state)

    assert result == "via block object"
    assert state.stopped_by == "finish"


# ============================================================ why conditions are checked before a turn (real State)


def test_add_user_message_makes_is_answered_false_so_loop_runs_again():
    # "Adding a new instruction (add_user_message) makes is_answered() false so the loop runs,
    #  and with nothing to do it stops at turn 0"
    fake = FakeModel(["first answer", "second answer"])
    agent = Agent(model=fake, reporter=None)
    state = State("Task")

    def body(a, s):
        a.think(s)

    think_loop = Loop(body, until=State.is_answered, limit=5)

    answer1 = think_loop(agent, state)
    assert answer1 == "first answer"
    assert state.turn == 1

    # Nothing to do: already is_answered(), so it must stop at turn 0 (uses no extra FakeModel reply)
    answer_again = think_loop(agent, state)
    assert answer_again == "first answer"
    assert state.turn == 1  # the body did not run
    assert fake.remaining == 1  # the second reply is still left

    # Adding a new instruction makes is_answered() false, so the loop actually runs
    state.add_user_message("Keep going")
    answer2 = think_loop(agent, state)
    assert answer2 == "second answer"
    assert state.turn == 2
    assert fake.remaining == 0


def test_until_accepts_builtin_and_custom_condition_together():
    # "Built-in questions like State.is_answered and user-made functions go in the same place"
    def never_over_budget(state):
        return False

    combo_loop = Loop(lambda a, s: a.think(s), until=[never_over_budget, State.is_answered], limit=5)
    fake = FakeModel(["final answer"])
    agent = Agent(model=fake, reporter=None)
    state = State("Task")

    answer = combo_loop(agent, state)

    assert answer == "final answer"
    assert state.stopped_by == "is_answered"


# ============================================================ body rules, @loop mistake-proofing errors


@pytest.mark.parametrize(
    "call_after_finish",
    [
        lambda agent, state: agent.think(state),
        lambda agent, state: agent.use_tools(state),
        lambda agent, state: agent.ask(state, "question"),
    ],
    ids=["think", "use_tools", "ask"],
)
def test_finish_then_think_or_use_tools_or_ask_raises_value_error(call_after_finish):
    # "Calling think or use_tools after finish() is an error" / "Fix: if a block called finish(),
    #  return from the body" - checks the error raised when a body breaks this rule
    agent = Agent(model=FakeModel([]), reporter=None)
    state = State("Task")
    state.finish("already done")

    with pytest.raises(ValueError):
        call_after_finish(agent, state)


# ============================================================ default loop


def test_default_loop_shape():
    assert default_loop.limit == 50
    assert [f.__name__ for f in default_loop.until] == ["is_answered"]
    assert default_loop.__name__ == "default_loop"
    assert default_loop.__wrapped__.__name__ == "default_loop"


def test_default_loop_body_uses_compact_if_full_block():
    # "The default loop contains this block (compact_if_full)"
    source = inspect.getsource(default_loop.body)
    assert "compact_if_full" in source


def test_agent_without_loop_uses_default_loop():
    # "If the loop is omitted... alpineagents.default_loop runs"
    agent = Agent(model=FakeModel(["answer"]), reporter=None)
    assert agent.loop is default_loop


def test_default_loop_runs_think_then_use_tools_until_answered():
    # Default loop: "compact_if_full(agent, state); agent.think(state); if state.wants_tools(): agent.use_tools(state)"
    # until=State.is_answered, limit=50
    @tool
    def echo(text: str) -> str:
        """Echo tool for tests."""
        return text

    fake = FakeModel([tool_call("echo", text="hi"), "final reply"])
    agent = Agent(model=fake, tools=[echo], reporter=None)
    state = State("Task")

    answer = agent.run(state)

    assert answer == "final reply"
    assert state.stopped_by == "is_answered"
    assert state.turn == 2  # turn 1: tool call, turn 2: final reply


def test_default_loop_copy_creates_independent_loop():
    # "To change only loop settings... make a new loop, e.g. Agent(loop=default_loop.copy(limit=200)); the original
    #  is unchanged"
    bigger = default_loop.copy(limit=200)

    assert bigger.limit == 200
    assert default_loop.limit == 50
    assert bigger.until == default_loop.until
    assert bigger.body is default_loop.body


# ============================================================ stop table


def test_stop_table_until_condition_sets_stopped_by_to_condition_name():
    # "until condition true before a turn | stop and return state.answer | condition function name (e.g. 'over_budget')"
    def over_budget(state):
        return state.turn >= 2

    fake = FakeModel(["first thought", "second thought"])

    @loop(until=over_budget, limit=10)
    def budget_loop(agent, state):
        agent.think(state)

    agent = Agent(model=fake, loop=budget_loop, reporter=None)
    state = State("Task")

    answer = agent.run(state)

    assert answer == state.answer == "second thought"
    assert state.stopped_by == "over_budget"
    assert state.turn == 2


def test_stop_table_finish_sets_stopped_by_finish():
    # "state.finish() called | stop when that turn ends and return state.answer... the whole run ends | 'finish'"
    @loop(until=State.is_answered, limit=5)
    def finishing_loop(agent, state):
        state.finish("finished answer")

    agent = Agent(model=FakeModel([]), loop=finishing_loop, reporter=None)
    state = State("Task")

    answer = agent.run(state)

    assert answer == "finished answer"
    assert state.stopped_by == "finish"


def test_stop_table_limit_sets_stopped_by_limit_and_returns_last_answer():
    # "limit reached | stop without an exception and return state.answer (None if no answer yet) | 'limit'"
    def never(state):
        return False

    fake = FakeModel(["thought one", "thought two"])

    @loop(until=never, limit=2)
    def think_only(agent, state):
        agent.think(state)

    agent = Agent(model=fake, loop=think_only, reporter=None)
    state = State("Task")

    answer = agent.run(state)

    assert answer == state.answer == "thought two"  # stopping at limit does not lose the answer already there
    assert state.stopped_by == "limit"
    assert state.turn == 2


# ============================================================ blocks.compact_if_full / CompactIfFull


def test_compact_if_full_default_instance_shape():
    # "compact_if_full = CompactIfFull()" - a ready-to-use lowercase instance, default at=0.6
    assert isinstance(compact_if_full, CompactIfFull)
    assert compact_if_full.at == 0.6
    assert compact_if_full.instructions is None


def test_compact_if_full_does_not_trigger_below_threshold():
    # "if state.context_used > 0.6: agent.compact(state)" - no compaction below the threshold
    fake = FakeModel(["short answer", "must not be used"], context_window=1_000_000)
    agent = Agent(model=fake, reporter=None)
    state = State("Task")
    agent.think(state)

    assert state.context_used <= 0.6
    before_remaining = fake.remaining

    compact_if_full(agent, state)

    assert fake.remaining == before_remaining  # agent.compact was not called
    assert state.context[-1].role == "assistant"  # not replaced by compaction (a summary user message)


#: Size that pushes context_used above 0.6 while the compaction request (with COMPACT_PROMPT) still fits.
_LONG_REPLY = "A long reply to fill the context window. " * 15
_OVER_060_WINDOW = 330


def test_compact_if_full_triggers_above_threshold():
    # "if state.context_used > 0.6: agent.compact(state)" - compacts above the threshold
    fake = FakeModel([_LONG_REPLY, "This is the summary"], context_window=_OVER_060_WINDOW)
    agent = Agent(model=fake, reporter=None)
    state = State("Task")
    agent.think(state)

    assert state.context_used > 0.6
    before_remaining = fake.remaining

    compact_if_full(agent, state)

    assert fake.remaining == before_remaining - 1  # agent.compact called the model once more
    assert state.context[-1].role == "user"  # replaced by start_from(summary)
    assert "This is the summary" in state.context[-1].text
    assert len(state.context) == 2  # "replaced by two: the original task + the summary"


def test_compact_if_full_instructions_reach_model_request():
    # "what to keep | task content | user (instructions)" - CompactIfFull(instructions=...) reaches the request
    fake = FakeModel([_LONG_REPLY, "summarized"], context_window=_OVER_060_WINDOW)
    agent = Agent(model=fake, reporter=None)
    state = State("Task")
    agent.think(state)

    block = CompactIfFull(instructions="Keep only the key points")
    block(agent, state)

    last_request = fake.requests[-1]
    assert "Keep only the key points" in last_request.messages[-1].text


def test_compact_if_full_custom_at_changes_trigger_point():
    # "Blocks that need settings are classes" - at sets 'when' to compact as a number visible in code
    fake = FakeModel(["short answer", "summary1"], context_window=1_000_000)
    agent = Agent(model=fake, reporter=None)
    state = State("Task")
    agent.think(state)

    assert state.context_used < 0.6  # would not be compacted at the default threshold (0.6)

    lenient = CompactIfFull(at=state.context_used / 2)
    lenient(agent, state)

    assert state.context[-1].role == "user"  # compaction happened because of the lowered threshold
    assert "summary1" in state.context[-1].text


def test_missing_compact_block_lets_context_overflow_error_surface():
    # "If a user-written loop has no compaction block, alpineagents.ContextTooLongError is raised when the
    #  context overflows. It never compacts silently" (checked here with the overflow FakeModel simulates)
    fake = FakeModel(["any answer"], context_window=1)

    @loop(until=State.is_answered, limit=5)
    def no_compact_loop(agent, state):
        agent.think(state)  # no compact_if_full

    agent = Agent(model=fake, loop=no_compact_loop, reporter=None)
    state = State("Task")

    with pytest.raises(ContextTooLongError):
        agent.run(state)
