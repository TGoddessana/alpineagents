"""Black-box tests for ``@loop``/``Loop``/``default_loop``.

Turns the rules of ARCHITECTURE.md "Loop contract" (stopping, blocks such as ``compact_if_full``/``CompactIfFull``)
and "Mistake-proofing errors" (the ``@loop`` items) into checks, one to one.
The tests come from the spec sentences, not from the implementation. No network (only ``FakeModel``).

The ``Loop`` contract is duck typing (``Callable[[Agent, State], Any]``), so the pure loop mechanics
(``until``/``limit`` checks, stop order, ``copy``) are checked with a tiny fake State/Agent.
Rules that need a real State and Model, like a user-written ``until`` function, ``add_message`` and
``agent.context_used``, are checked with a real ``State`` + ``Agent`` + ``FakeModel`` (``reporter=None``).
"""

from __future__ import annotations

import inspect

import pytest

from alpineagents import (
    Agent,
    CompactIfFull,
    ContextTooLongError,
    Loop,
    Message,
    State,
    StopEntry,
    StoppedByFinish,
    StoppedByLimit,
    StoppedByUntil,
    compact_if_full,
    default_loop,
    loop,
    tool,
)
from alpineagents.testing import FakeModel, tool_call


class FakeState:
    """Minimal fake State that only checks loop execution (for pure loop mechanics that need no real State).

    It has what ``Loop`` reads: ``stopped``, ``finished``, ``answer`` and the private ``_record_stop``."""

    def __init__(self):
        self.finished = False
        self._answer = None
        self.stopped = None

    @property
    def answer(self):
        return self._answer

    def finish(self, answer=None):
        self.finished = True
        self.stopped = StoppedByFinish()
        if answer is not None:
            self._answer = answer

    def _record_stop(self, stopped):
        self.stopped = stopped


def is_answered(state):
    """The until function of the fake-State tests: true once the fake State has an answer."""
    return state.answer is not None


def waiting_for_user(state: State) -> bool:
    """The user-written until function of SPEC section 6, for the tests with a real State."""
    if state.pending_calls or not state.messages:
        return False
    last = state.messages[-1]
    return last.role == "assistant" and not last.tool_calls


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
        Loop(lambda a, s: None, until=is_answered, limit=None)


def test_until_called_result_raises_typeerror():
    # "Passing a call result like until=waiting_for_user(state) -> pass waiting_for_user without parentheses"
    state = FakeState()
    with pytest.raises(TypeError):
        Loop(lambda a, s: None, until=is_answered(state), limit=10)


def test_until_can_be_a_method_of_any_object():
    # State has no predicate methods any more (is_answered, is_finished are gone), so there is no "bound to a State"
    # check: a bound method of your own object is a normal until, named after the method.
    class Budget:
        def __init__(self):
            self.spent = 5

        def over(self, state):
            return self.spent > 3

    budget = Budget()
    body_loop = Loop(lambda a, s: None, until=budget.over, limit=10)
    state = FakeState()
    body_loop(FakeAgent(), state)
    assert state.stopped == StoppedByUntil("over")


def test_state_no_longer_has_the_predicate_methods():
    # "State.is_answered no longer exists": the until functions are the user's own
    state = State(messages=[Message.user("Task")])
    for name in ("is_answered", "is_finished", "wants_tools"):
        assert not hasattr(state, name)
    assert not hasattr(State, "is_answered")


def test_until_lambda_warns():
    # "A lambda gives a warning" (its name does not show in state.stopped)
    with pytest.warns(UserWarning):
        Loop(lambda a, s: None, until=lambda s: True, limit=5)


def test_until_names_finish_and_limit_are_allowed():
    # The stop values are types now, so an until function named like another stop reason cannot collide.
    def finish(s):
        return True

    def limit(s):
        return True

    state = FakeState()
    Loop(lambda a, s: None, until=finish, limit=5)(FakeAgent(), state)
    assert state.stopped == StoppedByUntil("finish")
    Loop(lambda a, s: None, until=limit, limit=5)(FakeAgent(), state)
    assert state.stopped == StoppedByUntil("limit")


def test_until_uncallable_item_rejected():
    # "until takes one function or a list of functions" - the list items must be callable
    with pytest.raises(TypeError):
        Loop(lambda a, s: None, until=[is_answered, "not callable"], limit=5)


def test_until_accepts_tuple_as_well_as_list():
    # "until takes one State -> bool function or a list of functions" - a tuple works in the same place
    def cond_a(s):
        return False

    def cond_b(s):
        return True

    body_loop = Loop(lambda a, s: None, until=(cond_a, cond_b), limit=10)
    state = FakeState()
    body_loop(FakeAgent(), state)

    assert state.stopped == StoppedByUntil("cond_b")


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

    body_loop = Loop(bad_body, until=is_answered, limit=5)
    with pytest.raises(TypeError):
        body_loop(FakeAgent(), FakeState())


def test_stops_when_until_condition_true():
    # Rule 1: "Before a turn starts, stop if any until condition is true" / Rule 4: "return state.answer"
    calls = {"n": 0}

    def body(agent, state):
        calls["n"] += 1
        if calls["n"] == 2:
            state._answer = "answer text"

    body_loop = Loop(body, until=is_answered, limit=10)
    state = FakeState()
    result = body_loop(FakeAgent(), state)

    assert result == "answer text"
    assert state.stopped == StoppedByUntil("is_answered")
    assert calls["n"] == 2


def test_stops_on_finish():
    # "state.finish() is called -> stop when that turn ends and return state.answer" stopped == StoppedByFinish()
    def body(agent, state):
        state.finish("done!")

    body_loop = Loop(body, until=is_answered, limit=10)
    state = FakeState()
    result = body_loop(FakeAgent(), state)

    assert result == "done!"
    assert state.stopped == StoppedByFinish()


def test_zero_turn_stop_when_already_finished():
    # Rule 1: if it is already finished "before" the turn starts, the body never runs
    calls = {"n": 0}

    def body(agent, state):
        calls["n"] += 1

    already_finished = FakeState()
    already_finished.finish("pre")

    body_loop = Loop(body, until=is_answered, limit=10)
    result = body_loop(FakeAgent(), already_finished)

    assert calls["n"] == 0
    assert result == "pre"
    assert already_finished.stopped == StoppedByFinish()


def test_stops_on_limit_with_none_answer():
    # "After limit iterations... stop quietly with StoppedByLimit(limit)" / "None if there is no answer yet"
    calls = {"n": 0}

    def body(agent, state):
        calls["n"] += 1

    body_loop = Loop(body, until=is_answered, limit=3)
    state = FakeState()
    result = body_loop(FakeAgent(), state)

    assert result is None
    assert state.stopped == StoppedByLimit(3)
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
    assert state.stopped == StoppedByUntil("done")


def test_multiple_until_conditions_checked_in_order():
    # "until takes... a list of functions" - conditions are checked in order, and the true one's name is in
    # state.stopped
    def cond_a(s):
        return False

    def cond_b(s):
        return True

    body_loop = Loop(lambda a, s: None, until=[cond_a, cond_b], limit=10)
    state = FakeState()
    body_loop(FakeAgent(), state)

    assert state.stopped == StoppedByUntil("cond_b")


def test_copy_overrides_only_given_fields():
    # "A loop made by @loop has copy(until=, limit=)... the original is unchanged"
    def body(agent, state):
        pass

    original = Loop(body, until=is_answered, limit=5)
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

    body_loop = Loop(body, until=is_answered, limit=2)
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
    assert state.stopped == StoppedByFinish()


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

    outer = Loop(outer_body, until=is_answered, limit=5)
    state = FakeState()
    result = outer(FakeAgent(), state)

    assert counts["inner"] == 2
    assert result == "outer done"


def test_block_can_be_callable_object_not_just_function():
    # "A block is a function or __call__ object taking (agent, state), the same shape as a loop"
    class FinishingBlock:
        def __call__(self, agent, state):
            state.finish("via block object")

    body_loop = Loop(FinishingBlock(), until=is_answered, limit=5)
    state = FakeState()
    result = body_loop(FakeAgent(), state)

    assert result == "via block object"
    assert state.stopped == StoppedByFinish()


# ============================================================ why conditions are checked before a turn (real State)


def test_add_message_makes_the_until_false_so_loop_runs_again():
    # "Adding a new instruction (add_message) makes waiting_for_user false so the loop runs,
    #  and with nothing to do it stops at turn 0"
    fake = FakeModel(["first answer", "second answer"])
    agent = Agent(model=fake, reporter=None)
    state = State(messages=[Message.user("Task")])

    def body(a, s):
        a.think(s)

    think_loop = Loop(body, until=waiting_for_user, limit=5)

    answer1 = think_loop(agent, state)
    assert answer1 == "first answer"
    assert state.turn == 1

    # Nothing to do: the model already answered, so it must stop at turn 0 (uses no extra FakeModel reply)
    answer_again = think_loop(agent, state)
    assert answer_again == "first answer"
    assert state.turn == 1  # the body did not run
    assert fake.remaining == 1  # the second reply is still left

    # Adding a new instruction makes the until false, so the loop actually runs
    state.add_message(Message.user("Keep going"))
    answer2 = think_loop(agent, state)
    assert answer2 == "second answer"
    assert state.turn == 2
    assert fake.remaining == 0


def test_until_accepts_two_user_written_conditions_together():
    # "User-made functions go in the same list": any number of them, checked in order
    def never_over_budget(state):
        return False

    combo_loop = Loop(lambda a, s: a.think(s), until=[never_over_budget, waiting_for_user], limit=5)
    fake = FakeModel(["final answer"])
    agent = Agent(model=fake, reporter=None)
    state = State(messages=[Message.user("Task")])

    answer = combo_loop(agent, state)

    assert answer == "final answer"
    assert state.stopped == StoppedByUntil("waiting_for_user")


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
    state = State(messages=[Message.user("Task")])
    state.finish("already done")

    with pytest.raises(ValueError):
        call_after_finish(agent, state)


# ============================================================ default loop


def test_default_loop_shape():
    assert default_loop.limit == 50
    assert [f.__name__ for f in default_loop.until] == ["waiting_for_user"]
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
    # Default loop: "compact_if_full(agent, state); agent.think(state); if state.pending_calls: agent.use_tools(state)"
    # until=waiting_for_user, limit=50
    @tool
    def echo(text: str) -> str:
        """Echo tool for tests."""
        return text

    fake = FakeModel([tool_call("echo", text="hi"), "final reply"])
    agent = Agent(model=fake, tools=[echo], reporter=None)
    state = State(messages=[Message.user("Task")])

    answer = agent.run(state)

    assert answer == "final reply"
    assert state.stopped == StoppedByUntil("waiting_for_user")
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


def test_stop_table_until_condition_sets_stopped_to_condition_name():
    # "until condition true before a turn | stop and return state.answer | condition function name (e.g. 'over_budget')"
    def over_budget(state):
        return state.turn >= 2

    fake = FakeModel(["first thought", "second thought"])

    @loop(until=over_budget, limit=10)
    def budget_loop(agent, state):
        agent.think(state)

    agent = Agent(model=fake, loop=budget_loop, reporter=None)
    state = State(messages=[Message.user("Task")])

    answer = agent.run(state)

    assert answer == state.answer == "second thought"
    assert state.stopped == StoppedByUntil("over_budget")
    assert state.turn == 2


def test_stop_table_finish_sets_stopped_finish():
    # "state.finish() called | stop when that turn ends and return state.answer... the whole run ends | 'finish'"
    @loop(until=waiting_for_user, limit=5)
    def finishing_loop(agent, state):
        state.finish("finished answer")

    agent = Agent(model=FakeModel([]), loop=finishing_loop, reporter=None)
    state = State(messages=[Message.user("Task")])

    answer = agent.run(state)

    assert answer == "finished answer"
    assert state.stopped == StoppedByFinish("finished answer")


def test_stop_table_limit_sets_stopped_limit_and_returns_last_answer():
    # "limit reached | stop without an exception and return state.answer (None if no answer yet) | 'limit'"
    def never(state):
        return False

    fake = FakeModel(["thought one", "thought two"])

    @loop(until=never, limit=2)
    def think_only(agent, state):
        agent.think(state)

    agent = Agent(model=fake, loop=think_only, reporter=None)
    state = State(messages=[Message.user("Task")])

    answer = agent.run(state)

    assert answer == state.answer == "thought two"  # stopping at limit does not lose the answer already there
    assert state.stopped == StoppedByLimit(2)
    assert state.turn == 2


# ============================================================ stops are recorded as StopEntry (real State)


def _stops(state):
    """The ``content`` of every ``StopEntry`` in the history, in order (``None`` is a cleared stop)."""
    return [h.content for h in state.history if isinstance(h, StopEntry)]


def test_default_loop_stops_with_until_waiting_for_user_and_records_it_as_a_stop_entry():
    # "The default loops use a private until function named is_answered, so StoppedByUntil("waiting_for_user")
    # and the loop records the stop in the history instead of setting a field
    agent = Agent(model=FakeModel(["The answer"]), reporter=None)
    state = State(messages=[Message.user("Task")])

    assert agent.run(state) == "The answer"

    assert state.stopped == StoppedByUntil("waiting_for_user")
    assert str(state.stopped) == "stopped by waiting_for_user"
    assert isinstance(state.history[-1], StopEntry)
    assert state.history[-1].content == StoppedByUntil("waiting_for_user")
    assert _stops(state) == [StoppedByUntil("waiting_for_user")]


def test_default_loop_does_not_stop_while_the_reply_still_has_tool_calls():
    # the until needs "no pending calls and the last message is a reply without tool calls"
    @tool
    def echo(text: str) -> str:
        """Echo tool for tests."""
        return text

    fake = FakeModel([tool_call("echo", text="hi"), "done"])
    agent = Agent(model=fake, tools=[echo], reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.run(state)
    assert state.turn == 2
    assert _stops(state) == [StoppedByUntil("waiting_for_user")]  # one stop, at the end only


def test_default_loop_on_a_state_that_is_already_answered_stops_at_turn_zero():
    # A State whose last message is an answer is "answered": the loop stops before the first turn (and records it)
    agent = Agent(model=FakeModel(["never used"]), reporter=None)
    state = State(messages=[Message.user("Task"), Message.assistant("Already answered")])
    # (an imported assistant message is not a model reply of this State, so state.answer stays None)
    assert agent.run(state) is None
    assert state.turn == 0
    assert _stops(state) == [StoppedByUntil("waiting_for_user")]


def test_user_written_waiting_for_user_until_stops_a_loop_with_a_real_state():
    # SPEC section 6: the 5-line user-written version replaces State.is_answered
    @tool
    def echo(text: str) -> str:
        """Echo tool for tests."""
        return text

    @loop(until=waiting_for_user, limit=10)
    def coding(agent, state):
        agent.think(state)
        if state.pending_calls:
            agent.use_tools(state)

    fake = FakeModel([tool_call("echo", text="hi"), "Done, what next?"])
    agent = Agent(model=fake, tools=[echo], loop=coding, reporter=None)
    state = State(messages=[Message.user("Task")])

    assert agent.run(state) == "Done, what next?"
    assert state.stopped == StoppedByUntil("waiting_for_user")
    assert _stops(state) == [StoppedByUntil("waiting_for_user")]


def test_chat_style_loop_waits_for_the_user_and_runs_again_after_add_message():
    # the use the until exists for: stop when the model waits for the user, go on when the user writes
    fake = FakeModel(["Hello! What would you like?", "Sure, here it is."])

    @loop(until=waiting_for_user, limit=10)
    def chat(agent, state):
        agent.think(state)

    agent = Agent(model=fake, loop=chat, reporter=None)
    state = State(messages=[Message.user("Hi")])
    assert agent.run(state) == "Hello! What would you like?"
    state.add_message(Message.user("A summary please"))
    assert agent.run(state) == "Sure, here it is."
    assert _stops(state) == [StoppedByUntil("waiting_for_user")] * 2


def test_a_loop_limit_stop_is_recorded_as_a_stop_entry():
    def never(state):
        return False

    @loop(until=never, limit=2)
    def think_only(agent, state):
        agent.think(state)

    agent = Agent(model=FakeModel(["one", "two"]), loop=think_only, reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.run(state)
    assert _stops(state) == [StoppedByLimit(2)]
    assert state.stopped == StoppedByLimit(2)


def test_nested_loops_the_outer_loop_going_on_clears_the_inner_stop_with_stop_entry_none():
    # "An outer loop that goes on clears an inner loop's until/limit stop: StopEntry(content=None)"
    def never(state):
        return False

    @loop(until=never, limit=1)
    def inner(agent, state):
        agent.think(state)

    def after_two_turns(state):
        return state.turn >= 2

    seen_stopped = []

    @loop(until=after_two_turns, limit=10)
    def outer(agent, state):
        seen_stopped.append(state.stopped)  # what the body sees at the start of each outer turn
        inner(agent, state)

    agent = Agent(model=FakeModel(["one", "two"]), loop=outer, reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.run(state)

    assert _stops(state) == [
        StoppedByLimit(1),  # inner, first call
        None,  # the outer loop goes on: the inner stop is cleared
        StoppedByLimit(1),  # inner, second call
        StoppedByUntil("after_two_turns"),  # the outer loop's own stop
    ]
    assert seen_stopped == [None, None]  # the cleared stop is not visible to the body
    assert state.stopped == StoppedByUntil("after_two_turns")
    # the cleared stop is the history's fact too: replaying gives the same value
    assert State(history=state.history).stopped == state.stopped


def test_finish_in_a_loop_is_one_stop_entry_with_the_answer_and_the_loop_adds_none():
    # "finish recorded as StopEntry(StoppedByFinish(answer))"; the loop sees it is already recorded
    @loop(until=waiting_for_user, limit=5)
    def finishing_loop(agent, state):
        state.finish("finished answer")

    agent = Agent(model=FakeModel([]), loop=finishing_loop, reporter=None)
    state = State(messages=[Message.user("Task")])
    assert agent.run(state) == "finished answer"
    assert _stops(state) == [StoppedByFinish("finished answer")]
    assert state.finished and state.answer == "finished answer"


def test_finish_inside_a_nested_loop_ends_the_outer_loop_too():
    # a stop decided by the run (finish) is not cleared by an outer loop
    def never(state):
        return False

    @loop(until=never, limit=5)
    def inner(agent, state):
        agent.think(state)
        state.finish("inner decided")

    calls = []

    @loop(until=never, limit=5)
    def outer(agent, state):
        calls.append(1)
        inner(agent, state)

    agent = Agent(model=FakeModel(["thinking"]), loop=outer, reporter=None)
    state = State(messages=[Message.user("Task")])
    assert agent.run(state) == "inner decided"
    assert calls == [1]
    assert _stops(state) == [StoppedByFinish("inner decided")]


def test_a_new_run_starts_with_no_stop_even_after_a_stopped_one():
    # RunStartEntry resets state.stopped, so the previous reason does not stop the next run before it begins
    fake = FakeModel(["first", "second"])

    @loop(until=waiting_for_user, limit=5)
    def chat(agent, state):
        agent.think(state)

    agent = Agent(model=fake, loop=chat, reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.run(state)
    state.add_message(Message.user("More"))
    agent.run(state)
    assert state.turn == 2
    assert [h.kind for h in state.history if h.kind in ("run_start", "stop")] == [
        "run_start",
        "stop",
        "run_start",
        "stop",
    ]


# ============================================================ blocks.compact_if_full / CompactIfFull


def test_compact_if_full_default_instance_shape():
    # "compact_if_full = CompactIfFull()" - a ready-to-use lowercase instance, default at=0.6
    assert isinstance(compact_if_full, CompactIfFull)
    assert compact_if_full.at == 0.6
    assert compact_if_full.instructions is None


def test_compact_if_full_does_not_trigger_below_threshold():
    # "if agent.context_used(state) > 0.6: agent.compact(state)" - no compaction below the threshold
    fake = FakeModel(["short answer", "must not be used"], context_window=1_000_000)
    agent = Agent(model=fake, reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.think(state)

    assert agent.context_used(state) <= 0.6
    before_remaining = fake.remaining

    compact_if_full(agent, state)

    assert fake.remaining == before_remaining  # agent.compact was not called
    assert state.messages[-1].role == "assistant"  # not replaced by compaction (a summary user message)


#: Size that pushes context_used above 0.6 while the compaction request (with COMPACT_PROMPT) still fits.
_LONG_REPLY = "A long reply to fill the context window. " * 15
_OVER_060_WINDOW = 330


def test_compact_if_full_triggers_above_threshold():
    # "if agent.context_used(state) > 0.6: agent.compact(state)" - compacts above the threshold
    fake = FakeModel([_LONG_REPLY, "This is the summary"], context_window=_OVER_060_WINDOW)
    agent = Agent(model=fake, reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.think(state)

    assert agent.context_used(state) > 0.6
    before_remaining = fake.remaining

    compact_if_full(agent, state)

    assert fake.remaining == before_remaining - 1  # agent.compact called the model once more
    assert state.messages[-1].role == "user"  # replaced by compact(summary)
    assert "This is the summary" in state.messages[-1].text
    assert len(state.messages) == 2  # "replaced by two: the original task + the summary"


def test_compact_if_full_instructions_reach_model_request():
    # "what to keep | task content | user (instructions)" - CompactIfFull(instructions=...) reaches the request
    fake = FakeModel([_LONG_REPLY, "summarized"], context_window=_OVER_060_WINDOW)
    agent = Agent(model=fake, reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.think(state)

    block = CompactIfFull(instructions="Keep only the key points")
    block(agent, state)

    last_request = fake.requests[-1]
    assert "Keep only the key points" in last_request.messages[-1].text


def test_compact_if_full_custom_at_changes_trigger_point():
    # "Blocks that need settings are classes" - at sets 'when' to compact as a number visible in code
    fake = FakeModel(["short answer", "summary1"], context_window=1_000_000)
    agent = Agent(model=fake, reporter=None)
    state = State(messages=[Message.user("Task")])
    agent.think(state)

    assert agent.context_used(state) < 0.6  # would not be compacted at the default threshold (0.6)

    lenient = CompactIfFull(at=agent.context_used(state) / 2)
    lenient(agent, state)

    assert state.messages[-1].role == "user"  # compaction happened because of the lowered threshold
    assert "summary1" in state.messages[-1].text


def test_missing_compact_block_lets_context_overflow_error_surface():
    # "If a user-written loop has no compaction block, alpineagents.ContextTooLongError is raised when the
    #  context overflows. It never compacts silently" (checked here with the overflow FakeModel simulates)
    fake = FakeModel(["any answer"], context_window=1)

    @loop(until=waiting_for_user, limit=5)
    def no_compact_loop(agent, state):
        agent.think(state)  # no compact_if_full

    agent = Agent(model=fake, loop=no_compact_loop, reporter=None)
    state = State(messages=[Message.user("Task")])

    with pytest.raises(ContextTooLongError):
        agent.run(state)


def test_waiting_for_user_is_public_and_is_the_default_loops_until():
    import alpineagents
    from alpineagents import adefault_loop

    assert "waiting_for_user" in alpineagents.__all__
    assert default_loop.until == (alpineagents.waiting_for_user,)
    assert adefault_loop.until == (alpineagents.waiting_for_user,)
