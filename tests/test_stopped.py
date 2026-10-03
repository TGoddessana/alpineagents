"""state.stopped values (StoppedByUntil/Limit/Finish/Permission), when they are set, how loops stop on them, how a
Store saves and reads them (StopEntry in the log, ``stopped`` in the info dict), and ToolOutcomeKind. No network."""

from __future__ import annotations

import json
import warnings

import pytest

import alpineagents
from alpineagents import (
    Agent,
    FileStore,
    Reporter,
    Message,
    State,
    StateInfo,
    StoppedByFinish,
    StoppedByLimit,
    StoppedByPermission,
    StoppedByUntil,
    StopEntry,
    ToolOutcome,
    ToolOutcomeKind,
    loop,
    tool,
)
from alpineagents import _serial, types
from alpineagents import waiting_for_user
from alpineagents.testing import FakeModel, tool_call

CALL = tool_call("delete_file", path="main.py")
BY_PERMISSION = StoppedByPermission(CALL, "DecideByHuman()")


def make_agent(replies=("Done",), **settings) -> Agent:
    settings.setdefault("reporter", None)
    settings.setdefault("human", None)
    settings.setdefault("model", FakeModel(list(replies)))
    return Agent(**settings)


def task(text: str = "Task", **kwargs) -> State:
    return State(messages=[Message.user(text)], **kwargs)


def never(state: State) -> bool:
    return False


def always(state: State) -> bool:
    return True


def stop_by_permission(state: State) -> None:
    """What a permission's Denied(..., stop=True) does to state.stopped (without the pending calls it needs)."""
    state._record_stop(BY_PERMISSION)


# ================================================================ the values


def test_stop_values_are_exported_and_the_union_lives_in_types():
    for name in ("StoppedByUntil", "StoppedByLimit", "StoppedByFinish", "StoppedByPermission", "ToolOutcomeKind"):
        assert name in alpineagents.__all__
        assert getattr(alpineagents, name) is getattr(types, name)
        assert name in types.__all__
    assert "Stopped" in types.__all__
    assert "Stopped" not in alpineagents.__all__ and not hasattr(alpineagents, "Stopped")


def test_stop_values_compare_by_value_and_are_frozen():
    assert StoppedByUntil("waiting_for_user") == StoppedByUntil("waiting_for_user") != StoppedByUntil("other")
    assert StoppedByLimit(30) == StoppedByLimit(30) != StoppedByLimit(31)
    assert StoppedByFinish() == StoppedByFinish()
    assert StoppedByPermission(CALL, "X()") == StoppedByPermission(CALL, "X()") != StoppedByPermission(CALL, "Y()")
    with pytest.raises(AttributeError):
        StoppedByLimit(3).turns = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    "stopped, text",
    [
        (StoppedByUntil("waiting_for_user"), "stopped by waiting_for_user"),
        (StoppedByLimit(30), "stopped at limit 30"),
        (StoppedByFinish(), "stopped by finish"),
        (BY_PERMISSION, "stopped by permission DecideByHuman()"),
    ],
)
def test_str_is_a_short_form(stopped, text):
    assert str(stopped) == text


def test_state_starts_with_no_stop_and_str_shows_the_short_form():
    state = task()
    assert state.stopped is None

    @loop(until=never, limit=2)
    def idle(agent, state):
        pass

    idle(None, state)
    assert state.stopped == StoppedByLimit(2)
    assert str(state).splitlines()[-1] == "done: stopped at limit 2 (0 turns)"


# ================================================================ when stopped is set, and how loops stop on it


def test_until_functions_named_finish_or_limit_are_plain_until_names():
    def finish(state):
        return True

    state = task()
    loop(until=finish, limit=3)(lambda agent, state: None)(None, state)
    assert state.stopped == StoppedByUntil("finish")
    assert state.stopped != StoppedByFinish()


def test_lambda_warning_says_it_leaves_no_useful_name():
    with pytest.warns(UserWarning, match="no useful name in state.stopped"):
        loop(until=lambda s: False, limit=3)(lambda agent, state: None)


def test_loop_stops_on_a_permission_stop_before_the_next_turn():
    turns = []

    @loop(until=never, limit=5)
    def body(agent, state):
        turns.append(state)
        stop_by_permission(state)

    state = task()
    body(None, state)
    assert len(turns) == 1
    assert state.stopped == BY_PERMISSION


def test_a_stop_already_set_is_kept_before_until_is_checked():
    state = task()
    stop_by_permission(state)
    loop(until=always, limit=5)(lambda agent, state: None)(None, state)
    assert state.stopped == BY_PERMISSION

    state.finish("done")
    assert state.stopped == StoppedByFinish("done")  # finish replaces it right away
    loop(until=always, limit=5)(lambda agent, state: None)(None, state)
    assert state.stopped == StoppedByFinish("done")  # the finish stays; no until/limit stop replaces it


def test_a_permission_stop_stops_the_outer_loop_too():
    inner_turns = []

    @loop(until=never, limit=5)
    def inner(agent, state):
        inner_turns.append(1)
        stop_by_permission(state)

    outer = loop(until=never, limit=3)(inner)
    state = task()
    outer(None, state)
    assert len(inner_turns) == 1
    assert state.stopped == BY_PERMISSION


async def test_async_loop_stops_on_a_permission_stop():
    turns = []

    @loop(until=never, limit=5)
    async def body(agent, state):
        turns.append(1)
        stop_by_permission(state)

    state = task()
    await body(None, state)
    assert len(turns) == 1
    assert state.stopped == BY_PERMISSION


def test_run_clears_the_previous_stop():
    state = task()
    stop_by_permission(state)
    make_agent(["Done"]).run(state)
    # The stale request did not stop the loop before its first turn.
    assert state.turn == 1
    assert state.stopped == StoppedByUntil("waiting_for_user")


async def test_arun_clears_the_previous_stop():
    state = task()
    stop_by_permission(state)
    await make_agent(["Done"]).arun(state)
    assert state.turn == 1
    assert state.stopped == StoppedByUntil("waiting_for_user")


@pytest.mark.parametrize("use_async", [False, True])
async def test_run_that_raises_leaves_stopped_none_even_after_a_permission_stop(use_async):
    state = task()
    make_agent(["First"]).run(state)
    state.add_message(Message.user("Again"))

    if use_async:

        @loop(until=never, limit=5)
        async def failing(agent, state):
            stop_by_permission(state)
            raise RuntimeError("boom")

    else:

        @loop(until=never, limit=5)
        def failing(agent, state):
            stop_by_permission(state)
            raise RuntimeError("boom")

    ended = []

    class Ends(Reporter):
        def on_run_end(self, state, error):
            ended.append(state.stopped)

    with pytest.raises(RuntimeError):
        if use_async:
            await make_agent(loop=failing, reporter=Ends()).arun(state)
        else:
            make_agent(loop=failing, reporter=Ends()).run(state)
    assert state.stopped is None
    assert ended == [None]


def test_run_that_raises_after_finish_leaves_stopped_none_and_the_state_finished():
    @loop(until=never, limit=5)
    def failing(agent, state):
        state.finish("half done")
        raise RuntimeError("boom")

    state = task()
    with pytest.raises(RuntimeError):
        make_agent(loop=failing).run(state)
    assert state.stopped is None
    assert state.finished
    # Called directly, a loop still stops on a finished State.
    loop(until=never, limit=5)(lambda agent, state: None)(None, state)
    assert state.stopped == StoppedByFinish()  # recorded without an answer: the earlier one stays state.answer
    assert state.answer == "half done"


@tool
def submit(state: State, summary: str) -> str:
    """Submit the finished work"""
    state.finish(summary)
    return "submitted"


def test_finish_sets_stopped_right_away():
    state = task()
    state.finish()
    assert state.stopped == StoppedByFinish()
    assert "done: stopped by finish" in str(state)


def test_finish_answer_is_part_of_the_stop_and_the_stop_is_a_history_entry():
    state = task()
    state.finish({"summary": "all done", "files": ["a.py"]})
    assert state.stopped == StoppedByFinish({"summary": "all done", "files": ["a.py"]})
    assert str(state.stopped) == "stopped by finish"  # the str form does not show the answer
    [entry] = [e for e in state.history if isinstance(e, StopEntry)]
    assert entry.content == state.stopped
    assert state.answer == {"summary": "all done", "files": ["a.py"]}


def test_finish_in_a_tool_is_visible_in_the_same_turn_and_to_the_reporter():
    seen, reported = [], []

    class Ends(Reporter):
        def on_tool_end(self, state, call, result, outcome):
            reported.append(state.stopped)

    @loop(until=waiting_for_user, limit=5)
    def body(agent, state):
        assert state.stopped is None
        agent.think(state)
        if state.pending_calls:
            agent.use_tools(state)
            seen.append(state.stopped)

    agent = make_agent([tool_call("submit", summary="all done")], tools=[submit], loop=body, reporter=Ends())
    assert agent.run("Task") == "all done"
    assert seen == [StoppedByFinish("all done")]
    assert reported == [StoppedByFinish("all done")]


@pytest.mark.parametrize("use_async", [False, True])
async def test_a_loop_without_at_loop_stops_on_finish(use_async):
    turns = []

    def plain(agent, state):
        while state.stopped is None and not waiting_for_user(state):
            turns.append(state.turn)
            agent.think(state)
            if state.pending_calls:
                agent.use_tools(state)
        return state.answer

    async def aplain(agent, state):
        while state.stopped is None and not waiting_for_user(state):
            turns.append(state.turn)
            await agent.athink(state)
            if state.pending_calls:
                await agent.ause_tools(state)
        return state.answer

    agent = make_agent(
        [tool_call("submit", summary="all done"), "never asked"], tools=[submit], loop=aplain if use_async else plain
    )
    state = task()
    answer = await agent.arun(state) if use_async else agent.run(state)
    assert answer == "all done"
    assert turns == [0]
    assert state.stopped == StoppedByFinish("all done")


def test_a_nested_loop_stop_ends_only_that_loop_and_is_cleared_when_the_outer_goes_on():
    seen = []

    @loop(until=never, limit=2)
    def inner(agent, state):
        seen.append(state.stopped)

    outer = loop(until=never, limit=3)(inner)
    state = task()
    outer(None, state)
    # Each inner run starts with stopped None: the previous inner StoppedByLimit(2) was cleared by the outer loop.
    assert seen == [None] * 6
    assert state.stopped == StoppedByLimit(3)


def test_nothing_public_is_named_stop_requested():
    assert not hasattr(State, "stop_requested")
    assert not hasattr(task(), "stop_requested")
    assert not any("stop_request" in name for name in dir(task()))


# ================================================================ saving


@pytest.mark.parametrize(
    "stopped, saved",
    [
        (None, None),
        (StoppedByUntil("waiting_for_user"), {"kind": "until", "name": "waiting_for_user"}),
        (StoppedByLimit(30), {"kind": "limit", "turns": 30}),
        (StoppedByFinish(), {"kind": "finish", "answer": None}),
        (StoppedByFinish({"summary": "done"}), {"kind": "finish", "answer": {"summary": "done"}}),
        (
            BY_PERMISSION,
            {
                "kind": "permission",
                "call": {"name": "delete_file", "args": {"path": "main.py"}, "id": CALL.id},
                "permission": "DecideByHuman()",
            },
        ),
    ],
)
def test_stopped_is_saved_as_json_and_read_back(stopped, saved):
    data = _serial.stopped_to_dict(stopped)
    assert data == saved
    assert _serial.stopped_from_dict(json.loads(json.dumps(data))) == stopped


def test_unknown_saved_stop_kind_is_refused():
    with pytest.raises(ValueError, match="unknown stop kind"):
        _serial.stopped_from_dict({"kind": "timeout"})


def test_format_version_is_4():
    assert _serial.VERSION == 4


@pytest.mark.parametrize(
    "stopped",
    [
        None,
        StoppedByUntil("waiting_for_user"),
        StoppedByLimit(30),
        StoppedByFinish("answer"),
        BY_PERMISSION,
    ],
)
def test_a_stop_entry_survives_a_save_and_load(tmp_path, stopped):
    store = FileStore(tmp_path)
    state = task(id="s")
    state._record_stop(stopped)
    store.save(state)

    loaded = store.load("s")
    assert loaded.stopped == stopped
    assert isinstance(loaded.history[-1], StopEntry) and loaded.history[-1].content == stopped
    assert loaded.snapshot() == state.snapshot()


def test_permission_stop_survives_a_store_and_does_not_block_the_next_run(tmp_path):
    @loop(until=never, limit=5)
    def stops_after_one_turn(agent, state):
        agent.think(state)
        stop_by_permission(state)

    store = FileStore(tmp_path)
    state = task(id="perm")
    make_agent(["Thinking"], loop=stops_after_one_turn, store=store).run(state)
    assert state.stopped == BY_PERMISSION

    info = json.loads((tmp_path / "perm" / "info.json").read_text("utf-8"))
    assert info["v"] == 4 and info["stopped"]["kind"] == "permission"
    assert "stopped_by" not in info and "stopped_limit" not in info
    loaded = store.load("perm")
    assert loaded.stopped == BY_PERMISSION
    assert store.list()[0].stopped == BY_PERMISSION

    # A restored stop reason does not stop the next run: RunStartEntry resets it.
    loaded.add_message(Message.user("Go on"))
    assert make_agent(["Done"], store=store).run(loaded) == "Done"
    assert loaded.turn == 2
    assert loaded.stopped == StoppedByUntil("waiting_for_user")
    assert store.load("perm").stopped == StoppedByUntil("waiting_for_user")


def test_finish_in_a_tool_is_in_the_info_saved_after_that_tool(tmp_path):
    saved = []

    @loop(until=waiting_for_user, limit=5)
    def body(agent, state):
        agent.think(state)
        if state.pending_calls:
            agent.use_tools(state)
            # The loop has not stopped yet; the info was saved after the tool's result.
            saved.append(json.loads((tmp_path / "fin" / "info.json").read_text("utf-8")))

    store = FileStore(tmp_path)
    state = task(id="fin")
    make_agent([tool_call("submit", summary="all done")], tools=[submit], loop=body, store=store).run(state)
    assert saved[0]["stopped"] == {"kind": "finish", "answer": "all done"} and saved[0]["finished"] is True
    loaded = store.load("fin")
    assert loaded.stopped == StoppedByFinish("all done") and loaded.finished and loaded.answer == "all done"


def test_state_info_from_an_info_dict():
    info = {
        "v": 4,
        "first_message": "Hi",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:01+00:00",
        "turn": 3,
        "stopped": {"kind": "limit", "turns": 3},
        "finished": False,
    }
    state_info = StateInfo.from_info("x", info)
    assert state_info.id == "x"
    assert state_info.first_message == "Hi"
    assert state_info.turn == 3
    assert state_info.stopped == StoppedByLimit(3)


def test_info_from_the_older_stopped_by_format_is_refused():
    # 0.4 (format 3) kept stopped_by/stopped_limit in snapshot.json. 0.5 has no converter: it says so.
    info = {"v": 3, "task": "Hi", "turn": 3, "stopped_by": "limit", "stopped_limit": 3, "finished": False}
    with pytest.raises(ValueError, match="older alpineagents"):
        StateInfo.from_info("x", info)


# ================================================================ ToolOutcomeKind


def test_tool_outcome_kind_values():
    assert {kind.name: kind.value for kind in ToolOutcomeKind} == {
        "DONE": "done",
        "ERROR": "error",
        "INPUT_ERROR": "input_error",
        "ABORTED": "aborted",
        "INTERRUPTED": "interrupted",
        "DENIED": "denied",
        "CANCELLED": "cancelled",
    }


def test_tool_outcome_kind_compares_equal_to_the_old_strings():
    outcome = ToolOutcome(ToolOutcomeKind.DENIED, decided_by="DenyByName(['rm'])")
    assert outcome.kind == "denied"
    assert outcome.kind in ("denied", "cancelled")
    assert f"{outcome.kind}" == "denied"
    assert outcome.decided_by == "DenyByName(['rm'])"


def test_tool_outcome_turns_a_string_into_the_enum():
    outcome = ToolOutcome("aborted", TimeoutError())
    assert outcome.kind is ToolOutcomeKind.ABORTED
    assert outcome == ToolOutcome(ToolOutcomeKind.ABORTED, outcome.error)
    with pytest.raises(ValueError):
        ToolOutcome("exploded")


def test_decided_by_defaults_to_none():
    assert ToolOutcome("done").decided_by is None
    assert ToolOutcome(ToolOutcomeKind.CANCELLED).decided_by is None


def test_reporter_gets_enum_kinds_from_a_run():
    from alpineagents import Reporter, tool

    kinds = []

    class Kinds(Reporter):
        def on_tool_end(self, state, call, result, outcome):
            kinds.append(outcome.kind)

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        agent = make_agent([tool_call("add", a=1, b=2), tool_call("nope"), "3"], tools=[add], reporter=Kinds())
        agent.run("What is 1+2?")
    assert kinds == [ToolOutcomeKind.DONE, ToolOutcomeKind.INPUT_ERROR]
    assert all(type(kind) is ToolOutcomeKind for kind in kinds)
