"""state.stopped values (StoppedByUntil/Limit/Finish/Permission), when they are set, how loops stop on them, how a
Store saves and reads them (including snapshots from before they existed), and ToolOutcomeKind. No network."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

import alpineagents
from alpineagents import (
    Agent,
    FileStore,
    Reporter,
    SavedState,
    State,
    StoppedByFinish,
    StoppedByLimit,
    StoppedByPermission,
    StoppedByUntil,
    ToolOutcome,
    ToolOutcomeKind,
    loop,
    tool,
)
from alpineagents import _serial, types
from alpineagents.testing import FakeModel, tool_call

CALL = tool_call("delete_file", path="main.py")
BY_PERMISSION = StoppedByPermission(CALL, "DecideByHuman()")


def make_agent(replies=("Done",), **settings) -> Agent:
    settings.setdefault("reporter", None)
    settings.setdefault("human", None)
    settings.setdefault("model", FakeModel(list(replies)))
    return Agent(**settings)


def never(state: State) -> bool:
    return False


def always(state: State) -> bool:
    return True


def stop_by_permission(state: State) -> None:
    """What a permission's Denied(..., stop=True) does to state.stopped (without the pending calls it needs)."""
    with state._lock:
        state._stopped = BY_PERMISSION


# ================================================================ the values


def test_stop_values_are_exported_and_the_union_lives_in_types():
    for name in ("StoppedByUntil", "StoppedByLimit", "StoppedByFinish", "StoppedByPermission", "ToolOutcomeKind"):
        assert name in alpineagents.__all__
        assert getattr(alpineagents, name) is getattr(types, name)
        assert name in types.__all__
    assert "Stopped" in types.__all__
    assert "Stopped" not in alpineagents.__all__ and not hasattr(alpineagents, "Stopped")


def test_stop_values_compare_by_value_and_are_frozen():
    assert StoppedByUntil("is_answered") == StoppedByUntil("is_answered") != StoppedByUntil("other")
    assert StoppedByLimit(30) == StoppedByLimit(30) != StoppedByLimit(31)
    assert StoppedByFinish() == StoppedByFinish()
    assert StoppedByPermission(CALL, "X()") == StoppedByPermission(CALL, "X()") != StoppedByPermission(CALL, "Y()")
    with pytest.raises(AttributeError):
        StoppedByLimit(3).turns = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    "stopped, text",
    [
        (StoppedByUntil("is_answered"), "stopped by is_answered"),
        (StoppedByLimit(30), "stopped at limit 30"),
        (StoppedByFinish(), "stopped by finish"),
        (BY_PERMISSION, "stopped by permission DecideByHuman()"),
    ],
)
def test_str_is_a_short_form(stopped, text):
    assert str(stopped) == text


def test_state_starts_with_no_stop_and_str_shows_the_short_form():
    state = State("Task")
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

    state = State("Task")
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

    state = State("Task")
    body(None, state)
    assert len(turns) == 1
    assert state.stopped == BY_PERMISSION


def test_a_stop_already_set_is_kept_before_until_is_checked():
    state = State("Task")
    stop_by_permission(state)
    loop(until=always, limit=5)(lambda agent, state: None)(None, state)
    assert state.stopped == BY_PERMISSION

    state.finish("done")
    assert state.stopped == StoppedByFinish()  # finish replaces it right away
    loop(until=always, limit=5)(lambda agent, state: None)(None, state)
    assert state.stopped == StoppedByFinish()


def test_a_permission_stop_stops_the_outer_loop_too():
    inner_turns = []

    @loop(until=never, limit=5)
    def inner(agent, state):
        inner_turns.append(1)
        stop_by_permission(state)

    outer = loop(until=never, limit=3)(inner)
    state = State("Task")
    outer(None, state)
    assert len(inner_turns) == 1
    assert state.stopped == BY_PERMISSION


async def test_async_loop_stops_on_a_permission_stop():
    turns = []

    @loop(until=never, limit=5)
    async def body(agent, state):
        turns.append(1)
        stop_by_permission(state)

    state = State("Task")
    await body(None, state)
    assert len(turns) == 1
    assert state.stopped == BY_PERMISSION


def test_run_clears_the_previous_stop():
    state = State("Task")
    stop_by_permission(state)
    make_agent(["Done"]).run(state)
    # The stale request did not stop the loop before its first turn.
    assert state.turn == 1
    assert state.stopped == StoppedByUntil("is_answered")


async def test_arun_clears_the_previous_stop():
    state = State("Task")
    stop_by_permission(state)
    await make_agent(["Done"]).arun(state)
    assert state.turn == 1
    assert state.stopped == StoppedByUntil("is_answered")


@pytest.mark.parametrize("use_async", [False, True])
async def test_run_that_raises_leaves_stopped_none_even_after_a_permission_stop(use_async):
    state = State("Task")
    make_agent(["First"]).run(state)
    state.add_user_message("Again")

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

    state = State("Task")
    with pytest.raises(RuntimeError):
        make_agent(loop=failing).run(state)
    assert state.stopped is None
    assert state.is_finished()
    # Called directly, a loop still stops on a finished State.
    loop(until=never, limit=5)(lambda agent, state: None)(None, state)
    assert state.stopped == StoppedByFinish()


@tool
def submit(state: State, summary: str) -> str:
    """Submit the finished work"""
    state.finish(summary)
    return "submitted"


def test_finish_sets_stopped_right_away():
    state = State("Task")
    state.finish()
    assert state.stopped == StoppedByFinish()
    assert "done: stopped by finish" in str(state)


def test_finish_in_a_tool_is_visible_in_the_same_turn_and_to_the_reporter():
    seen, reported = [], []

    class Ends(Reporter):
        def on_tool_end(self, state, call, result, outcome):
            reported.append(state.stopped)

    @loop(until=State.is_answered, limit=5)
    def body(agent, state):
        assert state.stopped is None
        agent.think(state)
        if state.wants_tools():
            agent.use_tools(state)
            seen.append(state.stopped)

    agent = make_agent([tool_call("submit", summary="all done")], tools=[submit], loop=body, reporter=Ends())
    assert agent.run("Task") == "all done"
    assert seen == [StoppedByFinish()]
    assert reported == [StoppedByFinish()]


@pytest.mark.parametrize("use_async", [False, True])
async def test_a_loop_without_at_loop_stops_on_finish(use_async):
    turns = []

    def plain(agent, state):
        while state.stopped is None and not state.is_answered():
            turns.append(state.turn)
            agent.think(state)
            if state.wants_tools():
                agent.use_tools(state)
        return state.answer

    async def aplain(agent, state):
        while state.stopped is None and not state.is_answered():
            turns.append(state.turn)
            await agent.athink(state)
            if state.wants_tools():
                await agent.ause_tools(state)
        return state.answer

    agent = make_agent([tool_call("submit", summary="all done"), "never asked"], tools=[submit],
                       loop=aplain if use_async else plain)
    state = State("Task")
    answer = await agent.arun(state) if use_async else agent.run(state)
    assert answer == "all done"
    assert turns == [0]
    assert state.stopped == StoppedByFinish()


def test_a_nested_loop_stop_ends_only_that_loop_and_is_cleared_when_the_outer_goes_on():
    seen = []

    @loop(until=never, limit=2)
    def inner(agent, state):
        seen.append(state.stopped)

    outer = loop(until=never, limit=3)(inner)
    state = State("Task")
    outer(None, state)
    # Each inner run starts with stopped None: the previous inner StoppedByLimit(2) was cleared by the outer loop.
    assert seen == [None] * 6
    assert state.stopped == StoppedByLimit(3)


def test_nothing_public_is_named_stop_requested():
    assert not hasattr(State, "stop_requested")
    assert not hasattr(State("Task"), "stop_requested")
    assert not any("stop_request" in name for name in dir(State("Task")))


# ================================================================ saving


@pytest.mark.parametrize(
    "stopped, saved",
    [
        (None, None),
        (StoppedByUntil("is_answered"), {"kind": "until", "name": "is_answered"}),
        (StoppedByLimit(30), {"kind": "limit", "turns": 30}),
        (StoppedByFinish(), {"kind": "finish"}),
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
    assert _serial.stopped_from_snapshot({"stopped": json.loads(json.dumps(data))}) == stopped


def test_unknown_saved_stop_kind_is_refused():
    with pytest.raises(ValueError, match="unknown stop kind"):
        _serial.stopped_from_snapshot({"stopped": {"kind": "timeout"}})


def test_format_version_is_3():
    assert _serial.VERSION == 3


def test_permission_stop_survives_a_store_and_does_not_block_the_next_run(tmp_path):
    @loop(until=never, limit=5)
    def stops_after_one_turn(agent, state):
        agent.think(state)
        stop_by_permission(state)

    store = FileStore(tmp_path)
    state = State("Task", id="perm")
    make_agent(["Thinking"], loop=stops_after_one_turn, store=store).run(state)
    assert state.stopped == BY_PERMISSION

    snapshot = json.loads((tmp_path / "perm" / "snapshot.json").read_text("utf-8"))
    assert snapshot["v"] == 3 and snapshot["stopped"]["kind"] == "permission"
    assert "stopped_by" not in snapshot and "stopped_limit" not in snapshot
    loaded = store.load("perm")
    assert loaded.stopped == BY_PERMISSION
    assert store.list()[0].stopped == BY_PERMISSION

    # A restored stop reason does not stop the next run: run resets it at the start.
    loaded.add_user_message("Go on")
    assert make_agent(["Done"], store=store).run(loaded) == "Done"
    assert loaded.turn == 2
    assert loaded.stopped == StoppedByUntil("is_answered")
    assert store.load("perm").stopped == StoppedByUntil("is_answered")


def test_finish_in_a_tool_is_in_the_snapshot_saved_after_that_tool(tmp_path):
    saved = []

    @loop(until=State.is_answered, limit=5)
    def body(agent, state):
        agent.think(state)
        if state.wants_tools():
            agent.use_tools(state)
            # The loop has not stopped yet; the snapshot was saved after the tool's result.
            saved.append(json.loads((tmp_path / "fin" / "snapshot.json").read_text("utf-8")))

    store = FileStore(tmp_path)
    state = State("Task", id="fin")
    make_agent([tool_call("submit", summary="all done")], tools=[submit], loop=body, store=store).run(state)
    assert saved[0]["stopped"] == {"kind": "finish"} and saved[0]["finished"] is True
    loaded = store.load("fin")
    assert loaded.stopped == StoppedByFinish() and loaded.is_finished() and loaded.answer == "all done"


def _make_old(path: Path, stopped_by, stopped_limit, *, version=2) -> None:
    """Rewrites a saved snapshot the way alpineagents 0.3 wrote it."""
    snapshot = json.loads(path.read_text("utf-8"))
    del snapshot["stopped"]
    snapshot["v"] = version
    snapshot["stopped_by"] = stopped_by
    snapshot["stopped_limit"] = stopped_limit
    path.write_text(json.dumps(snapshot), "utf-8")


@pytest.mark.parametrize(
    "stopped_by, stopped_limit, expected",
    [
        ("is_answered", None, StoppedByUntil("is_answered")),
        ("finish", None, StoppedByFinish()),
        ("limit", 30, StoppedByLimit(30)),
        ("limit", None, StoppedByLimit(0)),
        (None, None, None),
    ],
)
@pytest.mark.parametrize("version", [1, 2])
def test_old_snapshot_with_stopped_by_loads(tmp_path, stopped_by, stopped_limit, expected, version):
    store = FileStore(tmp_path)
    make_agent(["Hello"], store=store).run(State("Hi", id="old"))
    _make_old(tmp_path / "old" / "snapshot.json", stopped_by, stopped_limit, version=version)

    loaded = store.load("old")
    assert loaded.stopped == expected
    assert loaded.answer == "Hello"
    [saved] = store.list()
    assert isinstance(saved, SavedState) and saved.stopped == expected

    # Continuing it clears the old reason and saves the new format.
    loaded.add_user_message("And then?")
    make_agent(["More"], store=store).run(loaded)
    snapshot = json.loads((tmp_path / "old" / "snapshot.json").read_text("utf-8"))
    assert snapshot["v"] == 3
    assert snapshot["stopped"] == {"kind": "until", "name": "is_answered"}
    assert store.load("old").stopped == StoppedByUntil("is_answered")


def test_saved_state_from_an_old_snapshot_dict():
    snapshot = {
        "task": "Hi",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:01+00:00",
        "turn": 3,
        "stopped_by": "limit",
        "stopped_limit": 3,
        "finished": False,
    }
    assert SavedState.from_snapshot("x", snapshot).stopped == StoppedByLimit(3)


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
