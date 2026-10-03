"""StateSnapshot and the State commands of SPEC-0.5 sections 2.1, 2.2 and 4.

- the keyword-only constructor (``messages=``, ``extra_data=``, ``history=``) and every error it raises
- ``StateSnapshot``: frozen, ``==``, what it holds
- ``snapshot()`` / ``fork()`` / ``restore()``
- ``edit_extra_data()``, ``add_message()``, ``compact()``, ``clear_tool_results()``
- the invariant ``State(history=s.history).snapshot() == s.snapshot()`` after every step of scripted scenarios

No network. ``FakeModel``/``FakeHuman`` and the internal helpers the Agent uses (``_begin_think``, ``_tool_result``,
...) drive the State.
"""

from __future__ import annotations

import dataclasses
import json
import re
import threading

import pytest

from alpineagents import (
    Agent,
    FileStore,
    Image,
    Message,
    Reporter,
    State,
    StateSnapshot,
    StoppedByFinish,
    StoppedByLimit,
    StoppedByPermission,
    StoppedByUntil,
    tool,
)
from alpineagents import _serial
from alpineagents._frozen import FrozenDict, FrozenList
from alpineagents.models.base import Model
from alpineagents.testing import FakeHuman, FakeModel, tool_call
from alpineagents.types import (
    AgentInfo,
    ContextChangeEntry,
    ExtraDataEntry,
    MessageEntry,
    ModelReplyEntry,
    ModelRequestEntry,
    Reply,
    Request,
    TextBlock,
    ToolCall,
    ToolOutcomeKind,
    ToolResultEntry,
    Usage,
)

MODEL = "fake/fake"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def task(text: str = "Task", **kwargs) -> State:
    return State(messages=[Message.user(text)], **kwargs)


def make_agent(replies=(), *, tools=(), human=None, reporter=None, **settings) -> Agent:
    return Agent(model=FakeModel(list(replies)), tools=list(tools), human=human, reporter=reporter, **settings)


def call(name: str = "bash", call_id: str = "c1", **args) -> ToolCall:
    return ToolCall(name=name, args=args or {"cmd": "ls"}, id=call_id)


def reply(*blocks, tokens: int | None = 100) -> Reply:
    parts = tuple(TextBlock(b) if isinstance(b, str) else b for b in blocks)
    return Reply(
        message=Message("assistant", parts),
        usage=Usage(input_tokens=10, output_tokens=5, requests=1),
        context_tokens=tokens,
    )


def think(state: State, *blocks) -> None:
    """One complete think without a model: request, then reply."""
    state._begin_think(MODEL)
    state._record_reply(reply(*blocks))


def assert_replays(state: State) -> None:
    """The invariant: a State rebuilt from the history has the same snapshot (public values and fold bookkeeping)."""
    snap = state.snapshot()
    rebuilt = State(history=state.history).snapshot()
    assert rebuilt == snap
    for field in dataclasses.fields(StateSnapshot):
        if field.name == "_memo":  # a cache of folds already done, not a value
            continue
        assert getattr(rebuilt, field.name) == getattr(snap, field.name), field.name


class SlowModel(Model):
    """A model that waits in respond() until released, so another thread can use the State meanwhile."""

    def __init__(self, text: str = "slow answer"):
        self.name = "slow"
        self.text = text
        self.release = threading.Event()
        self.entered = threading.Event()

    @property
    def context_window(self) -> int:
        return 100_000

    def respond(self, request: Request, on_text=None, on_event=None) -> Reply:
        self.entered.set()
        self.release.wait(timeout=5)
        return Reply(message=Message("assistant", (TextBlock(self.text),)), usage=Usage(requests=1))


def think_in_thread(agent: Agent, state: State, model: SlowModel):
    errors: list[BaseException] = []

    def body():
        try:
            agent.think(state)
        except BaseException as e:  # pragma: no cover - reported by the asserts
            errors.append(e)

    thread = threading.Thread(target=body)
    thread.start()
    assert model.entered.wait(timeout=5)
    return thread, errors


def echo_state(count: int):
    """A State after ``count`` think + use_tools turns of an ``echo`` tool. Returns (agent, state, calls)."""

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
    return agent, state, calls


# ===========================================================================
# The constructor
# ===========================================================================


def test_constructor_takes_no_positional_argument():
    with pytest.raises(TypeError, match=r"(?s)keyword arguments only.*State\(messages=\[Message\.user"):
        State("Fix calc.py")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="keyword arguments only"):
        State([Message.user("x")])  # type: ignore[call-arg]


def test_a_state_may_start_empty():
    state = State()
    assert state.history == ()
    assert state.messages == ()
    assert state.pending_calls == ()
    assert state.turn == 0
    assert state.finished is False
    assert state.stopped is None and state.answer is None
    assert dict(state.extra_data) == {}
    assert state.created_at is None and state.updated_at is None
    assert state.snapshot() == StateSnapshot()


def test_messages_are_recorded_as_one_import_entry_at_turn_zero():
    messages = [Message.user("Fix calc.py"), Message.assistant("Which test fails?"), Message.user("test_add")]
    state = State(messages=messages)

    [entry] = state.history
    assert isinstance(entry, ContextChangeEntry)
    assert entry.turn == 0
    change = entry.content
    assert change.kind == "import"
    assert change.messages == tuple(messages)
    assert change.before_tokens == 0 and change.after_tokens > 0
    assert state.messages == tuple(messages)
    assert state.turn == 0


def test_messages_may_be_any_iterable_of_messages():
    state = State(messages=(Message.user(t) for t in ["a", "b"]))
    assert [m.text for m in state.messages] == ["a", "b"]
    assert State(messages=(Message.user("x"),)).messages == (Message.user("x"),)


def test_empty_messages_record_nothing():
    assert State(messages=[]).history == ()
    assert State(messages=()).history == ()
    assert State(messages=[], extra_data={}).history == ()


@pytest.mark.parametrize("bad", ["Fix calc.py", b"bytes", Message.user("single, not a list")])
def test_messages_must_be_an_iterable_of_message_objects(bad):
    with pytest.raises(TypeError, match=r"State\(messages=\.\.\.\)"):
        State(messages=bad)  # type: ignore[arg-type]


def test_messages_that_is_not_iterable_is_a_type_error():
    with pytest.raises(TypeError):
        State(messages=42)  # type: ignore[arg-type]


def test_messages_that_is_not_iterable_gets_a_fix_message():
    with pytest.raises(TypeError, match=r"State\(messages=\.\.\.\)"):
        State(messages=42)  # type: ignore[arg-type]


def test_every_message_must_be_a_message_object():
    with pytest.raises(TypeError, match=r"(?s)takes Message objects only.*Message\.user"):
        State(messages=[Message.user("ok"), "not a message"])  # type: ignore[list-item]


def test_extra_data_is_recorded_as_one_entry_after_the_import():
    state = State(messages=[Message.user("Fix")], extra_data={"repo": "api", "n": 1})

    assert [e.kind for e in state.history] == ["context_change", "extra_data"]
    entry = state.history[1]
    assert isinstance(entry, ExtraDataEntry)
    assert entry.turn == 0
    assert entry.content == {"repo": "api", "n": 1}
    assert entry.removed == ()
    assert state.extra_data == {"repo": "api", "n": 1}


def test_extra_data_alone_makes_a_state_without_messages():
    state = State(extra_data={"repo": "api"})
    assert [e.kind for e in state.history] == ["extra_data"]
    assert state.messages == ()


def test_extra_data_is_copied_and_frozen():
    source = {"notes": ["a"], "cfg": {"depth": 1}}
    state = State(extra_data=source)
    source["notes"].append("b")
    source["cfg"]["depth"] = 2
    assert state.extra_data == {"notes": ["a"], "cfg": {"depth": 1}}
    assert isinstance(state.extra_data, FrozenDict)
    assert isinstance(state.extra_data["notes"], FrozenList)
    assert isinstance(state.extra_data["cfg"], FrozenDict)


def test_extra_data_tuples_become_lists():
    state = State(extra_data={"pair": (1, 2)})
    assert state.extra_data["pair"] == [1, 2]
    assert isinstance(state.extra_data["pair"], FrozenList)


@pytest.mark.parametrize(
    "value, fragment",
    [
        ({"seen": {1, 2}}, r"State\(extra_data=\.\.\.\)\['seen'\] must be JSON \(got set\)"),
        ({"when": object()}, r"\['when'\] must be JSON \(got object\)"),
        ({"nested": {"deep": [1, {2}]}}, r"\['nested'\]\['deep'\]\[1\] must be JSON"),
        ({"x": float("nan")}, r"\['x'\] must be JSON"),
        ({"x": float("inf")}, r"\['x'\] must be JSON"),
        ({1: "int key"}, "its key 1 is a int"),
    ],
)
def test_extra_data_must_be_json(value, fragment):
    with pytest.raises(TypeError, match=fragment):
        State(extra_data=value)


def test_extra_data_must_be_a_mapping():
    with pytest.raises(TypeError, match="takes a dict"):
        State(extra_data=["repo", "api"])  # type: ignore[arg-type]


def test_history_replays_another_states_history():
    source = task(extra_data={"repo": "api"})
    think(source, "answer")
    state = State(history=source.history)

    assert state.history == source.history
    assert state.snapshot() == source.snapshot()
    assert state.id != source.id


def test_history_may_be_any_iterable_of_entries():
    source = task()
    state = State(history=(entry for entry in source.history))
    assert state.snapshot() == source.snapshot()
    assert State(history=[]).snapshot() == StateSnapshot()


def test_history_cannot_be_combined_with_messages_or_extra_data():
    source = task()
    with pytest.raises(TypeError, match="history already holds them"):
        State(history=source.history, messages=[Message.user("x")])
    with pytest.raises(TypeError, match="history already holds them"):
        State(history=source.history, extra_data={"a": 1})
    with pytest.raises(TypeError, match="history already holds them"):
        State(history=source.history, extra_data={})
    # Empty messages are no messages.
    assert State(history=source.history, messages=()).snapshot() == source.snapshot()


def test_history_takes_history_entries_only():
    with pytest.raises(TypeError, match=r"history\[1\]"):
        State(history=[task().history[0], "not an entry"])  # type: ignore[list-item]


def test_an_inconsistent_history_names_the_entry_index():
    source = task()
    think(source, call(call_id="c1"))
    request, model_reply = source.history[-2:]
    result = ToolResultEntry(content="out", call=call(call_id="other"), outcome=ToolOutcomeKind.DONE, turn=1)

    # A tool result for a call that is not pending.
    with pytest.raises(ValueError, match=r"history entry 3 \(tool_result\)"):
        State(history=[*source.history, result])
    # A reply nobody asked for.
    with pytest.raises(ValueError, match=r"history entry 1 \(model_reply\)"):
        State(history=[source.history[0], model_reply])
    # A request while the last one is still waiting.
    with pytest.raises(ValueError, match=r"history entry 2 \(model_request\)"):
        State(history=[source.history[0], request, request])
    # A request while calls are pending.
    with pytest.raises(ValueError, match=r"history entry 3 \(model_request\)"):
        State(history=[*source.history, request])


def test_id_is_random_hex_when_omitted_and_unique():
    ids = {State().id for _ in range(20)}
    assert len(ids) == 20
    assert all(re.fullmatch(r"[0-9a-f]{32}", i) for i in ids)


def test_id_is_kept_as_given():
    assert State(id="bug-1").id == "bug-1"
    assert State(id="A_b-9").id == "A_b-9"
    assert State(id="x" * 128).id == "x" * 128


@pytest.mark.parametrize("bad", ["", "a b", "a/b", "../x", "dot.id", "x" * 129, "ü"])
def test_a_bad_id_is_a_value_error(bad):
    with pytest.raises(ValueError, match="letters, digits"):
        State(id=bad)


def test_a_non_string_id_is_a_type_error():
    with pytest.raises(TypeError, match="needs the id as a string"):
        State(id=123)  # type: ignore[arg-type]


def test_a_new_state_has_no_parent_and_is_its_own_root():
    state = State()
    assert state.parent is None
    assert state.root is state
    assert state.depth == 0


# ===========================================================================
# StateSnapshot
# ===========================================================================


def test_snapshot_holds_exactly_the_public_values():
    names = [f.name for f in dataclasses.fields(StateSnapshot) if not f.name.startswith("_")]
    assert names == [
        "history",
        "messages",
        "pending_calls",
        "turn",
        "usage",
        "extra_data",
        "finished",
        "answer",
        "stopped",
        "created_at",
        "updated_at",
    ]
    # And no method that changes anything (or anything else public).
    assert [n for n in dir(StateSnapshot()) if not n.startswith("_")] == sorted(names)


def test_private_fields_are_not_compared_and_not_shown():
    private = [f for f in dataclasses.fields(StateSnapshot) if f.name.startswith("_")]
    assert private
    assert all(not f.compare and not f.repr for f in private)


def test_snapshot_is_the_current_value_of_every_property():
    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return "x"

    agent = make_agent([tool_call("read_file", path="a"), "done"], tools=[read_file])
    state = task(extra_data={"k": 1})
    agent.run(state)
    snap = state.snapshot()

    for name in (
        "history",
        "messages",
        "pending_calls",
        "turn",
        "usage",
        "extra_data",
        "finished",
        "answer",
        "stopped",
        "created_at",
        "updated_at",
    ):
        assert getattr(state, name) == getattr(snap, name), name
    assert snap.turn == 2 and snap.answer == "done" and snap.stopped == StoppedByUntil("is_answered")


def test_snapshot_is_the_same_object_until_something_is_recorded():
    state = task()
    assert state.snapshot() is state.snapshot()
    before = state.snapshot()
    state.add_message(Message.user("more"))
    assert state.snapshot() is not before


def test_snapshot_never_changes_while_the_state_goes_on():
    agent = make_agent(["first", "second"])
    state = task()
    agent.think(state)
    snap = state.snapshot()
    history, messages, turn, usage = snap.history, snap.messages, snap.turn, snap.usage

    state.add_message(Message.user("more"))
    agent.think(state)
    state.finish("end")
    with state.edit_extra_data() as data:
        data["x"] = 1

    assert (snap.history, snap.messages, snap.turn, snap.usage) == (history, messages, turn, usage)
    assert snap.finished is False and snap.answer == "first" and dict(snap.extra_data) == {}
    assert snap.turn == 1 and len(snap.history) < len(state.history)


def test_snapshot_is_frozen():
    snap = task().snapshot()
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.turn = 1  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.messages = ()  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.brand_new_name = 1  # type: ignore[attr-defined]
    with pytest.raises(dataclasses.FrozenInstanceError):
        del snap.turn
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap._waiting = True  # type: ignore[misc]


def test_snapshot_values_are_read_only():
    snap = task(extra_data={"notes": ["a"], "cfg": {"x": 1}}).snapshot()
    with pytest.raises(TypeError):
        snap.extra_data["x"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        snap.extra_data["notes"].append("b")
    with pytest.raises(TypeError):
        snap.extra_data["cfg"]["x"] = 2
    with pytest.raises(TypeError):
        del snap.extra_data["notes"]
    assert isinstance(snap.history, tuple) and isinstance(snap.messages, tuple)
    assert isinstance(snap.pending_calls, tuple)


def test_snapshot_keyword_only_constructor():
    with pytest.raises(TypeError):
        StateSnapshot((), ())  # type: ignore[misc]
    assert StateSnapshot(turn=3).turn == 3


def test_snapshots_of_equal_histories_are_equal():
    state = task(extra_data={"k": [1, 2]})
    think(state, call())
    twin = State(history=state.history)
    assert state.snapshot() == twin.snapshot()
    assert state.snapshot() != task().snapshot()


def test_snapshot_equality_follows_every_public_field():
    state = task()
    base = state.snapshot()
    assert base == state.snapshot()

    state.add_message(Message.user("more"))
    assert base != state.snapshot()  # history and messages differ

    other = task()
    other.finish()
    assert other.snapshot() != task().snapshot()

    assert dataclasses.replace(base, turn=5) != base
    assert dataclasses.replace(base, usage=Usage(requests=1)) != base
    assert dataclasses.replace(base, answer="x") != base
    assert dataclasses.replace(base, stopped=StoppedByLimit(1)) != base
    assert dataclasses.replace(base, extra_data=FrozenDict(a=1)) != base


def test_snapshot_equality_ignores_the_fold_bookkeeping():
    state = task()
    base = state.snapshot()
    noisy = dataclasses.replace(
        base,
        _queued=(Message.user("held"),),
        _waiting=True,
        _late_notices=("x",),
        _last_reply_text="y",
        _finish_answer="z",
    )
    assert noisy == base
    assert "_queued" not in repr(noisy)


def test_a_snapshot_is_not_equal_to_other_types():
    snap = task().snapshot()
    assert snap != snap.history
    assert snap != "snapshot"
    assert snap is not None


def test_snapshots_of_histories_equal_apart_from_their_times_are_equal():
    first, second = task("Same"), task("Same")
    assert first.history == second.history  # entries ignore `at`
    assert first.snapshot() == second.snapshot()


def test_snapshot_repr_is_short():
    state = task()
    assert repr(state.snapshot()) == "<StateSnapshot turn=0 history=1 messages=1 pending=0>"
    state.finish()
    assert repr(state.snapshot()).endswith("pending=0 finished>")


# ===========================================================================
# fork()
# ===========================================================================


def test_fork_has_the_same_content_and_a_new_id():
    state = task(extra_data={"repo": "api"}, id="bug-1")
    think(state, "answer")
    fork = state.fork()

    assert fork is not state
    assert fork.id != "bug-1"
    assert re.fullmatch(r"[0-9a-f]{32}", fork.id)
    assert fork.history == state.history
    assert fork.snapshot() == state.snapshot()
    assert fork.messages == state.messages and fork.extra_data == state.extra_data
    assert fork.turn == state.turn and fork.answer == "answer"
    assert fork.fork().id != fork.id


def test_fork_is_a_fresh_root():
    state = task()
    fork = state.fork()
    assert fork.parent is None
    assert fork.root is fork
    assert fork.depth == 0


def test_fork_and_original_are_independent():
    state = task()
    fork = state.fork()

    fork.add_message(Message.user("only in the fork"))
    fork.finish("fork answer")
    with fork.edit_extra_data() as data:
        data["x"] = 1
    assert len(state.history) == 1 and state.finished is False and dict(state.extra_data) == {}

    state.add_message(Message.user("only in the original"))
    assert [m.text for m in fork.messages] == ["Task", "only in the fork"]
    assert [m.text for m in state.messages] == ["Task", "only in the original"]


def test_fork_is_not_bound_to_a_store(tmp_path):
    state = task(id="orig")
    store_a = FileStore(tmp_path / "a")
    store_b = FileStore(tmp_path / "b")
    store_a.save(state)

    fork = state.fork()
    store_b.save(fork)  # not bound to store_a
    store_a.save(state.fork())  # and a fork may go to the same store: it has its own id
    assert store_b.load(fork.id).snapshot() == fork.snapshot()
    with pytest.raises(ValueError, match="saved in another store"):
        store_b.save(state)  # the original stays bound to its store


def test_fork_is_not_running_and_has_no_agent():
    state = task()
    agent = make_agent()
    state._start_run(agent, AgentInfo(model=MODEL))
    try:
        fork = state.fork()
        assert fork._running is False
        assert fork._agent is None
        with pytest.raises(ValueError, match="already being run"):
            state._start_run(agent, AgentInfo(model=MODEL))
        fork._start_run(agent, AgentInfo(model=MODEL))  # a fork may run at the same time
        fork._end_run()
    finally:
        state._end_run()


def test_fork_in_the_middle_of_a_turn_keeps_the_pending_calls():
    @tool
    def bash(cmd: str) -> str:
        """Run."""
        return "ran"

    agent = make_agent([tool_call("bash", cmd="ls"), "done"], tools=[bash])
    state = task()
    agent.think(state)
    fork = state.fork()

    assert fork.pending_calls == state.pending_calls and fork.pending_calls
    agent.use_tools(fork)
    assert fork.pending_calls == () and state.pending_calls != ()
    agent.use_tools(state)
    assert [m.text for m in state.messages if m.role == "assistant"] == [""]
    assert_replays(fork)
    assert_replays(state)


def test_fork_while_the_model_is_being_waited_on_is_waiting_too():
    model = SlowModel()
    agent = Agent(model=model, reporter=None, human=None)
    state = task()
    thread, errors = think_in_thread(agent, state, model)
    try:
        fork = state.fork()
        assert fork.snapshot() == state.snapshot()
        assert fork.turn == 1 and fork.snapshot()._waiting
    finally:
        model.release.set()
        thread.join(timeout=5)
    assert not errors


# ===========================================================================
# restore()
# ===========================================================================


def test_restore_goes_back_to_the_snapshot():
    agent = make_agent(["first", "second"])
    state = task(extra_data={"step": 1})
    agent.think(state)
    before = state.snapshot()

    state.add_message(Message.user("more"))
    agent.think(state)
    with state.edit_extra_data() as data:
        data["step"] = 2
    state.finish("the end")

    state.restore(before)

    assert state.messages == before.messages
    assert state.turn == before.turn == 1
    assert state.extra_data == {"step": 1}
    assert state.finished is False
    assert state.answer == "first"
    assert state.stopped is None
    assert state.pending_calls == before.pending_calls == ()
    assert_replays(state)


def test_restore_keeps_all_history_and_adds_one_restore_entry():
    agent = make_agent(["first", "second"])
    state = task()
    agent.think(state)
    before = state.snapshot()
    agent.think(state)
    length = len(state.history)

    state.restore(before)

    assert state.history[:length] == before.history + state.history[len(before.history) : length]
    assert len(state.history) == length + 1
    entry = state.history[-1]
    assert isinstance(entry, ContextChangeEntry)
    assert entry.content.kind == "restore"
    assert entry.content.restored_to == len(before.history)
    assert entry.content.before_tokens > entry.content.after_tokens
    # The undone turn is still there to read.
    assert [e.content.text for e in state.history if isinstance(e, ModelReplyEntry)] == ["first", "second"]


def test_restore_keeps_usage_cumulative():
    agent = make_agent(["first", "second"])
    state = task()
    agent.think(state)
    before = state.snapshot()
    agent.think(state)
    spent = state.usage
    assert spent.requests == 2 and before.usage.requests == 1

    state.restore(before)

    assert state.usage == spent
    assert state.usage != before.usage
    # And a replay of the whole history gives the same cumulative value.
    assert State(history=state.history).usage == spent


def test_restore_reverts_the_turn_so_the_next_think_reuses_the_number():
    agent = make_agent(["first", "second", "third"])
    state = task()
    agent.think(state)
    before = state.snapshot()
    agent.think(state)
    assert state.turn == 2

    state.restore(before)
    assert state.turn == 1
    agent.think(state)
    assert state.turn == 2
    assert [m.text for m in state.messages if m.role == "assistant"] == ["first", "third"]
    requests = [e for e in state.history if isinstance(e, ModelRequestEntry)]
    assert [e.turn for e in requests] == [1, 2, 2]


def test_restore_to_the_latest_snapshot_changes_no_value_but_is_recorded():
    state = task()
    think(state, "a")
    snap = state.snapshot()

    state.restore(snap)

    assert state.messages == snap.messages and state.turn == snap.turn and state.answer == "a"
    assert len(state.history) == len(snap.history) + 1
    assert state.history[-1].content.restored_to == len(snap.history)


def test_restore_to_an_empty_snapshot():
    empty = State().snapshot()
    state = task(extra_data={"k": 1})
    think(state, "a")
    state.finish("x")

    state.restore(empty)

    assert state.messages == () and state.turn == 0 and state.finished is False
    assert dict(state.extra_data) == {} and state.answer is None
    assert len(state.history) > 4  # all of it is kept
    assert_replays(state)


def test_restore_after_finish_lets_the_state_run_again():
    agent = make_agent(["first", "second"])
    state = task()
    agent.think(state)
    before = state.snapshot()
    state.finish("done")
    with pytest.raises(ValueError):
        agent.think(state)

    state.restore(before)
    assert state.finished is False
    agent.think(state)
    assert state.answer == "second"


def test_restore_undoes_a_compaction_and_a_clear():
    _, state, _ = echo_state(3)
    before = state.snapshot()
    state.clear_tool_results(keep_last=0)
    state.compact("the summary")
    assert len(state.messages) == 2

    state.restore(before)

    assert state.messages == before.messages
    results = [b.content for m in state.messages for b in m.content if hasattr(b, "call_id")]
    assert results == ["result1", "result2", "result3"]
    assert_replays(state)


def test_restore_reverts_stopped():
    state = task()
    before = state.snapshot()
    state._record_stop(StoppedByLimit(3))
    assert state.stopped == StoppedByLimit(3)
    state.restore(before)
    assert state.stopped is None


def test_restore_can_go_to_any_earlier_snapshot_in_any_order():
    state = task()
    s0 = state.snapshot()
    think(state, "a")
    s1 = state.snapshot()
    think(state, "b")
    s2 = state.snapshot()

    state.restore(s1)
    assert state.turn == 1 and state.answer == "a"
    state.restore(s2)  # taken before the first restore: still a prefix
    assert state.turn == 2 and state.answer == "b"
    assert [m.text for m in state.messages] == ["Task", "a", "b"]
    state.restore(s0)
    assert [m.text for m in state.messages] == ["Task"] and state.turn == 0
    assert_replays(state)

    after_restores = state.snapshot()
    state.restore(s1)
    state.restore(after_restores)  # a snapshot taken after restores is a prefix too
    assert state.turn == 0 and [m.text for m in state.messages] == ["Task"]
    assert_replays(state)


def test_many_restores_replay_quickly():
    state = task()
    think(state, "a")
    snap = state.snapshot()
    for _ in range(60):
        state.restore(snap)
        state.add_message(Message.user("again"))
    assert_replays(state)


def test_restore_accepts_a_snapshot_with_an_equal_history_from_another_state():
    state = task()
    think(state, "a")
    fork = state.fork()
    snap = fork.snapshot()
    think(state, "b")

    state.restore(snap)  # fork's history is a prefix of state's (==)
    assert state.turn == 1 and state.answer == "a"


def test_restore_refuses_a_snapshot_of_another_state():
    state = task("one")
    other = task("two")
    think(other, "x")
    with pytest.raises(ValueError, match="restore takes a snapshot taken from this State") as info:
        state.restore(other.snapshot())
    assert "State(history=snapshot.history)" in str(info.value)
    assert len(state.history) == 1


def test_restore_refuses_a_snapshot_longer_than_the_history():
    state = task()
    fork = state.fork()
    think(fork, "more")
    with pytest.raises(ValueError, match="taken from this State"):
        state.restore(fork.snapshot())


def test_restore_refuses_what_is_not_a_snapshot():
    state = task()
    with pytest.raises(TypeError, match="takes a StateSnapshot"):
        state.restore(state.history)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        state.restore(None)  # type: ignore[arg-type]


def test_restore_refuses_pending_calls_and_records_nothing():
    state = task()
    before = state.snapshot()
    think(state, call())
    length = len(state.history)

    with pytest.raises(ValueError, match="calls are pending"):
        state.restore(before)
    assert len(state.history) == length and state.pending_calls


def test_restore_refuses_while_the_model_is_being_waited_on():
    model = SlowModel()
    agent = Agent(model=model, reporter=None, human=None)
    state = task()
    before = state.snapshot()
    thread, errors = think_in_thread(agent, state, model)
    try:
        with pytest.raises(ValueError, match="the model is being waited on"):
            state.restore(before)
    finally:
        model.release.set()
        thread.join(timeout=5)
    assert not errors
    assert state.answer == "slow answer"  # the think went on untouched
    state.restore(before)  # and now it is allowed


def test_restore_refuses_a_snapshot_taken_while_the_model_was_being_waited_on():
    model = SlowModel()
    agent = Agent(model=model, reporter=None, human=None)
    state = task()
    thread, errors = think_in_thread(agent, state, model)
    try:
        during = state.snapshot()
    finally:
        model.release.set()
        thread.join(timeout=5)
    assert not errors

    with pytest.raises(ValueError, match="taken while the model was being waited on"):
        state.restore(during)


def test_restore_refuses_while_a_compaction_is_under_way():
    state = task()
    before = state.snapshot()
    state._begin_compact()
    try:
        with pytest.raises(ValueError, match="a compaction is under way"):
            state.restore(before)
    finally:
        state._end_compact()
    state.restore(before)


def test_restore_to_a_snapshot_with_pending_calls_brings_them_back():
    ran = []

    @tool
    def bash(cmd: str) -> str:
        """Run."""
        ran.append(cmd)
        return f"ran {cmd}"

    agent = make_agent([tool_call("bash", cmd="ls"), "done"], tools=[bash])
    state = task()
    agent.think(state)
    mid_turn = state.snapshot()
    assert mid_turn.pending_calls
    agent.use_tools(state)
    assert state.pending_calls == ()

    state.restore(mid_turn)

    assert state.pending_calls == mid_turn.pending_calls
    assert state.messages == mid_turn.messages
    assert_replays(state)
    agent.use_tools(state)  # the call runs again; the model gets exactly one result for it
    assert ran == ["ls", "ls"]
    assert state.pending_calls == ()
    results = [b for m in state.messages for b in m.content if hasattr(b, "call_id")]
    assert [r.content for r in results] == ["ran ls"]
    assert_replays(state)


def test_restore_in_the_middle_of_parallel_results():
    state = task()
    think(state, call("a", "c1"), call("b", "c2"))
    state._tool_result(state.pending_calls[0], "ra", ToolOutcomeKind.DONE)
    partial = state.snapshot()
    assert [c.id for c in partial.pending_calls] == ["c2"]
    state._tool_result(state.pending_calls[0], "rb", ToolOutcomeKind.DONE)
    assert state.pending_calls == ()

    state.restore(partial)

    assert [c.id for c in state.pending_calls] == ["c2"]
    assert_replays(state)
    state._tool_result(state.pending_calls[0], "rb again", ToolOutcomeKind.DONE)
    last = state.messages[-1]
    assert [b.content for b in last.content] == ["ra", "rb again"]


def test_restore_moves_updated_at_but_not_created_at():
    state = task()
    created = state.created_at
    before = state.snapshot()
    think(state, "a")
    state.restore(before)
    assert state.created_at == created
    assert state.updated_at == state.history[-1].at
    assert state.updated_at >= before.updated_at


def test_restore_notifies_the_reporter_of_the_linked_agent():
    changes = []

    class Spy(Reporter):
        def on_context_change(self, state, change):
            changes.append(change)

    agent = make_agent(["first"], reporter=Spy())
    state = task()
    agent.think(state)  # links the agent to the State
    before = state.snapshot()
    state.add_message(Message.user("more"))
    state.restore(before)

    assert [c.kind for c in changes] == ["restore"]
    assert changes[0].restored_to == len(before.history)


def test_a_restored_state_runs_with_any_agent():
    state = task()
    before = state.snapshot()
    make_agent(["first"]).run(state)
    state.restore(before)
    assert make_agent(["again"]).run(state) == "again"
    assert [m.text for m in state.messages] == ["Task", "again"]
    assert_replays(state)


# ===========================================================================
# edit_extra_data()
# ===========================================================================


def test_edit_extra_data_gives_an_editable_deep_copy():
    state = task(extra_data={"notes": ["a"], "cfg": {"depth": 1}})
    with state.edit_extra_data() as data:
        assert type(data) is dict
        assert type(data["notes"]) is list
        assert type(data["cfg"]) is dict
        assert data == {"notes": ["a"], "cfg": {"depth": 1}}
        data["notes"].append("b")
        # Nothing is recorded before the block ends.
        assert state.extra_data["notes"] == ["a"]
        assert len(state.history) == 2
    assert state.extra_data["notes"] == ["a", "b"]


def test_edit_extra_data_deep_edits_record_the_changed_top_level_key():
    state = task(extra_data={"notes": {"bug": "a.py"}, "other": 1})
    with state.edit_extra_data() as data:
        data.setdefault("notes", {})["bug"] = "b.py"
        data["notes"]["extra"] = ["x"]

    [entry] = [e for e in state.history if isinstance(e, ExtraDataEntry)][1:]
    assert entry.content == {"notes": {"bug": "b.py", "extra": ["x"]}}  # the whole new value of the changed key
    assert "other" not in entry.content
    assert entry.removed == ()
    assert entry.turn == state.turn
    assert state.extra_data == {"notes": {"bug": "b.py", "extra": ["x"]}, "other": 1}
    assert_replays(state)


def test_edit_extra_data_adds_new_keys():
    state = task()
    with state.edit_extra_data() as data:
        data.setdefault("notes", {})["bug"] = "b.py"
        data["count"] = 3
    assert state.extra_data == {"notes": {"bug": "b.py"}, "count": 3}
    assert isinstance(state.extra_data["notes"], FrozenDict)
    assert_replays(state)


def test_edit_extra_data_deletes_are_recorded_as_removed():
    state = task(extra_data={"a": 1, "b": 2, "c": 3})
    with state.edit_extra_data() as data:
        del data["a"]
        data.pop("c")

    entry = state.history[-1]
    assert isinstance(entry, ExtraDataEntry)
    assert entry.content == {}
    assert entry.removed == ("a", "c")
    assert state.extra_data == {"b": 2}
    assert "a" not in state.extra_data
    assert_replays(state)


def test_edit_extra_data_delete_and_set_in_one_block():
    state = task(extra_data={"a": 1, "b": 2})
    with state.edit_extra_data() as data:
        del data["a"]
        data["b"] = 3
        data["c"] = 4

    entry = state.history[-1]
    assert entry.content == {"b": 3, "c": 4} and entry.removed == ("a",)
    assert state.extra_data == {"b": 3, "c": 4}
    assert_replays(state)


def test_edit_extra_data_clearing_everything():
    state = task(extra_data={"a": 1, "b": 2})
    with state.edit_extra_data() as data:
        data.clear()
    assert dict(state.extra_data) == {}
    assert state.history[-1].removed == ("a", "b")
    assert_replays(state)


def test_edit_extra_data_without_a_change_records_nothing():
    state = task(extra_data={"a": 1, "notes": ["x"], "cfg": {"p": 1, "q": 2}})
    length = len(state.history)
    updated = state.updated_at

    with state.edit_extra_data():
        pass
    with state.edit_extra_data() as data:
        data["a"] = 1  # the same value
        data["notes"].append("y")
        data["notes"].pop()  # back to the same
        data["cfg"] = {"q": 2, "p": 1}  # the same dict in another key order
        data["gone"] = 1
        del data["gone"]  # added and deleted again

    assert len(state.history) == length
    assert state.updated_at == updated


def test_edit_extra_data_tells_one_from_one_point_zero_and_true():
    state = task(extra_data={"a": 1})
    length = len(state.history)
    with state.edit_extra_data() as data:
        data["a"] = 1.0
    assert len(state.history) == length + 1
    with state.edit_extra_data() as data:
        data["a"] = True
    assert len(state.history) == length + 2
    assert state.extra_data["a"] is True


def test_edit_extra_data_accepts_tuples_as_lists():
    state = task()
    with state.edit_extra_data() as data:
        data["pair"] = (1, 2)
    assert state.extra_data["pair"] == [1, 2]
    assert isinstance(state.extra_data["pair"], FrozenList)


def test_edit_extra_data_values_are_frozen_once_recorded():
    state = task()
    with state.edit_extra_data() as data:
        data["notes"] = ["a"]
        local = data["notes"]
    local.append("b")  # the plain copy is the caller's; the State's value did not change
    assert state.extra_data["notes"] == ["a"]
    with pytest.raises(TypeError):
        state.extra_data["notes"].append("c")
    with pytest.raises(TypeError, match="edit_extra_data"):
        state.extra_data["x"] = 1


def test_edit_extra_data_earlier_snapshots_are_unaffected():
    state = task(extra_data={"a": 1})
    before = state.snapshot()
    with state.edit_extra_data() as data:
        data["a"] = 2
    assert before.extra_data == {"a": 1}
    assert state.extra_data == {"a": 2}


def test_edit_extra_data_must_end_with_json_and_names_the_key():
    state = task()
    with pytest.raises(
        TypeError,
        match=re.escape("state.extra_data['seen'] must be JSON (got set); use a list: sorted(seen)"),
    ):
        with state.edit_extra_data() as data:
            data["seen"] = {"a.py"}
    assert len(state.history) == 1 and dict(state.extra_data) == {}


@pytest.mark.parametrize(
    "value, fragment",
    [
        ({"deep": [1, {2}]}, re.escape("state.extra_data['k']['deep'][1] must be JSON (got set)")),
        (b"raw", "use text: base64"),
        (object(), r"state\.extra_data\['k'\] must be JSON \(got object\)"),
        (float("nan"), r"state\.extra_data\['k'\] must be JSON"),
        ({1: "x"}, "its key 1 is a int"),
    ],
)
def test_edit_extra_data_json_errors(value, fragment):
    state = task()
    with pytest.raises(TypeError, match=fragment):
        with state.edit_extra_data() as data:
            data["k"] = value
    assert dict(state.extra_data) == {}


def test_edit_extra_data_a_failed_validation_changes_nothing_and_frees_the_lock():
    state = task(extra_data={"keep": 1})
    with pytest.raises(TypeError):
        with state.edit_extra_data() as data:
            data["keep"] = 2
            data["bad"] = {1}
    assert state.extra_data == {"keep": 1}

    done = threading.Event()

    def later():
        with state.edit_extra_data() as data:
            data["after"] = True
        done.set()

    thread = threading.Thread(target=later)
    thread.start()
    assert done.wait(timeout=5)  # the lock was released
    thread.join()
    assert state.extra_data == {"keep": 1, "after": True}


def test_edit_extra_data_an_exception_in_the_block_changes_nothing():
    state = task(extra_data={"keep": 1})
    length = len(state.history)

    with pytest.raises(RuntimeError, match="boom"):
        with state.edit_extra_data() as data:
            data["keep"] = 2
            data["new"] = 3
            raise RuntimeError("boom")

    assert state.extra_data == {"keep": 1}
    assert len(state.history) == length
    with state.edit_extra_data() as data:  # and the lock is free again
        data["ok"] = 1
    assert state.extra_data == {"keep": 1, "ok": 1}


def test_edit_extra_data_keyboard_interrupt_in_the_block_changes_nothing():
    state = task()
    with pytest.raises(KeyboardInterrupt):
        with state.edit_extra_data() as data:
            data["x"] = 1
            raise KeyboardInterrupt
    assert dict(state.extra_data) == {}


def test_concurrent_edits_serialize_so_no_update_is_lost():
    state = task()
    threads_count, steps = 8, 100

    def worker():
        for _ in range(steps):
            with state.edit_extra_data() as data:
                data["n"] = data.get("n", 0) + 1

    threads = [threading.Thread(target=worker) for _ in range(threads_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert state.extra_data["n"] == threads_count * steps
    entries = [e for e in state.history if isinstance(e, ExtraDataEntry)]
    # One entry per edit, and the values count up one by one: the blocks ran one after the other.
    assert [e.content["n"] for e in entries] == list(range(1, threads_count * steps + 1))
    assert_replays(state)


def test_a_second_edit_waits_for_the_first_but_other_commands_do_not():
    state = task()
    inside, release, second_in = threading.Event(), threading.Event(), threading.Event()

    def first():
        with state.edit_extra_data() as data:
            data["a"] = 1
            inside.set()
            assert release.wait(timeout=5)

    def second():
        with state.edit_extra_data() as data:
            second_in.set()
            data["b"] = 2

    t1 = threading.Thread(target=first)
    t1.start()
    assert inside.wait(timeout=5)
    t2 = threading.Thread(target=second)
    t2.start()

    assert not second_in.wait(timeout=0.2)  # blocked by the first edit
    state.add_message(Message.user("not blocked"))  # the edit lock is not the State lock
    state.finish("not blocked either")
    release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert second_in.is_set()
    assert dict(state.extra_data) == {"a": 1, "b": 2}
    assert state.finished
    assert_replays(state)


def test_edit_extra_data_from_a_tool_while_the_agent_runs():
    @tool
    def note(state: State, text: str) -> str:
        """Take a note."""
        with state.edit_extra_data() as data:
            data.setdefault("notes", []).append(text)
        return "noted"

    agent = make_agent(
        [[tool_call("note", text="a"), tool_call("note", text="b"), tool_call("note", text="c")], "done"],
        tools=[note],
    )
    state = task()
    agent.run(state)
    assert sorted(state.extra_data["notes"]) == ["a", "b", "c"]
    assert_replays(state)


def test_edit_extra_data_is_undone_by_restore():
    state = task(extra_data={"a": 1})
    before = state.snapshot()
    with state.edit_extra_data() as data:
        data["a"] = 2
        data["b"] = 1
    state.restore(before)
    assert state.extra_data == {"a": 1}
    with state.edit_extra_data() as data:  # an edit after a restore is a patch on the restored value
        data["c"] = 3
    assert state.extra_data == {"a": 1, "c": 3}
    assert_replays(state)


# ===========================================================================
# add_message()
# ===========================================================================


def test_add_message_user_and_notice_kinds():
    state = task()
    state.add_message(Message.user("Now fix it"))
    state.add_message(Message.notice("The tests failed"))

    user, notice = state.history[-2:]
    assert isinstance(user, MessageEntry) and isinstance(notice, MessageEntry)
    assert (user.kind, user.content) == ("user", "Now fix it")
    assert (notice.kind, notice.content) == ("notice", "[notice] The tests failed")
    assert user.turn == notice.turn == 0
    assert [m.text for m in state.messages] == ["Task", "Now fix it", "[notice] The tests failed"]
    assert all(m.role == "user" for m in state.messages)


def test_add_message_kind_follows_the_text_prefix():
    state = task()
    state.add_message(Message.user("[notice] typed by hand"))
    state.add_message(Message.user("see [notice] later"))
    assert [e.kind for e in state.history[-2:]] == ["notice", "user"]


def test_add_message_records_the_turn_it_was_added_in():
    state = task()
    think(state, "a")
    state.add_message(Message.user("after turn one"))
    assert state.history[-1].turn == 1


def test_add_message_with_images():
    image = Image(PNG)
    state = task()
    state.add_message(Message.user("What is wrong in this chart?", image))
    state.add_message(Message.user("", image))  # an image alone is a message

    with_text, alone = state.history[-2:]
    assert with_text.kind == "user" and alone.kind == "user"
    assert with_text.content == (TextBlock("What is wrong in this chart?"), image)
    assert alone.content == (image,)
    assert state.messages[-2].content == (TextBlock("What is wrong in this chart?"), image)
    assert state.messages[-1].content == (image,)
    assert_replays(state)


def test_add_message_text_only_content_is_a_str():
    state = task()
    state.add_message(Message.user("plain"))
    assert state.history[-1].content == "plain"
    assert isinstance(state.history[-1].content, str)


def test_add_message_a_message_with_images_survives_a_store(tmp_path):
    store = FileStore(tmp_path)
    state = task(id="img")
    state.add_message(Message.user("Look", Image(PNG)))
    store.save(state)
    loaded = store.load("img")
    assert loaded.messages == state.messages
    assert loaded.snapshot() == state.snapshot()


def test_add_message_rejects_what_is_not_a_message():
    state = task()
    with pytest.raises(TypeError, match=r"(?s)takes a Message.*Message\.user"):
        state.add_message("plain text")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        state.add_message(None)  # type: ignore[arg-type]
    assert len(state.history) == 1


def test_add_message_rejects_assistant_messages():
    state = task()
    with pytest.raises(ValueError, match=r"(?s)user messages only.*State\(messages="):
        state.add_message(Message.assistant("from the model"))
    assert len(state.history) == 1 and len(state.messages) == 1


@pytest.mark.parametrize("text", ["", " ", "\n\t "])
def test_add_message_rejects_empty_text_without_an_image(text):
    state = task()
    with pytest.raises(ValueError, match="no text and no image"):
        state.add_message(Message.user(text))
    assert len(state.history) == 1


def test_add_message_rejects_messages_with_other_blocks():
    state = task()
    only_call = Message("user", (call(),))
    with pytest.raises(ValueError):
        state.add_message(only_call)
    mixed = Message("user", (TextBlock("hello"), call()))
    with pytest.raises(ValueError, match="text and images only"):
        state.add_message(mixed)
    assert len(state.history) == 1


def test_add_message_goes_straight_in_when_idle():
    state = task()
    state.add_message(Message.user("one"))
    think(state, "reply")
    state.add_message(Message.user("two"))
    assert [m.text for m in state.messages] == ["Task", "one", "reply", "two"]
    assert state.snapshot()._queued == ()


def test_add_message_is_held_while_the_model_is_being_waited_on():
    state = task()
    state._begin_think(MODEL)
    state.add_message(Message.user("typed while thinking"))
    state.add_message(Message.notice("and a notice"))

    # In history right away, in the messages only after the reply.
    assert [e.kind for e in state.history[-2:]] == ["user", "notice"]
    assert [m.text for m in state.messages] == ["Task"]
    assert len(state.snapshot()._queued) == 2

    state._record_reply(reply("the answer"))
    assert [m.text for m in state.messages] == ["Task", "the answer", "typed while thinking", "[notice] and a notice"]
    assert state.snapshot()._queued == ()
    assert_replays(state)


def test_add_message_is_held_while_calls_are_pending_in_arrival_order():
    state = task()
    think(state, call("a", "c1"), call("b", "c2"))
    state.add_message(Message.user("first"))
    state.add_message(Message.notice("second"))
    state.add_message(Message.user("third", Image(PNG)))
    assert [m.role for m in state.messages] == ["user", "assistant"]

    state._tool_result(state.pending_calls[1], "rb", ToolOutcomeKind.DONE)  # out of order
    state._tool_result(state.pending_calls[0], "ra", ToolOutcomeKind.DONE)

    assert [b.content for b in state.messages[-4].content] == ["ra", "rb"]  # in request order
    assert [m.text for m in state.messages[-3:]] == ["first", "[notice] second", "third"]
    assert state.messages[-1].content[-1] == Image(PNG)
    assert_replays(state)


def test_add_message_held_during_a_think_with_tool_calls_goes_after_the_results():
    state = task()
    state._begin_think(MODEL)
    state.add_message(Message.user("mid-think"))
    state._record_reply(reply(call()))
    assert state.messages[-1].role == "assistant"  # still held: calls are pending now
    state.add_message(Message.user("mid-turn"))
    state._tool_result(state.pending_calls[0], "out", ToolOutcomeKind.DONE)

    assert [m.text for m in state.messages][-2:] == ["mid-think", "mid-turn"]
    assert state.messages[-3].content[0].content == "out"
    assert_replays(state)


def test_add_message_held_during_a_failed_think_goes_in_after_the_rollback():
    state = task()
    state._begin_think(MODEL)
    state.add_message(Message.user("typed while thinking"))
    state._record_error(TimeoutError("timed out"))

    assert state.turn == 0
    assert [m.text for m in state.messages] == ["Task", "typed while thinking"]
    assert state.snapshot()._queued == ()
    assert_replays(state)


def test_a_tool_error_does_not_release_held_messages():
    state = task()
    think(state, call())
    state.add_message(Message.user("held"))
    state._record_error(RuntimeError("tool failed"), call=state.pending_calls[0])
    assert [m.text for m in state.messages] == ["Task", ""]  # the held message waits for the result
    state._tool_result(state.pending_calls[0], "failed", ToolOutcomeKind.ERROR)
    assert state.messages[-1].text == "held"


def test_add_message_held_during_a_reply_with_text_goes_right_after_it():
    state = task()
    state._begin_think(MODEL)
    state.add_message(Message.user("q2"))
    state._record_reply(reply("a1"))
    state.add_message(Message.user("q3"))
    assert [m.text for m in state.messages] == ["Task", "a1", "q2", "q3"]


def test_add_message_first_user_message_is_what_compact_keeps():
    state = State()
    state.add_message(Message.notice("not a user message"))
    state.add_message(Message.user("the task"))
    state.add_message(Message.user("later"))
    state.compact("summary")
    assert [m.text for m in state.messages] == ["the task", "[notice] Summary so far:\nsummary"]


def test_add_message_from_many_threads_records_every_message():
    state = task()

    def worker(n):
        for i in range(50):
            state.add_message(Message.user(f"w{n}-{i}"))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(state.history) == 401 and len(state.messages) == 401
    for n in range(8):  # each thread's messages stay in the order it added them
        mine = [m.text for m in state.messages if m.text.startswith(f"w{n}-")]
        assert mine == [f"w{n}-{i}" for i in range(50)]
    assert_replays(state)


# ===========================================================================
# compact(summary)
# ===========================================================================


def test_compact_keeps_the_first_user_message_and_the_summary():
    state = State(messages=[Message.user("Fix calc.py"), Message.assistant("ok"), Message.user("and add tests")])
    state.compact("what I know so far")

    assert [m.text for m in state.messages] == ["Fix calc.py", "[notice] Summary so far:\nwhat I know so far"]
    assert all(m.role == "user" for m in state.messages)
    entry = state.history[-1]
    assert isinstance(entry, ContextChangeEntry)
    change = entry.content
    assert change.kind == "compact"
    assert change.summary == "what I know so far"
    assert change.kept == 0
    assert change.usage is None
    assert change.before_tokens > 0 and change.after_tokens > 0
    assert_replays(state)


def test_compact_records_token_estimates_of_before_and_after():
    state = State(messages=[Message.user("x" * 8000), Message.assistant("y" * 8000), Message.user("z" * 8000)])
    state.compact("short")
    change = state.history[-1].content
    assert change.before_tokens > change.after_tokens > 0
    assert change.before_tokens > 4000


def test_compact_in_an_empty_state_leaves_only_the_summary():
    state = State()
    state.compact("only the summary")
    assert [m.text for m in state.messages] == ["[notice] Summary so far:\nonly the summary"]
    assert_replays(state)


def test_compact_again_replaces_the_old_summary():
    state = task("The task")
    state.compact("first summary")
    state.add_message(Message.user("more work"))
    state.compact("second summary")
    assert [m.text for m in state.messages] == ["The task", "[notice] Summary so far:\nsecond summary"]
    assert [e.content.kind for e in state.history if isinstance(e, ContextChangeEntry)] == [
        "import",
        "compact",
        "compact",
    ]
    assert_replays(state)


def test_compact_does_not_change_history_turn_or_usage():
    agent = make_agent(["answer"])
    state = task()
    agent.think(state)
    history = state.history
    turn, usage = state.turn, state.usage

    state.compact("summary")

    assert state.history[: len(history)] == history
    assert len(state.history) == len(history) + 1
    assert (state.turn, state.usage) == (turn, usage)
    assert state.answer == "answer"


def test_compact_keeps_the_messages_added_while_a_model_wrote_the_summary():
    state = task()
    state.add_message(Message.user("q1"))
    state._begin_compact()
    state.add_message(Message.user("q2 while compacting"))
    state.add_message(Message.notice("q3 while compacting"))
    spent = Usage(input_tokens=50, output_tokens=7, requests=1)
    state._compact_done("the summary", spent)
    state._end_compact()

    assert [m.text for m in state.messages] == [
        "Task",
        "[notice] Summary so far:\nthe summary",
        "q2 while compacting",
        "[notice] q3 while compacting",
    ]
    change = state.history[-1].content
    assert change.kept == 2
    assert change.usage == spent
    assert state.usage == spent  # the compaction request is counted
    assert_replays(state)  # fold keeps exactly the last `kept` messages


def test_compact_does_not_count_held_messages_as_kept():
    state = task()
    state._begin_compact()
    state._begin_think(MODEL)  # a think waiting on the model holds new messages
    state.add_message(Message.user("held"))
    state._compact_done("summary", None)
    state._end_compact()

    assert state.history[-1].content.kept == 0
    assert [m.text for m in state.messages] == ["Task", "[notice] Summary so far:\nsummary"]
    state._record_reply(reply("answer"))
    assert [m.text for m in state.messages][-2:] == ["answer", "held"]
    assert_replays(state)


def test_compact_by_hand_during_a_model_compaction_keeps_only_later_messages():
    state = task()
    state._begin_compact()
    state.add_message(Message.user("before the hand-written summary"))
    state.compact("hand-written")
    state.add_message(Message.user("after it"))
    state._compact_done("model summary", None)
    state._end_compact()

    assert [m.text for m in state.messages] == [
        "Task",
        "[notice] Summary so far:\nmodel summary",
        "after it",
    ]
    assert_replays(state)


def test_compact_second_begin_while_one_is_under_way_is_refused():
    state = task()
    state._begin_compact()
    try:
        with pytest.raises(ValueError, match="another compaction"):
            state._begin_compact()
    finally:
        state._end_compact()
    state._end_compact()  # idempotent
    state._begin_compact()
    state._end_compact()


def test_agent_compact_records_usage_and_kept():
    model = SlowModel("the model's summary")
    agent = Agent(model=model, reporter=None, human=None)
    state = task()
    state.add_message(Message.user("q1"))
    errors = []

    def run():
        try:
            agent.compact(state)
        except BaseException as e:  # pragma: no cover
            errors.append(e)

    thread = threading.Thread(target=run)
    thread.start()
    assert model.entered.wait(timeout=5)
    state.add_message(Message.user("typed during compaction"))
    model.release.set()
    thread.join(timeout=5)

    assert not errors
    assert [m.text for m in state.messages] == [
        "Task",
        "[notice] Summary so far:\nthe model's summary",
        "typed during compaction",
    ]
    change = state.history[-1].content
    assert change.kept == 1 and change.usage is not None and change.usage.requests == 1
    assert state.usage.requests == 1
    assert_replays(state)


@pytest.mark.parametrize("bad", ["", "   ", "\n"])
def test_compact_rejects_empty_summaries(bad):
    state = task()
    with pytest.raises(ValueError, match="empty summary"):
        state.compact(bad)
    assert len(state.history) == 1


@pytest.mark.parametrize("bad", [None, 3, b"bytes", ["a"]])
def test_compact_rejects_non_strings(bad):
    state = task()
    with pytest.raises(TypeError, match="needs summary to be a string"):
        state.compact(bad)  # type: ignore[arg-type]
    assert len(state.history) == 1


def test_compact_is_refused_while_calls_are_pending():
    state = task()
    think(state, call())
    length = len(state.history)
    with pytest.raises(ValueError, match="calls are pending"):
        state.compact("summary")
    assert len(state.history) == length


def test_compact_works_while_the_model_is_being_waited_on():
    # Allowed as before: the summary replaces the messages the request was made from.
    state = task()
    state._begin_think(MODEL)
    state.compact("summary")
    state._record_reply(reply("answer"))
    assert [m.text for m in state.messages] == ["Task", "[notice] Summary so far:\nsummary", "answer"]
    assert_replays(state)


# ===========================================================================
# clear_tool_results()
# ===========================================================================


def test_clear_tool_results_records_exactly_the_cleared_ids():
    _, state, calls = echo_state(4)
    state.clear_tool_results(keep_last=1)

    change = state.history[-1].content
    assert change.kind == "clear_tool_results"
    assert change.cleared == tuple(c.id for c in calls[:3])
    results = [b.content for m in state.messages for b in m.content if hasattr(b, "call_id")]
    assert results == ["(cleared: kept in history)"] * 3 + ["result4"]
    assert_replays(state)


def test_clear_tool_results_records_token_estimates_of_big_results():
    @tool
    def big(x: int) -> str:
        """Big result."""
        return "x" * 8000

    agent = make_agent([tool_call("big", x=1), tool_call("big", x=2)], tools=[big])
    state = task()
    for _ in range(2):
        agent.think(state)
        agent.use_tools(state)
    state.clear_tool_results(keep_last=1)
    change = state.history[-1].content
    assert change.before_tokens > change.after_tokens > 0
    assert change.before_tokens - change.after_tokens > 1000


def test_clear_tool_results_the_default_keeps_the_last_five():
    _, state, calls = echo_state(7)
    state.clear_tool_results()
    assert state.history[-1].content.cleared == tuple(c.id for c in calls[:2])


def test_clear_tool_results_counts_already_cleared_results_toward_keep_last():
    _, state, calls = echo_state(4)
    state.clear_tool_results(keep_last=3)
    assert state.history[-1].content.cleared == (calls[0].id,)
    length = len(state.history)

    state.clear_tool_results(keep_last=3)  # the first one is cleared already: nothing new
    assert len(state.history) == length

    state.clear_tool_results(keep_last=2)  # only the second one is new
    assert state.history[-1].content.cleared == (calls[1].id,)
    assert_replays(state)


def test_clear_tool_results_records_nothing_when_nothing_changes():
    _, state, _ = echo_state(2)
    length = len(state.history)
    state.clear_tool_results(keep_last=2)
    state.clear_tool_results(keep_last=10)
    assert len(state.history) == length
    assert State().snapshot() == State().snapshot()
    empty = State()
    empty.clear_tool_results(keep_last=0)
    assert empty.history == ()


def test_clear_tool_results_drops_the_token_anchors():
    _, state, _ = echo_state(2)
    assert any(m.tokens is not None for m in state.messages)
    state.clear_tool_results(keep_last=0)
    assert all(m.tokens is None for m in state.messages)
    assert_replays(state)


def test_clear_tool_results_only_blanks_tool_results():
    _, state, _ = echo_state(2)
    state.clear_tool_results(keep_last=0)
    assert [m.text for m in state.messages if m.role == "user" and m.text] == ["Task"]
    assert [m.role for m in state.messages] == ["user", "assistant", "user", "assistant", "user"]


def test_clear_tool_results_one_message_with_several_results():
    state = task()
    think(state, call("a", "c1"), call("b", "c2"), call("c", "c3"))
    for pending, text in zip(list(state.pending_calls), ["ra", "rb", "rc"], strict=True):
        state._tool_result(pending, text, ToolOutcomeKind.DONE)

    state.clear_tool_results(keep_last=1)

    assert [b.content for b in state.messages[-1].content] == [
        "(cleared: kept in history)",
        "(cleared: kept in history)",
        "rc",
    ]
    assert state.history[-1].content.cleared == ("c1", "c2")
    assert [e.content for e in state.history if isinstance(e, ToolResultEntry)] == ["ra", "rb", "rc"]
    assert_replays(state)


@pytest.mark.parametrize("bad", [-1, 1.5, "5", None, True])
def test_clear_tool_results_checks_keep_last(bad):
    state = task()
    with pytest.raises(ValueError, match="keep_last must be an integer >= 0"):
        state.clear_tool_results(keep_last=bad)  # type: ignore[arg-type]


def test_clear_tool_results_is_refused_while_calls_are_pending():
    state = task()
    think(state, call())
    with pytest.raises(ValueError, match="calls are pending"):
        state.clear_tool_results()


def test_clear_tool_results_is_undone_by_restore_and_survives_compact_order():
    _, state, _ = echo_state(3)
    state.clear_tool_results(keep_last=0)
    cleared = state.snapshot()
    state.compact("summary")
    state.restore(cleared)
    results = [b.content for m in state.messages for b in m.content if hasattr(b, "call_id")]
    assert results == ["(cleared: kept in history)"] * 3
    assert_replays(state)


# ===========================================================================
# The invariant: State(history=s.history).snapshot() == s.snapshot() after every step
# ===========================================================================


def test_invariant_after_every_step_of_a_scripted_scenario():
    """Drives the State with its commands and the internal helpers the Agent uses, checking the invariant after each
    step (including the steps that leave the State in the middle of a turn)."""
    agent = make_agent()
    info = AgentInfo(model=MODEL, name="scripted", tools=("bash",))
    c1, c2, c3, c4, c5 = (call("bash", f"c{i}", cmd=str(i)) for i in range(1, 6))
    kept: dict[str, StateSnapshot] = {}
    state = State(messages=[Message.user("Fix calc.py"), Message.assistant("Which test?")], extra_data={"repo": "api"})
    steps: list[tuple[str, object]] = []

    def step(label):
        def register(fn):
            steps.append((label, fn))
            return fn

        return register

    @step("add user message")
    def _():
        state.add_message(Message.user("test_add"))

    @step("add notice")
    def _():
        state.add_message(Message.notice("The tests failed"))

    @step("add image message")
    def _():
        state.add_message(Message.user("Look", Image(PNG)))

    @step("edit extra data")
    def _():
        with state.edit_extra_data() as data:
            data["notes"] = {"bug": "calc.py"}
            data["tmp"] = 1

    @step("delete extra data")
    def _():
        with state.edit_extra_data() as data:
            del data["tmp"]

    @step("failed edit changes nothing")
    def _():
        with pytest.raises(RuntimeError):
            with state.edit_extra_data() as data:
                data["x"] = 1
                raise RuntimeError

    @step("take snapshot s0")
    def _():
        kept["s0"] = state.snapshot()

    @step("run start")
    def _():
        state._start_run(agent, info)

    @step("think: request")
    def _():
        state._begin_think(MODEL)

    @step("message while waiting")
    def _():
        state.add_message(Message.user("typed while thinking"))

    @step("model event")
    def _():
        from alpineagents.types import ModelEvent

        state._record_model_event(ModelEvent("retry", "retrying", {"after": 1}))

    @step("think: reply with three calls")
    def _():
        state._record_reply(reply(c1, c2, c3))

    @step("message while pending")
    def _():
        state.add_message(Message.notice("while calls are pending"))

    @step("ask")
    def _():
        state._record_ask("What now?", {"answer": [1, 2]}, Usage(input_tokens=3, output_tokens=1, requests=1))

    @step("human")
    def _():
        state._record_human("Proceed?", "yes")

    @step("result for the last call first")
    def _():
        state._tool_result(c3, "r3", ToolOutcomeKind.DONE)

    @step("error result")
    def _():
        state._tool_result(c1, "boom", ToolOutcomeKind.ERROR, error=RuntimeError("boom"))

    @step("tool error entry")
    def _():
        state._record_error(RuntimeError("tool"), call=c2)

    @step("last result closes the turn")
    def _():
        state._tool_result(c2, "r2", ToolOutcomeKind.DONE)

    @step("late result of a closed call")
    def _():
        state._late_result(c1, "finished later", ToolOutcomeKind.DONE)

    @step("take snapshot s1")
    def _():
        kept["s1"] = state.snapshot()

    @step("think: request (late notice enters)")
    def _():
        state._begin_think(MODEL)

    @step("think fails: the turn is taken back")
    def _():
        state._record_error(TimeoutError("timed out"))

    @step("think again: reply with a call")
    def _():
        think(state, c4)

    @step("permission stop")
    def _():
        assert [c.id for c in state._stop_turn(c4, "no way", "DenyAll()")] == []

    @step("think: reply with a call, closed by an interrupt")
    def _():
        think(state, c5)
        assert [c.id for c in state._close_pending(KeyboardInterrupt())] == ["c5"]

    @step("loop stop and clear")
    def _():
        state._record_stop(StoppedByUntil("done"))
        state._record_stop(StoppedByLimit(3))
        state._record_stop(None)

    @step("final reply")
    def _():
        think(state, "all done")

    @step("run end")
    def _():
        state._end_run()

    @step("clear tool results")
    def _():
        state.clear_tool_results(keep_last=1)

    @step("compact")
    def _():
        state.compact("summary so far")

    @step("compaction in flight keeps a message")
    def _():
        state._begin_compact()
        state.add_message(Message.user("during compaction"))
        state._compact_done("model summary", Usage(input_tokens=9, output_tokens=2, requests=1))
        state._end_compact()

    @step("finish")
    def _():
        state.finish({"summary": "ok", "files": ["calc.py"]})

    @step("finish again without an answer")
    def _():
        state.finish()

    @step("restore s1")
    def _():
        state.restore(kept["s1"])

    @step("think after the restore")
    def _():
        think(state, c4)  # the same call id again in a new turn is fine

    @step("snapshot taken mid-turn")
    def _():
        kept["mid"] = state.snapshot()

    @step("result")
    def _():
        state._tool_result(c4, "r4", ToolOutcomeKind.DONE)

    @step("restore to the mid-turn snapshot")
    def _():
        state.restore(kept["mid"])
        assert state.pending_calls == (c4,)

    @step("result again")
    def _():
        state._tool_result(c4, "r4 again", ToolOutcomeKind.DONE)

    @step("restore s0 (before the first run)")
    def _():
        state.restore(kept["s0"])

    @step("restore to the beginning")
    def _():
        state.restore(State().snapshot())

    assert_replays(state)  # after the constructor
    for label, fn in steps:
        fn()
        try:
            assert_replays(state)
        except AssertionError:  # pragma: no cover - names the step that broke the invariant
            raise AssertionError(f"the invariant broke after the step {label!r}") from None
    assert len(steps) > 35
    assert state.messages == ()  # restored to the beginning
    assert len(state.history) > 40


def test_invariant_for_every_snapshot_of_the_scenario_not_only_the_last():
    state = task()
    snapshots = [state.snapshot()]
    state.add_message(Message.user("a"))
    snapshots.append(state.snapshot())
    think(state, call())
    snapshots.append(state.snapshot())
    state._tool_result(state.pending_calls[0], "r", ToolOutcomeKind.DONE)
    snapshots.append(state.snapshot())
    state.finish("x")
    snapshots.append(state.snapshot())

    for snap in snapshots:
        assert State(history=snap.history).snapshot() == snap
    # A snapshot's history is a prefix of the State's, so it replays to the snapshot itself.
    assert all(state.history[: len(s.history)] == s.history for s in snapshots)


def test_invariant_in_every_reporter_hook_of_real_runs():
    """Checks the invariant inside the Reporter hooks, which run in the middle of think and use_tools."""
    checked = []

    class Checker(Reporter):
        def _check(self, state, label):
            assert_replays(state)
            checked.append(label)

        def on_run_start(self, state):
            self._check(state, "run_start")

        def on_think_start(self, state):
            self._check(state, "think_start")  # waiting on the model

        def on_think_end(self, state, reply):
            self._check(state, "think_end")  # calls pending

        def on_tool_start(self, state, call):
            self._check(state, "tool_start")

        def on_tool_end(self, state, call, result, outcome):
            self._check(state, "tool_end")

        def on_context_change(self, state, change):
            self._check(state, "context_change")

        def on_run_end(self, state, error):
            self._check(state, "run_end")

    @tool
    def read_file(state: State, path: str) -> str:
        """Read a file."""
        with state.edit_extra_data() as data:
            data.setdefault("read", []).append(path)
        state.add_message(Message.notice(f"read {path}"))
        return f"contents of {path}"

    @tool
    def boom() -> str:
        """Fails."""
        raise RuntimeError("tool bug")

    @tool
    def submit(state: State, summary: str) -> str:
        """Submit."""
        state.finish(summary)
        return "submitted"

    agent = make_agent(
        [[tool_call("read_file", path="a.py"), tool_call("read_file", path="b.py")], "unused"],
        tools=[read_file, boom, submit],
        reporter=Checker(),
    )
    state = task(extra_data={"repo": "api"})
    agent.think(state)
    agent.use_tools(state)
    agent.run(state)  # a second run on the same State: the model answers "unused"
    assert_replays(state)

    agent2 = make_agent(
        [tool_call("submit", summary="all done")],
        tools=[read_file, boom, submit],
        reporter=Checker(),
    )
    state2 = task()
    assert agent2.run(state2) == "all done"
    assert state2.finished and state2.stopped == StoppedByFinish("all done")
    assert_replays(state2)

    failing = make_agent([TimeoutError("down"), "recovered"], reporter=Checker())
    state3 = task()
    with pytest.raises(TimeoutError):
        failing.run(state3)
    assert_replays(state3)
    assert failing.run(state3) == "recovered"
    assert_replays(state3)

    assert {"think_start", "think_end", "tool_start", "tool_end", "run_start", "run_end"} <= set(checked)


def test_invariant_across_agent_commands_ask_compact_clear_restore():
    @tool
    def echo(x: int) -> str:
        """Echo."""
        return f"result{x}"

    agent = make_agent(
        [tool_call("echo", x=1), "first", "an answer to the question", "the summary", "after"],
        tools=[echo],
        human=FakeHuman(["yes"]),
    )
    state = task()
    before = state.snapshot()
    agent.think(state)
    assert_replays(state)
    agent.use_tools(state)
    assert_replays(state)
    agent.think(state)
    assert_replays(state)
    agent.ask(state, "What do you think?")
    assert_replays(state)
    agent.ask_human(state, "Proceed?")
    assert_replays(state)
    state.clear_tool_results(keep_last=0)
    assert_replays(state)
    agent.compact(state)
    assert_replays(state)
    state.restore(before)
    assert_replays(state)
    state.add_message(Message.user("again"))
    agent.think(state)
    assert_replays(state)
    assert agent.model.remaining == 0
    assert state.fork().snapshot() == state.snapshot()


def test_invariant_with_a_json_round_trip_of_every_entry():
    state = task(extra_data={"a": [1, {"b": None}]})
    think(state, call("a", "c1"), call("b", "c2"))
    state._tool_result(state.pending_calls[0], "ra", ToolOutcomeKind.DONE)
    state.add_message(Message.user("held", Image(PNG)))
    state._tool_result(state.pending_calls[0], "rb", ToolOutcomeKind.INPUT_ERROR, error=ValueError("x"))
    state._record_ask("q", {"k": [1]}, Usage(requests=1))
    state._record_human("q", "a")
    think(state, "done")
    with state.edit_extra_data() as data:
        data["a"] = 2
        data["new"] = {"x": 1}
    state.compact("summary")
    state.clear_tool_results(keep_last=0)
    snap = state.snapshot()
    state.restore(snap)
    state.finish({"result": [1, 2]})
    state._record_stop(StoppedByPermission(call(), "DenyAll()"))

    entries = [
        _serial.entry_from_dict(json.loads(json.dumps(_serial.entry_to_dict(entry, seq))))
        for seq, entry in enumerate(state.history)
    ]
    assert entries == list(state.history)
    rebuilt = State(history=entries)
    assert rebuilt.snapshot() == state.snapshot()
    assert_replays(rebuilt)


def test_invariant_after_a_store_round_trip(tmp_path):
    store = FileStore(tmp_path)
    agent = make_agent(["first", "second"], store=store)
    state = task(extra_data={"repo": "api"}, id="scenario")
    agent.run(state)
    state.add_message(Message.user("and more"))
    before = state.snapshot()
    agent.run(state)
    state.restore(before)
    state.compact("summary")
    store.save(state)

    loaded = store.load("scenario")
    assert loaded.snapshot() == state.snapshot()
    assert_replays(loaded)
    loaded.add_message(Message.user("continue"))
    store.save(loaded)
    assert store.load("scenario").snapshot() == loaded.snapshot()
