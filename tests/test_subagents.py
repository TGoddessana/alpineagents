"""Subagents, the State side: the links recorded in history (``RunStartEntry.parent_id``/``parent_call_id``,
``ToolResultEntry.child_ids``/``usage``), how fold counts a subagent's usage, the saved format (version 5, which
still reads 4), ``state.parent_id``/``parent_call_id``, ``StateInfo.parent_id`` and ``state.child(child_id)``.

Histories are built entry by entry, so nothing here needs a model.

Every test keeps the two invariants: ``State(history=s.history).snapshot() == s.snapshot()``, and right after saving,
``store.load(id).snapshot() == state.snapshot()``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from alpineagents import (
    Agent,
    FileStore,
    Store,
    Message,
    ModelReplyEntry,
    ModelRequestEntry,
    Reply,
    RunStartEntry,
    State,
    StateInfo,
    ToolCall,
    ToolOutcomeKind,
    ToolResultEntry,
    Usage,
    _serial,
)
from alpineagents.testing import FakeModel
from alpineagents.store import Record
from alpineagents.types import AgentInfo, HistoryEntry

INFO = AgentInfo(model="fake/fake")


def call_of(name: str = "researcher", call_id: str = "c1") -> ToolCall:
    return ToolCall(name, {"prompt": "look"}, call_id)


def used(tokens: int, cost: float | None = 0.5) -> Usage:
    return Usage(input_tokens=tokens, output_tokens=1, requests=1, cost=cost)


def run_start(*, parent_id: str | None = None, parent_call_id: str | None = None, turn: int = 0) -> RunStartEntry:
    return RunStartEntry(content=INFO, parent_id=parent_id, parent_call_id=parent_call_id, turn=turn)


def asking(*calls: ToolCall, reply_usage: Usage | None = None) -> list[HistoryEntry]:
    """A request and a reply that asks for ``calls``, as a continuation of a State with one run_start."""
    reply = Reply(Message("assistant", calls), reply_usage or used(10))
    return [
        ModelRequestEntry(content="fake/fake", turn=1),
        ModelReplyEntry(content=reply, turn=1),
    ]


def result(
    call: ToolCall,
    content: str = "found",
    *,
    child_ids: Sequence[str] = (),
    usage: Usage | None = None,
    late: bool = False,
    outcome: ToolOutcomeKind = ToolOutcomeKind.DONE,
) -> ToolResultEntry:
    return ToolResultEntry(
        content=content, call=call, outcome=outcome, child_ids=tuple(child_ids), usage=usage, late=late, turn=1
    )


def rebuilt(entries: Sequence[HistoryEntry], **kwargs) -> State:
    """A State from a prefix with a user message (the import entry) and then ``entries``."""
    return State(history=State(messages=[Message.user("Go")]).history + tuple(entries), **kwargs)


def parent_state(child_ids: Sequence[str] = ("kid1",), usage: Usage | None = None, **kwargs) -> State:
    call = call_of()
    return rebuilt(
        [run_start(), *asking(call), result(call, child_ids=child_ids, usage=usage)],
        **kwargs,
    )


def child_state(parent: State, call_id: str = "c1", **kwargs) -> State:
    return rebuilt([run_start(parent_id=parent.id, parent_call_id=call_id)], **kwargs)


def assert_replays(state: State) -> None:
    assert State(history=state.history).snapshot() == state.snapshot()


def assert_loads_equal(store: FileStore, state: State) -> State:
    store.save(state)
    loaded = store.load(state.id)
    assert loaded.snapshot() == state.snapshot()
    return loaded


# ---------------------------------------------------------------- entry fields


def test_new_entry_fields_default_to_nothing():
    start = RunStartEntry(content=INFO, turn=0)
    assert (start.parent_id, start.parent_call_id) == (None, None)
    done = ToolResultEntry(content="x", call=call_of(), outcome=ToolOutcomeKind.DONE, turn=1)
    assert done.child_ids == ()
    assert done.usage is None


def test_child_ids_of_a_tool_result_become_a_tuple():
    entry = result(call_of(), child_ids=["a", "b"])
    assert entry.child_ids == ("a", "b")
    assert isinstance(entry.child_ids, tuple)


def test_the_new_fields_take_part_in_entry_equality():
    base = result(call_of())
    assert base == result(call_of())
    assert base != result(call_of(), child_ids=["a"])
    assert base != result(call_of(), usage=used(3))
    assert run_start() != run_start(parent_id="p", parent_call_id="c")


# ---------------------------------------------------------------- usage in fold


def test_a_tool_result_usage_is_added_to_the_state_usage():
    state = parent_state(usage=used(100, 1.0))
    # the parent's own reply (10 tokens, 0.5) plus what its subagent used
    assert state.usage == Usage(input_tokens=110, output_tokens=2, requests=2, cost=1.5)
    assert state.snapshot().usage == state.usage
    assert_replays(state)


def test_usage_of_a_result_without_subagents_changes_nothing():
    state = parent_state(child_ids=(), usage=None)
    assert state.usage == used(10)
    assert_replays(state)


def test_a_late_result_adds_its_usage_too():
    call = call_of()
    state = rebuilt([
        run_start(),
        *asking(call),
        result(call, "(interrupted by user)", outcome=ToolOutcomeKind.INTERRUPTED, child_ids=["kid1"]),
        result(call, "late answer", late=True, child_ids=["kid2"], usage=used(40, 0.25)),
    ])
    assert state.usage == Usage(input_tokens=50, output_tokens=2, requests=2, cost=0.75)
    assert [e.child_ids for e in state.history if isinstance(e, ToolResultEntry)] == [("kid1",), ("kid2",)]
    assert_replays(state)


def test_usage_with_an_unknown_cost_makes_the_total_unknown():
    state = parent_state(usage=used(5, None))
    assert state.usage.cost is None
    assert state.usage.requests == 2
    assert_replays(state)


def test_restore_does_not_undo_the_usage_of_a_subagent():
    call = call_of()
    state = rebuilt([run_start(), *asking(call, reply_usage=used(1))])
    before = state.snapshot()  # the call is still pending
    state._commit(result(call, child_ids=["kid1"], usage=used(100, 1.0)))
    total = state.usage
    assert total == Usage(input_tokens=101, output_tokens=2, requests=2, cost=1.5)

    state.restore(before)
    assert state.pending_calls == (call,)
    assert state.usage == total
    assert_replays(state)


# ---------------------------------------------------------------- parent_id / parent_call_id


def test_a_state_no_tool_call_started_has_no_parent():
    state = parent_state()
    assert (state.parent_id, state.parent_call_id) == (None, None)
    assert State().parent_id is None


def test_parent_id_and_parent_call_id_come_from_the_run_start():
    parent = parent_state()
    child = child_state(parent, "c9")
    assert child.parent_id == parent.id
    assert child.parent_call_id == "c9"
    assert child.snapshot() == State(history=child.history).snapshot()
    assert State(history=child.history).parent_id == parent.id


def test_the_first_link_stays_when_the_state_runs_again_from_elsewhere():
    first = rebuilt([run_start(parent_id="p1", parent_call_id="a")])
    again = State(
        history=first.history
        + (
            run_start(parent_id="p2", parent_call_id="b"),
            run_start(),
        )
    )
    assert (again.parent_id, again.parent_call_id) == ("p1", "a")
    assert_replays(again)


def test_a_link_in_a_later_run_start_counts_when_the_first_had_none():
    state = rebuilt([run_start(), run_start(parent_id="p", parent_call_id="c")])
    assert (state.parent_id, state.parent_call_id) == ("p", "c")


def test_restore_keeps_the_parent_link():
    state = rebuilt([run_start(parent_id="p", parent_call_id="c")])
    state.add_message(Message.user("more"))
    before_link = State(messages=[Message.user("Go")]).snapshot()  # a snapshot from before the run_start
    state.restore(before_link)
    assert (state.parent_id, state.parent_call_id) == ("p", "c")
    assert_replays(state)


def test_fork_and_rebuilding_keep_the_parent_link():
    child = rebuilt([run_start(parent_id="p", parent_call_id="c")])
    assert child.fork().parent_id == "p"
    assert State(history=child.history).parent_call_id == "c"


def test_parent_properties_are_read_only():
    state = State()
    with pytest.raises(AttributeError, match="parent_id is read-only"):
        state.parent_id = "x"  # type: ignore[misc]
    with pytest.raises(AttributeError, match="parent_call_id is read-only"):
        state.parent_call_id = "x"  # type: ignore[misc]


# ---------------------------------------------------------------- saved format


def test_the_format_version_is_5_and_4_is_still_read():
    assert _serial.VERSION == 5
    assert _serial.MIN_VERSION == 4
    for version in (4, 5):
        _serial.check_version("x", {"v": version})
    with pytest.raises(ValueError, match=r"older alpineagents \(format 3, 0\.4\.x\)"):
        _serial.check_version("x", {"v": 3})
    with pytest.raises(ValueError, match="newer alpineagents"):
        _serial.check_version("x", {"v": 6})


def test_new_entry_fields_survive_json():
    call = call_of()
    parent = parent_state(usage=used(7, 0.125))
    child = child_state(parent)
    for state in (parent, child):
        for index, entry in enumerate(state.history):
            data = json.loads(json.dumps(_serial.entry_to_dict(entry, index)))
            assert _serial.entry_from_dict(data) == entry
    done = next(e for e in parent.history if isinstance(e, ToolResultEntry))
    data = _serial.entry_to_dict(done, 0)
    assert data["child_ids"] == ["kid1"]
    assert data["usage"]["input_tokens"] == 7
    assert call.id == done.call.id
    start = next(e for e in child.history if isinstance(e, RunStartEntry))
    data = _serial.entry_to_dict(start, 0)
    assert (data["parent_id"], data["parent_call_id"]) == (parent.id, "c1")


def test_entries_without_the_new_fields_do_not_write_their_keys():
    call = call_of()
    state = rebuilt([run_start(), *asking(call), result(call)])
    for index, entry in enumerate(state.history):
        data = _serial.entry_to_dict(entry, index)
        assert not {"child_ids", "usage", "parent_id", "parent_call_id"} & set(data)


FORMAT_4_AT = "2026-10-14T15:15:59+00:00"
FORMAT_4_CALL = {"name": "researcher", "args": {"prompt": "look"}, "id": "c1"}


def format_4_entries() -> list[dict]:
    """Entries as format 4 wrote them, by hand: no parent_id, parent_call_id, child_ids or usage on them."""
    base = {"at": FORMAT_4_AT}
    return [
        {
            **base,
            "seq": 0,
            "kind": "run_start",
            "turn": 0,
            "content": {"name": None, "model": "fake/fake", "system_sha256": None, "tools": [], "mcp_servers": []},
        },
        {**base, "seq": 1, "kind": "model_request", "turn": 1, "content": "fake/fake"},
        {
            **base,
            "seq": 2,
            "kind": "model_reply",
            "turn": 1,
            "content": {
                "message": {"role": "assistant", "content": [{"type": "tool_call", **FORMAT_4_CALL}], "tokens": None},
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 1,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "requests": 1,
                    "cost": 0.5,
                },
                "context_tokens": None,
                "stop_reason": None,
                "model": None,
            },
        },
        {
            **base,
            "seq": 3,
            "kind": "tool_result",
            "turn": 1,
            "content": "found",
            "call": FORMAT_4_CALL,
            "outcome": "done",
        },
    ]


def test_format_4_entries_load_with_default_links():
    start, _, _, done = (_serial.entry_from_dict(data) for data in format_4_entries())
    assert isinstance(start, RunStartEntry)
    assert (start.parent_id, start.parent_call_id) == (None, None)
    assert isinstance(done, ToolResultEntry)
    assert (done.child_ids, done.usage) == ((), None)
    late = _serial.entry_from_dict({**format_4_entries()[3], "content": "late news", "late": True})
    assert isinstance(late, ToolResultEntry)
    assert (late.child_ids, late.usage, late.late) == ((), None, True)


def test_a_child_ids_string_is_refused_not_split_into_characters():
    with pytest.raises(TypeError, match="not one string"):
        ToolResultEntry(content="x", call=call_of(), outcome=ToolOutcomeKind.DONE, child_ids="kid1", turn=1)  # type: ignore[arg-type]
    data = {**format_4_entries()[3], "child_ids": "kid1"}
    with pytest.raises(ValueError, match="is damaged"):
        _serial.entry_from_dict(data)


def test_a_format_4_folder_loads_and_is_saved_on_as_format_5(tmp_path: Path):
    store = FileStore(tmp_path)
    state = rebuilt([], id="old")
    store.save(state)
    info_path = tmp_path / "old" / "info.json"
    info = json.loads(info_path.read_text("utf-8"))
    info["v"] = 4
    del info["parent_id"]
    info_path.write_text(json.dumps(info), "utf-8")
    # the log as format 4 wrote it: the import entry, then entries without any of the new keys
    log_path = tmp_path / "old" / "log.jsonl"
    lines = log_path.read_text("utf-8").splitlines()
    for seq, data in enumerate(format_4_entries(), start=len(lines)):
        lines.append(json.dumps({**data, "seq": seq}))
    log_path.write_text("\n".join(lines) + "\n", "utf-8")

    loaded = store.load("old")
    assert [type(e) for e in loaded.history[1:]] == [RunStartEntry, ModelRequestEntry, ModelReplyEntry, ToolResultEntry]
    assert loaded.parent_id is None
    assert loaded.usage.input_tokens == 10  # the reply only; the result adds nothing
    assert State(history=loaded.history).snapshot() == loaded.snapshot()
    [listed] = store.list()
    assert (listed.id, listed.parent_id) == ("old", None)

    loaded.add_message(Message.user("more"))
    store.save(loaded)
    assert json.loads(info_path.read_text("utf-8"))["v"] == 5
    assert store.load("old").snapshot() == loaded.snapshot()


def test_info_has_parent_id_and_a_version_5_info_without_it_is_damaged():
    info = _serial.info_to_dict(
        first_message=None, created_at=None, updated_at=None, turn=0, stopped=None, finished=False, parent_id="p"
    )
    assert info["v"] == 5
    assert info["parent_id"] == "p"
    assert _serial.info_from_dict("x", info)["parent_id"] == "p"
    broken = {k: v for k, v in info.items() if k != "parent_id"}
    with pytest.raises(ValueError, match="no parent_id"):
        _serial.info_from_dict("x", broken)
    assert _serial.info_from_dict("x", {**broken, "v": 4})["parent_id"] is None


# ---------------------------------------------------------------- store: StateInfo.parent_id


def test_store_fills_parent_id_in_the_info_and_the_listing(tmp_path: Path):
    store = FileStore(tmp_path)
    parent = parent_state(id="parent")
    child = child_state(parent, id="kid1")
    top = State(messages=[Message.user("Alone")], id="top")
    for state in (parent, child, top):
        store.save(state)

    record = store.read("kid1")
    assert record is not None and record.info["parent_id"] == "parent"
    listed = {info.id: info for info in store.list()}
    assert listed["kid1"].parent_id == "parent"
    assert listed["parent"].parent_id is None
    assert listed["top"].parent_id is None
    assert sorted(info.id for info in store.list() if info.parent_id is None) == ["parent", "top"]
    assert isinstance(listed["kid1"], StateInfo)


def test_parent_id_is_in_the_info_from_the_first_save_and_after_load(tmp_path: Path):
    store = FileStore(tmp_path)
    child = child_state(parent_state(id="parent"), id="kid1")
    loaded = assert_loads_equal(store, child)
    assert (loaded.parent_id, loaded.parent_call_id) == ("parent", "c1")
    assert_replays(loaded)


def test_state_info_parent_id_defaults_to_none():
    info = StateInfo(id="x", first_message=None, created_at=None, updated_at=None, turn=0, stopped=None, finished=False)
    assert info.parent_id is None


# ---------------------------------------------------------------- state.child


def test_child_returns_the_object_in_the_runtime_table():
    parent = parent_state()
    kid = child_state(parent, id="kid1")
    parent._add_child("c1", kid)
    assert parent.child("kid1") is kid


def test_child_finds_a_started_child_that_has_no_result_yet():
    parent = rebuilt([run_start(), *asking(call_of())])
    kid = child_state(parent)
    parent._add_child("c1", kid)
    assert parent.child(kid.id) is kid
    assert parent._children_of("c1") == (kid.id,)


def test_add_child_keeps_start_order_and_each_call_apart():
    parent = rebuilt([run_start()])
    first, second, other = (child_state(parent) for _ in range(3))
    parent._add_child("c1", first)
    parent._add_child("c1", second)
    parent._add_child("c1", first)  # the same child again is not listed twice
    parent._add_child("c2", other)
    assert parent._children_of("c1") == (first.id, second.id)
    assert parent._children_of("c2") == (other.id,)
    assert parent._children_of("nope") == ()


def test_child_of_an_unknown_id_is_a_key_error():
    parent = parent_state()
    with pytest.raises(KeyError, match="not a subagent"):
        parent.child("stranger")
    with pytest.raises(KeyError):
        State().child("anything")


def test_child_is_a_key_error_for_a_grandchild():
    parent = parent_state(child_ids=("kid1",))
    kid = child_state(parent, id="kid1")
    grandkid = child_state(kid, id="grand1")
    parent._add_child("c1", kid)
    kid._add_child("c1", grandkid)
    with pytest.raises(KeyError):
        parent.child("grand1")
    assert parent.child("kid1").child("grand1") is grandkid


def test_child_that_cannot_be_opened_says_how_to_fix_it():
    parent = parent_state()
    with pytest.raises(ValueError) as raised:
        parent.child("kid1")
    text = str(raised.value)
    assert "kid1" in text
    assert "no store" in text
    assert "Fix:" in text and "Agent(store=" in text


def test_a_rebuilt_state_has_no_runtime_table_and_no_store():
    parent = parent_state()
    kid = child_state(parent, id="kid1")
    parent._add_child("c1", kid)
    copy = State(history=parent.history)
    with pytest.raises(ValueError, match="no store"):
        copy.child("kid1")
    with pytest.raises(KeyError):
        copy.child("stranger")


def test_child_loads_from_the_store_after_store_load(tmp_path: Path):
    store = FileStore(tmp_path)
    parent = parent_state(id="parent")
    kid = child_state(parent, id="kid1")
    parent._add_child("c1", kid)
    store.save(kid)
    loaded_parent = assert_loads_equal(store, parent)

    opened = loaded_parent.child("kid1")
    assert opened is not kid
    assert opened.snapshot() == kid.snapshot()
    assert (opened.parent_id, opened.parent_call_id) == ("parent", "c1")
    assert_replays(opened)
    with pytest.raises(KeyError):
        loaded_parent.child("top-level")


def test_child_in_a_store_the_parent_is_not_bound_to_is_a_load_error(tmp_path: Path):
    store = FileStore(tmp_path)
    parent = parent_state(id="parent")
    loaded_parent = assert_loads_equal(store, parent)
    with pytest.raises(LookupError, match="no saved State"):
        loaded_parent.child("kid1")  # the child is a child, but its own store has it


def test_a_grandchild_is_reached_through_two_children(tmp_path: Path):
    store = FileStore(tmp_path)
    call = call_of()
    parent = parent_state(id="parent", child_ids=("kid1",))
    kid = rebuilt(
        [run_start(parent_id="parent", parent_call_id="c1"), *asking(call), result(call, child_ids=["grand1"])],
        id="kid1",
    )
    grand = child_state(kid, id="grand1")
    for state in (grand, kid):
        store.save(state)
    loaded = assert_loads_equal(store, parent)
    assert loaded.child("kid1").child("grand1").snapshot() == grand.snapshot()


def test_fork_carries_the_runtime_table_over():
    parent = parent_state()
    kid = child_state(parent, id="kid1")
    parent._add_child("c1", kid)
    copy = parent.fork()
    assert copy.id != parent.id
    assert copy.child("kid1") is kid
    assert copy.snapshot() == parent.snapshot()
    # the tables are separate: a child added to the copy later is not the original's
    other = child_state(parent, id="kid2")
    copy._add_child("c2", other)
    with pytest.raises(KeyError):
        parent.child("kid2")
    assert copy._children_of("c1") == ()  # which call started which child stays with the original
    assert_replays(copy)


def test_a_fork_of_a_loaded_state_opens_children_from_the_store(tmp_path: Path):
    store = FileStore(tmp_path)
    parent = parent_state(id="parent")
    kid = child_state(parent, id="kid1")
    store.save(kid)
    assert_loads_equal(store, parent)
    copy = store.load("parent").fork()
    assert copy.id != "parent"
    opened = copy.child("kid1")
    assert opened.snapshot() == kid.snapshot()
    assert opened is not copy.child("kid1")  # still not cached
    with pytest.raises(KeyError):
        copy.child("stranger")
    assert_replays(copy)


def test_a_fork_keeps_the_child_store_after_its_own_first_save(tmp_path: Path):
    store, other = FileStore(tmp_path / "a"), FileStore(tmp_path / "b")
    parent = parent_state(id="parent")
    kid = child_state(parent, id="kid1")
    store.save(kid)
    assert_loads_equal(store, parent)
    copy = store.load("parent").fork()
    other.save(copy)  # saved under its own id; the children stay where they are
    assert other.load(copy.id).snapshot() == copy.snapshot()
    assert copy.child("kid1").snapshot() == kid.snapshot()
    # the other store has no child, and the copy is bound to it, so only the carried-over store finds it
    assert copy._saved_store == other


async def test_a_fork_of_a_loaded_state_opens_children_with_achild():
    store = AsyncOnlyStore()
    parent = parent_state(id="parent")
    kid = child_state(parent, id="kid1")
    await store.asave(kid)
    await store.asave(parent)
    copy = (await store.aload("parent")).fork()
    assert (await copy.achild("kid1")).snapshot() == kid.snapshot()
    with pytest.raises(TypeError, match="only asynchronously"):
        copy.child("kid1")


def test_a_fork_of_an_unbound_state_has_no_store():
    parent = parent_state()
    with pytest.raises(ValueError, match="no store"):
        parent.fork().child("kid1")


def test_the_first_save_binds_the_store_children_are_opened_from(tmp_path: Path):
    store = FileStore(tmp_path)
    parent = parent_state(id="parent")
    kid = child_state(parent, id="kid1")
    store.save(kid)
    store.save(parent)
    assert parent.child("kid1").snapshot() == kid.snapshot()
    assert parent.fork().child("kid1").snapshot() == kid.snapshot()


def test_restore_keeps_the_runtime_table():
    parent = parent_state()
    kid = child_state(parent, id="kid1")
    parent._add_child("c1", kid)
    snapshot = parent.snapshot()
    parent.add_message(Message.user("more"))
    parent.restore(snapshot)
    assert parent.child("kid1") is kid
    assert_replays(parent)


def test_restore_to_before_the_result_still_opens_the_child():
    call = call_of()
    parent = rebuilt([run_start(), *asking(call, reply_usage=used(1))])
    kid = child_state(parent, id="kid1")
    parent._add_child("c1", kid)
    earlier = parent.snapshot()  # the call is still pending
    parent._commit(result(call, child_ids=["kid1"], usage=used(9)))
    assert parent.child("kid1") is kid
    assert earlier.pending_calls == (call,)
    assert_replays(parent)


def test_a_child_loaded_from_the_store_is_not_cached(tmp_path: Path):
    store = FileStore(tmp_path)
    parent = parent_state(id="parent")
    kid = child_state(parent, id="kid1")
    store.save(kid)
    loaded_parent = assert_loads_equal(store, parent)
    first, second = loaded_parent.child("kid1"), loaded_parent.child("kid1")
    assert first is not second
    assert first.snapshot() == second.snapshot() == kid.snapshot()

    # one still in the runtime table is the same object every time
    parent._add_child("c1", kid)
    assert parent.child("kid1") is parent.child("kid1") is kid


class AsyncOnlyStore(Store):
    """Implements only the async methods, so the sync ``load`` refuses."""

    def __init__(self) -> None:
        self.records: dict[str, Record] = {}
        self.loads: list[str] = []

    async def awrite(self, state_id, entries, info, *, create=False):
        await asyncio.sleep(0)
        record = self.records.setdefault(state_id, Record([], {}))
        record.entries.extend(json.loads(json.dumps(e)) for e in entries if e["seq"] >= len(record.entries))
        self.records[state_id] = Record(record.entries, json.loads(json.dumps(info)))

    async def aread(self, state_id):
        await asyncio.sleep(0)
        self.loads.append(state_id)
        return self.records.get(state_id)


async def test_achild_returns_the_object_in_the_runtime_table():
    parent = parent_state()
    kid = child_state(parent, id="kid1")
    parent._add_child("c1", kid)
    assert await parent.achild("kid1") is kid


async def test_achild_loads_from_an_async_only_store_and_does_not_cache():
    store = AsyncOnlyStore()
    parent = parent_state(id="parent")
    kid = child_state(parent, id="kid1")
    await store.asave(kid)
    await store.asave(parent)
    loaded_parent = await store.aload("parent")
    assert loaded_parent.snapshot() == parent.snapshot()

    with pytest.raises(TypeError, match="only asynchronously"):
        loaded_parent.child("kid1")
    first = await loaded_parent.achild("kid1")
    second = await loaded_parent.achild("kid1")
    assert first is not second
    assert first.snapshot() == second.snapshot() == kid.snapshot()
    assert (first.parent_id, first.parent_call_id) == ("parent", "c1")
    assert_replays(first)
    assert store.loads.count("kid1") == 2


async def test_achild_raises_the_same_errors_as_child(tmp_path: Path):
    parent = parent_state()
    with pytest.raises(KeyError, match="not a subagent"):
        await parent.achild("stranger")
    with pytest.raises(ValueError, match="no store") as raised:
        await parent.achild("kid1")
    assert "Agent(store=" in str(raised.value)

    store = FileStore(tmp_path)
    loaded_parent = assert_loads_equal(store, parent_state(id="p2"))
    with pytest.raises(LookupError, match="no saved State"):
        await loaded_parent.achild("kid1")
    with pytest.raises(KeyError):
        await loaded_parent.achild("stranger")


async def test_achild_works_with_a_sync_store(tmp_path: Path):
    store = FileStore(tmp_path)
    parent = parent_state(id="parent")
    kid = child_state(parent, id="kid1")
    store.save(kid)
    loaded_parent = assert_loads_equal(store, parent)
    assert (await loaded_parent.achild("kid1")).snapshot() == kid.snapshot()


def test_a_run_with_an_agent_keeps_the_invariants_with_the_new_fields(tmp_path: Path):
    """A normal run is unaffected: no links, no usage on results, and the state is saved and loaded as before."""
    store = FileStore(tmp_path)
    agent = Agent(model=FakeModel(["Hello"]), reporter=None, human=None, store=store)
    state = State(messages=[Message.user("Hi")], id="plain")
    agent.run(state)
    assert (state.parent_id, state.parent_call_id) == (None, None)
    assert_replays(state)
    loaded = assert_loads_equal(store, state)
    assert loaded.parent_id is None
    with pytest.raises(KeyError):
        loaded.child("anything")
    assert [i.parent_id for i in store.list()] == [None]
