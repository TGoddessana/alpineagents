"""Subagents, the State side: the links recorded in history (``RunStartEntry.parent_id``/``parent_call_id``,
``ToolResultEntry.child_ids``/``usage``), how fold counts a subagent's usage, the saved format (version 5, which
still reads 4), ``state.parent_id``/``parent_call_id``, ``StateInfo.parent_id`` and ``state.child(child_id)``.

Histories are built entry by entry, so nothing here needs a model.

Every test keeps the two invariants: ``State(history=s.history).snapshot() == s.snapshot()``, and right after saving,
``store.load(id).snapshot() == state.snapshot()``.
"""

from __future__ import annotations

import _thread
import asyncio
import io
import json
import threading
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from alpineagents import (
    Agent,
    FileStore,
    Store,
    Message,
    ModelReplyEntry,
    ModelRequestEntry,
    Price,
    Reply,
    NoHumanError,
    PermissionWarning,
    Reporter,
    RunStartEntry,
    State,
    StateInfo,
    StoppedByPermission,
    Terminal,
    ToolCall,
    ToolError,
    ToolOutcomeKind,
    ToolResultEntry,
    Usage,
    _serial,
    tool,
)
from alpineagents import terminal as terminal_module
from alpineagents.permissions import (
    AllowByDefault,
    AllowByName,
    AllowPermission,
    Allowed,
    DecideByHuman,
    DenyByName,
    DenyPermission,
)
from alpineagents.terminal import default_terminal
from alpineagents.testing import FakeHuman, FakeModel, tool_call
from alpineagents.store import Record
from alpineagents.human import Human
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


# ================================================================ linking runs started inside a tool call
#
# A run started inside a tool call of another run is its child: the tools below call another Agent's run/arun.


def agent_of(replies, **settings) -> Agent:
    settings.setdefault("reporter", None)
    settings.setdefault("human", None)
    return Agent(model=FakeModel(replies), **settings)


def user_state(text: str = "Task", **kwargs) -> State:
    return State(messages=[Message.user(text)], **kwargs)


def results_of(state: State) -> list[ToolResultEntry]:
    return [e for e in state.history if isinstance(e, ToolResultEntry)]


def starts_of(state: State) -> list[RunStartEntry]:
    return [e for e in state.history if isinstance(e, RunStartEntry)]


def own_usage(state: State) -> Usage:
    """What the State's own model requests used (not its subagents')."""
    total = Usage()
    for entry in state.history:
        if isinstance(entry, ModelReplyEntry):
            total = total + entry.content.usage
    return total


def assert_counts_children(state: State) -> None:
    """``state.usage`` is its own requests plus what its tool results say the subagents used."""
    total = own_usage(state)
    for entry in results_of(state):
        if entry.usage is not None:
            total = total + entry.usage
    assert state.usage == total
    assert_replays(state)


def link_of(child: State) -> tuple[str | None, str | None]:
    return child.parent_id, child.parent_call_id


def test_a_sync_run_inside_a_tool_is_a_linked_child():
    helper = agent_of(["inner answer"])

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    first = tool_call("delegate")
    agent = agent_of([first, "outer answer"], tools=[delegate])
    parent = user_state(id="parent")
    assert agent.run(parent) == "outer answer"

    (entry,) = results_of(parent)
    assert entry.outcome is ToolOutcomeKind.DONE and entry.content == "inner answer"
    assert len(entry.child_ids) == 1
    kid = parent.child(entry.child_ids[0])
    assert parent.child(kid.id) is kid
    assert link_of(kid) == ("parent", first.id)
    assert [(s.parent_id, s.parent_call_id) for s in starts_of(kid)] == [("parent", first.id)]
    assert entry.usage == kid.usage and kid.usage.requests == 1
    assert parent.usage == own_usage(parent) + kid.usage
    assert kid._depth == 1 and parent._depth == 0
    assert_counts_children(parent)
    assert_counts_children(kid)
    assert (parent.parent_id, parent.parent_call_id) == (None, None)


async def test_an_async_run_inside_an_async_tool_is_a_linked_child():
    helper = agent_of(["inner answer"])

    @tool
    async def delegate() -> str:
        """Ask the helper"""
        return await helper.arun("Inner task")

    first = tool_call("delegate")
    agent = agent_of([first, "outer answer"], tools=[delegate])
    parent = user_state(id="parent")
    assert await agent.arun(parent) == "outer answer"

    (entry,) = results_of(parent)
    (kid_id,) = entry.child_ids
    kid = parent.child(kid_id)
    assert link_of(kid) == ("parent", first.id)
    assert entry.usage == kid.usage
    assert_counts_children(parent)


async def test_a_sync_run_inside_a_sync_tool_of_an_async_run_is_linked():
    helper = agent_of(["inner answer"])

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    first = tool_call("delegate")
    agent = agent_of([first, "outer answer"], tools=[delegate])
    parent = user_state(id="parent")
    assert await agent.arun(parent) == "outer answer"
    (kid_id,) = results_of(parent)[0].child_ids
    assert link_of(parent.child(kid_id)) == ("parent", first.id)
    assert_counts_children(parent)


async def test_an_async_run_inside_a_sync_tool_started_with_asyncio_run_is_linked():
    # The tool makes its own event loop on its worker thread (the context is copied into it).
    helper = agent_of(["inner answer"])

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return asyncio.run(helper.arun("Inner task"))

    first = tool_call("delegate")
    agent = agent_of([first, "outer answer"], tools=[delegate])
    parent = user_state(id="parent")
    await agent.arun(parent)
    (kid_id,) = results_of(parent)[0].child_ids
    assert link_of(parent.child(kid_id)) == ("parent", first.id)


async def test_asyncio_gather_in_a_tool_starts_several_children_in_order():
    reviewers = [agent_of([f"review {i}"]) for i in range(3)]
    kids: list[State] = []

    @tool
    async def review_all() -> list[str]:
        """Review everything"""

        async def one(i: int) -> str:
            state = user_state(f"file {i}")
            kids.append(state)
            return await reviewers[i].arun(state)

        return list(await asyncio.gather(*(one(i) for i in range(3))))

    first = tool_call("review_all")
    agent = agent_of([first, "done"], tools=[review_all])
    parent = user_state(id="parent")
    await agent.arun(parent)

    (entry,) = results_of(parent)
    assert entry.child_ids == tuple(kid.id for kid in kids)  # start order
    for kid in kids:
        assert link_of(kid) == ("parent", first.id)
        assert parent.child(kid.id) is kid
    expected = Usage()
    for kid in kids:
        expected = expected + kid.usage
    assert entry.usage == expected and expected.requests == 3
    assert_counts_children(parent)


def test_several_sequential_children_in_one_tool_keep_their_start_order():
    helpers = [agent_of([f"answer {i}"]) for i in range(3)]
    kids = [user_state(f"q{i}") for i in range(3)]

    @tool
    def ask_all() -> str:
        """Ask them one after the other"""
        return ", ".join(helper.run(kid) for helper, kid in zip(helpers, kids, strict=True))

    first = tool_call("ask_all")
    agent = agent_of([first, "done"], tools=[ask_all])
    parent = user_state()
    agent.run(parent)
    (entry,) = results_of(parent)
    assert entry.child_ids == tuple(kid.id for kid in kids)
    assert entry.usage is not None and entry.usage.requests == 3
    assert_counts_children(parent)


def test_a_grandchild_chain_links_each_level_and_usage_reaches_the_top():
    grand_agent = agent_of(["deep answer"])
    grand = user_state("deepest")

    @tool
    def dig() -> str:
        """Go one level deeper"""
        return grand_agent.run(grand)

    mid_call = tool_call("dig")
    mid_agent = agent_of([mid_call, "mid answer"], tools=[dig])
    mid = user_state("middle")

    @tool
    def delegate() -> str:
        """Ask the middle agent"""
        return mid_agent.run(mid)

    top_call = tool_call("delegate")
    agent = agent_of([top_call, "top answer"], tools=[delegate])
    top = user_state(id="top")
    assert agent.run(top) == "top answer"

    assert link_of(mid) == ("top", top_call.id)
    assert link_of(grand) == (mid.id, mid_call.id)
    assert top.child(mid.id) is mid and mid.child(grand.id) is grand
    with pytest.raises(KeyError):
        top.child(grand.id)  # a grandchild is reached through its parent
    assert (grand._depth, mid._depth, top._depth) == (2, 1, 0)

    assert mid.usage == own_usage(mid) + grand.usage
    assert top.usage == own_usage(top) + mid.usage
    assert top.usage.requests == 5 and grand.usage.requests == 1
    for state in (top, mid, grand):
        assert_counts_children(state)


def test_parallel_calls_of_one_turn_each_get_their_own_children():
    barrier = threading.Barrier(2, timeout=5)
    helper_a, helper_b = agent_of(["a done"]), agent_of(["b done"])
    kids: dict[str, State] = {}

    @tool
    def tool_a() -> str:
        """A"""
        barrier.wait()  # both tools run at once
        kids["a"] = user_state("a")
        return helper_a.run(kids["a"])

    @tool
    def tool_b() -> str:
        """B"""
        barrier.wait()
        kids["b"] = user_state("b")
        return helper_b.run(kids["b"])

    call_a, call_b = tool_call("tool_a"), tool_call("tool_b")
    agent = agent_of([[call_a, call_b], "done"], tools=[tool_a, tool_b])
    parent = user_state()
    agent.run(parent)

    by_call = {entry.call.id: entry for entry in results_of(parent)}
    assert by_call[call_a.id].child_ids == (kids["a"].id,)
    assert by_call[call_b.id].child_ids == (kids["b"].id,)
    assert link_of(kids["a"]) == (parent.id, call_a.id)
    assert link_of(kids["b"]) == (parent.id, call_b.id)
    assert_counts_children(parent)


def test_a_run_in_a_thread_the_developer_made_is_not_linked():
    helper = agent_of(["x"] * 2)
    made: list[State] = []

    @tool
    def in_threads() -> str:
        """Runs agents on threads of its own"""
        first, second = user_state("t1"), user_state("t2")
        made.extend([first, second])
        thread = threading.Thread(target=helper.run, args=(first,))
        thread.start()
        thread.join()
        with ThreadPoolExecutor(1) as pool:
            pool.submit(helper.run, second).result()
        return "ok"

    agent = agent_of([tool_call("in_threads"), "done"], tools=[in_threads])
    parent = user_state()
    agent.run(parent)

    (entry,) = results_of(parent)
    assert entry.child_ids == () and entry.usage is None
    assert parent.usage == own_usage(parent)
    for state in made:
        assert link_of(state) == (None, None)
        assert state.usage.requests == 1
        with pytest.raises(KeyError):
            parent.child(state.id)
    assert_counts_children(parent)


def test_a_run_started_by_a_custom_loop_a_reporter_or_a_permission_is_not_linked():
    helper = agent_of(["x"] * 4)
    made: list[State] = []

    def start(label: str) -> None:
        state = user_state(label)
        made.append(state)
        helper.run(state)

    def custom_loop(agent: Agent, state: State) -> str:
        start("from the loop")
        agent.think(state)
        if state.pending_calls:
            agent.use_tools(state)
            agent.think(state)
        return state.answer

    class StartingReporter(Reporter):
        def on_run_start(self, state: State) -> None:
            start("from the reporter")

    class StartingPermission(AllowPermission):
        def check(self, state, call, tool):
            start("from the permission")
            return Allowed()

    @tool
    def noop() -> str:
        """Nothing"""
        return "ok"

    agent = agent_of(
        [tool_call("noop"), "done"],
        tools=[noop],
        loop=custom_loop,
        reporter=StartingReporter(),
        permissions=[StartingPermission()],
    )
    parent = user_state()
    agent.run(parent)

    assert len(made) == 3
    assert all(link_of(state) == (None, None) for state in made)
    assert results_of(parent)[0].child_ids == ()
    assert_counts_children(parent)


def test_a_run_started_by_the_loop_of_a_subagent_is_not_linked_to_the_tool_around_it():
    inner = agent_of(["x"])
    inner_state = user_state("inner")

    def kid_loop(agent: Agent, state: State) -> str:
        inner.run(inner_state)  # inside the subagent's loop, not inside a tool call
        agent.think(state)
        return state.answer

    kid_agent = agent_of(["kid answer"], loop=kid_loop)
    kid = user_state("kid")

    @tool
    def delegate() -> str:
        """Ask the subagent"""
        return kid_agent.run(kid)

    agent = agent_of([tool_call("delegate"), "done"], tools=[delegate])
    parent = user_state()
    agent.run(parent)

    assert results_of(parent)[0].child_ids == (kid.id,)
    assert link_of(inner_state) == (None, None)
    assert results_of(kid) == []
    assert_counts_children(parent)


def test_a_state_run_again_keeps_its_first_parent_and_records_each_origin():
    helper = agent_of(["one", "two", "three"])
    kid = user_state("kid")

    def delegating(name: str):
        @tool(name=name)
        def delegate() -> str:
            """Ask the helper"""
            return helper.run(kid)

        return delegate

    def run_under(parent_id: str, name: str) -> State:
        parent = user_state(id=parent_id)
        agent_of([tool_call(name), "done"], tools=[delegating(name)]).run(parent)
        return parent

    first = run_under("first", "ask_a")
    kid.add_message(Message.user("again"))
    second = run_under("second", "ask_b")
    kid.add_message(Message.user("and again"))
    helper.run(kid)  # outside any tool call

    assert kid.parent_id == "first"  # the first link stays
    origins = [(s.parent_id, s.parent_call_id) for s in starts_of(kid)]
    assert [parent for parent, _ in origins] == ["first", "second", None]
    assert results_of(first)[0].child_ids == results_of(second)[0].child_ids == (kid.id,)
    assert_replays(kid)
    # Each parent counts only what the child used inside its own call
    assert results_of(first)[0].usage is not None and results_of(second)[0].usage is not None
    assert results_of(first)[0].usage.requests == 1 and results_of(second)[0].usage.requests == 1
    assert_counts_children(first)
    assert_counts_children(second)


def test_a_state_that_ran_before_counts_only_what_it_used_inside_the_call():
    helper = agent_of(["one", "two"])
    kid = user_state("kid")
    helper.run(kid)  # used a request before any parent
    before = kid.usage
    kid.add_message(Message.user("more"))

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run(kid)

    agent = agent_of([tool_call("delegate"), "done"], tools=[delegate])
    parent = user_state()
    agent.run(parent)
    (entry,) = results_of(parent)
    assert entry.usage is not None and entry.usage.requests == 1
    assert kid.usage.requests == before.requests + 1
    assert parent.usage == own_usage(parent) + entry.usage
    assert_counts_children(parent)


def test_a_call_id_used_again_in_a_later_turn_does_not_take_over_the_earlier_children():
    helper = agent_of(["one", "two"])
    kids: list[State] = []

    @tool
    def delegate() -> str:
        """Ask the helper"""
        kid = user_state()
        kids.append(kid)
        return helper.run(kid)

    same_id = [ToolCall("delegate", {}, "reused") for _ in range(2)]
    agent = agent_of([same_id[0], same_id[1], "done"], tools=[delegate])
    parent = user_state()
    agent.run(parent)

    first, second = results_of(parent)
    assert first.child_ids == (kids[0].id,) and second.child_ids == (kids[1].id,)
    assert parent._children_of("reused") == ()
    assert_counts_children(parent)


# ---------------------------------------------------------------- how a call closes


def test_a_tool_error_result_still_lists_its_children():
    helper = agent_of(["inner"])

    @tool
    def delegate() -> str:
        """Ask the helper, then give up"""
        helper.run("Inner task")
        raise ToolError("could not use it")

    agent = agent_of([tool_call("delegate"), "done"], tools=[delegate])
    parent = user_state()
    agent.run(parent)
    (entry,) = results_of(parent)
    assert entry.outcome is ToolOutcomeKind.ERROR
    assert len(entry.child_ids) == 1 and entry.usage == parent.child(entry.child_ids[0]).usage
    assert_counts_children(parent)


def test_a_tool_that_raises_still_lists_its_children_on_the_aborted_result():
    helper = agent_of(["inner"])

    @tool
    def delegate() -> str:
        """Ask the helper, then crash"""
        helper.run("Inner task")
        raise RuntimeError("boom")

    agent = agent_of([tool_call("delegate"), "never"], tools=[delegate])
    parent = user_state()
    with pytest.raises(RuntimeError):
        agent.run(parent)
    (entry,) = results_of(parent)
    assert entry.outcome is ToolOutcomeKind.ABORTED
    assert len(entry.child_ids) == 1 and entry.usage is not None and entry.usage.requests == 1
    assert parent.usage == own_usage(parent) + entry.usage
    assert_counts_children(parent)


def test_a_keyboard_interrupt_in_a_tool_lists_its_children_on_the_interrupted_result():
    helper = agent_of(["inner"])

    @tool
    def delegate() -> str:
        """Ask the helper, then get interrupted"""
        helper.run("Inner task")
        raise KeyboardInterrupt

    agent = agent_of([tool_call("delegate"), "never"], tools=[delegate])
    parent = user_state()
    with pytest.raises(KeyboardInterrupt):
        agent.run(parent)
    (entry,) = results_of(parent)
    assert entry.outcome is ToolOutcomeKind.INTERRUPTED
    assert len(entry.child_ids) == 1 and entry.usage is not None
    assert_counts_children(parent)


class AllowEverything(AllowPermission):
    def check(self, state, call, tool):
        return Allowed()


def test_a_denied_call_has_no_children():
    helper = agent_of(["x"])

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    agent = agent_of(
        [tool_call("delegate"), "done"], tools=[delegate], permissions=[DenyByName(["delegate"]), AllowEverything()]
    )
    parent = user_state()
    agent.run(parent)
    (entry,) = results_of(parent)
    assert entry.outcome is ToolOutcomeKind.DENIED
    assert entry.child_ids == () and entry.usage is None
    assert helper.model.remaining == 1  # type: ignore[attr-defined]
    assert_counts_children(parent)


def test_a_late_result_repeats_the_child_ids_and_counts_only_what_came_after_the_call_closed():
    def slow_reply(request):
        time.sleep(0.05)  # let the interrupt land while the subagent waits for its model
        _thread.interrupt_main()
        time.sleep(0.3)
        return "inner answer"

    helper = agent_of([slow_reply])
    kid = user_state("kid")

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run(kid)

    agent = agent_of([tool_call("delegate"), "never"], tools=[delegate])
    parent = user_state()
    with pytest.raises(KeyboardInterrupt):
        agent.run(parent)

    (closing,) = [e for e in results_of(parent) if not e.late]
    assert closing.outcome is ToolOutcomeKind.INTERRUPTED
    assert closing.child_ids == (kid.id,)
    assert closing.usage == Usage()  # the subagent had used nothing yet when the call closed

    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not any(e.late for e in results_of(parent)):
        time.sleep(0.02)
    (late,) = [e for e in results_of(parent) if e.late]
    assert late.child_ids == (kid.id,)
    assert late.usage == kid.usage and kid.usage.requests == 1
    assert parent.usage == own_usage(parent) + kid.usage  # counted once
    assert_counts_children(parent)


def test_a_late_result_for_a_call_with_no_children_has_no_usage():
    parent = user_state(id="parent")
    call = call_of("slow", "c1")
    parent._start_run(Agent(model=FakeModel([]), reporter=None, human=None), INFO)
    parent._begin_think("fake/fake")
    parent._record_reply(Reply(Message("assistant", (call,)), used(10)))
    parent._close_pending(KeyboardInterrupt())
    parent._late_result(call, "finished", ToolOutcomeKind.DONE)
    late = results_of(parent)[-1]
    assert late.late and late.child_ids == () and late.usage is None
    assert_replays(parent)


def test_a_child_that_starts_after_its_call_closed_is_counted_by_the_late_result():
    parent = user_state(id="parent")
    call = call_of("slow", "c1")
    parent._start_run(Agent(model=FakeModel([]), reporter=None, human=None), INFO)
    parent._begin_think("fake/fake")
    parent._record_reply(Reply(Message("assistant", (call,)), used(10)))
    first = child_state(parent, "c1")
    parent._add_child("c1", first, call)
    parent._close_pending(KeyboardInterrupt())
    assert parent._children_of("c1") == ()  # a later call that reuses the id starts from nothing

    second = child_state(parent, "c1")
    parent._add_child("c1", second, call)  # the tool kept running and started another one after the close
    assert parent._children_of("c1") == ()
    parent._late_result(call, "finished", ToolOutcomeKind.DONE)
    late = results_of(parent)[-1]
    assert late.late and late.child_ids == (first.id, second.id)
    assert parent.child(second.id) is second
    assert_replays(parent)


def test_a_child_that_starts_after_a_call_closed_without_children_is_counted_by_the_late_result():
    parent = user_state(id="parent")
    call = call_of("slow", "c1")
    parent._start_run(Agent(model=FakeModel([]), reporter=None, human=None), INFO)
    parent._begin_think("fake/fake")
    parent._record_reply(Reply(Message("assistant", (call,)), used(10)))
    parent._close_pending(KeyboardInterrupt())  # the tool had not started a subagent yet
    assert results_of(parent)[-1].child_ids == () and results_of(parent)[-1].usage is None

    kid = child_state(parent, "c1")
    parent._add_child("c1", kid, call)  # the tool kept running and started one after the close
    assert parent._children_of("c1") == ()  # not left in the list a later call with this id would take
    kid._start_run(Agent(model=FakeModel([]), reporter=None, human=None), INFO)
    kid._begin_think("fake/fake")
    kid._record_reply(Reply(Message.assistant("done"), used(7)))
    parent._late_result(call, "finished", ToolOutcomeKind.DONE)
    late = results_of(parent)[-1]
    assert late.late and late.child_ids == (kid.id,)
    assert late.usage == kid.usage and kid.usage.requests == 1
    assert parent.child(kid.id) is kid
    assert parent.usage == own_usage(parent) + kid.usage  # counted once
    assert_replays(parent)

    # A later call that reuses the id does not adopt the child.
    parent._begin_think("fake/fake")
    parent._record_reply(Reply(Message("assistant", (call_of("slow", "c1"),)), used(10)))
    assert parent._children_of("c1") == ()


def test_a_former_subagent_that_runs_on_its_own_is_at_depth_zero_again():
    helper = agent_of(["inner", "again"])

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    agent = agent_of([tool_call("delegate"), "done"], tools=[delegate])
    parent = user_state(id="parent")
    agent.run(parent)
    (entry,) = results_of(parent)
    kid = parent.child(entry.child_ids[0])
    assert kid.depth == 1

    kid.add_message(Message.user("Once more"))
    helper.run(kid)  # outside any tool call: not a subagent run
    assert kid.depth == 0
    assert (starts_of(kid)[-1].parent_id, kid.parent_id) == (None, "parent")  # first link kept, this origin recorded
    assert_replays(kid)


# ---------------------------------------------------------------- usage


def test_usage_of_children_is_added_once_even_with_a_second_turn():
    helper = agent_of(["a", "b"])

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    agent = agent_of([tool_call("delegate"), tool_call("delegate"), "done"], tools=[delegate])
    parent = user_state()
    agent.run(parent)
    assert [len(e.child_ids) for e in results_of(parent)] == [1, 1]
    assert parent.usage.requests == 5  # three of the parent, one for each of the two children
    assert_counts_children(parent)


def test_cost_adds_up_across_models_with_different_prices():
    helper = Agent(
        model=FakeModel(["inner"], name="cheap", price=Price(input=1.0, output=1.0)), reporter=None, human=None
    )

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    model = FakeModel([tool_call("delegate"), "done"], name="dear", price=Price(input=100.0, output=100.0))
    agent = Agent(model=model, tools=[delegate], reporter=None, human=None)
    parent = user_state()
    agent.run(parent)

    (entry,) = results_of(parent)
    kid = parent.child(entry.child_ids[0])
    assert entry.usage is not None and entry.usage.cost == pytest.approx(kid.usage.cost)
    assert parent.usage.cost == pytest.approx(own_usage(parent).cost + kid.usage.cost)
    assert kid.usage.cost is not None and own_usage(parent).cost > 50 * kid.usage.cost
    assert_counts_children(parent)


def test_an_unknown_price_anywhere_makes_the_total_cost_unknown():
    helper = agent_of(["inner"])  # no price: its cost is None

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    model = FakeModel([tool_call("delegate"), "done"], price=Price(input=1.0, output=1.0))
    agent = Agent(model=model, tools=[delegate], reporter=None, human=None)
    parent = user_state()
    agent.run(parent)
    (entry,) = results_of(parent)
    assert own_usage(parent).cost is not None
    assert entry.usage is not None and entry.usage.cost is None
    assert parent.usage.cost is None
    assert_counts_children(parent)


def test_a_price_of_zero_keeps_the_total_cost_known():
    free = Price(input=0, output=0)
    helper = Agent(model=FakeModel(["inner"], price=free), reporter=None, human=None)

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    model = FakeModel([tool_call("delegate"), "done"], price=Price(input=1.0, output=1.0))
    agent = Agent(model=model, tools=[delegate], reporter=None, human=None)
    parent = user_state()
    agent.run(parent)
    assert parent.usage.cost is not None
    assert parent.usage.cost == pytest.approx(own_usage(parent).cost)


def test_restore_keeps_the_usage_of_children_and_the_link_survives_it():
    helper = agent_of(["inner"])

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    agent = agent_of([tool_call("delegate"), "done"], tools=[delegate])
    parent = user_state()
    start = parent.snapshot()
    agent.run(parent)
    spent = parent.usage
    assert spent.requests == 3

    parent.restore(start)
    assert parent.usage == spent  # tokens that were spent cannot be given back
    assert len(parent.history) > len(start.history)
    (entry,) = results_of(parent)
    assert parent.child(entry.child_ids[0]).parent_id == parent.id
    assert_replays(parent)


def test_a_saved_parent_loads_with_the_children_and_usage_on_its_results(tmp_path: Path):
    store = FileStore(tmp_path)
    helper = agent_of(["inner"])
    kid = user_state("kid", id="kid1")

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run(kid)

    agent = agent_of([tool_call("delegate"), "done"], tools=[delegate], store=store)
    parent = user_state(id="parent")
    agent.run(parent)

    loaded = assert_loads_equal(store, parent)
    (entry,) = results_of(loaded)
    assert entry.child_ids == ("kid1",)
    assert entry.usage == results_of(parent)[0].usage and entry.usage is not None
    assert loaded.usage == parent.usage
    assert_replays(parent)


# ================================================================ the run context: human, reporter, store
#
# A subagent that was not given a human, reporter or store uses the ones the run that started it uses. A value it was
# given (``None`` included) is its own.


class Recorder(Reporter):
    """Remembers which State each notification was about: ``(method, state id, parent id of that State)``."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, str | None]] = []
        self._lock = threading.Lock()

    def _note(self, method: str, state: State) -> None:
        with self._lock:
            self.events.append((method, state.id, state.parent_id))

    def on_run_start(self, state):
        self._note("run_start", state)

    def on_think_start(self, state):
        self._note("think_start", state)

    def on_tool_start(self, state, call):
        self._note("tool_start", state)

    def on_tool_end(self, state, call, result, outcome):
        self._note("tool_end", state)

    def on_run_end(self, state, error):
        self._note("run_end", state)

    def of(self, state: State) -> list[str]:
        return [method for method, state_id, _ in self.events if state_id == state.id]


def asks_the_human(agent_box: list, question: str = "Proceed?"):
    """A tool for the Agent that ends up in ``agent_box[0]``: it asks that Agent's human."""

    @tool
    def check(state: State) -> str:
        """Ask the person"""
        return agent_box[0].ask_human(state, question)

    return check


def test_a_subagent_without_a_reporter_or_human_uses_those_of_the_run_that_started_it():
    report, human = Recorder(), FakeHuman(["go"])
    box: list = []
    helper = Agent(model=FakeModel([tool_call("check"), "inner"]), tools=[asks_the_human(box)])
    box.append(helper)

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    agent = Agent(model=FakeModel([tool_call("delegate"), "outer"]), tools=[delegate], reporter=report, human=human)
    parent = user_state(id="parent")
    assert agent.run(parent) == "outer"

    (entry,) = results_of(parent)
    kid = parent.child(entry.child_ids[0])
    # the reporter got the child State in every callback, run_start and run_end included
    assert report.of(kid) == ["run_start", "think_start", "tool_start", "tool_end", "think_start", "run_end"]
    assert {parent_id for _, state_id, parent_id in report.events if state_id == kid.id} == {"parent"}
    assert {parent_id for _, state_id, parent_id in report.events if state_id == "parent"} == {None}
    # the question went to the human of the parent run and was recorded in the child's history
    assert human.questions == ["Proceed?"]
    assert [e.kind for e in kid.history if e.kind == "human"] == ["human"]
    assert not [e for e in parent.history if e.kind == "human"]
    assert_replays(parent)
    assert_replays(kid)


def test_a_subagent_with_its_own_reporter_and_human_keeps_them():
    parent_report, own_report = Recorder(), Recorder()
    parent_human, own_human = FakeHuman(["parent"]), FakeHuman(["own"])
    box: list = []
    helper = Agent(
        model=FakeModel([tool_call("check"), "inner"]),
        tools=[asks_the_human(box)],
        reporter=own_report,
        human=own_human,
    )
    box.append(helper)

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    agent = Agent(
        model=FakeModel([tool_call("delegate"), "outer"]),
        tools=[delegate],
        reporter=parent_report,
        human=parent_human,
    )
    parent = user_state(id="parent")
    agent.run(parent)

    kid = parent.child(results_of(parent)[0].child_ids[0])
    assert own_report.of(kid)[0] == "run_start" and own_report.of(kid)[-1] == "run_end"
    assert parent_report.of(kid) == []
    assert own_human.questions == ["Proceed?"] and parent_human.questions == []


def test_a_subagent_given_reporter_none_stays_silent_while_the_parent_reports():
    report = Recorder()
    helper = agent_of(["inner"])  # reporter=None given on purpose

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    agent = Agent(model=FakeModel([tool_call("delegate"), "outer"]), tools=[delegate], reporter=report, human=None)
    parent = user_state(id="parent")
    agent.run(parent)
    kid = parent.child(results_of(parent)[0].child_ids[0])
    assert report.of(kid) == [] and report.of(parent)


def test_a_subagent_given_human_none_has_no_human_even_when_the_parent_has_one():
    box: list = []
    helper = Agent(
        model=FakeModel([tool_call("check"), "inner"]), tools=[asks_the_human(box)], reporter=None, human=None
    )
    box.append(helper)
    human = FakeHuman(["go"])

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    agent = Agent(model=FakeModel([tool_call("delegate"), "outer"]), tools=[delegate], reporter=None, human=human)
    with pytest.raises(NoHumanError):
        agent.run(user_state())
    assert human.questions == []


def test_a_subagent_that_inherits_no_human_has_none_and_a_decide_by_human_fails_before_it_starts():
    helper = Agent(model=FakeModel(["inner"]), reporter=None, permissions=[DecideByHuman()])  # human is inherited

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    agent = Agent(model=FakeModel([tool_call("delegate"), "outer"]), tools=[delegate], reporter=None, human=None)
    parent = user_state(id="parent")
    with pytest.raises(NoHumanError, match="human=None"):
        agent.run(parent)
    # the helper never started: the check comes before its RunStartEntry
    assert parent._children == {}
    assert results_of(parent)[0].outcome is ToolOutcomeKind.ABORTED


def test_the_no_human_error_of_a_subagent_that_inherits_none_names_the_parent_run():
    helper = Agent(model=FakeModel(["inner"]), reporter=None, permissions=[DecideByHuman()])

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    agent = Agent(model=FakeModel([tool_call("delegate"), "outer"]), tools=[delegate], reporter=None, human=None)
    with pytest.raises(NoHumanError) as info:
        agent.run(user_state())
    assert "this subagent has no human to ask" in str(info.value)
    assert "the run that started it has human=None" in str(info.value)
    assert "give one to the Agent that runs it" in str(info.value)


def test_ask_human_in_a_subagent_that_inherits_no_human_names_the_parent_run():
    box: list = []
    helper = Agent(model=FakeModel([tool_call("check"), "inner"]), tools=[asks_the_human(box)], reporter=None)
    box.append(helper)

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    agent = Agent(model=FakeModel([tool_call("delegate"), "outer"]), tools=[delegate], reporter=None, human=None)
    with pytest.raises(NoHumanError, match="called by a subagent with no human to ask") as info:
        agent.run(user_state())
    assert "the run that started it has human=None" in str(info.value)


def test_ask_human_on_an_agent_that_has_human_none_keeps_its_wording():
    agent = Agent(model=FakeModel(["a"]), reporter=None, human=None)
    with pytest.raises(NoHumanError, match="was called on an Agent with human=None"):
        agent.ask_human(user_state("Hi"), "Proceed?")


def test_the_context_passes_down_a_chain_and_a_middle_agent_with_its_own_value_decides_for_its_children():
    top_report, middle_report = Recorder(), Recorder()
    leaf = Agent(model=FakeModel(["leaf"]))
    middle_human = FakeHuman(["m"])

    @tool
    def ask_leaf() -> str:
        """Ask the leaf"""
        return leaf.run("Leaf task")

    middle = Agent(model=FakeModel([tool_call("ask_leaf"), "middle"]), tools=[ask_leaf], reporter=middle_report)

    @tool
    def ask_middle() -> str:
        """Ask the middle"""
        return middle.run("Middle task")

    top = Agent(
        model=FakeModel([tool_call("ask_middle"), "top"]),
        tools=[ask_middle],
        reporter=top_report,
        human=middle_human,
    )
    root = user_state(id="root")
    top.run(root)

    mid = root.child(results_of(root)[0].child_ids[0])
    (leaf_id,) = results_of(mid)[0].child_ids
    kid = mid.child(leaf_id)
    # the middle Agent has its own reporter and the top human; the leaf has neither, so it gets the middle's
    # reporter (not the top's) and the human the middle run uses
    assert top_report.of(mid) == []
    assert middle_report.of(mid)[0] == "run_start"
    assert top_report.of(kid) == []
    assert middle_report.of(kid)[0] == "run_start" and middle_report.of(kid)[-1] == "run_end"
    assert top_report.of(root)


def test_a_custom_loop_that_calls_think_and_use_tools_sees_the_same_context():
    report, human = Recorder(), FakeHuman(["go"])
    box: list = []

    def two_turns(agent: Agent, state: State):
        agent.think(state)
        agent.use_tools(state)
        agent.think(state)
        return state.answer

    helper = Agent(model=FakeModel([tool_call("check"), "inner"]), tools=[asks_the_human(box)], loop=two_turns)
    box.append(helper)

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    agent = Agent(
        model=FakeModel([tool_call("delegate"), "outer"]),
        tools=[delegate],
        loop=two_turns,
        reporter=report,
        human=human,
    )
    parent = user_state(id="parent")
    agent.run(parent)

    kid = parent.child(results_of(parent)[0].child_ids[0])
    assert report.of(kid) == ["run_start", "think_start", "tool_start", "tool_end", "think_start", "run_end"]
    assert human.questions == ["Proceed?"]
    assert_replays(kid)


def test_a_state_made_inside_a_subagent_run_uses_the_context_of_that_run(tmp_path: Path):
    report, human, store = Recorder(), FakeHuman(["go"]), FileStore(tmp_path)
    box: list = []
    forks: list[State] = []

    def tries_on_a_fork(agent: Agent, state: State):
        attempt = state.fork()
        forks.append(attempt)
        agent.think(attempt)
        agent.use_tools(attempt)
        agent.think(attempt)
        return attempt.answer

    helper = Agent(model=FakeModel([tool_call("check"), "inner"]), tools=[asks_the_human(box)], loop=tries_on_a_fork)
    box.append(helper)

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    agent = Agent(
        model=FakeModel([tool_call("delegate"), "outer"]), tools=[delegate], reporter=report, human=human, store=store
    )
    agent.run(user_state(id="parent"))

    (attempt,) = forks
    # the fork is not the State of the run, but it is notified, asked and saved like the run's State
    assert report.of(attempt) == ["think_start", "tool_start", "tool_end", "think_start"]
    assert human.questions == ["Proceed?"]
    assert {info.id for info in store.list()} >= {attempt.id}
    assert store.load(attempt.id).snapshot() == attempt.snapshot()
    assert_replays(attempt)


def test_the_run_context_does_not_outlive_the_run():
    report = Recorder()
    agent = Agent(model=FakeModel(["a", "b"]), reporter=report, human=None)
    state = user_state("Hi")
    agent.run(state)
    assert agent._active_context(state) is None
    assert agent._active_context(State(history=state.history)) is None


def test_another_agent_called_on_a_state_inside_a_run_uses_its_own_settings():
    report, other_report = Recorder(), Recorder()
    other = Agent(model=FakeModel(["from other"]), reporter=other_report, human=None)

    def mixed(agent: Agent, state: State):
        agent.think(state)
        other.think(state)  # another Agent: not the one running, so its reporter, not the run's
        return state.answer

    agent = Agent(model=FakeModel(["from agent"]), loop=mixed, reporter=report, human=None)
    state = user_state("Hi")
    agent.run(state)
    assert report.of(state) == ["run_start", "think_start", "run_end"]
    assert other_report.of(state) == ["think_start"]


def test_a_run_started_outside_a_tool_call_does_not_inherit(monkeypatch: pytest.MonkeyPatch):
    default = Terminal(output=io.StringIO(), input=lambda prompt: "go")
    monkeypatch.setattr(terminal_module, "_default_terminal", default)
    report = Recorder()
    helper = Agent(model=FakeModel(["inner"]))  # defaults, so the shared Terminal

    def starts_it(agent: Agent, state: State):
        helper.run("Inner task")  # in the loop, not in a tool call
        agent.think(state)
        return state.answer

    agent = Agent(model=FakeModel(["outer"]), loop=starts_it, reporter=report, human=None)
    state = user_state(id="parent")
    agent.run(state)
    assert [e for e in report.events if e[0] == "run_start"] == [("run_start", "parent", None)]
    assert default._output.getvalue()  # type: ignore[union-attr]  # the helper reported to the shared Terminal


async def test_an_async_subagent_inherits_the_context_too():
    report, human = Recorder(), FakeHuman(["go"])
    box: list = []
    helper = Agent(model=FakeModel([tool_call("check"), "inner"]), tools=[asks_the_human(box)])
    box.append(helper)

    @tool
    async def delegate() -> str:
        """Ask the helper"""
        return await helper.arun("Inner task")

    agent = Agent(model=FakeModel([tool_call("delegate"), "outer"]), tools=[delegate], reporter=report, human=human)
    parent = user_state(id="parent")
    assert await agent.arun(parent) == "outer"
    kid = parent.child(results_of(parent)[0].child_ids[0])
    assert report.of(kid)[0] == "run_start" and report.of(kid)[-1] == "run_end"
    assert human.questions == ["Proceed?"]
    assert_replays(kid)


def test_agent_properties_still_show_the_agents_own_settings():
    helper = Agent(model=FakeModel(["inner"]))
    assert helper.reporter is default_terminal() and helper.human is default_terminal()
    assert helper.store is None


# ---------------------------------------------------------------- saving


def test_children_are_saved_under_their_own_id_in_the_store_of_the_parent_run(tmp_path: Path):
    store = FileStore(tmp_path)
    helper = agent_of(["inner"])  # no store of its own

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    agent = agent_of([tool_call("delegate"), "outer"], tools=[delegate], store=store)
    parent = user_state(id="parent")
    agent.run(parent)

    kid = parent.child(results_of(parent)[0].child_ids[0])
    assert {info.id for info in store.list()} == {"parent", kid.id}
    assert {info.id: info.parent_id for info in store.list()} == {"parent": None, kid.id: "parent"}
    loaded_kid = store.load(kid.id)
    assert loaded_kid.snapshot() == kid.snapshot()
    assert (loaded_kid.parent_id, loaded_kid.parent_call_id) == (kid.parent_id, kid.parent_call_id)
    assert_loads_equal(store, parent)
    assert_replays(kid)
    assert_replays(parent)


def test_a_child_is_saved_while_it_runs_at_the_normal_save_points(tmp_path: Path):
    store = FileStore(tmp_path)
    seen: list[tuple[str, int, int]] = []

    @tool
    def peek(state: State) -> str:
        """Look at what the store has of this State and of its parent"""
        for saved in (store.load(state.id), store.load(state.parent_id)):
            seen.append((saved.id, saved.turn, sum(isinstance(e, ModelReplyEntry) for e in saved.history)))
        return "peeked"

    helper = agent_of([tool_call("peek"), "inner"], tools=[peek])

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    agent = agent_of([tool_call("delegate"), "outer"], tools=[delegate], store=store)
    parent = user_state(id="parent")
    agent.run(parent)

    kid = parent.child(results_of(parent)[0].child_ids[0])
    # the reply of turn 1 (asking for the tool) was saved before the tool ran, for the child and for its parent
    assert seen == [(kid.id, 1, 1), ("parent", 1, 1)]
    assert store.load(kid.id).snapshot() == kid.snapshot()
    assert_replays(kid)


def test_a_child_with_its_own_store_is_saved_there_and_not_in_the_parents(tmp_path: Path):
    parent_store, own_store = FileStore(tmp_path / "parent"), FileStore(tmp_path / "own")
    helper = agent_of(["inner"], store=own_store)

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    agent = agent_of([tool_call("delegate"), "outer"], tools=[delegate], store=parent_store)
    parent = user_state(id="parent")
    agent.run(parent)

    kid = parent.child(results_of(parent)[0].child_ids[0])
    assert [info.id for info in parent_store.list()] == ["parent"]
    assert [info.id for info in own_store.list()] == [kid.id]
    assert own_store.load(kid.id).snapshot() == kid.snapshot()
    # the loaded parent is bound to the parent store only, so it cannot open a child that lives in another store
    loaded = parent_store.load("parent")
    with pytest.raises(LookupError):
        loaded.child(kid.id)


def test_a_child_without_a_store_in_a_run_without_a_store_is_not_saved():
    helper = agent_of(["inner"])

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    agent = agent_of([tool_call("delegate"), "outer"], tools=[delegate])
    parent = user_state()
    agent.run(parent)
    assert parent.child(results_of(parent)[0].child_ids[0])._saved_store is None


def test_after_store_load_the_parent_opens_its_child_from_the_bound_store(tmp_path: Path):
    store = FileStore(tmp_path)
    helper = agent_of(["inner"])

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    agent = agent_of([tool_call("delegate"), "outer"], tools=[delegate], store=store)
    parent = user_state(id="parent")
    agent.run(parent)
    (child_id,) = results_of(parent)[0].child_ids
    kid = parent.child(child_id)

    loaded = store.load("parent")
    opened = loaded.child(child_id)
    assert opened is not kid and opened.snapshot() == kid.snapshot()
    assert (opened.parent_id, opened.parent_call_id) == ("parent", results_of(parent)[0].call.id)
    assert loaded.usage == parent.usage and loaded.usage.requests == 3  # 2 of its own + the child's


def test_a_child_loaded_from_the_store_is_a_normal_state_that_can_be_continued(tmp_path: Path):
    store = FileStore(tmp_path)
    helper = agent_of(["inner", "more"], store=store)

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    agent = agent_of([tool_call("delegate"), "outer"], tools=[delegate], store=store)
    parent = user_state(id="parent")
    agent.run(parent)
    (child_id,) = results_of(parent)[0].child_ids

    loaded = store.load(child_id)
    loaded.add_message(Message.user("Go on"))
    assert helper.run(loaded) == "more"
    assert loaded.parent_id == "parent"  # the first link stays
    assert store.load(child_id).snapshot() == loaded.snapshot()
    assert_replays(loaded)


def test_grandchildren_are_saved_too_and_the_chain_opens_after_load(tmp_path: Path):
    store = FileStore(tmp_path)
    leaf = agent_of(["leaf"])

    @tool
    def ask_leaf() -> str:
        """Ask the leaf"""
        return leaf.run("Leaf task")

    middle = agent_of([tool_call("ask_leaf"), "middle"], tools=[ask_leaf])

    @tool
    def ask_middle() -> str:
        """Ask the middle"""
        return middle.run("Middle task")

    top = agent_of([tool_call("ask_middle"), "top"], tools=[ask_middle], store=store)
    root = user_state(id="root")
    top.run(root)

    mid = root.child(results_of(root)[0].child_ids[0])
    kid = mid.child(results_of(mid)[0].child_ids[0])
    assert {info.id: info.parent_id for info in store.list()} == {"root": None, mid.id: "root", kid.id: mid.id}
    loaded = store.load("root")
    assert loaded.child(mid.id).child(kid.id).snapshot() == kid.snapshot()
    assert kid.usage.requests == 1 and mid.usage.requests == 3  # mid: 2 requests of its own + the leaf's 1
    assert_loads_equal(store, root)


def test_a_failing_save_of_a_child_run_stops_that_run_like_any_other(tmp_path: Path):
    class FailingStore(FileStore):
        def write(self, state_id, entries, info, *, create=False):
            if state_id != "parent":
                raise OSError("disk full")
            return super().write(state_id, entries, info, create=create)

    helper = agent_of(["inner"])

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    agent = agent_of([tool_call("delegate"), "outer"], tools=[delegate], store=FailingStore(tmp_path))
    parent = user_state(id="parent")
    with pytest.raises(OSError, match="disk full"):
        agent.run(parent)
    # the child never started its run (the first save failed), but the parent's call is still closed in its history
    assert results_of(parent)[0].outcome is ToolOutcomeKind.ABORTED
    assert_replays(parent)


async def test_an_async_run_saves_children_with_asave(tmp_path: Path):
    store = AsyncOnlyStore()
    helper = agent_of(["inner"])

    @tool
    async def delegate() -> str:
        """Ask the helper"""
        return await helper.arun("Inner task")

    agent = agent_of([tool_call("delegate"), "outer"], tools=[delegate], store=store)
    parent = user_state(id="parent")
    await agent.arun(parent)
    (child_id,) = results_of(parent)[0].child_ids
    kid = await store.aload(child_id)
    assert kid.snapshot() == parent.child(child_id).snapshot()
    assert (await (await store.aload("parent")).achild(child_id)).snapshot() == kid.snapshot()


async def test_a_sync_run_in_a_tool_cannot_inherit_a_store_that_only_works_asynchronously():
    store = AsyncOnlyStore()
    helper = agent_of(["inner"])

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")  # a worker thread: sync run is allowed, the inherited store is not

    agent = agent_of([tool_call("delegate"), "outer"], tools=[delegate], store=store)
    parent = user_state(id="parent")
    with pytest.raises(TypeError, match="works only asynchronously"):
        await agent.arun(parent)
    assert parent._children == {}


def test_state_info_of_a_top_level_run_and_a_subagent_differ_by_parent_id(tmp_path: Path):
    store = FileStore(tmp_path)
    helper = agent_of(["inner"])

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    agent = agent_of([tool_call("delegate"), "outer"], tools=[delegate], store=store)
    agent.run(user_state(id="parent"))
    agent_of(["alone"], store=store).run(user_state(id="another"))
    top_level = sorted(info.id for info in store.list() if info.parent_id is None)
    assert top_level == ["another", "parent"]


# ---------------------------------------------------------------- permissions
#
# ``permissions=`` applies to every tool call of the run, the calls inside its subagents included. A subagent's calls
# are checked by the DenyPermission of the runs above plus its own list, or the parent run's whole list when it has
# none. Each permission asks the human of the Agent that wrote it.


@tool
def peek() -> str:
    """Peek at the files"""
    return "peeked"


def passing_on(helper: Agent, name: str = "delegate"):
    """A tool for the Agent above ``helper`` that runs it (the helper becomes a subagent)."""

    @tool(name=name)
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    return delegate


def passing_on_async(helper: Agent, name: str = "delegate"):
    @tool(name=name)
    async def delegate() -> str:
        """Ask the helper"""
        return await helper.arun("Inner task")

    return delegate


def kid_agent(replies, **settings) -> Agent:
    """A subagent that peeks once. It gives no reporter or human, so it takes those of the run above."""
    return Agent(model=FakeModel(replies), tools=[peek], **settings)


def kid_of(state: State, index: int = 0) -> State:
    return state.child(results_of(state)[index].child_ids[0])


class Outcomes(Reporter):
    """Remembers how every tool call ended: ``(state id, tool name, outcome kind, decided_by)``."""

    def __init__(self) -> None:
        self.ends: list[tuple[str, str, ToolOutcomeKind, str | None]] = []
        self._lock = threading.Lock()

    def on_tool_end(self, state, call, result, outcome):
        with self._lock:
            self.ends.append((state.id, call.name, outcome.kind, outcome.decided_by))

    def of(self, state: State) -> list[tuple[str, ToolOutcomeKind, str | None]]:
        return [(name, kind, by) for state_id, name, kind, by in self.ends if state_id == state.id]


NO_PEEKING = DenyByName(["peek"], reason="no peeking")


def human_questions(state: State) -> list[str]:
    return [e.content.question for e in state.history if e.kind == "human"]


@pytest.mark.parametrize("own", [None, [AllowByDefault()]])
def test_a_deny_of_the_parent_blocks_the_call_of_a_subagent(own):
    report = Outcomes()
    helper = kid_agent(([tool_call("peek"), "inner"]), permissions=own)
    agent = Agent(
        model=FakeModel([tool_call("delegate"), "outer"]),
        tools=[passing_on(helper)],
        permissions=[NO_PEEKING, AllowByDefault()],
        reporter=report,
        human=None,
    )
    parent = user_state(id="parent")
    assert agent.run(parent) == "outer"

    kid = kid_of(parent)
    (peeked,) = results_of(kid)
    assert peeked.outcome is ToolOutcomeKind.DENIED and peeked.content == "no peeking"
    # the rule that decided is the parent's, and it is told about the child's call
    assert report.of(kid) == [("peek", ToolOutcomeKind.DENIED, repr(NO_PEEKING))]
    assert report.of(parent) == [("delegate", ToolOutcomeKind.DONE, None)]
    assert_counts_children(parent)
    assert_counts_children(kid)


def test_a_subagent_without_permissions_runs_under_the_whole_list_of_the_parent_run():
    parent_human, own_human = FakeHuman(["yes"]), FakeHuman([])
    helper = kid_agent([tool_call("peek"), "inner"], human=own_human)
    agent = Agent(
        model=FakeModel([tool_call("delegate"), "outer"]),
        tools=[passing_on(helper)],
        permissions=[AllowByName(["delegate"]), DecideByHuman()],
        reporter=None,
        human=parent_human,
    )
    parent = user_state(id="parent")
    assert agent.run(parent) == "outer"

    kid = kid_of(parent)
    # the parent's DecideByHuman asked the parent's person, not the one the subagent was given
    assert parent_human.questions == ["Run peek()?"]
    assert own_human.questions == []
    assert results_of(kid)[0].outcome is ToolOutcomeKind.DONE
    # the question and the answer are in the history of the State that made the call
    assert human_questions(kid) == ["Run peek()?"] and human_questions(parent) == []
    assert_replays(parent)
    assert_replays(kid)


def test_a_no_from_the_person_stops_the_subagent_only_and_is_recorded_in_its_history():
    report, human = Outcomes(), FakeHuman(["no"])
    helper = kid_agent([tool_call("peek"), "never asked again"])
    agent = Agent(
        model=FakeModel([tool_call("delegate"), "outer"]),
        tools=[passing_on(helper)],
        permissions=[AllowByName(["delegate"]), DecideByHuman()],
        reporter=report,
        human=human,
    )
    parent = user_state(id="parent")
    assert agent.run(parent) == "outer"

    kid = kid_of(parent)
    assert isinstance(kid.stopped, StoppedByPermission) and kid.stopped.permission == "DecideByHuman()"
    assert kid.stopped.call.name == "peek"
    assert not isinstance(parent.stopped, StoppedByPermission)
    assert report.of(kid) == [("peek", ToolOutcomeKind.DENIED, "DecideByHuman()")]
    assert human.questions == ["Run peek()?"]
    assert_replays(parent)
    assert_replays(kid)


def test_a_subagent_with_a_list_of_its_own_is_not_asked_about_by_the_allow_and_decide_rules_of_the_parent():
    human = FakeHuman([])
    helper = kid_agent([tool_call("peek"), "inner"], permissions=[AllowByDefault()])
    agent = Agent(
        model=FakeModel([tool_call("delegate"), "outer"]),
        tools=[passing_on(helper)],
        permissions=[AllowByName(["delegate"]), DecideByHuman()],  # it would ask about peek, and say no to the rest
        reporter=None,
        human=human,
    )
    parent = user_state(id="parent")
    agent.run(parent)
    kid = kid_of(parent)
    assert results_of(kid)[0].outcome is ToolOutcomeKind.DONE
    assert human.questions == []
    assert_replays(kid)


def test_the_implicit_deny_the_rest_of_a_parent_list_does_not_bind_a_subagent_with_its_own_list():
    helper = kid_agent([tool_call("peek"), "inner"], permissions=[AllowByDefault()])
    agent = Agent(
        model=FakeModel([tool_call("delegate"), "outer"]),
        tools=[passing_on(helper)],
        permissions=[AllowByName(["delegate"])],  # nothing else is allowed for the parent's own calls
        reporter=None,
        human=None,
    )
    parent = user_state()
    agent.run(parent)
    assert results_of(kid_of(parent))[0].outcome is ToolOutcomeKind.DONE


def test_a_subagent_with_an_empty_list_has_every_call_denied_with_a_warning():
    helper = kid_agent([tool_call("peek"), "inner"], permissions=[])
    agent = Agent(
        model=FakeModel([tool_call("delegate"), "outer"]),
        tools=[passing_on(helper)],
        permissions=[AllowByDefault()],
        reporter=None,
        human=None,
    )
    parent = user_state()
    with pytest.warns(PermissionWarning, match="peek"):
        agent.run(parent)
    kid = kid_of(parent)
    assert results_of(kid)[0].outcome is ToolOutcomeKind.DENIED
    assert results_of(parent)[0].outcome is ToolOutcomeKind.DONE


def test_a_deny_of_the_parent_decides_before_the_empty_list_of_a_subagent_so_nothing_warns():
    report = Outcomes()
    helper = kid_agent([tool_call("peek"), "inner"], permissions=[])
    agent = Agent(
        model=FakeModel([tool_call("delegate"), "outer"]),
        tools=[passing_on(helper)],
        permissions=[NO_PEEKING, AllowByDefault()],
        reporter=report,
        human=None,
    )
    parent = user_state()
    with warnings.catch_warnings():
        warnings.simplefilter("error", PermissionWarning)
        agent.run(parent)
    assert report.of(kid_of(parent)) == [("peek", ToolOutcomeKind.DENIED, repr(NO_PEEKING))]


@pytest.mark.parametrize("leaf_permissions", [None, [AllowByDefault()]])
def test_a_deny_of_the_top_run_reaches_a_grandchild(leaf_permissions):
    report = Outcomes()
    leaf = kid_agent([tool_call("peek"), "leaf"], permissions=leaf_permissions)
    middle = Agent(
        model=FakeModel([tool_call("delegate_leaf"), "middle"]),
        tools=[passing_on(leaf, "delegate_leaf")],
        permissions=[AllowByDefault()],  # its own list: the allow rules of the top run are not asked
    )
    top = Agent(
        model=FakeModel([tool_call("delegate_middle"), "top"]),
        tools=[passing_on(middle, "delegate_middle")],
        permissions=[NO_PEEKING, AllowByDefault()],
        reporter=report,
        human=None,
    )
    state = user_state(id="top")
    assert top.run(state) == "top"
    middle_state = kid_of(state)
    leaf_state = kid_of(middle_state)
    assert [e.outcome for e in results_of(middle_state)] == [ToolOutcomeKind.DONE]
    assert report.of(leaf_state) == [("peek", ToolOutcomeKind.DENIED, repr(NO_PEEKING))]
    assert_replays(state)
    assert_replays(middle_state)
    assert_replays(leaf_state)


def test_the_denials_of_every_run_above_apply_top_run_first():
    order: list[str] = []

    class Noting(DenyPermission):
        def __init__(self, label: str) -> None:
            self.label = label

        def check(self, state, call, tool):
            if call.name == "peek":
                order.append(self.label)
            return None

        def __repr__(self) -> str:
            return f"Noting({self.label!r})"

    leaf = kid_agent([tool_call("peek"), "leaf"], permissions=[Noting("leaf"), AllowByDefault()])
    middle = Agent(
        model=FakeModel([tool_call("delegate_leaf"), "middle"]),
        tools=[passing_on(leaf, "delegate_leaf")],
        permissions=[Noting("middle"), AllowByDefault()],
    )
    top = Agent(
        model=FakeModel([tool_call("delegate_middle"), "top"]),
        tools=[passing_on(middle, "delegate_middle")],
        permissions=[AllowByDefault(), Noting("top")],  # a deny sits first wherever it is written
        reporter=None,
        human=None,
    )
    top.run(user_state())
    assert order == ["top", "middle", "leaf"]


def test_a_denial_is_not_asked_twice_when_the_subagent_takes_the_whole_list():
    asked: list[str] = []

    class Counting(DenyPermission):
        def check(self, state, call, tool):
            asked.append(call.name)
            return None

    helper = kid_agent([tool_call("peek"), "inner"])  # permissions=None: the list of the parent run
    agent = Agent(
        model=FakeModel([tool_call("delegate"), "outer"]),
        tools=[passing_on(helper)],
        permissions=[Counting(), AllowByDefault()],
        reporter=None,
        human=None,
    )
    agent.run(user_state())
    assert asked == ["delegate", "peek"]


def test_without_permissions_anywhere_nothing_is_checked():
    helper = kid_agent([tool_call("peek"), "inner"])
    agent = Agent(
        model=FakeModel([tool_call("delegate"), "outer"]), tools=[passing_on(helper)], reporter=None, human=None
    )
    parent = user_state()
    agent.run(parent)
    assert results_of(kid_of(parent))[0].outcome is ToolOutcomeKind.DONE


def test_a_subagent_with_a_list_of_its_own_under_a_parent_without_one_uses_only_its_own():
    report = Outcomes()
    helper = kid_agent([tool_call("peek"), "inner"], permissions=[NO_PEEKING, AllowByDefault()])
    agent = Agent(
        model=FakeModel([tool_call("delegate"), "outer"]), tools=[passing_on(helper)], reporter=report, human=None
    )
    parent = user_state()
    agent.run(parent)
    assert report.of(parent) == [("delegate", ToolOutcomeKind.DONE, None)]
    assert report.of(kid_of(parent)) == [("peek", ToolOutcomeKind.DENIED, repr(NO_PEEKING))]


def test_a_decide_by_human_of_the_subagent_asks_the_human_of_the_subagent():
    parent_human, own_human = FakeHuman([]), FakeHuman(["yes"])
    helper = kid_agent([tool_call("peek"), "inner"], human=own_human, permissions=[DecideByHuman()])
    agent = Agent(
        model=FakeModel([tool_call("delegate"), "outer"]),
        tools=[passing_on(helper)],
        permissions=[AllowByDefault()],
        reporter=None,
        human=parent_human,
    )
    parent = user_state()
    agent.run(parent)
    assert own_human.questions == ["Run peek()?"] and parent_human.questions == []
    assert results_of(kid_of(parent))[0].outcome is ToolOutcomeKind.DONE


def test_a_decide_by_human_of_a_subagent_without_a_human_of_its_own_asks_the_human_of_the_parent_run():
    parent_human = FakeHuman(["yes"])
    helper = kid_agent([tool_call("peek"), "inner"], permissions=[DecideByHuman()])
    agent = Agent(
        model=FakeModel([tool_call("delegate"), "outer"]),
        tools=[passing_on(helper)],
        permissions=[AllowByDefault()],
        reporter=None,
        human=parent_human,
    )
    parent = user_state()
    agent.run(parent)
    assert parent_human.questions == ["Run peek()?"]
    assert human_questions(kid_of(parent)) == ["Run peek()?"]


def test_a_decide_by_human_with_a_human_of_its_own_keeps_asking_it_inside_a_subagent():
    parent_human, approvals = FakeHuman([]), FakeHuman(["yes"])
    helper = kid_agent([tool_call("peek"), "inner"])
    agent = Agent(
        model=FakeModel([tool_call("delegate"), "outer"]),
        tools=[passing_on(helper)],
        permissions=[AllowByName(["delegate"]), DecideByHuman(human=approvals)],
        reporter=None,
        human=parent_human,
    )
    parent = user_state()
    agent.run(parent)
    assert approvals.questions == ["Run peek()?"] and parent_human.questions == []


def test_a_permission_asks_the_human_of_the_run_that_wrote_it_down_a_chain():
    top_human, middle_human, leaf_human = FakeHuman(["yes"]), FakeHuman(["yes"]), FakeHuman([])
    leaf = kid_agent([tool_call("peek"), "leaf"], human=leaf_human)  # no permissions: the whole list of the middle
    middle = Agent(
        model=FakeModel([tool_call("delegate_leaf"), "middle"]),
        tools=[passing_on(leaf, "delegate_leaf")],
        permissions=[AllowByName(["delegate_leaf"]), DecideByHuman()],
        human=middle_human,
    )
    top = Agent(
        model=FakeModel([tool_call("delegate_middle"), "top"]),
        tools=[passing_on(middle, "delegate_middle")],
        permissions=[AllowByDefault()],
        reporter=None,
        human=top_human,
    )
    state = user_state()
    top.run(state)
    middle_state = kid_of(state)
    # the middle run wrote the DecideByHuman, so its person is asked about the leaf's call, and the leaf's person is not
    assert middle_human.questions == ["Run peek()?"]
    assert top_human.questions == [] and leaf_human.questions == []
    assert human_questions(kid_of(middle_state)) == ["Run peek()?"]


def test_a_decide_by_human_of_the_top_run_asks_the_top_person_for_a_grandchild_that_has_a_human_in_between():
    top_human, middle_human = FakeHuman(["yes"]), FakeHuman([])
    leaf = kid_agent([tool_call("peek"), "leaf"])
    middle = Agent(  # no permissions: the whole list of the top run, with the top run's person
        model=FakeModel([tool_call("delegate_leaf"), "middle"]),
        tools=[passing_on(leaf, "delegate_leaf")],
        human=middle_human,
    )
    top = Agent(
        model=FakeModel([tool_call("delegate_middle"), "top"]),
        tools=[passing_on(middle, "delegate_middle")],
        permissions=[AllowByName(["delegate_middle", "delegate_leaf"]), DecideByHuman()],
        reporter=None,
        human=top_human,
    )
    state = user_state()
    top.run(state)
    assert top_human.questions == ["Run peek()?"] and middle_human.questions == []


class AsyncOnlyDeny(DenyPermission):
    async def acheck(self, state, call, tool):
        return None


class AsyncHuman(Human):
    def __init__(self, answers):
        self.answers = list(answers)
        self.questions: list[str] = []

    async def aask(self, state, prompt, returns=str):
        self.questions.append(prompt)
        return self.answers.pop(0)


@pytest.mark.parametrize("own", [None, [AllowByDefault()]])
async def test_an_async_only_permission_of_the_parent_run_fails_a_sync_subagent_before_it_starts(own):
    helper = kid_agent(["inner"], permissions=own)  # even a list of its own keeps the parent's denials
    agent = Agent(
        model=FakeModel([tool_call("delegate"), "outer"]),
        tools=[passing_on(helper)],  # a sync tool: the subagent is started with the sync run
        permissions=[AsyncOnlyDeny(), AllowByDefault()],
        reporter=None,
        human=None,
    )
    parent = user_state()
    with pytest.raises(TypeError, match="checks calls only asynchronously"):
        await agent.arun(parent)
    assert parent._children == {}  # the check comes before the RunStartEntry
    assert results_of(parent)[0].outcome is ToolOutcomeKind.ABORTED


async def test_the_same_permission_is_fine_for_an_async_subagent_and_the_checks_run_with_acheck():
    helper = kid_agent([tool_call("peek"), "inner"])
    agent = Agent(
        model=FakeModel([tool_call("delegate"), "outer"]),
        tools=[passing_on_async(helper)],
        permissions=[AsyncOnlyDeny(), AllowByDefault()],
        reporter=None,
        human=None,
    )
    parent = user_state()
    assert await agent.arun(parent) == "outer"
    assert results_of(kid_of(parent))[0].outcome is ToolOutcomeKind.DONE


async def test_a_human_of_the_parent_run_that_answers_only_asynchronously_fails_a_sync_subagent_that_would_ask_it():
    human = AsyncHuman([])
    helper = kid_agent(["inner"])  # the whole list of the parent run: its DecideByHuman asks the parent's person
    agent = Agent(
        model=FakeModel([tool_call("delegate"), "outer"]),
        tools=[passing_on(helper)],
        permissions=[AllowByName(["delegate"]), DecideByHuman()],
        reporter=None,
        human=human,
    )
    parent = user_state()
    with pytest.raises(TypeError, match="answers only asynchronously"):
        await agent.arun(parent)
    assert parent._children == {} and human.questions == []


async def test_a_sync_subagent_with_a_list_of_its_own_does_not_ask_the_async_human_of_the_parent_run():
    human = AsyncHuman([])
    helper = kid_agent([tool_call("peek"), "inner"], permissions=[AllowByDefault()])
    agent = Agent(
        model=FakeModel([tool_call("delegate"), "outer"]),
        tools=[passing_on(helper)],
        permissions=[AllowByName(["delegate"]), DecideByHuman()],
        reporter=None,
        human=human,
    )
    parent = user_state()
    assert await agent.arun(parent) == "outer"
    assert human.questions == []


async def test_an_async_subagent_is_checked_by_the_parent_list_and_asks_the_parent_human_with_aask():
    human = AsyncHuman(["yes"])
    helper = kid_agent([tool_call("peek"), "inner"])
    agent = Agent(
        model=FakeModel([tool_call("delegate"), "outer"]),
        tools=[passing_on_async(helper)],
        permissions=[AllowByName(["delegate"]), DecideByHuman()],
        reporter=None,
        human=human,
    )
    parent = user_state()
    assert await agent.arun(parent) == "outer"
    kid = kid_of(parent)
    assert human.questions == ["Run peek()?"]
    assert results_of(kid)[0].outcome is ToolOutcomeKind.DONE
    assert human_questions(kid) == ["Run peek()?"]
    assert_replays(kid)


def test_no_human_for_a_decide_by_human_of_the_subagent_itself_fails_before_it_starts():
    helper = kid_agent(["inner"], human=None, permissions=[DecideByHuman()])
    agent = Agent(
        model=FakeModel([tool_call("delegate"), "outer"]),
        tools=[passing_on(helper)],
        reporter=None,
        human=FakeHuman([]),
    )
    parent = user_state()
    with pytest.raises(NoHumanError, match="human=None"):
        agent.run(parent)
    assert parent._children == {}


def test_a_decide_by_human_of_the_parent_does_not_need_a_human_of_the_subagent():
    helper = kid_agent([tool_call("peek"), "inner"], human=None)
    agent = Agent(
        model=FakeModel([tool_call("delegate"), "outer"]),
        tools=[passing_on(helper)],
        permissions=[AllowByName(["delegate"]), DecideByHuman()],
        reporter=None,
        human=FakeHuman(["yes"]),
    )
    parent = user_state()
    assert agent.run(parent) == "outer"
    assert results_of(kid_of(parent))[0].outcome is ToolOutcomeKind.DONE


def test_another_agent_called_on_a_state_inside_a_run_checks_with_its_own_permissions():
    other = Agent(
        model=FakeModel([tool_call("peek"), "from other"]),
        tools=[peek],
        permissions=[NO_PEEKING, AllowByDefault()],
        reporter=None,
        human=None,
    )

    def mixed(agent: Agent, state: State):
        other.think(state)  # another Agent: its own list, not the one of the run
        other.use_tools(state)
        return state.answer

    agent = Agent(
        model=FakeModel(["x"]), tools=[peek], loop=mixed, permissions=[AllowByDefault()], reporter=None, human=None
    )
    state = user_state()
    agent.run(state)
    assert results_of(state)[0].outcome is ToolOutcomeKind.DENIED


def test_permissions_of_the_agent_still_show_its_own_list_only():
    helper = kid_agent(["inner"])
    agent = Agent(
        model=FakeModel(["x"]), tools=[passing_on(helper)], permissions=[NO_PEEKING], reporter=None, human=None
    )
    assert helper.permissions == () and agent.permissions == (NO_PEEKING,)
