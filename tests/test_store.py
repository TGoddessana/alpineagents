"""Store tests: save/load, the Store.write contract, replaying every entry on load, closing a State saved in the
middle of a turn, the format version check, FileStore on disk, and what happens when saving fails. No network.

A crash is simulated by copying a FileStore folder in the middle of a run: the copy is what a process that died
at that moment would leave on disk.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import shutil
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel

from alpineagents import (
    Agent,
    ErrorEntry,
    FileStore,
    Image,
    Message,
    MessageEntry,
    ModelReplyEntry,
    ModelRequestEntry,
    Reply,
    State,
    StateInfo,
    StoppedByFinish,
    StoppedByUntil,
    Store,
    ToolOutcomeKind,
    ToolResultEntry,
    Usage,
    loop,
    tool,
)
from alpineagents import _serial
from alpineagents.errors import RateLimitError
from alpineagents.state import STOPPED_WAITING, UNSAVED_RESULT
from alpineagents.store import Record
from alpineagents.testing import FakeModel, tool_call


def make_agent(replies=("Done",), **settings) -> Agent:
    settings.setdefault("reporter", None)
    settings.setdefault("human", None)
    if "model" not in settings:
        settings["model"] = FakeModel(list(replies))
    return Agent(**settings)


def user_state(text: str = "Go", **kwargs) -> State:
    return State(messages=[Message.user(text)], **kwargs)


def crash_copy(store: FileStore, state_id: str, into: Path) -> FileStore:
    """What is on disk right now, as another FileStore."""
    shutil.copytree(Path(store._root) / state_id, into / state_id, dirs_exist_ok=True)
    return FileStore(into)


def waiting_for_user(state: State) -> bool:
    if state.pending_calls or not state.messages:
        return False
    last = state.messages[-1]
    return last.role == "assistant" and not last.tool_calls


def an_info(**changes) -> dict:
    """A valid info dict, for tests that call ``Store.write`` by hand."""
    values = {
        "first_message": None,
        "created_at": None,
        "updated_at": None,
        "turn": 0,
        "stopped": None,
        "finished": False,
    }
    return dict(_serial.info_to_dict(**{**values, **changes}))


def entry_dicts(*entries) -> list[dict]:
    return [_serial.entry_to_dict(entry, seq) for seq, entry in enumerate(entries)]


class MemoryStore(Store):
    """A Store in a dict, following the write contract (create, skip known seqs). Keeps what it was asked."""

    def __init__(self) -> None:
        self.states: dict[str, Record] = {}
        self.writes = 0
        self.calls: list[tuple[str, list[int], bool]] = []  # (state id, seqs, create)

    def write(self, state_id, entries, info, *, create=False):
        self.writes += 1
        self.calls.append((state_id, [e["seq"] for e in entries], create))
        if create and state_id in self.states:
            raise ValueError(f"{state_id} exists")
        if not create and state_id not in self.states:
            raise LookupError(state_id)
        record = self.states.setdefault(state_id, Record([], {}))
        have = len(record.entries)
        record.entries.extend(json.loads(json.dumps(e)) for e in entries if e["seq"] >= have)
        self.states[state_id] = Record(record.entries, json.loads(json.dumps(info)))

    def read(self, state_id):
        return self.states.get(state_id)

    def list(self):
        return sorted(
            (StateInfo.from_info(state_id, record.info) for state_id, record in self.states.items()),
            key=lambda info: info.updated_at or info.created_at or 0,
            reverse=True,
        )


class FailingStore(MemoryStore):
    """Fails writes (with a new ``error``, writing nothing) while ``failing`` is set, or when
    ``fail_when(entries, info)`` is true."""

    def __init__(self) -> None:
        super().__init__()
        self.failing = False
        self.fail_when = None
        self.error = lambda: OSError("disk full")
        self.failed = 0

    def write(self, state_id, entries, info, *, create=False):
        if self.failing or (self.fail_when is not None and self.fail_when(entries, info)):
            self.failed += 1
            raise self.error()
        super().write(state_id, entries, info, create=create)


class InfoFailingStore(FileStore):
    """Writes the entries, then fails before the info while ``failing`` is set (a half-done save)."""

    failing = False

    def write(self, state_id, entries, info, *, create=False):
        if self.failing:
            self._entries_only(state_id, entries)
            raise OSError("disk full")
        super().write(state_id, entries, info, create=create)

    def _entries_only(self, state_id, entries):
        from alpineagents.store import _lines, _write_file

        log = self._root / state_id / "log.jsonl"
        have = len(log.read_text("utf-8").splitlines())
        _write_file(log, _lines([e for e in entries if e["seq"] >= have]), append=True)


# ================================================================ State id


def test_state_id_is_random_by_default():
    assert user_state().id != user_state().id
    assert len(user_state().id) == 32


def test_state_id_given():
    assert user_state(id="bug-hunt_1").id == "bug-hunt_1"


@pytest.mark.parametrize("bad", ["", "../x", "a.b", "a b", "x" * 129])
def test_state_id_rejects_unsafe_characters(bad):
    with pytest.raises(ValueError, match="letters, digits"):
        user_state(id=bad)


def test_state_id_must_be_a_string():
    with pytest.raises(TypeError):
        user_state(id=3)


# ================================================================ round trip


@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


def test_run_saves_and_load_rebuilds_the_state(tmp_path):
    store = FileStore(tmp_path)
    agent = make_agent([tool_call("add", a=1, b=2), "It is 3"], tools=[add], store=store)
    state = State(id="sum", messages=[Message.user("What is 1+2?")], extra_data={"seen": ["x", {"n": 1}]})
    agent.run(state)

    loaded = store.load("sum")
    assert loaded.id == "sum"
    assert loaded.history == state.history
    assert [e.at for e in loaded.history] == [e.at for e in state.history]
    assert loaded.snapshot() == state.snapshot()
    assert loaded.messages == state.messages
    assert loaded.turn == state.turn == 2
    assert loaded.usage == state.usage
    assert loaded.answer == "It is 3"
    assert loaded.stopped == StoppedByUntil("waiting_for_user")
    assert loaded.extra_data == {"seen": ["x", {"n": 1}]}
    assert loaded.created_at == state.created_at and loaded.updated_at == state.updated_at
    assert not loaded.finished and loaded.pending_calls == ()


def test_loaded_state_continues(tmp_path):
    store = FileStore(tmp_path)
    make_agent(["Hello"], store=store).run(user_state("Hi", id="chat"))

    state = store.load("chat")
    state.add_message(Message.user("And then?"))
    fake = FakeModel(["More"])
    make_agent(model=fake, store=store).run(state)
    assert [m.text for m in fake.requests[0].messages] == ["Hi", "Hello", "And then?"]

    again = store.load("chat")
    assert again.answer == "More"
    assert again.history == state.history


def test_history_entry_is_error_round_trips(tmp_path):
    @tool
    def broken() -> str:
        """Always fails"""
        raise RuntimeError("boom")

    store = FileStore(tmp_path)
    state = user_state("Go", id="err")
    with pytest.raises(RuntimeError):
        make_agent([tool_call("broken")], tools=[broken], store=store).run(state)
    closed = [e for e in state.history if e.kind == "tool_result"]
    assert closed and closed[-1].is_error
    loaded = store.load("err")
    # The exception object is not saved (``error`` is None, and ignored by ==); the entry text keeps its type and
    # message.
    assert loaded.history == state.history
    assert all(e.error is None for e in loaded.history if hasattr(e, "error"))
    assert [e.content for e in loaded.history if e.kind == "error"] == ["RuntimeError: boom"]
    assert loaded.snapshot() == state.snapshot()


def test_answers_become_plain_values(tmp_path):
    @dataclass
    class Review:
        ok: bool

    class Verdict(BaseModel):
        score: int

    store = FileStore(tmp_path)
    agent = make_agent(['{"ok": true}'], store=store)
    state = user_state("Review", id="rev")
    agent.ask(state, "ok?", returns=Review)
    state.finish(Verdict(score=3))
    store.save(state)

    loaded = store.load("rev")
    assert loaded.answer == {"score": 3}
    assert [e.content.answer for e in loaded.history if e.kind == "ask"] == [{"ok": True}]
    assert loaded.finished
    assert loaded.stopped == StoppedByFinish(answer={"score": 3})


def test_extra_data_that_is_not_json_is_refused_when_it_is_set():
    """The notepad takes JSON only, so a save never meets a value it cannot write."""
    state = user_state()
    with pytest.raises(TypeError, match=r"state.extra_data\['seen'\] must be JSON \(got set\)"):
        with state.edit_extra_data() as data:
            data["seen"] = {1, 2}
    assert state.extra_data == {}


@pytest.mark.parametrize("value", [{1: "a"}, float("nan"), object()])
def test_other_values_that_are_not_json(value):
    state = user_state()
    with pytest.raises(TypeError, match="must be JSON"):
        with state.edit_extra_data() as data:
            data["x"] = value
    assert len(state.history) == 1  # only the import


def test_load_replays_everything_to_an_equal_snapshot(tmp_path):
    """The invariant of 0.5: history is the only truth, so a loaded State has the very same snapshot."""
    store = FileStore(tmp_path)
    review = {"ok": True}
    agent = make_agent(
        [tool_call("add", a=1, b=2), "It is 3", json.dumps(review), "Summary of the work"],
        tools=[add],
        store=store,
    )
    state = State(
        id="rich",
        messages=[Message.user("What is 1+2?", Image(b"\x89PNG\r\n\x1a\nfake", "image/png"))],
        extra_data={"repo": "api", "n": 1},
    )
    state.add_message(Message.notice("the tests are red"))
    with state.edit_extra_data() as data:
        data["notes"] = {"bug": "b.py"}
        del data["repo"]
    agent.run(state)
    state.add_message(Message.user("Second task"))
    state.clear_tool_results(keep_last=0)
    before = state.snapshot()
    agent.ask(state, "ok?", returns=dict)
    agent.compact(state)
    state.add_message(Message.user("After the summary"))
    state.restore(before)
    state.add_message(Message.user("Third task"))
    state.finish({"summary": "done"})
    store.save(state)

    loaded = store.load("rich")
    assert loaded.snapshot() == state.snapshot()
    assert loaded.history == state.history
    assert State(history=loaded.history).snapshot() == loaded.snapshot()
    assert loaded.extra_data == {"n": 1, "notes": {"bug": "b.py"}}
    assert loaded.finished and loaded.answer == {"summary": "done"}
    assert {e.kind for e in loaded.history} >= {
        "context_change",
        "extra_data",
        "run_start",
        "model_request",
        "model_reply",
        "tool_result",
        "ask",
        "stop",
        "user",
        "notice",
    }


def test_load_a_state_with_no_history(tmp_path):
    store = FileStore(tmp_path)
    store.save(State(id="empty"))
    loaded = store.load("empty")
    assert loaded.history == () and loaded.messages == ()
    assert loaded.snapshot() == State().snapshot()


# ================================================================ save and asave


def test_store_save_writes_what_is_new_and_skips_the_rest():
    store = MemoryStore()
    state = user_state("Sum", id="s")
    store.save(state)
    assert store.calls == [("s", [0], True)]
    store.save(state)  # nothing new: no write at all
    assert store.writes == 1

    state.add_message(Message.user("more"))
    with state.edit_extra_data() as data:
        data["k"] = 1
    store.save(state)
    assert store.calls[-1] == ("s", [1, 2], False)
    assert len(store.states["s"].entries) == len(state.history) == 3
    assert store.load("s").snapshot() == state.snapshot()


def test_store_save_without_an_agent_then_the_agent_continues():
    store = MemoryStore()
    state = user_state("Hi", id="manual")
    store.save(state)
    make_agent(["Hello"], store=store).run(state)
    assert len(store.states["manual"].entries) == len(state.history)
    assert store.states["manual"].info["turn"] == 1


def test_store_save_takes_a_state():
    with pytest.raises(TypeError, match="takes a State"):
        MemoryStore().save("not a state")


async def test_store_asave_and_aload(tmp_path):
    store = FileStore(tmp_path)
    state = user_state("Go", id="as")
    await store.asave(state)
    state.add_message(Message.user("more"))
    await store.asave(state)
    loaded = await store.aload("as")
    assert loaded.history == state.history
    assert [s.id for s in store.list()] == ["as"]


async def test_asave_with_an_async_only_store():
    store = AsyncMemoryStore()
    state = user_state("Go", id="aa")
    await store.asave(state)
    assert (await store.aload("aa")).snapshot() == state.snapshot()
    with pytest.raises(TypeError, match="only asynchronously"):
        store.save(user_state("Another", id="ab"))  # a sync save has no write to call


def test_saves_after_each_step_and_skips_unchanged_states():
    store = MemoryStore()
    agent = make_agent([tool_call("add", a=1, b=2), "3"], tools=[add], store=store)
    state = user_state("Sum", id="s")
    agent.run(state)
    record = store.states["s"]
    assert len(record.entries) == len(state.history)
    assert record.info["turn"] == 2
    assert record.info["stopped"] == {"kind": "until", "name": "waiting_for_user"}
    writes = store.writes
    store.save(state)
    assert store.writes == writes  # nothing new


def test_the_run_start_entry_records_the_agent(tmp_path):
    store = FileStore(tmp_path)
    make_agent(store=store, name="old", system="Be brief").run(user_state("Go", id="w"))
    loaded = store.load("w")
    [start] = [e for e in loaded.history if e.kind == "run_start"]
    assert start.content.name == "old"
    assert start.content.model == "fake/fake"
    assert start.content.system_sha256 is not None


def test_agent_store_must_be_a_store_object(tmp_path):
    with pytest.raises(TypeError, match="not a Store object"):
        make_agent(store=FileStore)
    assert make_agent(store=FileStore(tmp_path)).copy(name="x").store is not None


async def test_sync_store_inside_an_async_run_writes_off_the_event_loop():
    """A Store with only ``write`` works with ``arun``: the Agent calls ``asave``, whose default runs ``write`` on a
    worker thread, so a blocking write never stalls the event loop."""
    main = threading.get_ident()
    seen: list[int] = []

    class SyncStore(MemoryStore):
        def write(self, state_id, entries, info, *, create=False):
            seen.append(threading.get_ident())
            super().write(state_id, entries, info, create=create)

    store = SyncStore()
    agent = make_agent([tool_call("add", a=2, b=2), "4"], tools=[add], store=store)
    state = user_state("Sum", id="sa")
    await agent.arun(state)
    assert seen and main not in seen
    assert (await store.aload("sa")).snapshot() == state.snapshot()


# ================================================================ StateInfo (store.list)


def test_state_info_fields_from_list(tmp_path):
    store = FileStore(tmp_path)
    make_agent(["A"], store=store).run(user_state("First\nwith a second line", id="one"))
    state = user_state("Second", id="two")
    make_agent(["B"], store=store).run(state)
    state.finish({"x": 1})
    store.save(state)

    infos = {info.id: info for info in store.list()}
    one, two = infos["one"], infos["two"]
    assert isinstance(one, StateInfo)
    assert one.first_message == "First\nwith a second line"  # the full text, not cut
    assert (one.turn, one.stopped, one.finished) == (1, StoppedByUntil("waiting_for_user"), False)
    assert (two.first_message, two.turn, two.finished) == ("Second", 1, True)
    assert two.stopped == StoppedByFinish(answer={"x": 1})
    saved = store.load("two")
    assert (two.created_at, two.updated_at) == (saved.created_at, saved.updated_at)
    assert two.created_at is not None and two.created_at <= two.updated_at
    assert not hasattr(two, "task")


def test_state_info_first_message_is_the_first_user_message(tmp_path):
    store = FileStore(tmp_path)
    state = State(id="a", messages=[Message.user("imported first"), Message.assistant("ok")])
    state.add_message(Message.user("added later"))
    store.save(state)
    only_image = State(id="b", messages=[Message.user("", Image(b"png", "image/png"))])
    store.save(only_image)
    added = State(id="c")
    added.add_message(Message.user("starts here"))
    store.save(added)
    empty = State(id="d")
    store.save(empty)

    infos = {info.id: info for info in store.list()}
    assert infos["a"].first_message == "imported first"
    assert infos["b"].first_message is None  # nothing to show
    assert infos["c"].first_message == "starts here"
    assert infos["d"].first_message is None
    assert infos["d"].created_at is None and infos["d"].updated_at is None  # no history yet
    assert (infos["d"].turn, infos["d"].stopped, infos["d"].finished) == (0, None, False)


def test_state_info_from_info_for_store_implementers():
    store = MemoryStore()
    state = user_state("Hi", id="m")
    make_agent(["Hello"], store=store).run(state)
    [info] = store.list()
    assert info == StateInfo.from_info("m", store.states["m"].info)
    assert (info.id, info.first_message, info.turn, info.stopped) == ("m", "Hi", 1, StoppedByUntil("waiting_for_user"))
    assert info.created_at == state.created_at and info.updated_at == state.updated_at


def test_state_info_from_info_refuses_other_formats_and_missing_keys():
    info = an_info()
    with pytest.raises(ValueError, match="newer alpineagents"):
        StateInfo.from_info("x", {**info, "v": 5})
    with pytest.raises(ValueError, match=r"damaged: its info has no turn, finished"):
        StateInfo.from_info("x", {k: v for k, v in info.items() if k not in ("turn", "finished")})


# ================================================================ Store.write contract (a custom Store)


def test_write_contract_of_a_custom_store():
    store = MemoryStore()
    agent = make_agent([tool_call("add", a=1, b=2), "3"], tools=[add], store=store)
    state = user_state("Sum", id="c")
    agent.run(state)

    # The first write creates, every later one only adds, and the entries continue without gaps.
    assert store.calls[0][2] is True
    assert all(create is False for _, _, create in store.calls[1:])
    seen = 0
    for _, seqs, _ in store.calls:
        assert seqs == list(range(seen, seen + len(seqs)))
        seen += len(seqs)
    assert seen == len(state.history)

    record = store.states["c"]
    assert [e["seq"] for e in record.entries] == list(range(len(state.history)))
    # info: exactly the StateInfo fields (but id) and the format version, no messages, no history.
    assert set(record.info) == {"v", "first_message", "created_at", "updated_at", "turn", "stopped", "finished"}
    assert record.info["v"] == _serial.VERSION == 4
    assert json.loads(json.dumps(record.info)) == record.info
    assert record.info["first_message"] == "Sum" and record.info["turn"] == 2


def test_write_gets_the_info_with_every_batch():
    store = MemoryStore()
    state = user_state("Go", id="i")
    infos = []
    original = store.write
    store.write = lambda *a, **k: (infos.append(dict(a[2])), original(*a, **k))[1]
    store.save(state)
    state.add_message(Message.user("more"))
    state.finish("done")
    store.save(state)
    assert [i["finished"] for i in infos] == [False, True]
    assert infos[1]["stopped"] == {"kind": "finish", "answer": "done"}


def test_first_save_of_an_id_the_store_has_is_refused_by_the_store():
    store = MemoryStore()
    store.save(user_state("First", id="dup"))
    with pytest.raises(ValueError, match="dup exists"):
        store.save(user_state("Second", id="dup"))
    assert store.load("dup").messages[0].text == "First"


def test_a_failed_write_is_sent_again_at_the_next_save():
    store = FailingStore()
    state = user_state("Go", id="r")
    store.failing = True
    with pytest.raises(OSError):
        store.save(state)
    store.failing = False
    state.add_message(Message.user("more"))
    store.save(state)
    assert store.calls == [("r", [0, 1], True)]  # still the create, now with both entries
    assert len(store.load("r").history) == 2


def test_custom_store_loads_what_it_was_given():
    store = MemoryStore()
    agent = make_agent([tool_call("add", a=1, b=2), "3"], tools=[add], store=store)
    state = user_state("Sum", id="l")
    agent.run(state)
    assert store.load("l").snapshot() == state.snapshot()


# ================================================================ ids and binding


def test_new_state_with_a_saved_id_is_refused_before_the_model_is_called(tmp_path):
    store = FileStore(tmp_path)
    make_agent(store=store).run(user_state("First", id="dup"))
    fake = FakeModel(["Done"])
    with pytest.raises(ValueError, match="already saved") as info:
        make_agent(model=fake, store=store).run(user_state("Second", id="dup"))
    assert "store.load" in str(info.value)
    assert fake.requests == []
    assert store.load("dup").messages[0].text == "First"


def test_a_state_is_saved_in_one_store_only(tmp_path):
    first, second = FileStore(tmp_path / "a"), FileStore(tmp_path / "b")
    state = user_state("Go")
    make_agent(store=first).run(state)
    state.add_message(Message.user("more"))
    with pytest.raises(ValueError, match="saved in another store"):
        make_agent(store=second).run(state)
    with pytest.raises(ValueError, match="saved in another store"):
        second.save(state)


def test_two_file_stores_for_one_folder_are_the_same_store(tmp_path):
    make_agent(["One"], store=FileStore(tmp_path)).run(user_state("Go", id="same"))
    state = FileStore(tmp_path).load("same")
    state.add_message(Message.user("again"))
    make_agent(["Two"], store=FileStore(str(tmp_path) + "/.")).run(state)
    assert FileStore(tmp_path).load("same").answer == "Two"


# ================================================================ format version


def _edit_info(tmp_path: Path, state_id: str, edit) -> None:
    path = tmp_path / state_id / "info.json"
    info = json.loads(path.read_text("utf-8"))
    edit(info)
    path.write_text(json.dumps(info), "utf-8")


def test_load_refuses_format_3_from_0_4(tmp_path):
    store = FileStore(tmp_path)
    make_agent(store=store).run(user_state("Hi", id="old"))
    _edit_info(tmp_path, "old", lambda i: i.update(v=3))
    with pytest.raises(ValueError) as raised:
        store.load("old")
    assert str(raised.value).splitlines()[0] == (
        "saved State 'old' was saved by an older alpineagents (format 3, 0.4.x); 0.5 cannot load it"
    )
    assert "Fix:" in str(raised.value)


def test_load_refuses_an_older_format_without_the_release(tmp_path):
    store = FileStore(tmp_path)
    store.save(user_state("Hi", id="older"))
    _edit_info(tmp_path, "older", lambda i: i.update(v=2))
    with pytest.raises(ValueError) as raised:
        store.load("older")
    assert str(raised.value).splitlines()[0] == (
        "saved State 'older' was saved by an older alpineagents (format 2); 0.5 cannot load it"
    )


@pytest.mark.parametrize("version", [_serial.VERSION + 1, 99])
def test_load_refuses_a_newer_format(tmp_path, version):
    store = FileStore(tmp_path)
    make_agent(store=store).run(user_state("Hi", id="new"))
    _edit_info(tmp_path, "new", lambda i: i.update(v=version))
    with pytest.raises(ValueError) as raised:
        store.load("new")
    assert str(raised.value).splitlines()[0] == (
        f"saved State 'new' was saved by a newer alpineagents (format {version}); this alpineagents reads format 4"
    )
    assert "upgrade alpineagents" in str(raised.value)


def test_load_checks_the_format_before_reading_entries(tmp_path):
    """A State from another format says so, instead of failing on an entry it cannot read."""
    store = MemoryStore()
    store.states["x"] = Record([{"seq": 0, "kind": "something_new"}], {"v": 3})
    with pytest.raises(ValueError, match=r"older alpineagents \(format 3, 0\.4\.x\)"):
        store.load("x")


async def test_aload_refuses_other_formats_too():
    store = MemoryStore()
    store.states["x"] = Record([], {"v": 5})
    with pytest.raises(ValueError, match="newer alpineagents"):
        await store.aload("x")


def test_load_refuses_a_0_4_folder_with_the_format_message(tmp_path):
    """0.4 kept ``snapshot.json`` (with ``v: 3``) beside the log and had no ``info.json``."""
    folder = tmp_path / "legacy"
    folder.mkdir()
    (folder / "log.jsonl").write_text('{"seq": 0, "kind": "user", "turn": 0, "content": "Hi"}\n', "utf-8")
    (folder / "snapshot.json").write_text(json.dumps({"v": 3, "task": "Hi", "turn": 0}), "utf-8")
    with pytest.raises(ValueError, match=r"older alpineagents \(format 3, 0\.4\.x\); 0\.5 cannot load it"):
        FileStore(tmp_path).load("legacy")


def test_load_refuses_info_without_a_version(tmp_path):
    store = FileStore(tmp_path)
    store.save(user_state("Hi", id="nov"))
    _edit_info(tmp_path, "nov", lambda i: i.pop("v"))
    with pytest.raises(ValueError, match="no format version"):
        store.load("nov")


def test_load_refuses_a_folder_without_info(tmp_path):
    store = FileStore(tmp_path)
    store.save(user_state("Hi", id="torn"))
    (tmp_path / "torn" / "info.json").unlink()
    with pytest.raises(ValueError, match="damaged"):
        store.load("torn")


def test_load_refuses_entries_with_a_gap():
    store = MemoryStore()
    entries = entry_dicts(
        MessageEntry(kind="user", content="a", turn=0), MessageEntry(kind="user", content="b", turn=0)
    )
    entries[1]["seq"] = 2
    store.states["gap"] = Record(entries, an_info())
    with pytest.raises(ValueError, match="history entry 1 has seq 2"):
        store.load("gap")


def test_load_refuses_a_history_that_cannot_be_replayed():
    """A reply with no request before it is not a history any State could have had."""
    store = MemoryStore()
    reply = Reply(Message.assistant("hi"), Usage())
    store.states["bad"] = Record(entry_dicts(ModelReplyEntry(content=reply, turn=1)), an_info())
    with pytest.raises(ValueError, match="saved State 'bad' is damaged"):
        store.load("bad")


def test_load_missing_id(tmp_path):
    with pytest.raises(LookupError, match="no saved State"):
        FileStore(tmp_path).load("nope")


# ================================================================ closing a State saved mid-turn


def _history_waiting_on_the_model() -> list:
    return [
        MessageEntry(kind="user", content="Find the bug", turn=0),
        ModelRequestEntry(content="fake/fake", turn=1),
    ]


def _history_with_pending_calls(*calls) -> list:
    reply = Reply(Message("assistant", tuple(calls)), Usage(requests=1))
    return [
        MessageEntry(kind="user", content="Fix it", turn=0),
        ModelRequestEntry(content="fake/fake", turn=1),
        ModelReplyEntry(content=reply, turn=1),
    ]


def test_state_saved_waiting_on_the_model_is_closed_on_load():
    store = MemoryStore()
    waiting = State(id="w", history=_history_waiting_on_the_model())
    assert waiting.turn == 1
    store.save(waiting)

    loaded = store.load("w")
    assert loaded.turn == 0  # the request was taken back
    assert [m.text for m in loaded.messages] == ["Find the bug"]
    assert len(loaded.history) == 3
    closing = loaded.history[-1]
    assert isinstance(closing, ErrorEntry)
    assert closing.content == STOPPED_WAITING == "(process stopped while waiting for the model)"
    assert closing.error is None
    # What was saved is untouched: the closing entry exists only in the loaded State until it is saved.
    assert len(store.states["w"].entries) == 2


def test_state_saved_with_pending_calls_is_closed_in_request_order():
    first, second = tool_call("add", a=1, b=2), tool_call("add", a=3, b=4)
    store = MemoryStore()
    store.save(State(id="p", history=_history_with_pending_calls(first, second)))

    loaded = store.load("p")
    assert loaded.pending_calls == ()
    closing = loaded.history[3:]
    assert [type(e) for e in closing] == [ToolResultEntry, ToolResultEntry]
    assert [e.call.id for e in closing] == [first.id, second.id]
    assert all(e.outcome == ToolOutcomeKind.ABORTED and e.content == UNSAVED_RESULT for e in closing)
    assert all(e.is_error for e in closing)
    last = loaded.messages[-1]
    assert last.role == "user"
    assert [b.call_id for b in last.content] == [first.id, second.id]
    assert [b.content for b in last.content] == [UNSAVED_RESULT, UNSAVED_RESULT]
    assert not waiting_for_user(loaded)


def test_closing_entries_are_saved_at_the_next_save():
    call = tool_call("add", a=1, b=2)
    store = MemoryStore()
    store.save(State(id="n", history=_history_with_pending_calls(call)))
    assert len(store.states["n"].entries) == 3

    loaded = store.load("n")
    assert len(loaded.history) == 4
    store.save(loaded)
    assert store.calls[-1] == ("n", [3], False)
    again = store.load("n")
    assert again.history == loaded.history  # nothing left to close
    assert again.snapshot() == loaded.snapshot()
    store.save(again)
    assert store.writes == 2  # and nothing to write


def test_a_state_saved_waiting_closes_and_saves_the_same_way():
    store = MemoryStore()
    store.save(State(id="ws", history=_history_waiting_on_the_model()))
    loaded = store.load("ws")
    store.save(loaded)
    again = store.load("ws")
    assert again.history == loaded.history and len(again.history) == 3
    assert again.turn == 0


def test_a_state_that_ended_cleanly_gets_no_closing_entries(tmp_path):
    store = FileStore(tmp_path)
    state = user_state("Go", id="clean")
    make_agent(["Done"], store=store).run(state)
    assert len(store.load("clean").history) == len(state.history)


def test_a_state_saved_waiting_with_a_held_message_keeps_it():
    """A message added while the model was awaited sits behind the request; taking the request back releases it."""
    store = MemoryStore()
    waiting = State(id="held", history=_history_waiting_on_the_model())
    waiting.add_message(Message.user("hurry"))
    assert [m.text for m in waiting.messages] == ["Find the bug"]  # held
    store.save(waiting)
    loaded = store.load("held")
    assert [m.text for m in loaded.messages] == ["Find the bug", "hurry"]
    assert loaded.history[-1].content == STOPPED_WAITING


def test_loaded_state_that_was_waiting_can_run_again():
    store = MemoryStore()
    store.save(State(id="again", history=_history_waiting_on_the_model()))
    loaded = store.load("again")
    fake = FakeModel(["Found it"])
    assert make_agent(model=fake, store=store).run(loaded) == "Found it"
    assert [m.text for m in fake.requests[0].messages] == ["Find the bug"]
    assert store.load("again").snapshot() == loaded.snapshot()


# ================================================================ crash: what is on disk mid-run


def test_crash_while_a_tool_runs_closes_the_call_as_unknown(tmp_path):
    live = FileStore(tmp_path / "live")
    state = user_state("Send the report", id="mail")
    crashed: list[FileStore] = []

    @tool
    def send_email(to: str) -> str:
        """Send an email"""
        crashed.append(crash_copy(live, "mail", tmp_path / "crash"))
        return "sent"

    agent = make_agent([tool_call("send_email", to="boss"), "Done"], tools=[send_email], store=live)
    agent.run(state)

    loaded = crashed[0].load("mail")
    assert loaded.turn == 1
    assert loaded.pending_calls == ()
    last = loaded.history[-1]
    assert (last.kind, last.content, last.outcome, last.is_error) == (
        "tool_result",
        UNSAVED_RESULT,
        ToolOutcomeKind.ABORTED,
        True,
    )
    assert last.call.name == "send_email"
    assert [m.role for m in loaded.messages] == ["user", "assistant", "user"]
    assert loaded.messages[-1].content[0].content == UNSAVED_RESULT
    assert not waiting_for_user(loaded)

    # Continuing sends the unknown result to the model, and the closing entry gets saved.
    fake = FakeModel(["I will check the outbox"])
    saved_len = len(loaded.history)
    make_agent(model=fake, tools=[send_email], store=crashed[0]).run(loaded)
    assert UNSAVED_RESULT in str(fake.requests[0].messages[-1].content)
    after = crashed[0].load("mail")
    assert after.history[:saved_len] == loaded.history[:saved_len]
    assert after.history == loaded.history


def test_crash_keeps_results_saved_per_tool(tmp_path):
    live = FileStore(tmp_path / "live")
    crashed: list[FileStore] = []

    @tool
    def fast() -> str:
        """Fast"""
        return "fast result"

    @tool
    def slow() -> str:
        """Slow"""
        log = tmp_path / "live" / "par" / "log.jsonl"
        deadline = time.monotonic() + 5
        while "fast result" not in log.read_text() and time.monotonic() < deadline:
            time.sleep(0.01)
        crashed.append(crash_copy(live, "par", tmp_path / "crash"))
        return "slow result"

    agent = make_agent([[tool_call("fast"), tool_call("slow")], "Done"], tools=[fast, slow], store=live)
    agent.run(user_state("Both", id="par"))

    loaded = crashed[0].load("par")
    results = {e.call.name: e.content for e in loaded.history if e.kind == "tool_result"}
    assert results == {"fast": "fast result", "slow": UNSAVED_RESULT}
    blocks = loaded.messages[-1].content
    assert [b.content for b in blocks] == ["fast result", UNSAVED_RESULT]  # request order


def test_crash_while_waiting_on_the_model_drops_that_think(tmp_path):
    live = FileStore(tmp_path / "live")
    state = user_state("Find the bug", id="think")
    crashed: list[FileStore] = []

    def during(request):
        state.add_message(Message.user("hurry"))  # another thread would do this; held until the reply
        live.save(state)
        crashed.append(crash_copy(live, "think", tmp_path / "crash"))
        return "Done"

    agent = make_agent([during, "Ok"], store=live)
    agent.run(state)

    loaded = crashed[0].load("think")
    assert loaded.turn == 0
    assert [m.text for m in loaded.messages] == ["Find the bug", "hurry"]
    assert loaded.history[-1].content == STOPPED_WAITING
    assert loaded.history[-2].content == "hurry"
    assert [e.kind for e in loaded.history if e.kind == "model_request"] == ["model_request"]


def test_crash_replays_messages_held_during_think_after_the_results(tmp_path):
    live = FileStore(tmp_path / "live")
    state = user_state("Fix it", id="held")
    crashed: list[FileStore] = []

    def during(request):
        state.add_message(Message.user("also run the tests"))
        return tool_call("probe")

    @tool
    def probe() -> str:
        """Probe"""
        crashed.append(crash_copy(live, "held", tmp_path / "crash"))
        return "ok"

    agent = make_agent([during, "Done"], tools=[probe], store=live)
    agent.run(state)

    loaded = crashed[0].load("held")
    task, reply, results, held = loaded.messages
    assert task.text == "Fix it"
    assert [call.name for call in reply.tool_calls] == ["probe"]
    assert [block.content for block in results.content] == [UNSAVED_RESULT]
    assert held.text == "also run the tests"


def test_entries_without_their_info_still_load_in_full(tmp_path):
    """A save that wrote the entries and then failed on the info leaves a log ahead of ``info.json``. Load reads
    only the log (the info is for listing), so it replays everything."""
    store = InfoFailingStore(tmp_path)
    agent = make_agent(["First answer"], store=store)
    state = user_state("Go", id="b")
    agent.run(state)

    store.failing = True
    state.add_message(Message.user("A second question"))
    state.compact("A summary")
    with pytest.raises(OSError):
        store.save(state)

    loaded = FileStore(tmp_path).load("b")
    assert loaded.history == state.history
    assert loaded.snapshot() == state.snapshot()
    assert [m.text for m in loaded.messages][-1].endswith("A summary")
    # The listing is one save behind, which is fine: it is only a listing.
    assert FileStore(tmp_path).list()[0].turn == 1


def test_system_exit_closes_calls_and_saves(tmp_path):
    """What the SIGTERM handler in the guide does: SystemExit ends the run like an exception."""
    store = FileStore(tmp_path)

    @loop(until=waiting_for_user, limit=5)
    def stopped(agent, state):
        agent.think(state)
        raise SystemExit(143)

    state = user_state("Go", id="term")
    with pytest.raises(SystemExit):
        make_agent([tool_call("add", a=1, b=1)], tools=[add], loop=stopped, store=store).run(state)
    loaded = FileStore(tmp_path).load("term")
    assert [e.content for e in loaded.history if e.kind == "tool_result"] == ["(aborted: SystemExit)"]
    assert loaded.pending_calls == ()
    assert loaded.snapshot() == state.snapshot()


def test_torn_last_line_is_ignored_and_repaired(tmp_path):
    store = FileStore(tmp_path)
    agent = make_agent(["One", "Two"], store=store)
    state = user_state("Go", id="torn")
    agent.run(state)
    log = tmp_path / "torn" / "log.jsonl"
    with open(log, "a") as file:
        file.write('{"seq": 99, "kind": "us')

    reopened = FileStore(tmp_path)
    loaded = reopened.load("torn")
    assert loaded.history == state.history

    loaded.add_message(Message.user("again"))
    make_agent(["Two"], store=reopened).run(loaded)
    lines = log.read_text().splitlines()
    assert all(json.loads(line) for line in lines)
    assert FileStore(tmp_path).load("torn").answer == "Two"


# ================================================================ failures while saving


def test_failed_save_during_a_failing_run_is_a_note_on_the_original():
    store = FailingStore()

    def rate_limited(request):
        store.failing = True  # from here on the disk is full
        raise RateLimitError("429")

    agent = make_agent([rate_limited], store=store)
    state = user_state("Go")
    with pytest.raises(RateLimitError) as info:
        agent.run(state)
    assert info.value.__notes__ == ["saving the state also failed: OSError: disk full"]


def test_interrupt_while_saving_wins_over_the_original():
    store = FailingStore()

    def rate_limited(request):
        store.failing = True
        store.error = KeyboardInterrupt
        raise RateLimitError("429")

    agent = make_agent([rate_limited], store=store)
    state = user_state("Go")
    with pytest.raises(KeyboardInterrupt) as info:
        agent.run(state)
    chain = []
    error: BaseException | None = info.value
    while error is not None:
        chain.append(type(error))
        error = error.__context__
    assert RateLimitError in chain


def test_failed_save_at_the_end_of_a_good_run_raises():
    store = FailingStore()
    store.fail_when = lambda entries, info: any(e["kind"] == "stop" for e in entries)
    state = user_state("Go")
    with pytest.raises(OSError, match="disk full") as info:
        make_agent(["Done"], store=store).run(state)
    assert not getattr(info.value, "__notes__", None)  # the retry failed the same way: no note
    assert state.answer == "Done"


def test_failed_first_save_means_the_run_never_started():
    store = FailingStore()
    store.failing = True
    fake = FakeModel(["Done"])
    state = user_state("Go")
    with pytest.raises(OSError, match="disk full"):
        make_agent(model=fake, store=store).run(state)
    assert fake.requests == []
    store.failing = False
    assert make_agent(["Done"], store=store).run(state) == "Done"  # the State can be run again


def test_failed_save_after_think_does_not_roll_the_reply_back():
    store = FailingStore()
    agent = make_agent([tool_call("add", a=1, b=1)], tools=[add], store=store)
    state = user_state("Go")
    store.save(state)

    original = store.write

    def fail_after_think(state_id, entries, info, *, create=False):
        if any(e["kind"] == "model_reply" for e in entries):
            raise OSError("disk full")
        original(state_id, entries, info, create=create)

    store.write = fail_after_think
    with pytest.raises(OSError):
        agent.think(state)
    assert state.turn == 1
    assert state.pending_calls  # the reply stays; nothing was rolled back
    assert state.history[-1].kind == "model_reply"


def test_failed_save_before_the_model_takes_the_request_back():
    store = FailingStore()
    fake = FakeModel(["Done"])
    agent = make_agent(model=fake, store=store)
    state = user_state("Go")
    store.save(state)
    store.failing = True
    with pytest.raises(OSError):
        agent.think(state)
    assert fake.requests == []
    assert state.turn == 0  # the request was never sent, and is taken back


def test_per_tool_save_failures_do_not_stop_the_tools():
    store = FailingStore()
    failures = [0]

    def fail_twice(entries, info):
        # The saves right after each result fail; the save at the end of use_tools works.
        if any(e["kind"] == "tool_result" for e in entries) and failures[0] < 2:
            failures[0] += 1
            return True
        return False

    store.fail_when = fail_twice
    ran = []

    @tool(parallel=False)
    def work(n: int) -> str:
        """Work"""
        ran.append(n)
        return f"done {n}"

    agent = make_agent([[tool_call("work", n=1), tool_call("work", n=2)], "Done"], tools=[work], store=store)
    state = user_state("Go", id="t")
    agent.run(state)
    assert ran == [1, 2]
    assert store.failed == 2
    assert len(store.states["t"].entries) == len(state.history)


def test_create_is_all_or_nothing(tmp_path):
    store = FileStore(tmp_path)
    store.write("x", entry_dicts(MessageEntry(kind="user", content="a", turn=0)), an_info(), create=True)
    with pytest.raises(ValueError, match="already saved"):
        store.write("x", [], an_info(), create=True)
    assert [p.name for p in tmp_path.iterdir()] == ["x"]


def test_write_without_create_needs_a_known_id(tmp_path):
    with pytest.raises(LookupError, match="no saved State"):
        FileStore(tmp_path).write("ghost", [], an_info())


def test_write_skips_entries_it_already_has(tmp_path):
    store = FileStore(tmp_path)
    state = user_state("Go", id="dedupe")
    agent = make_agent(["Done"], store=store)
    agent.run(state)
    entries = store.read("dedupe").entries
    FileStore(tmp_path).write("dedupe", entries, an_info())  # a retried write
    assert store.read("dedupe").entries == entries


def test_a_deleted_state_is_not_written_again(tmp_path):
    store = FileStore(tmp_path)
    state = user_state("Go", id="gone")
    store.save(state)
    store.delete("gone")
    state.add_message(Message.user("more"))
    with pytest.raises(LookupError, match="no saved State"):
        store.save(state)


# ================================================================ FileStore: layout, list, delete


def test_file_store_layout_on_disk(tmp_path):
    store = FileStore(tmp_path)
    agent = make_agent([tool_call("add", a=1, b=2), "3"], tools=[add], store=store)
    state = user_state("Sum", id="lay")
    agent.run(state)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["lay"]
    assert sorted(p.name for p in (tmp_path / "lay").iterdir()) == ["info.json", "log.jsonl"]  # no snapshot.json

    lines = (tmp_path / "lay" / "log.jsonl").read_text("utf-8").splitlines()
    entries = [json.loads(line) for line in lines]
    assert [e["seq"] for e in entries] == list(range(len(state.history)))
    assert [e["kind"] for e in entries] == [e.kind for e in state.history]
    assert all({"seq", "kind", "turn", "at", "content"} <= set(e) for e in entries)
    assert "v" not in entries[0]  # the version is in the info only

    info = json.loads((tmp_path / "lay" / "info.json").read_text("utf-8"))
    assert info["v"] == 4
    assert set(info) == {"v", "first_message", "created_at", "updated_at", "turn", "stopped", "finished"}
    assert info["first_message"] == "Sum" and info["turn"] == 2 and info["finished"] is False
    assert info["stopped"] == {"kind": "until", "name": "waiting_for_user"}
    assert store.read("lay") == Record(entries, info)


def test_file_store_list_skips_0_4_folders_and_other_folders(tmp_path):
    store = FileStore(tmp_path)
    make_agent(["A"], store=store).run(user_state("New", id="new"))
    legacy = tmp_path / "old-state"
    legacy.mkdir()
    (legacy / "log.jsonl").write_text("", "utf-8")
    (legacy / "snapshot.json").write_text(json.dumps({"v": 3}), "utf-8")
    (tmp_path / ".new-staging").mkdir()
    (tmp_path / "notes.txt").write_text("not a state", "utf-8")

    assert [s.id for s in store.list()] == ["new"]
    with pytest.raises(ValueError, match="format 3"):
        store.load("old-state")


def test_list_and_delete(tmp_path):
    store = FileStore(tmp_path)
    make_agent(["A"], store=store).run(user_state("First", id="one"))
    make_agent(["B"], store=store).run(user_state("Second", id="two"))
    saved = store.list()
    assert [s.id for s in saved] == ["two", "one"]  # most recently updated first
    assert isinstance(saved[0], StateInfo)
    assert (saved[0].first_message, saved[0].turn, saved[0].stopped, saved[0].finished) == (
        "Second",
        1,
        StoppedByUntil("waiting_for_user"),
        False,
    )

    store.delete("one")
    store.delete("one")  # already gone: nothing happens
    assert [s.id for s in store.list()] == ["two"]
    assert FileStore(tmp_path / "missing").list() == []


def test_files_are_private(tmp_path):
    store = FileStore(tmp_path)
    make_agent(store=store).run(user_state("Go", id="p"))
    for name in ("log.jsonl", "info.json"):
        assert stat.S_IMODE(os.stat(tmp_path / "p" / name).st_mode) == 0o600


def test_info_is_replaced_on_every_save_and_the_log_only_grows(tmp_path):
    store = FileStore(tmp_path)
    state = user_state("Go", id="grow")
    store.save(state)
    log = tmp_path / "grow" / "log.jsonl"
    first = log.read_text("utf-8")
    state.add_message(Message.user("more"))
    state.finish("done")
    store.save(state)
    assert log.read_text("utf-8").startswith(first)
    assert len(log.read_text("utf-8").splitlines()) == 3
    assert json.loads((tmp_path / "grow" / "info.json").read_text("utf-8"))["finished"] is True
    assert not [p for p in (tmp_path / "grow").iterdir() if p.name.startswith(".")]  # no staging files left


# ================================================================ Store role


def test_store_needs_write_and_read():
    class Nothing(Store):
        pass

    with pytest.raises(TypeError, match="neither write nor awrite"):
        Nothing()


def test_list_and_delete_are_optional():
    class Minimal(MemoryStore):
        list = Store.list

    store = Minimal()
    with pytest.raises(NotImplementedError):
        store.list()
    with pytest.raises(NotImplementedError):
        store.delete("x")


class AsyncMemoryStore(Store):
    def __init__(self) -> None:
        self.inner = MemoryStore()

    async def awrite(self, state_id, entries, info, *, create=False):
        await asyncio.sleep(0)
        self.inner.write(state_id, entries, info, create=create)

    async def aread(self, state_id):
        return self.inner.read(state_id)


def test_async_only_store_refuses_sync_run():
    with pytest.raises(TypeError, match="only asynchronously"):
        make_agent(store=AsyncMemoryStore()).run("Go")


async def test_arun_with_async_store():
    store = AsyncMemoryStore()
    agent = make_agent([tool_call("add", a=2, b=2), "4"], tools=[add], store=store)
    state = user_state("Sum", id="a")
    await agent.arun(state)
    loaded = await store.aload("a")
    assert loaded.history == state.history
    assert loaded.snapshot() == state.snapshot()
    assert loaded.answer == "4"


async def test_arun_with_file_store(tmp_path):
    store = FileStore(tmp_path)
    state = user_state("Go", id="af")
    await make_agent(["Done"], store=store).arun(state)
    assert (await store.aload("af")).history == state.history
    assert [s.id for s in await store.alist()] == ["af"]
    await store.adelete("af")
    assert store.list() == []


async def test_async_load_closes_a_state_saved_mid_turn():
    store = AsyncMemoryStore()
    await store.asave(State(id="aw", history=_history_waiting_on_the_model()))
    loaded = await store.aload("aw")
    assert loaded.turn == 0 and loaded.history[-1].content == STOPPED_WAITING


async def test_sync_save_inside_async_run_is_refused():
    from alpineagents._async import ASYNC_RUN

    store = MemoryStore()
    token = ASYNC_RUN.set(asyncio.get_running_loop())
    try:
        with pytest.raises(TypeError, match="asave"):
            store.save(user_state("Go"))
    finally:
        ASYNC_RUN.reset(token)
    await store.asave(user_state("Go", id="ok"))  # the async version is the way


def test_an_entry_equals_itself_after_a_round_trip():
    """Entries compare equal after a save and load even though ``error`` and ``at`` are not compared."""
    entry = ErrorEntry(content="ValueError: x", error=ValueError("x"), turn=0)
    back = _serial.entry_from_dict(json.loads(json.dumps(_serial.entry_to_dict(entry, 0))))
    assert back == entry and back.error is None
    assert dataclasses.replace(entry, error=None) == back
