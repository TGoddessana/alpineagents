"""Store tests: saving as a run goes, loading, replaying steps saved after the last snapshot, and what happens when
saving fails. No network.

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
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel

from alpineagents import (
    Agent,
    FileStore,
    Message,
    Reply,
    ResumeWarning,
    SavedState,
    State,
    Store,
    Usage,
    loop,
    tool,
)
from alpineagents import _serial
from alpineagents.errors import RateLimitError
from alpineagents.state import UNSAVED_RESULT, UNSAVED_STEPS
from alpineagents.store import Record
from alpineagents.testing import FakeModel, tool_call


def make_agent(replies=("Done",), **settings) -> Agent:
    settings.setdefault("reporter", None)
    settings.setdefault("human", None)
    if "model" not in settings:
        settings["model"] = FakeModel(list(replies))
    return Agent(**settings)


def crash_copy(store: FileStore, state_id: str, into: Path) -> FileStore:
    """What is on disk right now, as another FileStore."""
    shutil.copytree(Path(store._root) / state_id, into / state_id, dirs_exist_ok=True)
    return FileStore(into)


class MemoryStore(Store):
    """A Store in a dict, following the write contract (create, skip known seqs). Counts writes."""

    def __init__(self) -> None:
        self.states: dict[str, Record] = {}
        self.writes = 0

    def write(self, state_id, entries, snapshot, *, create=False):
        self.writes += 1
        if create and state_id in self.states:
            raise ValueError(f"{state_id} exists")
        if not create and state_id not in self.states:
            raise LookupError(state_id)
        record = self.states.setdefault(state_id, Record([], None))
        have = len(record.entries)
        record.entries.extend(json.loads(json.dumps(e)) for e in entries if e["seq"] >= have)
        if snapshot is not None:
            self.states[state_id] = Record(record.entries, json.loads(json.dumps(snapshot)))

    def read(self, state_id):
        return self.states.get(state_id)


class FailingStore(MemoryStore):
    """Fails writes (with a new ``error``, writing nothing) while ``failing`` is set, or when
    ``fail_when(entries, snapshot)`` is true."""

    def __init__(self) -> None:
        super().__init__()
        self.failing = False
        self.fail_when = None
        self.error = lambda: OSError("disk full")
        self.failed = 0

    def write(self, state_id, entries, snapshot, *, create=False):
        if self.failing or (self.fail_when is not None and self.fail_when(entries, snapshot)):
            self.failed += 1
            raise self.error()
        super().write(state_id, entries, snapshot, create=create)


class SnapshotFailingStore(FileStore):
    """Writes the entries, then fails before the snapshot while ``failing`` is set (a half-done save)."""

    failing = False

    def write(self, state_id, entries, snapshot, *, create=False):
        if self.failing and snapshot is not None:
            super().write(state_id, entries, None, create=create)
            raise OSError("disk full")
        super().write(state_id, entries, snapshot, create=create)


# ================================================================ State id


def test_state_id_is_random_by_default():
    assert State("a").id != State("a").id
    assert len(State("a").id) == 32


def test_state_id_given():
    assert State("a", id="bug-hunt_1").id == "bug-hunt_1"


@pytest.mark.parametrize("bad", ["", "../x", "a.b", "a b", "x" * 129])
def test_state_id_rejects_unsafe_characters(bad):
    with pytest.raises(ValueError, match="letters, digits"):
        State("a", id=bad)


def test_state_id_must_be_a_string():
    with pytest.raises(TypeError):
        State("a", id=3)


# ================================================================ round trip


@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


def test_run_saves_and_load_rebuilds_the_state(tmp_path):
    store = FileStore(tmp_path)
    agent = make_agent([tool_call("add", a=1, b=2), "It is 3"], tools=[add], store=store)
    state = State("What is 1+2?", id="sum")
    state.data["seen"] = ["x", {"n": 1}]
    agent.run(state)

    loaded = store.load("sum")
    assert loaded.id == "sum"
    assert loaded.task == state.task
    assert loaded.history == state.history
    assert [e.at for e in loaded.history] == [e.at for e in state.history]
    assert loaded.context == state.context
    assert loaded.turn == state.turn == 2
    assert loaded.usage == state.usage
    assert loaded.answer == "It is 3"
    assert loaded.stopped_by == "is_answered"
    assert loaded.data == {"seen": ["x", {"n": 1}]}
    assert loaded.created_at == state.created_at and loaded.updated_at == state.updated_at
    assert loaded.is_answered()


def test_loaded_state_continues(tmp_path):
    store = FileStore(tmp_path)
    make_agent(["Hello"], store=store).run(State("Hi", id="chat"))

    state = store.load("chat")
    state.add_user_message("And then?")
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
    state = State("Go", id="err")
    with pytest.raises(RuntimeError):
        make_agent([tool_call("broken")], tools=[broken], store=store).run(state)
    closed = [e for e in state.history if e.kind == "tool_result"]
    assert closed and closed[-1].is_error
    loaded = store.load("err")
    # The exception object is not saved; the entry text keeps its type and message.
    assert loaded.history == tuple(dataclasses.replace(e, error=None) for e in state.history)
    assert [e.content for e in loaded.history if e.kind == "error"] == ["RuntimeError: boom"]


def test_answers_become_plain_values(tmp_path):
    @dataclass
    class Review:
        ok: bool

    class Verdict(BaseModel):
        score: int

    store = FileStore(tmp_path)
    agent = make_agent(['{"ok": true}'], store=store)
    state = State("Review", id="rev")
    agent.ask(state, "ok?", returns=Review)
    state.finish(Verdict(score=3))
    agent.save(state)

    loaded = store.load("rev")
    assert loaded.answer == {"score": 3}
    assert [e.content.answer for e in loaded.history if e.kind == "ask"] == [{"ok": True}]
    assert loaded.is_finished()


def test_data_that_is_not_json_fails_before_the_model_is_called(tmp_path):
    fake = FakeModel(["Done"])
    state = State("Go")
    state.data["seen"] = {1, 2}
    with pytest.raises(TypeError, match=r"state.data\['seen'\] holds set"):
        make_agent(model=fake, store=FileStore(tmp_path)).run(state)
    assert fake.requests == []


@pytest.mark.parametrize("value", [{1: "a"}, float("nan"), object()])
def test_other_values_that_are_not_json(tmp_path, value):
    state = State("Go")
    state.data["x"] = value
    with pytest.raises(TypeError, match="cannot be saved as JSON"):
        make_agent(store=FileStore(tmp_path)).save(state)


# ================================================================ save points


def test_saves_after_each_step_and_skips_unchanged_snapshots():
    store = MemoryStore()
    agent = make_agent([tool_call("add", a=1, b=2), "3"], tools=[add], store=store)
    state = State("Sum", id="s")
    agent.run(state)
    record = store.states["s"]
    assert len(record.entries) == len(state.history)
    assert record.snapshot["history_len"] == len(state.history)
    assert record.snapshot["stopped_by"] == "is_answered"
    writes = store.writes
    agent.save(state)
    assert store.writes == writes  # nothing new


def test_snapshot_records_the_agent_and_resume_warns(tmp_path):
    store = FileStore(tmp_path)
    make_agent(store=store, name="old").run(State("Go", id="w"))
    state = store.load("w")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make_agent(store=store, name="new").run(state)
    [warning] = [w.message for w in caught if isinstance(w.message, ResumeWarning)]
    assert warning.changes["name"] == ("old", "new")


def test_save_needs_a_store():
    with pytest.raises(ValueError, match="needs a store"):
        make_agent().save(State("Go"))


def test_store_must_be_a_store_object(tmp_path):
    with pytest.raises(TypeError, match="not a Store object"):
        make_agent(store=FileStore)
    assert make_agent(store=FileStore(tmp_path)).copy(name="x").store is not None


# ================================================================ ids and binding


def test_new_state_with_a_saved_id_is_refused_before_the_model_is_called(tmp_path):
    store = FileStore(tmp_path)
    make_agent(store=store).run(State("First", id="dup"))
    fake = FakeModel(["Done"])
    with pytest.raises(ValueError, match="already saved") as info:
        make_agent(model=fake, store=store).run(State("Second", id="dup"))
    assert "store.load" in str(info.value)
    assert fake.requests == []
    assert store.load("dup").task == "First"


def test_a_state_is_saved_in_one_store_only(tmp_path):
    first, second = FileStore(tmp_path / "a"), FileStore(tmp_path / "b")
    state = State("Go")
    make_agent(store=first).run(state)
    state.add_user_message("more")
    with pytest.raises(ValueError, match="saved in another store"):
        make_agent(store=second).run(state)


def test_two_file_stores_for_one_folder_are_the_same_store(tmp_path):
    make_agent(["One"], store=FileStore(tmp_path)).run(State("Go", id="same"))
    state = FileStore(tmp_path).load("same")
    state.add_user_message("again")
    make_agent(["Two"], store=FileStore(str(tmp_path) + "/.")).run(state)
    assert FileStore(tmp_path).load("same").answer == "Two"


def test_load_missing_id(tmp_path):
    with pytest.raises(LookupError, match="no saved State"):
        FileStore(tmp_path).load("nope")


# ================================================================ crash: replaying the tail


def test_crash_while_a_tool_runs_closes_the_call_as_unknown(tmp_path):
    live = FileStore(tmp_path / "live")
    state = State("Send the report", id="mail")
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
    assert (last.kind, last.content, last.is_error) == ("tool_result", UNSAVED_RESULT, True)
    assert last.call.name == "send_email"
    assert [m.role for m in loaded.context] == ["user", "assistant", "user"]
    assert loaded.context[-1].content[0].content == UNSAVED_RESULT
    assert not loaded.is_answered()

    # Continuing sends the unknown result to the model, and the closing entry gets saved.
    fake = FakeModel(["I will check the outbox"])
    make_agent(model=fake, tools=[send_email], store=crashed[0]).run(loaded)
    assert UNSAVED_RESULT in str(fake.requests[0].messages[-1].content)
    assert crashed[0].load("mail").history == loaded.history


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
    agent.run(State("Both", id="par"))

    loaded = crashed[0].load("par")
    results = {e.call.name: e.content for e in loaded.history if e.kind == "tool_result"}
    assert results == {"fast": "fast result", "slow": UNSAVED_RESULT}
    blocks = loaded.context[-1].content
    assert [b.content for b in blocks] == ["fast result", UNSAVED_RESULT]  # request order


def test_crash_while_waiting_on_the_model_drops_that_think(tmp_path):
    live = FileStore(tmp_path / "live")
    state = State("Find the bug", id="think")
    crashed: list[FileStore] = []

    def during(request):
        state.add_user_message("hurry")  # another thread would do this; held until the reply
        agent.save(state)
        crashed.append(crash_copy(live, "think", tmp_path / "crash"))
        return "Done"

    agent = make_agent([during, "Ok"], store=live)
    agent.run(state)

    loaded = crashed[0].load("think")
    assert loaded.turn == 0
    assert [m.text for m in loaded.context] == ["Find the bug", "hurry"]
    assert loaded.history[-1].content == "hurry"


def test_crash_replays_messages_held_during_think_after_the_results(tmp_path):
    live = FileStore(tmp_path / "live")
    state = State("Fix it", id="held")
    crashed: list[FileStore] = []

    def during(request):
        state.add_user_message("also run the tests")
        return tool_call("probe")

    @tool
    def probe() -> str:
        """Probe"""
        crashed.append(crash_copy(live, "held", tmp_path / "crash"))
        return "ok"

    agent = make_agent([during, "Done"], tools=[probe], store=live)
    agent.run(state)

    loaded = crashed[0].load("held")
    task, reply, results, held = loaded.context
    assert task.text == "Fix it"
    assert [call.name for call in reply.tool_calls] == ["probe"]
    assert [block.content for block in results.content] == [UNSAVED_RESULT]
    assert held.text == "also run the tests"


def test_steps_that_cannot_be_replayed_fall_back_to_the_snapshot(tmp_path):
    store = SnapshotFailingStore(tmp_path)
    agent = make_agent(["First answer"], store=store)
    state = State("Go", id="b")
    agent.run(state)
    snapshot_context = state.context

    store.failing = True
    state.start_from("A summary")
    with pytest.raises(OSError):
        agent.save(state)

    loaded = FileStore(tmp_path).load("b")
    assert loaded.history[:-1] == state.history
    assert loaded.history[-1].content == UNSAVED_STEPS
    assert loaded.context == snapshot_context + (loaded.context[-1],)
    assert loaded.context[-1].text == UNSAVED_STEPS


def test_fallback_names_calls_without_results(tmp_path):
    store = FileStore(tmp_path)
    state = State("Go", id="bc")
    make_agent(["First answer"], store=store).run(state)
    saved = len(state.history)

    # Steps after the snapshot that cannot be replayed (a start_from), then a reply whose call has no result,
    # written as a save whose snapshot never landed.
    state.start_from("summary")
    state._begin_think()
    state._record_reply(Reply(Message("assistant", (tool_call("pay", amount=5),)), Usage(requests=1)))
    entries = [_serial.entry_to_dict(entry, seq) for seq, entry in enumerate(state.history)][saved:]
    store.write("bc", entries, None)

    loaded = FileStore(tmp_path).load("bc")
    notice = loaded.context[-1].text
    assert notice.startswith(UNSAVED_STEPS)
    assert 'pay(amount=5)' in notice
    assert loaded.usage.requests == state.usage.requests
    assert loaded.pending_calls == ()


def test_system_exit_closes_calls_and_saves(tmp_path):
    """What the SIGTERM handler in the guide does: SystemExit ends the run like an exception."""
    store = FileStore(tmp_path)

    @loop(until=State.is_answered, limit=5)
    def stopped(agent, state):
        agent.think(state)
        raise SystemExit(143)

    state = State("Go", id="term")
    with pytest.raises(SystemExit):
        make_agent([tool_call("add", a=1, b=1)], tools=[add], loop=stopped, store=store).run(state)
    loaded = FileStore(tmp_path).load("term")
    assert [e.content for e in loaded.history if e.kind == "tool_result"] == ["(aborted: SystemExit)"]
    assert loaded.pending_calls == ()


def test_torn_last_line_is_ignored_and_repaired(tmp_path):
    store = FileStore(tmp_path)
    agent = make_agent(["One", "Two"], store=store)
    state = State("Go", id="torn")
    agent.run(state)
    log = tmp_path / "torn" / "log.jsonl"
    with open(log, "a") as file:
        file.write('{"seq": 99, "kind": "us')

    reopened = FileStore(tmp_path)
    loaded = reopened.load("torn")
    assert loaded.history == state.history

    loaded.add_user_message("again")
    make_agent(["Two"], store=reopened).run(loaded)
    lines = log.read_text().splitlines()
    assert all(json.loads(line) for line in lines)
    assert FileStore(tmp_path).load("torn").answer == "Two"


# ================================================================ failures while saving


def test_failed_save_during_a_failing_run_is_a_note_on_the_original():
    store = FailingStore()
    agent = make_agent([RateLimitError("429")], store=store)
    state = State("Go")
    agent.save(state)
    store.failing = True
    with pytest.raises(RateLimitError) as info:
        agent.run(state)
    assert info.value.__notes__ == ["saving the state also failed: OSError: disk full"]


def test_interrupt_while_saving_wins_over_the_original():
    store = FailingStore()
    agent = make_agent([RateLimitError("429")], store=store)
    state = State("Go")
    agent.save(state)
    store.failing = True
    store.error = KeyboardInterrupt
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
    store.fail_when = lambda entries, snapshot: snapshot is not None and snapshot["stopped_by"] is not None
    state = State("Go")
    with pytest.raises(OSError, match="disk full") as info:
        make_agent(["Done"], store=store).run(state)
    assert not getattr(info.value, "__notes__", None)  # the retry failed the same way: no note
    assert state.answer == "Done"


def test_failed_save_after_think_does_not_roll_the_reply_back():
    store = FailingStore()
    agent = make_agent([tool_call("add", a=1, b=1)], tools=[add], store=store)
    state = State("Go")
    agent.save(state)

    original = store.write

    def fail_after_think(state_id, entries, snapshot, *, create=False):
        if any(e["kind"] == "reply" for e in entries):
            raise OSError("disk full")
        original(state_id, entries, snapshot, create=create)

    store.write = fail_after_think
    with pytest.raises(OSError):
        agent.think(state)
    assert state.turn == 1
    assert state.wants_tools()  # the reply stays; nothing was rolled back


def test_per_tool_save_failures_do_not_stop_the_tools():
    store = FailingStore()
    # Saves while calls are pending (no snapshot) fail; the save at the end of use_tools works.
    store.fail_when = lambda entries, snapshot: snapshot is None and any(e["kind"] == "tool_result" for e in entries)
    ran = []

    @tool(parallel=False)
    def work(n: int) -> str:
        """Work"""
        ran.append(n)
        return f"done {n}"

    agent = make_agent([[tool_call("work", n=1), tool_call("work", n=2)], "Done"], tools=[work], store=store)
    state = State("Go", id="t")
    agent.run(state)
    assert ran == [1, 2]
    assert store.failed >= 1
    assert len(store.states["t"].entries) == len(state.history)


def test_create_is_all_or_nothing(tmp_path):
    store = FileStore(tmp_path)
    store.write("x", [{"seq": 0, "kind": "user", "turn": 0, "at": "2026-01-01T00:00:00+00:00", "content": "a"}],
                None, create=True)
    with pytest.raises(ValueError, match="already saved"):
        store.write("x", [], None, create=True)
    assert [p.name for p in tmp_path.iterdir()] == ["x"]


def test_write_skips_entries_it_already_has(tmp_path):
    store = FileStore(tmp_path)
    state = State("Go", id="dedupe")
    agent = make_agent(["Done"], store=store)
    agent.run(state)
    entries = store.read("dedupe").entries
    FileStore(tmp_path).write("dedupe", entries, None)  # a retried write
    assert store.read("dedupe").entries == entries


# ================================================================ list, delete, files


def test_list_and_delete(tmp_path):
    store = FileStore(tmp_path)
    make_agent(["A"], store=store).run(State("First", id="one"))
    make_agent(["B"], store=store).run(State("Second", id="two"))
    saved = store.list()
    assert [s.id for s in saved] == ["two", "one"]
    assert isinstance(saved[0], SavedState)
    assert (saved[0].task, saved[0].turn, saved[0].stopped_by, saved[0].finished) == ("Second", 1, "is_answered", False)

    store.delete("one")
    store.delete("one")  # already gone: nothing happens
    assert [s.id for s in store.list()] == ["two"]
    assert FileStore(tmp_path / "missing").list() == []


def test_files_are_private(tmp_path):
    store = FileStore(tmp_path)
    make_agent(store=store).run(State("Go", id="p"))
    for name in ("log.jsonl", "snapshot.json"):
        assert stat.S_IMODE(os.stat(tmp_path / "p" / name).st_mode) == 0o600


# ================================================================ Store role


def test_store_needs_write_and_read():
    class Nothing(Store):
        pass

    with pytest.raises(TypeError, match="neither write nor awrite"):
        Nothing()


def test_list_and_delete_are_optional():
    store = MemoryStore()
    with pytest.raises(NotImplementedError):
        store.list()
    with pytest.raises(NotImplementedError):
        store.delete("x")


class AsyncMemoryStore(Store):
    def __init__(self) -> None:
        self.inner = MemoryStore()

    async def awrite(self, state_id, entries, snapshot, *, create=False):
        await asyncio.sleep(0)
        self.inner.write(state_id, entries, snapshot, create=create)

    async def aread(self, state_id):
        return self.inner.read(state_id)


def test_async_only_store_refuses_sync_run():
    with pytest.raises(TypeError, match="only asynchronously"):
        make_agent(store=AsyncMemoryStore()).run("Go")


async def test_arun_with_async_store():
    store = AsyncMemoryStore()
    agent = make_agent([tool_call("add", a=2, b=2), "4"], tools=[add], store=store)
    state = State("Sum", id="a")
    await agent.arun(state)
    loaded = await store.aload("a")
    assert loaded.history == state.history
    assert loaded.answer == "4"


async def test_arun_with_file_store(tmp_path):
    store = FileStore(tmp_path)
    state = State("Go", id="af")
    await make_agent(["Done"], store=store).arun(state)
    assert (await store.aload("af")).history == state.history
    assert [s.id for s in await store.alist()] == ["af"]
    await store.adelete("af")
    assert store.list() == []


async def test_sync_save_inside_async_run_is_refused():
    from alpineagents._async import ASYNC_RUN

    agent = make_agent(store=MemoryStore())
    token = ASYNC_RUN.set(asyncio.get_running_loop())
    try:
        with pytest.raises(TypeError, match="asave"):
            agent.save(State("Go"))
    finally:
        ASYNC_RUN.reset(token)
