"""When history entries were recorded: ``HistoryEntry.at`` and ``State.created_at``/``updated_at``.

The clock is frozen by replacing ``alpineagents.types._now``, so every stamp is known. No network.
"""

import itertools
import threading
from datetime import datetime, timedelta, timezone

import pytest

import alpineagents.types as types_module
from alpineagents import Agent, Message, State, tool
from alpineagents.testing import FakeHuman, FakeModel, tool_call
from alpineagents.types import MessageEntry

T0 = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)


def minute(n):
    return T0 + timedelta(minutes=n)


@pytest.fixture
def clock(monkeypatch):
    """Each read of the clock returns the next minute after T0: T0, T0+1m, T0+2m, ..."""
    ticks = itertools.count()
    monkeypatch.setattr(types_module, "_now", lambda: minute(next(ticks)))


def task(text="Task"):
    return State(messages=[Message.user(text)])


def test_every_entry_is_stamped_when_it_is_recorded(clock):
    @tool
    def run_tests(state: State) -> str:
        """Run the tests"""
        # Added while the call is pending: stamped now, but goes into the messages after the result.
        state.add_message(Message.user("Also check the docs"))
        return "3 passed"

    fake = FakeModel([tool_call("run_tests"), "All good"])
    state = task("Check the repo")
    Agent(model=fake, tools=[run_tests], reporter=None, human=None).run(state)

    assert [(e.kind, e.at) for e in state.history] == [
        ("context_change", minute(0)),  # the import of the starting messages
        ("run_start", minute(1)),
        ("model_request", minute(2)),
        ("model_reply", minute(3)),
        ("user", minute(4)),
        ("tool_result", minute(5)),
        ("model_request", minute(6)),
        ("model_reply", minute(7)),
        ("stop", minute(8)),
    ]
    assert state.created_at == minute(0)
    assert state.updated_at == minute(8)


def test_created_at_is_the_first_entry_and_updated_at_follows_the_latest_entry(clock):
    agent = Agent(model=FakeModel(["Answer"]), human=FakeHuman(["yes"]), reporter=None)
    state = task()
    assert state.created_at == state.updated_at == minute(0)

    agent.run(state)
    after_run = len(state.history) - 1
    assert state.updated_at == minute(after_run)

    agent.ask_human(state, "Proceed?")
    assert state.history[-1].kind == "human"
    assert state.updated_at == minute(after_run + 1)

    state.compact("Summary so far")
    assert state.history[-1].kind == "context_change"
    assert state.updated_at == minute(after_run + 2)
    assert state.created_at == minute(0)


def test_an_empty_state_has_no_times_until_something_is_recorded(clock):
    state = State()
    assert state.created_at is None and state.updated_at is None
    state.add_message(Message.user("First"))
    assert state.created_at == state.updated_at == minute(0)


def test_changing_extra_data_moves_updated_at_but_no_change_does_not(clock):
    state = task()
    with state.edit_extra_data() as data:
        data["notes"] = ["first"]
    assert state.history[-1].kind == "extra_data"
    assert state.updated_at == minute(1)
    assert state.created_at == minute(0)

    with state.edit_extra_data() as data:  # nothing changed: no entry, no stamp
        data.setdefault("notes", ["first"])
    assert state.updated_at == minute(1)
    assert len(state.history) == 2

    with state.edit_extra_data() as data:
        data["count"] = 0
    assert state.updated_at == minute(2)


def test_equality_ignores_at():
    first = MessageEntry(kind="user", content="Find the bug", turn=0, at=minute(0))
    later = MessageEntry(kind="user", content="Find the bug", turn=0, at=minute(5))
    assert first == later
    assert MessageEntry(kind="user", content="Find the bug", turn=0) == first


def test_a_rebuilt_state_keeps_the_stamps(clock):
    state = task()
    state.add_message(Message.notice("Something happened"))
    rebuilt = State(history=state.history)
    assert [e.at for e in rebuilt.history] == [minute(0), minute(1)]
    assert rebuilt.created_at == minute(0) and rebuilt.updated_at == minute(1)
    assert rebuilt.snapshot() == state.snapshot()


def test_at_is_timezone_aware_utc():
    before = datetime.now(timezone.utc)
    state = task()
    state.add_message(Message.notice("Something happened"))
    after = datetime.now(timezone.utc)

    assert len(state.history) == 2
    for entry in state.history:
        assert entry.at.tzinfo is timezone.utc
        assert entry.at.utcoffset() == timedelta(0)
        assert before <= entry.at <= after


def test_entries_from_several_threads_are_stamped_in_history_order(clock):
    # The stamp is taken inside the State lock, so no thread can stamp first and append later.
    state = task()

    def worker(n):
        for i in range(100):
            state.add_message(Message.notice(f"worker {n} step {i}"))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stamps = [entry.at for entry in state.history]
    assert len(stamps) == 801
    assert stamps == sorted(stamps)
    assert state.updated_at == stamps[-1]
