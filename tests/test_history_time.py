"""When history entries were recorded: ``HistoryEntry.at`` and ``State.created_at``/``updated_at``.

The clock is frozen by replacing ``alpineagents.types._now``, so every stamp is known. No network.
"""

import itertools
import threading
from datetime import datetime, timedelta, timezone

import pytest

import alpineagents.types as types_module
from alpineagents import Agent, State, tool
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


def test_every_entry_is_stamped_when_it_is_recorded(clock):
    @tool
    def run_tests(state: State) -> str:
        """Run the tests"""
        # Added while the call is pending: stamped now, but goes into the context after the result.
        state.add_user_message("Also check the docs")
        return "3 passed"

    fake = FakeModel([tool_call("run_tests"), "All good"])
    state = State("Check the repo")
    Agent(model=fake, tools=[run_tests], reporter=None, human=None).run(state)

    assert [(e.kind, e.at) for e in state.history] == [
        ("user", minute(0)),
        ("reply", minute(1)),
        ("user", minute(2)),
        ("tool_result", minute(3)),
        ("reply", minute(4)),
    ]
    assert state.created_at == minute(0)
    assert state.updated_at == minute(4)


def test_created_at_is_the_task_and_updated_at_follows_the_latest_entry(clock):
    agent = Agent(model=FakeModel(["Answer"]), human=FakeHuman(["yes"]), reporter=None)
    state = State("Task")
    assert state.created_at == state.updated_at == minute(0)

    agent.run(state)
    assert state.updated_at == minute(1)

    agent.ask_human(state, "Proceed?")
    assert state.history[-1].kind == "human"
    assert state.updated_at == minute(2)

    state.start_from("Summary so far")
    assert state.history[-1].kind == "context_change"
    assert state.updated_at == minute(3)
    assert state.created_at == minute(0)


def test_changing_data_alone_does_not_move_updated_at(clock):
    state = State("Task")
    state.data["notes"] = ["first"]
    state.data.setdefault("count", 0)
    assert state.updated_at == minute(0)
    assert len(state.history) == 1


def test_equality_ignores_at():
    first = MessageEntry(kind="user", content="Find the bug", turn=0, at=minute(0))
    later = MessageEntry(kind="user", content="Find the bug", turn=0, at=minute(5))
    assert first == later
    assert MessageEntry(kind="user", content="Find the bug", turn=0) == first


def test_at_is_timezone_aware_utc():
    before = datetime.now(timezone.utc)
    state = State("Task")
    state.add_notice("Something happened")
    after = datetime.now(timezone.utc)

    for entry in state.history:
        assert entry.at.tzinfo is timezone.utc
        assert entry.at.utcoffset() == timedelta(0)
        assert before <= entry.at <= after


def test_entries_from_several_threads_are_stamped_in_history_order(clock):
    # The stamp is taken inside the State lock, so no thread can stamp first and append later.
    state = State("Task")

    def worker(n):
        for i in range(100):
            state.add_notice(f"worker {n} step {i}")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stamps = [entry.at for entry in state.history]
    assert len(stamps) == 801
    assert stamps == sorted(stamps)
    assert state.updated_at == stamps[-1]
