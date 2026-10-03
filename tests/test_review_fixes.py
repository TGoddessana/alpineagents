"""Regression tests for the second review round: error advice, context_used checks, a repeated exception object,
FileStore and special characters, Message content as a tuple."""

from __future__ import annotations

import pytest

from alpineagents import Agent, FileStore, Message, State
from alpineagents.errors import ProviderError
from alpineagents.testing import FakeModel
from alpineagents.types import TextBlock, ToolResultBlock


def agent_for(*replies) -> Agent:
    return Agent(FakeModel(list(replies)), reporter=None)


# ---------------------------------------------------------------- advice in errors


def test_run_on_finished_state_does_not_advise_fork():
    state = State(messages=[Message.user("hi")])
    state.finish("x")
    with pytest.raises(ValueError) as info:
        agent_for("a").run(state)
    assert "fork" in str(info.value)  # it says that a fork is finished too
    assert "State(messages=state.messages)" in str(info.value)
    assert state.fork().finished  # which is true
    with pytest.raises(ValueError):
        agent_for("a").run(state.fork())
    assert agent_for("a").run(State(messages=state.messages)) == "a"  # the advice works


def test_running_error_and_think_waiting_error_do_not_advise_fork():
    state = State(messages=[Message.user("hi")])
    agent = agent_for("a", "b")
    agent._start_run(state, "run")  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(ValueError) as info:
        agent._start_run(state, "run")  # pyright: ignore[reportPrivateUsage]
    assert "state.fork()" not in str(info.value)

    waiting = State(messages=[Message.user("hi")])
    waiting._begin_think(agent._agent_info_now())  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(ValueError) as info:
        agent.think(waiting)
    assert "state.fork()" not in str(info.value)


# ---------------------------------------------------------------- context_used


def test_context_used_checks_its_argument_and_names_itself():
    agent = agent_for("x")
    state = State(messages=[Message.user("hi")])
    with pytest.raises(TypeError, match="context_used"):
        agent.context_used(state.snapshot())  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------- the same exception object failing twice


def test_same_exception_object_failing_twice_rolls_back_both_times():
    err = ProviderError("boom")
    agent = agent_for(err, err, "ok")
    state = State(messages=[Message.user("x")])
    for _ in range(2):
        with pytest.raises(ProviderError):
            agent.think(state)
    assert state.turn == 0
    assert [e.kind for e in state.history].count("error") == 2
    agent.think(state)  # no longer stuck waiting on the model
    assert state.messages[-1].text == "ok"


def test_run_after_failed_think_does_not_record_the_same_error_twice():
    err = ProviderError("boom")
    agent = agent_for(err)
    state = State(messages=[Message.user("x")])
    with pytest.raises(ProviderError):
        agent.run(state)
    assert [e.kind for e in state.history].count("error") == 1


# ---------------------------------------------------------------- FileStore and special characters


@pytest.mark.parametrize("char", [" ", " ", "\x85", "\x0b", "\x0c", "\x1c"])
def test_filestore_keeps_line_separator_characters_inside_entries(tmp_path, char):
    store = FileStore(tmp_path)
    state = State(id="a", messages=[Message.user(f"line one{char}line two")], extra_data={"k": f"v{char}w"})
    store.save(state)
    state.add_message(Message.user(f"more{char}"))
    FileStore(tmp_path).save(state)  # another FileStore object counts the entries already on disk
    loaded = store.load("a")
    assert loaded.snapshot() == state.snapshot()
    assert loaded.messages[0].text == f"line one{char}line two"


def test_filestore_saves_and_loads_a_lone_surrogate(tmp_path):
    store = FileStore(tmp_path)
    state = State(id="x", messages=[Message.user("name: \udcff.txt")])
    store.save(state)
    state.add_message(Message.user("ok"))
    store.save(state)  # does not stay broken
    loaded = store.load("x")
    assert loaded.messages[0].text == "name: \udcff.txt"
    assert loaded.snapshot() == state.snapshot()


def test_filestore_log_stays_readable_utf8(tmp_path):
    store = FileStore(tmp_path)
    store.save(State(id="k", messages=[Message.user("안녕하세요")]))
    assert "안녕하세요" in (tmp_path / "k" / "log.jsonl").read_text("utf-8")


# ---------------------------------------------------------------- Message content is a tuple


def test_message_list_content_becomes_a_tuple(tmp_path):
    state = State(
        id="l2", messages=[Message("user", [TextBlock("hi")]), Message("assistant", [TextBlock("yo")])]  # pyright: ignore[reportArgumentType]
    )
    assert isinstance(state.messages[0].content, tuple)
    store = FileStore(tmp_path)
    store.save(state)
    assert store.load("l2").snapshot() == state.snapshot()
    with pytest.raises(AttributeError):
        state.messages[0].content.append(TextBlock("MUT"))  # pyright: ignore[reportAttributeAccessIssue]
    assert state.history[0].content.messages[0].text == "hi"  # pyright: ignore[reportAttributeAccessIssue]


def test_message_content_string_is_refused():
    with pytest.raises(TypeError, match="Message.user"):
        Message("user", "hi")  # pyright: ignore[reportArgumentType]


def test_tool_result_block_list_content_becomes_a_tuple():
    block = ToolResultBlock("c1", [TextBlock("t")])  # pyright: ignore[reportArgumentType]
    assert block.content == (TextBlock("t"),)
    assert ToolResultBlock("c1", "text").content == "text"
