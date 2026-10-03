"""Errors that guide people coming from 0.4 to the 0.5 State. They give guidance only: the old names do not work."""

from __future__ import annotations

import copy
import pickle  # noqa: S403 (a State with only a history in it, loaded back in the same test)

import pytest

from alpineagents import Message, State

STATE_ARGS = {"messages": [Message.user("Task")]}


def make() -> State:
    return State(**STATE_ARGS)


# ---------------------------------------------------------------- State(...) keywords


def test_task_keyword_points_to_messages():
    with pytest.raises(TypeError) as e:
        State(task="Fix it")  # type: ignore[call-arg]
    text = str(e.value)
    assert "State has no task in 0.5" in text
    assert "State(messages=[Message.user(...)])" in text
    assert "Fix:" in text and "Example:" in text


def test_context_keyword_points_to_messages():
    with pytest.raises(TypeError, match=r"context=.*messages="):
        State(context=[Message.user("x")])  # type: ignore[call-arg]


def test_data_keyword_points_to_extra_data():
    with pytest.raises(TypeError, match=r"data=.*extra_data="):
        State(data={"a": 1})  # type: ignore[call-arg]


@pytest.mark.parametrize("name", ["turn", "usage", "stopped", "finished", "answer", "pending_calls"])
def test_derived_keywords_say_they_come_from_history(name):
    with pytest.raises(TypeError) as e:
        State(**{name: 1})  # type: ignore[arg-type]
    assert f"{name}=" in str(e.value)
    assert "derived from history" in str(e.value)
    assert "history=" in str(e.value)


def test_other_unknown_keyword_lists_the_accepted_ones():
    with pytest.raises(TypeError) as e:
        State(mesages=[])  # type: ignore[call-arg]
    text = str(e.value)
    assert "'mesages'" in text
    for accepted in ("id", "messages", "extra_data", "history"):
        assert accepted in text


def test_positional_argument_error_is_kept():
    with pytest.raises(TypeError, match="keyword arguments only"):
        State("Fix it")  # type: ignore[call-arg]


def test_valid_keywords_still_work():
    state = State(id="x", messages=[Message.user("a")], extra_data={"k": 1})
    assert state.id == "x" and state.extra_data == {"k": 1}
    assert State(history=state.history).messages == state.messages


# ---------------------------------------------------------------- assigning attributes


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("data", "edit_extra_data()"),
        ("context", "state.messages is read-only; use state.add_message(...) / state.compact(...)"),
        ("task", "removed; the first message is state.messages[0]"),
        ("lock", "removed; edit_extra_data() is atomic"),
    ],
)
def test_assigning_a_removed_name_names_the_replacement(name, expected):
    state = make()
    with pytest.raises(AttributeError) as e:
        setattr(state, name, 1)
    assert expected in str(e.value)
    assert "Fix:" in str(e.value)
    assert not hasattr(state, name)


@pytest.mark.parametrize(
    ("name", "command"),
    [
        ("messages", "add_message"),
        ("extra_data", "edit_extra_data"),
        ("history", "history only grows"),
        ("pending_calls", "use_tools"),
        ("turn", "think"),
        ("usage", "think"),
        ("finished", "finish"),
        ("answer", "finish"),
        ("stopped", "finish"),
        ("created_at", "history entry"),
        ("updated_at", "history entry"),
        ("id", "State(id="),
    ],
)
def test_assigning_a_read_only_property_names_the_command(name, command):
    state = make()
    before = state.snapshot()
    with pytest.raises(AttributeError) as e:
        setattr(state, name, 1)
    text = str(e.value)
    assert f"state.{name} is read-only" in text
    assert "only through its commands" in text
    assert command in text
    assert state.snapshot() == before


def test_assigning_a_typo_says_the_attribute_does_not_exist():
    state = make()
    with pytest.raises(AttributeError) as e:
        state.mesages = []  # type: ignore[attr-defined]
    assert "State has no attribute 'mesages' and does not accept new attributes" in str(e.value)
    assert not hasattr(state, "mesages")


def test_assigning_over_a_method_is_refused():
    state = make()
    with pytest.raises(AttributeError, match="State method"):
        state.add_message = None  # type: ignore[method-assign]
    state.add_message(Message.user("still works"))


def test_assigning_a_removed_method_name_points_to_the_replacement():
    state = make()
    with pytest.raises(AttributeError, match="state.add_message"):
        state.add_user_message = None  # type: ignore[attr-defined]


def test_names_starting_with_an_underscore_can_be_assigned():
    state = make()
    state._mine = 1  # type: ignore[attr-defined]
    assert state._mine == 1  # type: ignore[attr-defined]


def test_internal_paths_do_not_assign_public_names():
    """Copy, pickle-free replay, fork, store load and subclasses keep working with the guard in place."""
    state = make()
    state.finish("done")
    assert state.fork().answer == "done"
    assert State(history=state.history).snapshot() == state.snapshot()
    assert copy.copy(state).history == state.history

    class MyState(State):
        pass

    mine = MyState(messages=[Message.user("q")], extra_data={"a": 1})
    assert mine.fork().extra_data == {"a": 1}
    with pytest.raises(AttributeError):
        mine.note = "x"  # type: ignore[attr-defined]


def test_load_from_store_goes_through_the_guard(tmp_path):
    from alpineagents import FileStore

    store = FileStore(tmp_path)
    state = State(id="s", messages=[Message.user("q")])
    state.finish("a")
    store.save(state)
    assert store.load("s").answer == "a"


def test_a_state_with_a_history_pickles_the_state_free_way():
    # a State holds a lock, so it is not picklable; its history is
    state = make()
    assert State(history=pickle.loads(pickle.dumps(state.history))).snapshot() == state.snapshot()


# ---------------------------------------------------------------- reading removed names


@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        ("data", "extra_data"),
        ("context", "state.messages"),
        ("task", "state.messages[0]"),
        ("lock", "edit_extra_data()"),
        ("is_answered", "until function"),
        ("wants_tools", "pending_calls"),
        ("is_finished", "state.finished"),
        ("add_user_message", "Message.user"),
        ("add_notice", "Message.notice"),
        ("start_from", "compact"),
        ("context_used", "agent.context_used(state)"),
        ("context_tokens", "agent.context_tokens(state)"),
    ],
)
def test_reading_a_removed_name_names_the_replacement(name, replacement):
    state = make()
    with pytest.raises(AttributeError) as e:
        getattr(state, name)
    assert replacement in str(e.value)
    assert "Fix:" in str(e.value)
    assert not hasattr(state, name)


def test_reading_a_missing_name_is_the_usual_attribute_error():
    state = make()
    with pytest.raises(AttributeError, match="'State' object has no attribute 'nothing'"):
        state.nothing  # type: ignore[attr-defined]  # noqa: B018
    assert getattr(state, "nothing", "default") == "default"


def test_old_names_do_not_work_as_aliases():
    state = make()
    for name in ("data", "context", "task", "lock"):
        assert not hasattr(state, name)
    with pytest.raises(TypeError):
        State(task="x")  # type: ignore[call-arg]
