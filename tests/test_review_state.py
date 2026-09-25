"""Regression tests for State defects found in review (state.py)."""

from __future__ import annotations

import sys
import threading

import pytest

from alpineagents.agent import Agent
from alpineagents.models.base import Model
from alpineagents.state import State
from alpineagents.testing import FakeModel
from alpineagents.types import Message, Reply, Request, TextBlock, ToolCall, Usage


def _reply_with_call(call: ToolCall) -> Reply:
    return Reply(message=Message("assistant", (call,)), usage=Usage())


# ---------------------------------------------------------------- a late result attaches to its call, not to its id


def test_late_result_must_not_attach_to_new_calls_reusing_the_same_id():
    state = State("task")

    # Turn 1: bash(cmd="1.0"), id "call_0". Closed by an interrupt (Ctrl+C), but the worker thread keeps running.
    call_turn1 = ToolCall(name="bash", args={"cmd": "1.0"}, id="call_0")
    state._begin_think()
    state._record_reply(_reply_with_call(call_turn1))
    state._record_tool_result(call_turn1, "(interrupted by user)", is_error=True)

    # Turn 2: the server does not give ids, so the first call is "call_0" again.
    call_turn2 = ToolCall(name="bash", args={"cmd": "2.0"}, id="call_0")
    state._begin_think()
    state._record_reply(_reply_with_call(call_turn2))
    assert state.pending_calls == (call_turn2,)

    # Turn 1's worker thread finishes now.
    state._record_late_result(call_turn1, "done 1.0")

    assert state.pending_calls == (call_turn2,), "turn 2's call must still be waiting for its own result"
    late = [e for e in state.history if e.kind == "tool_result" and e.late]
    assert len(late) == 1
    assert late[0].call is call_turn1 and late[0].content == "done 1.0"

    state._record_tool_result(call_turn2, "done 2.0")
    normal = [e.content for e in state.history if e.kind == "tool_result" and not e.late]
    assert "done 2.0" in normal
    assert "done 1.0" not in normal

    # The late result enters the context as a notice right before the next think.
    state._begin_think()
    assert state.context[-1].text == '[notice] interrupted bash(cmd="1.0") finished later: done 1.0'


def test_late_result_for_other_tool_with_same_id_is_not_misattributed():
    state = State("task")
    make = ToolCall(name="bash", args={"cmd": "make"}, id="call_0")
    state._begin_think()
    state._record_reply(_reply_with_call(make))
    state._record_tool_result(make, "(interrupted by user)", is_error=True)

    read = ToolCall(name="read_file", args={"path": "a"}, id="call_0")
    state._begin_think()
    state._record_reply(_reply_with_call(read))

    state._record_late_result(make, "exit 0")

    assert state.pending_calls == (read,)
    entry = state.history[-1]
    assert entry.kind == "tool_result" and entry.late and entry.call is make


def test_late_result_for_the_same_pending_call_object_is_a_normal_result():
    # If the loop caught the interrupt and the call is still pending, it is recorded as a normal result.
    state = State("task")
    call = ToolCall(name="bash", args={"cmd": "x"}, id="call_0")
    state._begin_think()
    state._record_reply(_reply_with_call(call))

    state._record_late_result(call, "ok")

    assert state.pending_calls == ()
    entry = state.history[-1]
    assert entry.kind == "tool_result" and not entry.late and entry.content == "ok"
    assert state.context[-1].content[0].content == "ok"


# ---------------------------------------------------------------- messages added during think/compact


class SlowModel(Model):
    """A model that waits in respond() until released (another thread changes the State meanwhile)."""

    def __init__(self, text: str = "first answer"):
        self.name = "slow"
        self.text = text
        self.release = threading.Event()
        self.entered = threading.Event()
        self.requests: list[Request] = []

    @property
    def context_window(self) -> int:
        return 100_000

    def respond(self, request: Request, on_text=None, on_event=None) -> Reply:
        self.requests.append(request)
        self.entered.set()
        self.release.wait(timeout=5)
        return Reply(message=Message("assistant", (TextBlock(self.text),)), usage=Usage(requests=1))


def _run_in_thread(target):
    errors: list[BaseException] = []

    def body():
        try:
            target()
        except BaseException as e:  # pragma: no cover - for failure reporting
            errors.append(e)

    thread = threading.Thread(target=body)
    thread.start()
    return thread, errors


def test_user_message_added_during_think_goes_after_the_reply():
    model = SlowModel()
    agent = Agent(model=model, reporter=None, human=None)
    state = State("task")

    thread, errors = _run_in_thread(lambda: agent.think(state))
    assert model.entered.wait(timeout=5)
    state.add_user_message("NEW QUESTION")
    # It is in the history right away.
    assert state.history[-1].kind == "user"
    model.release.set()
    thread.join(timeout=5)
    assert not errors

    # The model did not see NEW QUESTION.
    assert [m.text for m in model.requests[0].messages] == ["task"]
    assert [(m.role, m.text) for m in state.context] == [
        ("user", "task"),
        ("assistant", "first answer"),
        ("user", "NEW QUESTION"),
    ]
    assert not state.is_answered()


def test_is_answered_is_false_while_a_message_is_deferred_during_think():
    state = State("task")
    state._begin_think()
    state._record_reply(Reply(message=Message("assistant", (TextBlock("a"),)), usage=Usage()))
    assert state.is_answered()
    state._begin_think()
    state.add_user_message("more")
    assert not state.is_answered()


def test_message_added_during_think_with_tool_calls_goes_after_the_results():
    state = State("task")
    call = ToolCall(name="bash", args={"cmd": "x"}, id="c1")
    state._begin_think()
    state.add_notice("mid-think")
    state._record_reply(_reply_with_call(call))
    assert state.context[-1].role == "assistant"
    state._record_tool_result(call, "out")
    assert [m.text for m in state.context][-1] == "[notice] mid-think"
    assert state.context[-2].content[0].content == "out"


def test_message_added_during_failed_think_survives_the_rollback():
    state = State("task")
    checkpoint = state._begin_think()
    state.add_user_message("typed while thinking")
    state._rollback(checkpoint, RuntimeError("boom"))
    assert [m.text for m in state.context] == ["task", "typed while thinking"]
    # After the rollback it goes into the context right away (not deferred).
    state.add_user_message("after")
    assert state.context[-1].text == "after"


def test_user_message_added_during_compact_is_not_dropped():
    model = SlowModel(text="the summary")
    agent = Agent(model=model, reporter=None, human=None)
    state = State("task")
    state.add_user_message("q1")

    thread, errors = _run_in_thread(lambda: agent.compact(state))
    assert model.entered.wait(timeout=5)
    state.add_user_message("q3 typed during compaction")
    model.release.set()
    thread.join(timeout=5)
    assert not errors

    texts = [m.text for m in state.context]
    assert texts[0] == "task"
    assert "the summary" in texts[1]
    assert texts[2:] == ["q3 typed during compaction"]


def test_compact_with_no_concurrent_messages_leaves_task_and_summary_only():
    agent = Agent(model=FakeModel(["summary"]), reporter=None, human=None)
    state = State("task")
    state.add_user_message("q1")
    agent.compact(state)
    assert len(state.context) == 2
    # A message added after compaction finishes goes in only once.
    state.add_user_message("q2")
    assert [m.text for m in state.context][2:] == ["q2"]


def test_start_from_ignores_compact_tracking():
    state = State("task")
    state._begin_compact()  # a compact still waiting on the model
    state.add_user_message("q1")
    state.start_from("summary")
    assert len(state.context) == 2


def test_ensure_no_pending_does_not_start_compact_tracking():
    state = State("task")
    state._ensure_no_pending("compact")
    assert state._compact_added is None


def test_failed_compact_ends_tracking_right_away():
    agent = Agent(model=FakeModel([RuntimeError("provider down")]), reporter=None, human=None)
    state = State("task")
    with pytest.raises(RuntimeError):
        agent.compact(state)
    assert state._compact_added is None
    state.add_user_message("q1")
    assert [m.text for m in state.context] == ["task", "q1"]


# ---------------------------------------------------------------- empty text is rejected early


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_state_rejects_empty_task(text):
    with pytest.raises(ValueError, match="got an empty task"):
        State(text)


@pytest.mark.parametrize("text", ["", "   "])
def test_add_user_message_rejects_empty_text(text):
    state = State("do something")
    with pytest.raises(ValueError, match="Fix:"):
        state.add_user_message(text)
    assert len(state.history) == 1
    assert len(state.context) == 1


def test_add_notice_deny_and_start_from_reject_empty_text():
    state = State("do something")
    with pytest.raises(ValueError):
        state.add_notice(" ")
    with pytest.raises(ValueError):
        state.start_from("")
    call = ToolCall(name="bash", args={}, id="c1")
    state._begin_think()
    state._record_reply(_reply_with_call(call))
    with pytest.raises(ValueError):
        state.deny(call, "")
    assert state.pending_calls == (call,)


# ---------------------------------------------------------------- single reads of state.data


def test_dict_snapshot_of_state_data_is_atomic_under_concurrent_mutation():
    state = State("t")
    for k in range(20):
        state.data[k] = k

    stop = threading.Event()

    def mutator():
        i = 0
        while not stop.is_set():
            k = i % 20
            state.data.pop(k, None)
            state.data[k] = i
            i += 1

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    threads = [threading.Thread(target=mutator, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    try:
        for _ in range(3000):
            dict(state.data)
            {**state.data}
            {} | state.data
            state.data | {}
            list(state.data)
    finally:
        stop.set()
        for t in threads:
            t.join()
        sys.setswitchinterval(old)


def test_state_data_iterates_and_copies_like_a_dict():
    state = State("t")
    state.data.update(a=1, b=2)
    assert list(state.data) == ["a", "b"]
    assert dict(state.data) == {"a": 1, "b": 2}
    assert type(dict(state.data)) is dict
    assert {**state.data, "c": 3} == {"a": 1, "b": 2, "c": 3}


# ---------------------------------------------------------------- context_used on the first turn


def test_context_used_is_known_after_claim_even_before_the_first_think():
    state = State("x" * 600_000)
    agent = Agent(model=FakeModel(["ok"], context_window=200_000), reporter=None, human=None)
    state._claim(agent, context_window=200_000)
    assert state.context_used > 1.0


def test_compact_if_full_can_fire_on_turn_one():
    # A starting task above 60% of the context window but smaller than the window: it must be compacted before turn 1.
    state = State("x" * 400_000)
    fake = FakeModel(["summary", "ok"], context_window=200_000)
    agent = Agent(model=fake, reporter=None, human=None)
    assert agent.run(state) == "ok"
    assert len(fake.requests) == 2
    assert any(e.kind == "context_change" for e in state.history)


def test_run_notes_the_window_without_claiming_ownership():
    state = State("x" * 400_000)
    owner = Agent(model=FakeModel(["a"], context_window=200_000), reporter=None, human=None)
    other = Agent(model=FakeModel(["b"], context_window=200_000), reporter=None, human=None)

    def only_other_thinks(agent, state):
        other.think(state)
        return state.answer

    # The Agent that ran does not become the owner if it does not think: other, which thought, is the owner.
    assert owner.copy(loop=only_other_thinks).run(state) == "b"
    assert state.context_used > 0
