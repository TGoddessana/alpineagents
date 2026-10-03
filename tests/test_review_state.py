"""Regression tests for State defects found in review (state.py), ported to the 0.5 API."""

from __future__ import annotations

import json
import sys
import threading

import pytest

from alpineagents.agent import Agent
from alpineagents import waiting_for_user
from alpineagents.models.base import Model
from alpineagents.state import State
from alpineagents.testing import FakeModel
from alpineagents.types import Message, Reply, Request, TextBlock, ToolCall, ToolOutcomeKind, Usage

MODEL = "fake/fake"


def task(text: str = "task", **kwargs) -> State:
    return State(messages=[Message.user(text)], **kwargs)


def _reply_with_call(call: ToolCall) -> Reply:
    return Reply(message=Message("assistant", (call,)), usage=Usage())


def _text_reply(text: str) -> Reply:
    return Reply(message=Message("assistant", (TextBlock(text),)), usage=Usage())


# ---------------------------------------------------------------- a late result attaches to its call, not to its id


def test_late_result_must_not_attach_to_new_calls_reusing_the_same_id():
    state = task()

    # Turn 1: bash(cmd="1.0"), id "call_0". Closed by an interrupt (Ctrl+C), but the worker thread keeps running.
    call_turn1 = ToolCall(name="bash", args={"cmd": "1.0"}, id="call_0")
    state._begin_think(MODEL)
    state._record_reply(_reply_with_call(call_turn1))
    state._tool_result(call_turn1, "(interrupted by user)", ToolOutcomeKind.INTERRUPTED)

    # Turn 2: the server does not give ids, so the first call is "call_0" again.
    call_turn2 = ToolCall(name="bash", args={"cmd": "2.0"}, id="call_0")
    state._begin_think(MODEL)
    state._record_reply(_reply_with_call(call_turn2))
    assert state.pending_calls == (call_turn2,)

    # Turn 1's worker thread finishes now.
    state._late_result(call_turn1, "done 1.0", ToolOutcomeKind.DONE)

    assert state.pending_calls == (call_turn2,), "turn 2's call must still be waiting for its own result"
    late = [e for e in state.history if e.kind == "tool_result" and e.late]
    assert len(late) == 1
    assert late[0].call is call_turn1 and late[0].content == "done 1.0"

    state._tool_result(call_turn2, "done 2.0", ToolOutcomeKind.DONE)
    normal = [e.content for e in state.history if e.kind == "tool_result" and not e.late]
    assert "done 2.0" in normal
    assert "done 1.0" not in normal

    # The late result enters the messages as a notice right before the next think.
    state._begin_think(MODEL)
    assert state.messages[-1].text == '[notice] interrupted bash(cmd="1.0") finished later: done 1.0'


def test_late_result_for_other_tool_with_same_id_is_not_misattributed():
    state = task()
    make = ToolCall(name="bash", args={"cmd": "make"}, id="call_0")
    state._begin_think(MODEL)
    state._record_reply(_reply_with_call(make))
    state._tool_result(make, "(interrupted by user)", ToolOutcomeKind.INTERRUPTED)

    read = ToolCall(name="read_file", args={"path": "a"}, id="call_0")
    state._begin_think(MODEL)
    state._record_reply(_reply_with_call(read))

    state._late_result(make, "exit 0", ToolOutcomeKind.DONE)

    assert state.pending_calls == (read,)
    entry = state.history[-1]
    assert entry.kind == "tool_result" and entry.late and entry.call is make


def test_late_result_for_the_same_pending_call_object_is_a_normal_result():
    # If the loop caught the interrupt and the call is still pending, it is recorded as a normal result.
    state = task()
    call = ToolCall(name="bash", args={"cmd": "x"}, id="call_0")
    state._begin_think(MODEL)
    state._record_reply(_reply_with_call(call))

    state._late_result(call, "ok", ToolOutcomeKind.DONE)

    assert state.pending_calls == ()
    entry = state.history[-1]
    assert entry.kind == "tool_result" and not entry.late and entry.content == "ok"
    assert state.messages[-1].content[0].content == "ok"


def test_late_results_replay_the_same_way():
    # The late notice is folded from history, so a rebuilt State has it too.
    state = task()
    call = ToolCall(name="bash", args={"cmd": "1.0"}, id="call_0")
    state._begin_think(MODEL)
    state._record_reply(_reply_with_call(call))
    state._tool_result(call, "(interrupted by user)", ToolOutcomeKind.INTERRUPTED)
    state._late_result(call, "done 1.0", ToolOutcomeKind.DONE)
    state._begin_think(MODEL)

    rebuilt = State(history=state.history)
    assert rebuilt.messages == state.messages
    assert rebuilt.snapshot() == state.snapshot()


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
    state = task()

    thread, errors = _run_in_thread(lambda: agent.think(state))
    assert model.entered.wait(timeout=5)
    state.add_message(Message.user("NEW QUESTION"))
    # It is in the history right away, and the State is waiting on the model (the request is the entry before).
    assert state.history[-1].kind == "user"
    assert state.history[-2].kind == "model_request"
    assert [m.text for m in state.messages] == ["task"]
    model.release.set()
    thread.join(timeout=5)
    assert not errors

    # The model did not see NEW QUESTION.
    assert [m.text for m in model.requests[0].messages] == ["task"]
    assert [(m.role, m.text) for m in state.messages] == [
        ("user", "task"),
        ("assistant", "first answer"),
        ("user", "NEW QUESTION"),
    ]
    assert not waiting_for_user(state)


def test_default_until_is_false_when_a_message_was_deferred_during_think():
    state = task()
    state._begin_think(MODEL)
    state._record_reply(_text_reply("a"))
    assert waiting_for_user(state)
    state._begin_think(MODEL)
    state.add_message(Message.user("more"))
    state._record_reply(_text_reply("b"))
    # "more" arrived while the model answered "b", so it comes after it and the loop must go on.
    assert [m.text for m in state.messages][-2:] == ["b", "more"]
    assert not waiting_for_user(state)


def test_message_added_during_think_with_tool_calls_goes_after_the_results():
    state = task()
    call = ToolCall(name="bash", args={"cmd": "x"}, id="c1")
    state._begin_think(MODEL)
    state.add_message(Message.notice("mid-think"))
    state._record_reply(_reply_with_call(call))
    assert state.messages[-1].role == "assistant"
    state._tool_result(call, "out", ToolOutcomeKind.DONE)
    assert [m.text for m in state.messages][-1] == "[notice] mid-think"
    assert state.messages[-2].content[0].content == "out"


def test_message_added_during_failed_think_survives_the_rollback():
    state = task()
    state._begin_think(MODEL)
    state.add_message(Message.user("typed while thinking"))
    state._record_error(RuntimeError("boom"))
    assert [m.text for m in state.messages] == ["task", "typed while thinking"]
    assert state.turn == 0
    # After the rollback it goes into the messages right away (not deferred).
    state.add_message(Message.user("after"))
    assert state.messages[-1].text == "after"


def test_user_message_added_during_compact_is_not_dropped():
    model = SlowModel(text="the summary")
    agent = Agent(model=model, reporter=None, human=None)
    state = task()
    state.add_message(Message.user("q1"))

    thread, errors = _run_in_thread(lambda: agent.compact(state))
    assert model.entered.wait(timeout=5)
    state.add_message(Message.user("q3 typed during compaction"))
    model.release.set()
    thread.join(timeout=5)
    assert not errors

    texts = [m.text for m in state.messages]
    assert texts[0] == "task"
    assert "the summary" in texts[1]
    assert texts[2:] == ["q3 typed during compaction"]
    # The entry says how many trailing messages it kept, and a rebuilt State agrees.
    assert state.history[-1].content.kept == 1
    assert State(history=state.history).messages == state.messages


def test_compact_with_no_concurrent_messages_leaves_task_and_summary_only():
    agent = Agent(model=FakeModel(["summary"]), reporter=None, human=None)
    state = task()
    state.add_message(Message.user("q1"))
    agent.compact(state)
    assert len(state.messages) == 2
    assert state.history[-1].content.kept == 0
    # A message added after compaction finishes goes in only once.
    state.add_message(Message.user("q2"))
    assert [m.text for m in state.messages][2:] == ["q2"]


def test_state_compact_ignores_compact_tracking():
    state = task()
    state._begin_compact()  # a compact still waiting on the model
    state.add_message(Message.user("q1"))
    state.compact("summary")
    assert len(state.messages) == 2
    assert state.history[-1].content.kept == 0
    state._end_compact()


def test_ensure_no_pending_does_not_start_compact_tracking():
    state = task()
    state._ensure_no_pending("compact")
    assert state._compacting is None


def test_failed_compact_ends_tracking_right_away():
    agent = Agent(model=FakeModel([RuntimeError("provider down")]), reporter=None, human=None)
    state = task()
    with pytest.raises(RuntimeError):
        agent.compact(state)
    assert state._compacting is None
    state.add_message(Message.user("q1"))
    assert [m.text for m in state.messages] == ["task", "q1"]


# ---------------------------------------------------------------- empty text is rejected early


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_run_rejects_an_empty_prompt(text):
    # agent.run("") used to fail in State(""); the check is now at the start of run.
    agent = Agent(model=FakeModel(["x"]), reporter=None, human=None)
    with pytest.raises(ValueError, match="got an empty prompt"):
        agent.run(text)


@pytest.mark.parametrize("text", ["", "   "])
def test_add_message_rejects_empty_text(text):
    state = task("do something")
    with pytest.raises(ValueError, match="Fix:"):
        state.add_message(Message.user(text))
    assert len(state.history) == 1  # the import
    assert len(state.messages) == 1


def test_add_message_rejects_an_empty_notice():
    state = task("do something")
    with pytest.raises(ValueError):
        state.add_message(Message.notice(" "))


def test_compact_rejects_empty_text():
    state = task("do something")
    with pytest.raises(ValueError):
        state.compact("")
    with pytest.raises(ValueError):
        state.compact("  ")
    with pytest.raises(TypeError):
        state.compact(None)  # type: ignore[arg-type]
    assert len(state.history) == 1


# ---------------------------------------------------------------- single reads of state.extra_data


def test_dict_read_of_extra_data_is_atomic_under_concurrent_edits():
    # Each edit sets two keys together; a reader that gets a torn value would see different numbers.
    state = task(extra_data={"a": 0, "b": 0})
    for k in range(20):
        with state.edit_extra_data() as data:
            data[f"k{k}"] = k

    stop = threading.Event()
    torn: list[dict] = []

    def mutator():
        i = 0
        while not stop.is_set():
            i += 1
            with state.edit_extra_data() as data:
                data["a"] = i
                data["b"] = i
                data.pop(f"k{i % 20}", None)
                data[f"k{i % 20}"] = i

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    threads = [threading.Thread(target=mutator, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    try:
        for _ in range(2000):
            copy = dict(state.extra_data)
            {**state.extra_data}
            {} | state.extra_data
            state.extra_data | {}
            list(state.extra_data)
            if copy["a"] != copy["b"]:
                torn.append(copy)
    finally:
        stop.set()
        for t in threads:
            t.join()
        sys.setswitchinterval(old)
    assert torn == []
    assert State(history=state.history).snapshot() == state.snapshot()


def test_extra_data_iterates_and_copies_like_a_dict():
    state = task(extra_data={"a": 1, "b": 2})
    assert list(state.extra_data) == ["a", "b"]
    assert dict(state.extra_data) == {"a": 1, "b": 2}
    assert type(dict(state.extra_data)) is dict
    assert {**state.extra_data, "c": 3} == {"a": 1, "b": 2, "c": 3}
    assert json.dumps(state.extra_data) == '{"a": 1, "b": 2}'
    assert isinstance(state.extra_data, dict)


# ---------------------------------------------------------------- context size on the first turn


def test_context_used_is_known_before_the_first_think():
    state = task("x" * 600_000)
    agent = Agent(model=FakeModel(["ok"], context_window=200_000), reporter=None, human=None)
    assert agent.context_used(state) > 1.0
    assert agent.context_tokens(state) > 200_000


def test_compact_if_full_can_fire_on_turn_one():
    # A starting task above 60% of the context window but smaller than the window: it must be compacted before turn 1.
    state = task("x" * 400_000)
    fake = FakeModel(["summary", "ok"], context_window=200_000)
    agent = Agent(model=fake, reporter=None, human=None)
    assert agent.run(state) == "ok"
    assert len(fake.requests) == 2
    assert any(e.kind == "context_change" and e.content.kind == "compact" for e in state.history)


def test_run_with_a_loop_that_thinks_with_another_agent_uses_that_agents_answer():
    # 0.4's owner rule is gone: the Agent that ran the State and the one that thought may differ.
    state = task("x" * 400_000)
    runner = Agent(model=FakeModel(["a"], context_window=200_000), reporter=None, human=None)
    other = Agent(model=FakeModel(["b"], context_window=200_000), reporter=None, human=None)

    def only_other_thinks(agent, state):
        other.think(state)
        return state.answer

    assert runner.copy(loop=only_other_thinks).run(state) == "b"
    assert runner.context_used(state) > 0
    assert other.context_used(state) > 0
    assert runner.model.requests == []  # the runner's own model was never asked
