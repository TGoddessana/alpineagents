"""Black-box e2e tests: grounded in the spec's sentences. They look at the spec, not the implementation.

Covers:
- Reporter: showing progress (on_* order and arguments, reporter=None)
- Human: asking a person and getting an answer (human=None -> NoHumanError, FakeHuman, queuing questions)
- Terminal: coordinating output and questions (output shape, limit warning)
- Test doubles (FakeModel, FakeHuman, tool_call)
- Complex example code (permission check, plan then execute, chat, coder and reviewer, tests)
- Running the first two overview examples (minus subagents/MCP/skills) end to end with FakeModel/FakeHuman

No test uses a real provider that goes over the network (FakeModel/FakeHuman only).
"""

from __future__ import annotations

import io
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from alpineagents import (
    Agent,
    ContextTooLongError,
    NoHumanError,
    Reply,
    Reporter,
    State,
    Terminal,
    Usage,
    compact_if_full,
    default_loop,
    loop,
    tool,
)
from alpineagents.testing import FakeHuman, FakeModel, tool_call
from alpineagents.types import Message, Request, TextBlock


def _byte_size(text: str) -> str:
    """UTF-8 byte size as the terminal shows it (``512B``, ``1.2KB``, ``3.4MB``). Computes expected values."""
    n = len(text.encode("utf-8"))
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


# ======================================================================
# Test doubles: FakeModel, FakeHuman, tool_call
# ("testing.py": ARCHITECTURE.md "Test doubles")
# ======================================================================


def test_tool_call_creates_unique_ids_each_time():
    """"tool_call(name, **args)`` -> ``id`` is ``"call_"`` + 12 hex digits (uuid4), newly made each time."""
    a = tool_call("read_file", path="main.py")
    b = tool_call("read_file", path="main.py")
    assert a.name == "read_file"
    assert a.args == {"path": "main.py"}
    assert re.fullmatch(r"call_[0-9a-f]{12}", a.id)
    assert a.id != b.id  # newly made each time


def test_fake_model_str_item_is_text_only_reply():
    """"str`` -> a text reply (no tool calls)"""
    fake = FakeModel(["The bug is on line 3"])
    reply = fake.respond(Request(None, (Message.user("Find the bug"),)))
    assert reply.text == "The bug is on line 3"
    assert reply.tool_calls == ()
    assert reply.stop_reason == "end_turn"


def test_fake_model_toolcall_item_is_tool_use_reply():
    """"ToolCall`` -> a reply with only that call"""
    call = tool_call("read_file", path="main.py")
    fake = FakeModel([call])
    reply = fake.respond(Request(None, (Message.user("Find the bug"),)))
    assert reply.tool_calls == (call,)
    assert reply.text == ""
    assert reply.stop_reason == "tool_use"


def test_fake_model_list_item_mixes_blocks_in_order():
    """"list`` / ``tuple`` (mixing ``str``, ``ToolCall``, ``RawBlock``) -> a reply with those blocks in order"""
    call = tool_call("read_file", path="main.py")
    fake = FakeModel([["Let me read the file.", call]])
    chunks: list[str] = []
    reply = fake.respond(Request(None, (Message.user("Find the bug"),)), on_text=chunks.append)
    assert [type(b).__name__ for b in reply.message.content] == ["TextBlock", "ToolCall"]
    assert reply.text == "Let me read the file."
    assert reply.tool_calls == (call,)
    assert chunks == ["Let me read the file."]  # calls on_text(text) once per text block


def test_fake_model_reply_item_passthrough():
    """"Reply`` -> as is"""
    custom = Reply(message=Message("assistant", (TextBlock("A fixed reply"),)), usage=Usage(), stop_reason="end_turn")
    fake = FakeModel([custom])
    reply = fake.respond(Request(None, (Message.user("Question"),)))
    assert reply is custom


def test_fake_model_exception_item_is_raised():
    """"BaseException`` instance -> raises that exception (for testing error paths)"""
    boom = TimeoutError("slow")
    fake = FakeModel([boom])
    with pytest.raises(TimeoutError) as exc_info:
        fake.respond(Request(None, (Message.user("Question"),)))
    assert exc_info.value is boom


def test_fake_model_callable_item_receives_request():
    """"callable -> calls ``item(request)`` and converts the result by the rules above"""
    fake = FakeModel([lambda request: f"Saw {len(request.messages)} messages"])
    request = Request(None, (Message.user("one"), Message("assistant", (TextBlock("two"),))))
    reply = fake.respond(request)
    assert reply.text == "Saw 2 messages"


def test_fake_model_records_requests_in_order():
    """"requests``: a list of the received Requests, in order (for checking)."""
    fake = FakeModel(["First reply", "Second reply"])
    r1 = Request(None, (Message.user("First question"),))
    r2 = Request(None, (Message.user("Second question"),))
    fake.respond(r1)
    fake.respond(r2)
    assert fake.requests == [r1, r2]


def test_fake_model_context_too_long_error_does_not_consume_item():
    """"If the estimated context exceeds ``context_window``, ``ContextTooLongError``"" (the item is not used)."""
    fake = FakeModel(["Never used"], context_window=1)
    long_request = Request(None, (Message.user("very long " * 50),))
    with pytest.raises(ContextTooLongError):
        fake.respond(long_request)
    assert fake.remaining == 1  # the item is not used


def test_fake_model_exhausted_raises_runtime_error():
    """"If no items are left, ``RuntimeError``("FakeModel: ran out of prepared replies (all N used)"...)"""
    fake = FakeModel(["Only one"])
    fake.respond(Request(None, (Message.user("First question"),)))
    with pytest.raises(RuntimeError, match=r"FakeModel: ran out of prepared replies \(all 1 used\)"):
        fake.respond(Request(None, (Message.user("Second question"),)))


def test_fake_human_returns_answers_in_order_and_records_questions():
    """"FakeHuman(answers)``: returns the answers in turn, converted by ``parse_answer``.

    ``questions``: the questions received, in order."""
    human = FakeHuman(["yes", "no"])
    state = State("Task")
    assert human.ask(state, "Run it?", returns=bool) is True
    assert human.ask(state, "Continue?", returns=bool) is False
    assert human.questions == ["Run it?", "Continue?"]
    assert human.remaining == 0


def test_fake_human_skips_invalid_answer_and_uses_next():
    """"If it does not fit (``ValueError``), the next answer is used, as if the person answered again."""
    human = FakeHuman(["dunno", "yes"])
    state = State("Task")
    assert human.ask(state, "Continue?", returns=bool) is True
    assert human.remaining == 0  # both answers were used


def test_fake_human_exhausted_raises_runtime_error():
    """"When answers run out, ``RuntimeError``("FakeHuman: ran out of prepared answers", with the last question)."""
    human = FakeHuman([])
    state = State("Task")
    with pytest.raises(RuntimeError, match="FakeHuman: ran out of prepared answers"):
        human.ask(state, "Any questions?", returns=str)


def test_fake_human_queues_concurrent_questions():
    """Human: "Questions are asked one at a time. When several places ask at once, Human queues them."

    Checks with several threads that FakeHuman is thread-safe (``questions``/``_index`` go out one at a time).
    """
    n = 50
    human = FakeHuman([str(i) for i in range(n)])
    state = State("Task")
    results: list[str] = []
    results_lock = threading.Lock()

    def worker() -> None:
        answer = human.ask(state, "Tell me a number", returns=str)
        with results_lock:
            results.append(answer)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # If they were queued, no answer goes out twice and none is missing.
    assert sorted(results, key=int) == [str(i) for i in range(n)]
    assert len(human.questions) == n
    assert human.remaining == 0


# ======================================================================
# Reporter: on_* order and arguments
# (ARCHITECTURE.md "Reporter notifications: who and when")
# ======================================================================


class RecordingReporter(Reporter):
    """Snapshots the values at the moment each notification is called.

    State is mutable, so looking at it later shows only the final values.
    """

    def __init__(self) -> None:
        self.events: list[tuple] = []

    def on_run_start(self, state):
        self.events.append(("on_run_start", state))

    def on_think_start(self, state):
        self.events.append(("on_think_start", state.turn))

    def on_text(self, state, chunk):
        self.events.append(("on_text", chunk))

    def on_think_end(self, state, reply):
        self.events.append(("on_think_end", reply))

    def on_tool_start(self, state, call):
        self.events.append(("on_tool_start", call.id, call.name))

    def on_tool_end(self, state, call, result, outcome):
        self.events.append(("on_tool_end", call.id, result, outcome.kind))

    def on_context_change(self, state, change):
        self.events.append(("on_context_change", change.kind, change.before_tokens, change.after_tokens))

    def on_run_end(self, state, error):
        self.events.append(("on_run_end", error))

    def names(self) -> list[str]:
        return [e[0] for e in self.events]


@tool
def _read_file_fixed(path: str) -> str:
    """A test file-reading tool that returns fixed content."""
    return "hello"


def test_on_run_start_before_loop_and_on_run_end_in_finally_on_success():
    """"on_run_start / on_run_end(state, error) | Agent.run | before the loop / finally``"""
    reporter = RecordingReporter()
    fake = FakeModel(["Done"])
    agent = Agent(model=fake, reporter=reporter, human=None)
    state = State("Task")
    agent.run(state)
    assert reporter.names()[0] == "on_run_start"
    assert reporter.events[0][1] is state
    assert reporter.names()[-1] == "on_run_end"
    assert reporter.events[-1][1] is None  # error=None on a normal finish


def test_on_run_end_called_with_error_when_run_raises():
    """"Even when an error occurs, on_run_end is always called in finally. error is that exception as is."""
    reporter = RecordingReporter()
    boom = RuntimeError("model failure")
    fake = FakeModel([boom])
    agent = Agent(model=fake, reporter=reporter, human=None)
    state = State("Task")
    with pytest.raises(RuntimeError):
        agent.run(state)
    assert reporter.names()[0] == "on_run_start"
    assert reporter.names()[-1] == "on_run_end"
    assert reporter.events[-1][1] is boom
    # Exception during think means rollback: on_think_end is not called (only "right after the reply is recorded")
    assert "on_think_end" not in reporter.names()
    # One error entry stays in history
    assert any(e.kind == "error" and e.error is boom for e in state.history)


def test_on_think_start_before_respond_and_turn_already_incremented():
    """"on_think_start(state) | Agent.think, Agent.ask | right before respond (for think, after _begin_think)``.

    So when think calls it, ``state.turn`` is already this turn's number.
    """
    reporter = RecordingReporter()
    fake = FakeModel(["Answer"])
    agent = Agent(model=fake, reporter=reporter, human=None)
    state = State("Task")
    assert state.turn == 0
    agent.think(state)
    think_start_events = [e for e in reporter.events if e[0] == "on_think_start"]
    assert think_start_events == [("on_think_start", 1)]  # turn 1 is already counted


def test_on_text_called_once_per_text_block_with_chunk():
    """"on_text(state, chunk) | per text chunk``"""
    reporter = RecordingReporter()
    fake = FakeModel(["Hello, nice to meet you"])
    agent = Agent(model=fake, reporter=reporter, human=None)
    state = State("Say hello")
    agent.think(state)
    text_events = [e for e in reporter.events if e[0] == "on_text"]
    assert text_events == [("on_text", "Hello, nice to meet you")]


def test_on_think_end_called_right_after_reply_recorded_with_reply_object():
    """"on_think_end(state, reply) | Agent.think, Agent.ask | right after the reply finishes and is recorded``"""
    reporter = RecordingReporter()
    fake = FakeModel(["Final answer"])
    agent = Agent(model=fake, reporter=reporter, human=None)
    state = State("Task")
    agent.think(state)
    order = reporter.names()
    assert order == ["on_think_start", "on_text", "on_think_end"]
    think_end = [e for e in reporter.events if e[0] == "on_think_end"][0]
    assert isinstance(think_end[1], Reply)
    assert think_end[1].text == "Final answer"
    # It must already be recorded too
    assert state.context[-1].text == "Final answer"


def test_on_tool_start_and_on_tool_end_wrap_execution_in_order():
    """"on_tool_start/on_tool_end | _runner.run_calls (main thread) |
    right before running / right after the result is recorded``"""
    reporter = RecordingReporter()
    call = tool_call("_read_file_fixed", path="main.py")
    fake = FakeModel([call, "There is a bug on line 3"])
    agent = Agent(model=fake, tools=[_read_file_fixed], reporter=reporter, human=None)
    state = State("Find the bug")
    agent.run(state)

    names = reporter.names()
    tool_start_i = names.index("on_tool_start")
    tool_end_i = names.index("on_tool_end")
    assert tool_start_i < tool_end_i
    # After think (turn 1), use_tools runs, and only then does the turn-2 think come
    assert names == [
        "on_run_start",
        "on_think_start",
        "on_think_end",
        "on_tool_start",
        "on_tool_end",
        "on_think_start",
        "on_text",
        "on_think_end",
        "on_run_end",
    ]
    start_event = reporter.events[tool_start_i]
    end_event = reporter.events[tool_end_i]
    assert start_event[1] == call.id
    assert start_event[2] == "_read_file_fixed"
    assert end_event[1] == call.id
    assert end_event[2] == "(done)" or end_event[2]  # the result string (same as what is sent to the model)
    assert end_event[2] == "hello"  # the string _read_file_fixed returned, as is
    assert end_event[3] == "done"


def test_on_tool_end_for_denied_call_has_no_matching_on_tool_start():
    """"on_tool_end (denied) | State.deny | right after recording (outside the lock)`` -- without ``on_tool_start``.

    (ARCHITECTURE.md "Reporter notifications: who and when": State.deny calls on_tool_end for a denied call,
    without on_tool_start.)
    """

    @tool
    def write_file(path: str, content: str) -> str:
        """Writes to a file."""
        return "written"

    @loop(until=State.is_answered, limit=5)
    def coding_with_permission(agent, state):
        agent.think(state)
        if state.wants_tools():
            for call in state.pending_calls:
                state.deny(call, "The user denied it")
            agent.use_tools(state)

    reporter = RecordingReporter()
    human = FakeHuman([])
    call = tool_call("write_file", path="a.txt", content="x")
    fake = FakeModel([call, "Gave up"])
    agent = Agent(
        model=fake, tools=[write_file], loop=coding_with_permission, reporter=reporter, human=human
    )
    state = State("Write the file")
    agent.run(state)

    assert "on_tool_start" not in reporter.names()  # denied, so the tool was not called
    tool_end_events = [e for e in reporter.events if e[0] == "on_tool_end"]
    assert tool_end_events == [("on_tool_end", call.id, "The user denied it", "denied")]
    denied = [e for e in state.history if e.kind == "denied"]
    assert len(denied) == 1 and denied[0].content == "The user denied it"


def test_on_context_change_called_after_compact_with_before_after_tokens():
    """"on_context_change(state, change) | State(_rollback, _replace_context, clear_tool_results) |
    right after recording (outside the lock)``"""
    reporter = RecordingReporter()
    fake = FakeModel(["Summary: read the files so far"])
    agent = Agent(model=fake, reporter=reporter, human=None)
    state = State("Keep going with the long task")
    agent.compact(state)

    change_events = [e for e in reporter.events if e[0] == "on_context_change"]
    assert len(change_events) == 1
    _, kind, before, after = change_events[0]
    assert kind == "compact"
    assert isinstance(before, int) and isinstance(after, int)
    assert state.context[-1].text.startswith("[notice] Summary so far:")


def test_reporter_none_is_silent_and_does_not_change_behavior():
    """"With reporter=None nothing is shown.", "The agent behaves the same without a Reporter."""
    call = tool_call("_read_file_fixed", path="main.py")

    reporter = RecordingReporter()
    agent_with_reporter = Agent(
        model=FakeModel([call, "Answer"]), tools=[_read_file_fixed], reporter=reporter, human=None
    )
    state_with_reporter = State("Find the bug")
    agent_with_reporter.run(state_with_reporter)

    agent_silent = Agent(
        model=FakeModel([call, "Answer"]), tools=[_read_file_fixed], reporter=None, human=None
    )
    state_silent = State("Find the bug")
    agent_silent.run(state_silent)  # must behave the same, with no exception

    assert state_silent.answer == state_with_reporter.answer == "Answer"
    assert state_silent.turn == state_with_reporter.turn
    assert state_silent.stopped_by == state_with_reporter.stopped_by == "is_answered"
    assert len(reporter.events) > 0  # control: with a reporter, notifications really pile up


# ======================================================================
# Human: human=None -> NoHumanError, check_returns
# ======================================================================


def test_ask_human_with_human_none_raises_no_human_error():
    """"With human=None ... ask_human raises alpineagents.NoHumanError."""
    agent = Agent(model=FakeModel([]), human=None, reporter=None)
    state = State("Task")
    with pytest.raises(NoHumanError):
        agent.ask_human(state, "Run it?")


def test_ask_human_unsupported_returns_type_raises_type_error():
    """"ask_human(returns=...) with an unsupported type`` -> ``TypeError`` (returns is str/bool/Literal only)."""
    agent = Agent(model=FakeModel([]), human=FakeHuman(["any answer"]), reporter=None)
    state = State("Task")
    with pytest.raises(TypeError):
        agent.ask_human(state, "How many?", returns=int)


def test_ask_human_records_question_and_answer_in_history_not_context():
    """"The question and answer stay in history as human and do not go into the context."""
    agent = Agent(model=FakeModel([]), human=FakeHuman(["yes"]), reporter=None)
    state = State("Task")
    before_context = state.context
    value = agent.ask_human(state, "Continue?", returns=bool)
    assert value is True
    assert state.context == before_context  # the context does not change
    human_entries = [e for e in state.history if e.kind == "human"]
    assert len(human_entries) == 1
    assert human_entries[0].content.question == "Continue?"
    assert human_entries[0].content.answer is True


# ======================================================================
# Terminal: output shape and limit warning
# (ARCHITECTURE.md "Terminal: coordinating output and questions")
# ======================================================================


def test_terminal_end_to_end_output_shape_for_tool_then_answer():
    """Reproduces the example output shape (thinking/tool/done) with a real Agent.run."""
    out = io.StringIO()
    terminal = Terminal(output=out)
    call = tool_call("_read_file_fixed", path="main.py")
    fake = FakeModel([call, "The bug is on line 3"])
    agent = Agent(model=fake, tools=[_read_file_fixed], reporter=terminal, human=None)
    state = State("Find the bug")
    agent.run(state)

    expected = (
        "[turn 1] thinking\n"
        '  tool _read_file_fixed(path="main.py")\n'
        f"  done {_byte_size('hello')}\n"
        "[turn 2] thinking\n"
        "The bug is on line 3\n"
        "done: is_answered (2 turns)\n"
    )
    assert out.getvalue() == expected


def test_terminal_shows_limit_warning_line_end_to_end():
    """"The default Terminal() shows a run stopped by limit as a warning line
    (``done: reached limit(50), the task may be unfinished``)``"""

    @tool
    def noop() -> None:
        """Does nothing."""
        return None

    @loop(until=State.is_answered, limit=1)
    def one_turn_loop(agent, state):
        agent.think(state)
        if state.wants_tools():
            agent.use_tools(state)

    out = io.StringIO()
    terminal = Terminal(output=out)
    fake = FakeModel([tool_call("noop")])  # only calls a tool, never answers -> is_answered stays false
    agent = Agent(model=fake, tools=[noop], loop=one_turn_loop, reporter=terminal, human=None)
    state = State("A task that never ends")
    agent.run(state)

    assert state.stopped_by == "limit"
    assert "done: reached limit(1), the task may be unfinished (1 turn)\n" in out.getvalue()
    assert out.getvalue().splitlines()[-1] == "done: reached limit(1), the task may be unfinished (1 turn)"


# ======================================================================
# Complex example code
# ======================================================================


DANGEROUS = {"write_file"}
_write_calls: list[tuple[str, str]] = []


@tool
def write_file(path: str, content: str) -> str:
    """Writes to a file."""
    _write_calls.append((path, content))
    return "written"


def ask_permission(agent: Agent, state: State) -> None:
    """The "permission check" block as is. The 'always allow' list lives on ``state.root``."""
    root = state.root
    for call in state.pending_calls:
        if call.name not in DANGEROUS or call.name in root.data.get("always_allow", []):
            continue
        answer = agent.ask_human(
            state, f"Run {call.name}({call.args})?", returns=Literal["yes", "no", "always"]
        )
        if answer == "always":
            with root.lock:
                root.data.setdefault("always_allow", []).append(call.name)
        elif answer == "no":
            state.deny(call, "The user denied it")


@loop(until=State.is_answered, limit=10)
def coding(agent: Agent, state: State):
    """The ``coding`` loop from the overview. Several "complex example code" examples reuse this loop."""
    agent.think(state)
    if state.wants_tools():
        ask_permission(agent, state)
        agent.use_tools(state)


def test_permission_block_denies_dangerous_call():
    """"A denied call has a result, so use_tools skips it."""
    _write_calls.clear()
    call = tool_call("write_file", path="a.txt", content="x")
    fake = FakeModel([call, "Gave up"])
    human = FakeHuman(["no"])
    agent = Agent(model=fake, tools=[write_file], loop=coding, human=human, reporter=None)
    state = State("Write the file")
    agent.run(state)

    assert _write_calls == []  # the tool was not actually called
    assert human.questions == ["Run write_file({'path': 'a.txt', 'content': 'x'})?"]
    denied = [e for e in state.history if e.kind == "denied"]
    assert len(denied) == 1


def test_permission_block_always_allow_skips_asking_again_same_turn():
    """"The 'always allow' list lives on the top-level State (state.root).

    The next call to the same tool in the same turn is not asked again (parallel calls see the same list too).
    """
    _write_calls.clear()
    calls = [
        tool_call("write_file", path="a.txt", content="x"),
        tool_call("write_file", path="b.txt", content="y"),
    ]
    fake = FakeModel([calls, "All done"])  # request two calls together in one turn
    human = FakeHuman(["always"])
    agent = Agent(model=fake, tools=[write_file], loop=coding, human=human, reporter=None)
    state = State("Write two files")
    agent.run(state)

    assert sorted(_write_calls) == [("a.txt", "x"), ("b.txt", "y")]
    assert len(human.questions) == 1  # the second call hit always_allow and was not asked
    assert state.root.data.get("always_allow") == ["write_file"]


def test_plan_then_execute_runs_each_step_with_coding_loop():
    """"Plan then execute": calls loops in turn. limit is counted anew for each step."""

    @dataclass
    class Plan:
        steps: list[str]

    def plan_then_execute(agent: Agent, state: State):
        plan = agent.ask(state, "Split the task into 3-7 steps", returns=Plan)
        for step in plan.steps:
            state.add_user_message(f"Next step: {step}")
            default_loop(agent, state)
            if state.is_finished():
                break
        return state.answer

    fake = FakeModel(
        [
            '{"steps": ["List the files", "Fix the bug"]}',
            "Step 1 done",
            "Step 2 done",
        ]
    )
    agent = Agent(model=fake, loop=plan_then_execute, reporter=None, human=None)
    state = State("Fix the bug")
    answer = agent.run(state)

    assert answer == "Step 2 done"
    assert state.answer == "Step 2 done"
    assert state.turn == 2  # think happens only in the two steps (ask does not advance the turn)
    ask_entries = [e for e in state.history if e.kind == "ask"]
    assert len(ask_entries) == 1
    assert ask_entries[0].content.answer.steps == ["List the files", "Fix the bug"]


def test_chat_loop_answers_then_quits_on_command():
    """"Chat": the outer loop is the human's turn, the inner loop is the agent's turn."""

    @loop(until=State.is_finished, limit=1000)
    def chat(agent: Agent, state: State):
        default_loop(agent, state)  # answer the current message
        text = agent.ask_human(state, ">")
        if text == "quit":
            state.finish()
            return
        state.add_user_message(text)

    fake = FakeModel(["Hello! How can I help?", "Yes, there is"])
    human = FakeHuman(["One more question?", "quit"])
    agent = Agent(model=fake, loop=chat, human=human, reporter=None)
    state = State("Hi")
    agent.run(state)

    assert state.stopped_by == "finish"
    assert state.answer == "Yes, there is"
    assert human.questions == [">", ">"]
    assert len([e for e in state.history if e.kind == "human"]) == 2


def test_coder_and_reviewer_pair_finishes_once_approved():
    """"Coder and reviewer": the coder is the State's only owner.

    The reviewer only reads the same context with ask."""

    @dataclass
    class Review:
        approved: bool
        comments: str

    reviewer_fake = FakeModel(
        [
            '{"approved": false, "comments": "Add tests"}',
            '{"approved": true, "comments": "Looks good"}',
        ]
    )
    reviewer = Agent(model=reviewer_fake, system="You are a meticulous code reviewer", reporter=None, human=None)

    @loop(until=State.is_finished, limit=5)
    def pair(agent: Agent, state: State):
        default_loop(agent, state)
        review = reviewer.ask(state, "Review the changes so far", returns=Review)
        if review.approved:
            state.finish()
        else:
            state.add_user_message(f"Reviewer comments: {review.comments}")

    coder_fake = FakeModel(["First fix done", "Tests added"])
    coder = Agent(model=coder_fake, loop=pair, reporter=None, human=None)
    state = State("Implement the feature")
    coder.run(state)

    assert state.stopped_by == "finish"
    assert state.answer == "Tests added"
    ask_entries = [e for e in state.history if e.kind == "ask"]
    assert len(ask_entries) == 2
    assert ask_entries[0].content.answer.approved is False
    assert ask_entries[1].content.answer.approved is True
    # The reviewer comments went into the coder's conversation as a user message
    # (the contexts do not mix; only the coder's State has it)
    assert any("Reviewer comments: Add tests" in m.text for m in state.context)


def test_reads_file_then_answers():
    """The test example from "complex example code", as is."""

    @tool
    def read_file(path: str) -> str:
        """Reads a file's contents."""
        return "def main():\n    pass\n"

    base_agent = Agent(model=FakeModel([]), tools=[read_file], reporter=None, human=None)
    fake = FakeModel(
        [
            tool_call("read_file", path="main.py"),
            "The bug is on line 3",
        ]
    )
    state = State("Find the bug")
    base_agent.copy(model=fake, reporter=None).run(state)

    assert state.answer == "The bug is on line 3"
    assert state.turn == 2
    assert state.stopped_by == "is_answered"


# ======================================================================
# The first two overview examples (only subagents/MCP/skills left out)
# ======================================================================


def test_overview_example_one_smallest_agent_end_to_end():
    """"The smallest agent": ``Agent(model=..., tools=[web_search])``, ``agent.run("...")``."""

    @tool
    def web_search(query: str) -> str:
        """Searches the web."""
        return "Already exists: alpineagents is an agent framework registered on PyPI"

    fake = FakeModel(
        [
            tool_call("web_search", query="alpineagents pypi"),
            "The package already exists. Checked on PyPI.",
        ]
    )
    agent = Agent(model=fake, tools=[web_search], reporter=None, human=None)
    answer = agent.run("Find out whether a Python package called alpineagents already exists")

    assert answer == "The package already exists. Checked on PyPI."
    assert fake.remaining == 0


def test_overview_example_two_full_loop_with_filesystem(tmp_path):
    """"Everything in use" example, run end to end without the subagent (researcher)/MCP/skills.

    The ``FileSystem`` tool object, a custom ``coding`` loop (with ``compact_if_full``),
    ``state.stopped_by`` and ``state.usage.cost`` are all checked as is.
    """
    (tmp_path / "main.py").write_text("def main():\n    return None  # bug: no null check\n")

    class FileSystem:
        def __init__(self, root: str = "."):
            self.root = Path(root)

        @tool
        def read_file(self, path: str) -> str:
            """Reads a file's contents."""
            file = self.root / path
            if not file.exists():
                return f"No such file: {path}"
            return file.read_text()

    @loop(until=State.is_answered, limit=50)
    def full_coding_loop(agent: Agent, state: State):
        compact_if_full(agent, state)
        agent.think(state)
        if state.wants_tools():
            agent.use_tools(state)

    fake = FakeModel(
        [
            tool_call("read_file", path="main.py"),
            "Found the bug: missing null check",
        ]
    )
    agent = Agent(
        model=fake,
        system="You are a coding assistant",
        tools=[FileSystem(root=str(tmp_path))],
        loop=full_coding_loop,
        reporter=None,
        human=None,
    )
    state = State("Find the bug in this repo")
    answer = agent.run(state)

    assert answer == "Found the bug: missing null check"
    assert state.stopped_by == "is_answered"
    # An unknown price is None (unknown and 0 are kept apart)
    assert state.usage.cost is None
