"""Regression tests for defects found in review in the Reporter/Human/Terminal group (terminal.py). No network."""

from __future__ import annotations

import _thread
import io
import threading
import time

import pytest

from alpineagents import Agent, State, tool
from alpineagents.terminal import Terminal
from alpineagents.testing import FakeModel, tool_call


# ======================================================================
# When Ctrl+C arrives, an abandoned question must not block run() from finishing
# ======================================================================


class _DangerousTools:
    """A class-based tool that holds the agent (same as the reported scenario)."""

    def __init__(self) -> None:
        self.agent: Agent | None = None
        self.executed = False

    @tool
    def dangerous_tool(self, state: State) -> str:
        """Checks with the human, then does something dangerous."""
        assert self.agent is not None
        # Blocks inside Terminal.ask (it holds the question lock, but must release the output lock).
        ok = self.agent.ask_human(state, "Do the dangerous thing?", bool)
        if ok:
            self.executed = True
        return "done"


def test_ctrl_c_during_pending_question_does_not_hang_run():
    """After Ctrl+C, an abandoned question in a tool thread must not block run() from finishing (on_run_end).

    If Terminal.ask held the output lock until it got an answer, the on_run_end -> _write_line call made by
    Agent.run's finally would wait on the same lock and hang too (the reported defect).
    """
    blocked = threading.Event()  # set once ask() is inside input() holding the lock
    release_answer = threading.Event()  # simulates the human's (late) answer

    def fake_input(prompt: str) -> str:
        blocked.set()
        release_answer.wait(timeout=10)
        return "yes"

    terminal = Terminal(input=fake_input, show_text=False)

    tools = _DangerousTools()
    agent = Agent(
        model=FakeModel([tool_call("dangerous_tool")]),
        tools=[tools],
        reporter=terminal,
        human=terminal,
    )
    tools.agent = agent

    def interrupt_when_blocked() -> None:
        assert blocked.wait(timeout=5), "the tool thread never reached input()"
        time.sleep(0.1)
        _thread.interrupt_main()

    def answer_late() -> None:
        assert blocked.wait(timeout=5)
        time.sleep(2.0)
        release_answer.set()

    threading.Thread(target=interrupt_when_blocked, daemon=True).start()
    threading.Thread(target=answer_late, daemon=True).start()

    start = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            agent.run("Do the task")
    finally:
        release_answer.set()  # cleanup: release the tool thread even if the test fails
    elapsed = time.monotonic() - start

    # Sync tools/questions must be closed without waiting (spec): run() must finish well before the late
    # answer (2 seconds later) instead of waiting for it.
    assert elapsed < 1.0, (
        f"agent.run() took {elapsed:.2f}s to raise KeyboardInterrupt after Ctrl+C -- "
        "on_run_end seems to have waited for the output lock held by the abandoned question"
    )


def test_ctrl_c_does_not_let_abandoned_tool_finish_its_action_first():
    """The abandoned tool's 'dangerous action' must not run before run() finishes."""
    blocked = threading.Event()
    release_answer = threading.Event()

    def fake_input(prompt: str) -> str:
        blocked.set()
        release_answer.wait(timeout=10)
        return "yes"

    terminal = Terminal(input=fake_input, show_text=False)
    tools = _DangerousTools()
    agent = Agent(
        model=FakeModel([tool_call("dangerous_tool")]),
        tools=[tools],
        reporter=terminal,
        human=terminal,
    )
    tools.agent = agent

    def interrupt_when_blocked() -> None:
        assert blocked.wait(timeout=5)
        time.sleep(0.1)
        _thread.interrupt_main()

    threading.Thread(target=interrupt_when_blocked, daemon=True).start()

    try:
        with pytest.raises(KeyboardInterrupt):
            agent.run("Do the task")
    finally:
        release_answer.set()

    # When run() finishes first (without waiting for the abandoned question's answer), the dangerous action
    # must not have happened yet.
    assert tools.executed is False


# ======================================================================
# The compaction line must come after the new turn's header
# ======================================================================

_LONG_REPLY = "Long filler text. " * 35


def test_compaction_line_appears_after_new_turn_header():
    @tool
    def echo(text: str) -> str:
        """Echo tool for tests."""
        return _LONG_REPLY

    # Turn 1: tool call (echo returns a large result, pushing context_used above 0.6)
    # Turn 2: the final answer, after compaction steps in
    # Sizes: the context after turn 1 (~267 tokens) must be above 0.6 * context_window, and the compaction request
    # (that context + COMPACT_PROMPT, ~83 tokens) must still fit in context_window.
    fake = FakeModel(
        [tool_call("echo", text="hi"), "Here is the summary", "Final answer"],
        context_window=400,
    )
    buf = io.StringIO()
    term = Terminal(output=buf, input=lambda p: "")
    agent = Agent(model=fake, tools=[echo], reporter=term, human=term)
    state = State("Task")

    answer = agent.run(state)
    # Confirms compaction really happened (otherwise "Here is the summary" would be the answer).
    assert answer == "Final answer"

    lines = buf.getvalue().splitlines()
    think2_idx = lines.index("[turn 2] thinking")
    compact_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("context compacted"))

    assert compact_idx > think2_idx, (
        "As in the spec's output sample, the 'context compacted' line must come after the new turn header "
        f"'[turn 2] thinking', but it came first at line {compact_idx} (header at line {think2_idx}): {lines!r}"
    )
    # It must be the line right after the header (no other line from the previous turn in between).
    assert compact_idx == think2_idx + 1


# ======================================================================
# Even when output= is not sys.stdout, the question and the re-ask hint must go to the same stream
# ======================================================================


class _StubState:
    def __init__(self, *, turn=1, depth=0):
        self.turn = turn
        self.depth = depth


def test_default_input_writes_prompt_to_configured_output_not_real_stdout(monkeypatch, capsys):
    """Without ``input=`` (the real ``builtins.input``), the question text must also go to ``output=``.

    Before, ``builtins.input`` wrote the prompt straight to the real ``sys.stdout``, so when ``output=`` was
    another stream (a log, say) the question went to the screen and the re-ask hint went to the log.
    """
    log = io.StringIO()
    terminal = Terminal(output=log)  # input=None -> the real builtins.input
    state = _StubState(turn=1)

    monkeypatch.setattr("sys.stdin", io.StringIO("not sure\nyes\n"))

    result = terminal.ask(state, "Continue?", returns=bool)
    assert result is True

    captured = capsys.readouterr()
    log_value = log.getvalue()

    # The question and the re-ask hint must both be in the same (configured) output stream.
    assert "Continue? (yes/no)" in log_value
    assert "Answer 'yes' or 'no'" in log_value

    # Nothing leaks to the real stdout.
    assert "Continue? (yes/no)" not in captured.out


def test_next_reporter_line_is_not_glued_to_prompt(monkeypatch, capsys):
    """After asking with the default Terminal() (output=None -> sys.stdout), the next line must not stick to the
    prompt (with a non-tty stdin, no newline is echoed when the person types an answer)."""
    terminal = Terminal()  # output=None -> sys.stdout, input=None -> builtins.input
    state = _StubState(turn=2)

    monkeypatch.setattr("sys.stdin", io.StringIO("yes\n"))
    terminal.ask(state, "Continue?", returns=bool)

    terminal.on_think_start(state)

    out = capsys.readouterr().out
    lines = out.split("\n")
    assert not any("Continue?" in line and "thinking" in line for line in lines), (
        f"the prompt and the next output line are stuck on one line: {out!r}"
    )


class _TtyStringIO(io.StringIO):
    def isatty(self):
        return True


def test_tty_answer_does_not_leave_a_blank_line(monkeypatch):
    """When answering in a terminal, the Enter newline is already on screen; another newline from Terminal
    would leave a blank line."""
    out = _TtyStringIO()
    terminal = Terminal(output=out)
    state = _StubState(turn=2)
    monkeypatch.setattr("sys.stdin", _TtyStringIO("yes\n"))

    assert terminal.ask(state, "Continue?", returns=bool) is True
    terminal.on_think_start(state)

    # The screen shows "Continue? (yes/no) yes⏎" (echo), and the turn header comes right on the next line.
    assert out.getvalue() == "Continue? (yes/no) [turn 2] thinking\n"


def test_empty_string_answer_is_asked_again():
    """An empty answer for ``returns=str`` is asked again, so a chat loop (``text = agent.ask_human(state, ">")``
    then ``add_user_message(text)``) does not stop on a single Enter."""
    answers = iter(["", "   ", "hello"])
    out = io.StringIO()
    terminal = Terminal(output=out, input=lambda prompt: next(answers))
    assert terminal.ask(_StubState(), ">") == "hello"
    assert "Empty answers are not accepted" in out.getvalue()


def test_fake_human_skips_empty_string_answer():
    from alpineagents.testing import FakeHuman

    human = FakeHuman(["", "quit"])
    assert human.ask(_StubState(), ">") == "quit"
