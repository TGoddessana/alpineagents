"""Terminal tests. State is still being built in another unit, so these use a small stub object with only the
attributes needed (turn, depth, usage, stopped_by, stopped_limit).
"""

from __future__ import annotations

import io

import pytest

from alpineagents.terminal import Terminal, default_terminal
from alpineagents.testing import tool_call
from alpineagents.types import ContextChange, ToolOutcome, Usage


class StubState:
    def __init__(self, *, turn=1, depth=0, usage=None, stopped_by=None, stopped_limit=None):
        self.turn = turn
        self.depth = depth
        self.usage = usage if usage is not None else Usage()
        self.stopped_by = stopped_by
        self.stopped_limit = stopped_limit


DONE = ToolOutcome("done")


def test_think_start_shows_turn_and_indent():
    out = io.StringIO()
    terminal = Terminal(output=out)
    terminal.on_think_start(StubState(turn=1))
    terminal.on_think_start(StubState(turn=2, depth=1))
    assert out.getvalue() == "[turn 1] thinking\n  [turn 2] thinking\n"


def test_text_streams_and_think_end_closes_line():
    out = io.StringIO()
    terminal = Terminal(output=out)
    state = StubState(turn=1)
    terminal.on_think_start(state)
    terminal.on_text(state, "Let me look at ")
    terminal.on_text(state, "the repo layout first.")
    terminal.on_think_end(state, reply=None)
    assert out.getvalue() == "[turn 1] thinking\nLet me look at the repo layout first.\n"


def test_text_disabled_when_show_text_false():
    out = io.StringIO()
    terminal = Terminal(output=out, show_text=False)
    state = StubState(turn=1)
    terminal.on_text(state, "should not show")
    assert out.getvalue() == ""


def test_think_end_no_output_when_already_at_line_start():
    out = io.StringIO()
    terminal = Terminal(output=out)
    state = StubState(turn=1)
    terminal.on_think_start(state)
    terminal.on_think_end(state, reply=None)
    assert out.getvalue() == "[turn 1] thinking\n"


def test_tool_start_and_completed():
    out = io.StringIO()
    terminal = Terminal(output=out)
    state = StubState(turn=1)
    call = tool_call("read_file", path="main.py")
    terminal.on_tool_start(state, call)
    terminal.on_tool_end(state, call, "x" * 1200, DONE)
    assert out.getvalue() == (
        '  tool read_file(path="main.py")\n'
        "  done 1.2KB\n"
    )


@pytest.mark.parametrize(
    "size, expected",
    [
        (100, "100B"),
        (1200, "1.2KB"),
        (3_565_158, "3.4MB"),
    ],
)
def test_tool_end_size_formatting(size, expected):
    out = io.StringIO()
    terminal = Terminal(output=out)
    state = StubState(turn=1)
    call = tool_call("read_file", path="x")
    terminal.on_tool_end(state, call, "x" * size, DONE)
    assert out.getvalue() == f"  done {expected}\n"


def test_tool_end_input_error():
    out = io.StringIO()
    terminal = Terminal(output=out)
    state = StubState(turn=1)
    call = tool_call("read_file", path="x")
    terminal.on_tool_end(state, call, "(input error: path is missing)", ToolOutcome("input_error"))
    assert out.getvalue() == "  (input error: path is missing)\n"


@pytest.mark.parametrize(
    "result, outcome",
    [
        ("(interrupted by user)", ToolOutcome("interrupted", KeyboardInterrupt())),
        ("(aborted: TimeoutError)", ToolOutcome("aborted", TimeoutError())),
    ],
)
def test_tool_end_closed_call_shows_interruption_not_size(result, outcome):
    out = io.StringIO()
    terminal = Terminal(output=out)
    call = tool_call("bash", command="make build")
    terminal.on_tool_end(StubState(turn=1), call, result, outcome)
    assert out.getvalue() == f"  aborted bash: {result}\n"


def test_tool_end_denied():
    out = io.StringIO()
    terminal = Terminal(output=out)
    call = tool_call("bash", command="rm -rf /")
    terminal.on_tool_end(StubState(turn=1), call, "The user denied it", ToolOutcome("denied"))
    assert out.getvalue() == "  denied bash: The user denied it\n"


@pytest.mark.parametrize("result", ["(aborted: nothing to do)", "(input error in the log)", "(interrupted by user)"])
def test_tool_end_done_result_is_not_read_as_a_failure(result):
    # The display follows the outcome, not the text: a tool whose normal output looks like a failure phrase
    # still shows as done.
    out = io.StringIO()
    terminal = Terminal(output=out)
    call = tool_call("bash", command="echo")
    terminal.on_tool_end(StubState(turn=1), call, result, DONE)
    assert out.getvalue().startswith("  done ")


def test_context_change_compact_shows_cache_tail():
    # on_context_change does not write right away; it queues the line. The compaction line must come after the
    # next turn's header (it must not look attached to the previous turn's output). The rule is covered in detail
    # in test_review_io.py::test_compaction_line_appears_after_new_turn_header.
    out = io.StringIO()
    terminal = Terminal(output=out)
    state = StubState(turn=2)
    terminal.on_context_change(state, ContextChange("compact", 121_000, 18_000))
    assert out.getvalue() == ""
    terminal.on_think_start(StubState(turn=3))
    assert out.getvalue() == "[turn 3] thinking\n  context compacted: 121k → 18k tokens (cache rebuilds)\n"


def test_context_change_clear_tool_results_no_tail():
    # If the run ends with no next think, on_run_end flushes it (otherwise it would be lost).
    out = io.StringIO()
    terminal = Terminal(output=out)
    state = StubState(turn=1)
    terminal.on_context_change(state, ContextChange("clear_tool_results", 900, 300))
    assert out.getvalue() == ""
    terminal.on_run_end(state, None)
    assert out.getvalue() == "  tool results cleared: 900 → 300 tokens\ndone: (1 turn)\n"


def test_context_change_rollback():
    out = io.StringIO()
    terminal = Terminal(output=out)
    state = StubState(turn=1)
    terminal.on_context_change(state, ContextChange("rollback", 500, 200))
    terminal.on_run_end(state, None)
    assert out.getvalue() == "  context rolled back: 500 → 200 tokens\ndone: (1 turn)\n"


def test_run_end_normal_with_cost_and_cache_rate():
    out = io.StringIO()
    terminal = Terminal(output=out)
    usage = Usage(input_tokens=100, cache_read_tokens=525, cost=0.42)
    state = StubState(turn=5, usage=usage, stopped_by="is_answered")
    terminal.on_run_end(state, None)
    assert out.getvalue() == "done: is_answered (5 turns, ~$0.42, cache hit 84%)\n"


def test_run_end_normal_without_cost_or_cache():
    out = io.StringIO()
    terminal = Terminal(output=out)
    state = StubState(turn=2, usage=Usage(), stopped_by="finish")
    terminal.on_run_end(state, None)
    assert out.getvalue() == "done: finish (2 turns)\n"


def test_run_end_limit_is_a_warning_line():
    out = io.StringIO()
    terminal = Terminal(output=out)
    state = StubState(turn=50, stopped_by="limit", stopped_limit=50)
    terminal.on_run_end(state, None)
    assert out.getvalue() == "done: reached limit(50), the task may be unfinished (50 turns)\n"


def test_run_end_keyboard_interrupt():
    out = io.StringIO()
    terminal = Terminal(output=out)
    state = StubState(turn=3)
    terminal.on_run_end(state, KeyboardInterrupt())
    assert out.getvalue() == "done: interrupted by user (3 turns)\n"


def test_run_end_other_exception():
    out = io.StringIO()
    terminal = Terminal(output=out)
    state = StubState(turn=2)
    terminal.on_run_end(state, TimeoutError("slow"))
    assert out.getvalue() == "done: error TimeoutError: slow (2 turns)\n"


def test_run_end_breaks_mid_line_first():
    out = io.StringIO()
    terminal = Terminal(output=out)
    state = StubState(turn=1, stopped_by="is_answered")
    terminal.on_text(state, "stopped mid-line")
    terminal.on_run_end(state, None)
    assert out.getvalue() == "stopped mid-line\ndone: is_answered (1 turn)\n"


def test_run_end_without_stopped_by_does_not_show_none():
    # A loop without @loop (a plain function) leaves no stopped_by. Python's None must not show on screen.
    out = io.StringIO()
    terminal = Terminal(output=out)
    terminal.on_run_end(StubState(turn=2), None)
    assert out.getvalue() == "done: (2 turns)\n"


def test_ask_returns_parsed_value():
    out = io.StringIO()
    prompts = []

    def fake_input(p):
        prompts.append(p)
        return "yes"

    terminal = Terminal(output=out, input=fake_input)
    state = StubState(turn=1)
    assert terminal.ask(state, "Run it?", returns=bool) is True
    assert prompts == ["Run it? (yes/no) "]


def test_ask_reasks_on_invalid_answer():
    out = io.StringIO()
    answers = iter(["not sure", "yes"])
    terminal = Terminal(output=out, input=lambda p: next(answers))
    state = StubState(turn=1)
    assert terminal.ask(state, "Continue?", returns=bool) is True
    assert "Answer 'yes' or 'no'" in out.getvalue()


def test_ask_no_choices_for_str():
    out = io.StringIO()
    prompts = []

    def fake_input(p):
        prompts.append(p)
        return "body"

    terminal = Terminal(output=out, input=fake_input)
    state = StubState(turn=1)
    assert terminal.ask(state, "What should I do?", returns=str) == "body"
    assert prompts == ["What should I do? "]


def test_ask_flushes_mid_line_text_first():
    out = io.StringIO()
    terminal = Terminal(output=out, input=lambda p: "yes")
    state = StubState(turn=1)
    terminal.on_text(state, "still thinking")
    terminal.ask(state, "Continue?", returns=bool)
    assert out.getvalue() == "still thinking\n"


def test_ask_propagates_eof_and_keyboard_interrupt():
    terminal_eof = Terminal(output=io.StringIO(), input=lambda p: (_ for _ in ()).throw(EOFError()))
    with pytest.raises(EOFError):
        terminal_eof.ask(StubState(), "question", returns=str)

    terminal_kb = Terminal(
        output=io.StringIO(), input=lambda p: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        terminal_kb.ask(StubState(), "question", returns=str)


def test_default_terminal_is_a_singleton():
    assert default_terminal() is default_terminal()
