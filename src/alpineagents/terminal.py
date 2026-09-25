"""Terminal: the default Reporter and Human.

The two roles share one terminal, so a single object coordinates them: while asking, it holds back output from
other threads and shows the question. This is the only module in the package that uses ``print``/``input``
(output goes through ``output.write``). Contract: ARCHITECTURE.md "Terminal: coordinating output and questions".
"""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TextIO

from .human import Human, describe_choices, parse_answer
from .reporter import Reporter
from .types import format_call

if TYPE_CHECKING:
    from .state import State
    from .types import ContextChange, ModelEvent, Reply, ToolCall, ToolOutcome

__all__ = ["Terminal", "default_terminal"]

#: on_context_change kind → display name.
_CHANGE_NAMES: dict[str, str] = {
    "compact": "context compacted",
    "start_from": "context restarted",
    "clear_tool_results": "tool results cleared",
    "rollback": "context rolled back",
}
#: Kinds that get the "cache rebuilds" note (the ones that replace the whole context).
_CACHE_RESET_KINDS = {"compact", "start_from"}


def _format_tokens(n: int) -> str:
    """``121k`` style at 1000 and above, otherwise as is."""
    if n >= 1000:
        return f"{n / 1000:.0f}k"
    return str(n)


def _format_size(text: str) -> str:
    """UTF-8 byte size: ``512B``, ``1.2KB``, ``3.4MB``."""
    size = len(text.encode("utf-8"))
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / (1024 * 1024):.1f}MB"


def _turns(n: int) -> str:
    """``1 turn``, ``2 turns``."""
    return f"{n} turn" if n == 1 else f"{n} turns"


class Terminal(Reporter, Human):
    """Shows progress in the terminal and asks the human in the terminal.

    Example output::

        [turn 1] thinking
        Let me look at the repo layout first.
          tool read_file(path="main.py")
          done 1.2KB
        [turn 2] thinking
          context compacted: 121k → 18k tokens (cache rebuilds)
        done: is_answered (5 turns, ~$0.42, cache hit 84%)
    """

    def __init__(
        self,
        *,
        output: TextIO | None = None,
        input: Callable[[str], str] | None = None,
        show_text: bool = True,
    ) -> None:
        """If ``output`` is None, ``sys.stdout`` is looked up on every write (so tests' capsys can capture it).
        If ``input`` is None, ``builtins.input``. With ``show_text=False``, model text streaming is not shown.

        Holds two locks: the output lock (``threading.RLock``, all writes) and the question lock (one question
        at a time).
        """
        self._output = output
        self._input = input
        self.show_text = show_text
        self._output_lock = threading.RLock()
        self._question_lock = threading.Lock()
        # Whether we are at the start of a line (where the indent must be written again). Starts True.
        self._at_line_start = True
        # Lines queued by on_context_change and not shown yet: (depth, text).
        # Flushed right after the next on_think_start turn header, or just before
        # on_run_end if the run ends first.
        self._pending_context_lines: list[tuple[int, str]] = []

    # ------------------------------------------------------------ internal helpers

    def _out(self) -> TextIO:
        return self._output if self._output is not None else sys.stdout

    def _write_line(self, state: State, text: str) -> None:
        """Take the output lock, end the current line if mid-line, then write indent + ``text`` + newline."""
        with self._output_lock:
            out = self._out()
            if not self._at_line_start:
                out.write("\n")
            out.write("  " * state.depth + text + "\n")
            out.flush()
            self._at_line_start = True

    def _flush_pending_context_lines(self) -> None:
        """Flush the lines queued by ``on_context_change`` now (if any)."""
        with self._output_lock:
            pending = self._pending_context_lines
            self._pending_context_lines = []
            if not pending:
                return
            out = self._out()
            for depth, text in pending:
                if not self._at_line_start:
                    out.write("\n")
                out.write("  " * depth + text + "\n")
                self._at_line_start = True
            out.flush()

    # ------------------------------------------------------------ Reporter

    def on_run_start(self, state: State) -> None:
        """Prints nothing."""

    def on_think_start(self, state: State) -> None:
        """One line: ``{indent}[turn {state.turn}] thinking``. The indent is ``"  " * state.depth``.

        Right after the header, flushes the lines deferred by ``on_context_change`` (if any), so a
        compaction line shows up under the new turn header.
        """
        self._write_line(state, f"[turn {state.turn}] thinking")
        self._flush_pending_context_lines()

    def on_text(self, state: State, chunk: str) -> None:
        """If ``show_text``, writes the chunk as is (indent at each line start). Remembers an unfinished line."""
        if not self.show_text or not chunk:
            return
        indent = "  " * state.depth
        with self._output_lock:
            out = self._out()
            parts = chunk.split("\n")
            for i, part in enumerate(parts):
                if self._at_line_start:
                    out.write(indent)
                    self._at_line_start = False
                out.write(part)
                if i < len(parts) - 1:
                    out.write("\n")
                    self._at_line_start = True
            out.flush()

    def on_think_end(self, state: State, reply: Reply) -> None:
        """Ends the line if the text stopped mid-line. Prints nothing otherwise."""
        with self._output_lock:
            if not self._at_line_start:
                out = self._out()
                out.write("\n")
                out.flush()
                self._at_line_start = True

    def on_tool_start(self, state: State, call: ToolCall) -> None:
        """``{indent}  tool {format_call(call)}``."""
        self._write_line(state, f"  tool {format_call(call)}")

    def on_tool_end(self, state: State, call: ToolCall, result: str, outcome: ToolOutcome) -> None:
        """By ``outcome.kind``: input_error ``  {result}``; aborted or interrupted ``  aborted {call.name}: {result}``
        (e.g. ``(aborted: TimeoutError)``, ``(interrupted by user)``); denied ``  denied {call.name}: {result}``;
        error ``  error {call.name}: {first line of result, at most 100 characters}``;
        done ``  done {size}`` (UTF-8 bytes: ``512B``, ``1.2KB``, ``3.4MB``).
        """
        kind = outcome.kind
        if kind == "input_error":
            text = f"  {result}"
        elif kind == "error":
            text = f"  error {call.name}: {_first_line(result)}"
        elif kind in ("aborted", "interrupted"):
            text = f"  aborted {call.name}: {result}"
        elif kind == "denied":
            text = f"  denied {call.name}: {result}"
        else:
            text = f"  done {_format_size(result)}"
        self._write_line(state, text)

    def on_context_change(self, state: State, change: ContextChange) -> None:
        """``  {name}: {before} → {after} tokens`` (tokens as ``121k`` at 1000 and above).

        Names: compact ``context compacted`` (followed by `` (cache rebuilds)``), start_from
        ``context restarted`` (same tail), clear_tool_results ``tool results cleared``, rollback
        ``context rolled back``.

        Queued instead of written right away: many calls, like ``compact_if_full``, change the context just
        before the next ``think``, so writing immediately would attach the line to the previous turn's output
        (it belongs under the next turn header). Flushed after the next ``on_think_start`` writes its header,
        or, if ``run`` ends first, by ``on_run_end`` before it writes the last line.
        """
        name = _CHANGE_NAMES.get(change.kind, change.kind)
        tail = " (cache rebuilds)" if change.kind in _CACHE_RESET_KINDS else ""
        before = _format_tokens(change.before_tokens)
        after = _format_tokens(change.after_tokens)
        text = f"  {name}: {before} → {after} tokens{tail}"
        with self._output_lock:
            self._pending_context_lines.append((state.depth, text))

    def on_model_event(self, state: State, event: ModelEvent) -> None:
        """``  model: {event.message}`` (ends the current line first if mid-line)."""
        self._write_line(state, f"  model: {event.message}")

    def on_run_end(self, state: State, error: BaseException | None) -> None:
        """One last line (ends the current line first if mid-line):

        - Normal: ``done: {stopped_by} ({turns}[, ~${cost:.2f}][, cache hit {rate:.0%}])``
          (cost is left out when None; the cache hit rate is left out when None or 0. For a loop without
          ``@loop``, ``stopped_by`` is ``None``: ``done: (2 turns)``). ``{turns}`` is ``1 turn`` / ``n turns``.
        - ``stopped_by == "limit"``: ``done: reached limit({state.stopped_limit}), the task may be unfinished
          ({turns})``
        - ``KeyboardInterrupt``/``CancelledError``: ``done: interrupted by user ({turns})``
        - Any other exception: ``done: error {type(error).__name__}: {error} ({turns})``

        Before that, flushes any lines ``on_context_change`` still has queued because no next ``think`` came
        (e.g. the last compaction happened after this run's last turn; otherwise they would be lost).
        """
        self._flush_pending_context_lines()
        turns = _turns(state.turn)
        if error is not None:
            if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)):
                text = f"done: interrupted by user ({turns})"
            else:
                text = f"done: error {type(error).__name__}: {error} ({turns})"
        elif state.stopped_by == "limit":
            text = f"done: reached limit({state.stopped_limit}), the task may be unfinished ({turns})"
        else:
            parts = [turns]
            cost = state.usage.cost
            if cost is not None:
                parts.append(f"~${cost:.2f}")
            rate = state.usage.cache_hit_rate
            if rate:
                parts.append(f"cache hit {rate:.0%}")
            reason = state.stopped_by
            head = f"done: {reason}" if reason is not None else "done:"
            text = f"{head} ({', '.join(parts)})"
        self._write_line(state, text)

    # ------------------------------------------------------------ Human

    def ask(self, state: State, prompt: str, returns: Any = str) -> Any:
        """Takes the question lock (concurrent questions wait in line) and asks. **The output lock is released
        while ``input`` is called**, so that if this question is abandoned without an answer (e.g. the tool was
        interrupted with Ctrl+C), the last line from ``on_run_end`` does not get stuck waiting on the lock too.

        - Ends the current line first if mid-line. The question is indented by ``"  " * state.depth``.
        - If no ``input`` was given (``builtins.input`` is used), we write the question ourselves to this
          Terminal's ``output``: ``builtins.input`` always writes its prompt to the real ``sys.stdout``, so with
          a different ``output=`` stream the question and the re-ask message would end up in different places.
          When ``input=`` is given, that function decides how to show the prompt (it gets the full text with
          ``{describe_choices(returns)}`` appended), so we leave it alone.
        - If ``parse_answer(answer, returns)`` raises ``ValueError``, writes its message on one line and asks
          again.
        - ``EOFError`` and ``KeyboardInterrupt`` propagate as is.
        """
        indent = "  " * state.depth
        choices = describe_choices(returns)
        shown = f"{prompt} ({choices}) " if choices else f"{prompt} "
        full_prompt = indent + shown
        custom_input = self._input

        with self._question_lock:
            if custom_input is None:
                self._show_prompt(full_prompt)
            else:
                with self._output_lock:
                    out = self._out()
                    if not self._at_line_start:
                        out.write("\n")
                        out.flush()
                        self._at_line_start = True
            while True:
                # Called without the lock: if it blocks forever here (an abandoned question), other output
                # does not stop.
                if custom_input is not None:
                    answer = custom_input(full_prompt)
                else:
                    answer = input()
                    self._end_input_line()
                try:
                    value = parse_answer(answer, returns)
                except ValueError as e:
                    self._show_reask_error(indent, e, full_prompt if custom_input is None else None)
                    continue
                return value

    def _end_input_line(self) -> None:
        """Return to the start of a line after ``builtins.input`` has read one.

        When typing in a terminal (tty), the newline from Enter is already on screen, so none is written
        (otherwise every question would leave a blank line). If input is a pipe or ``output`` is not a terminal,
        there is no echo, so the newline is written here.
        """
        with self._output_lock:
            out = self._out()
            if not self._at_line_start and not _echoes_to(out):
                out.write("\n")
                out.flush()
            self._at_line_start = True

    def _show_prompt(self, full_prompt: str) -> None:
        """Write ``full_prompt`` directly to this Terminal's ``output`` (no newline)."""
        with self._output_lock:
            out = self._out()
            if not self._at_line_start:
                out.write("\n")
            out.write(full_prompt)
            out.flush()
            self._at_line_start = False

    def _show_reask_error(self, indent: str, error: ValueError, reprompt: str | None) -> None:
        with self._output_lock:
            out = self._out()
            if not self._at_line_start:
                out.write("\n")
            out.write(f"{indent}{error}\n")
            out.flush()
            self._at_line_start = True
        if reprompt is not None:
            self._show_prompt(reprompt)


def _echoes_to(out: TextIO) -> bool:
    """Whether the line the human typed (and the Enter) already shows on the same terminal as ``out``."""
    try:
        return sys.stdin.isatty() and out.isatty()
    except (AttributeError, ValueError, OSError):
        return False


_default_terminal: Terminal | None = None
_default_terminal_lock = threading.Lock()


def default_terminal() -> Terminal:
    """Default ``reporter``/``human`` for ``Agent``. Only one is made per process (thread-safely)."""
    global _default_terminal
    if _default_terminal is None:
        with _default_terminal_lock:
            if _default_terminal is None:
                _default_terminal = Terminal()
    return _default_terminal


def _first_line(text: str, limit: int = 100) -> str:
    """The first line of ``text``, cut to ``limit`` characters with ``…``."""
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line if len(line) <= limit else line[: limit - 1] + "…"
