"""The Reporter role: shows progress.

Every method does nothing. To change the display, subclass and override only the methods you need.
A Reporter does not change State and does not affect the flow. Exceptions it raises are not swallowed; they propagate.

Who calls what, and when (ARCHITECTURE.md "Reporter notifications: who and when"):

- ``on_run_start``/``on_run_end``: ``Agent.run``
- ``on_think_start``/``on_text``/``on_think_end``: ``Agent.think``, ``Agent.ask``
- ``on_tool_start``/``on_tool_end``: ``Agent.use_tools`` (on the main thread). For a denied call,
  ``State.deny`` calls ``on_tool_end`` (without ``on_tool_start``).
- ``on_context_change``: called inside the State method that changed the context, after releasing the lock.
- ``on_model_event``: when the Model reports via ``on_event`` during ``respond``/``compact``, right after the
  Agent records it.

May be called from several threads at once (parallel tools in one turn). Implementations must be thread-safe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import State
    from .types import ContextChange, ModelEvent, Reply, ToolCall, ToolOutcome

__all__ = ["Reporter"]


class Reporter:
    """A watch-only observer. Every default implementation does nothing."""

    def on_run_start(self, state: State) -> None:
        """When ``run()`` starts."""

    def on_think_start(self, state: State) -> None:
        """Right before a request to the model (``ask`` included). In ``think``, ``state.turn`` is already this turn."""

    def on_text(self, state: State, chunk: str) -> None:
        """As model text arrives, chunk by chunk."""

    def on_think_end(self, state: State, reply: Reply) -> None:
        """Right after the reply ends and is recorded (``ask`` records nothing to the context but calls it then too)."""

    def on_tool_start(self, state: State, call: ToolCall) -> None:
        """Right before a tool runs."""

    def on_tool_end(self, state: State, call: ToolCall, result: str, outcome: ToolOutcome) -> None:
        """Right after it finishes. ``result`` is the string sent to the model; for a denied call, the denial reason.

        ``outcome`` says how it ended (``done``, ``input_error``, ``aborted``, ``interrupted``, ``denied``). Branch on
        it, not on ``result``: result strings are written for the model and may change.
        """

    def on_context_change(self, state: State, change: ContextChange) -> None:
        """When the context changes: compaction, ``start_from``, clearing tool results, rollback after an exception."""

    def on_model_event(self, state: State, event: ModelEvent) -> None:
        """When the Model reports something outside the reply (e.g. falling back to another model).

        Called while waiting for the model's reply.
        """

    def on_run_end(self, state: State, error: BaseException | None) -> None:
        """When ``run()`` ends. On a normal finish ``error`` is ``None``; the stop reason is ``state.stopped_by``."""
