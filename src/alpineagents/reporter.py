"""The Reporter role: shows progress.

Every method does nothing. To change the display, subclass and override only the methods you need.
A Reporter does not change State and does not affect the flow. Exceptions it raises are not swallowed; they propagate.

Who calls what, and when (ARCHITECTURE.md "Reporter notifications: who and when"):

- ``on_run_start``/``on_run_end``: ``Agent.run``
- ``on_think_start``/``on_text``/``on_think_end``: ``Agent.think``, ``Agent.ask``
- ``on_tool_start``/``on_tool_end``: ``Agent.use_tools`` (on the main thread). A call a permission denied or
  cancelled gets ``on_tool_end`` without ``on_tool_start``.
- ``on_context_change``: called inside the State method that changed the messages, after releasing the lock, on
  the Reporter of the Agent linked to the State.
- ``on_model_event``: when the Model reports via ``on_event`` during ``respond``/``compact``, right after the
  Agent records it.

May be called from several threads at once (parallel tools in one turn). Implementations must be thread-safe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import State
    from .types import ContextChange, ModelEvent, Reply, ToolCall, ToolOutcome, ToolResultContent

__all__ = ["Reporter"]


class Reporter:
    """Shows progress. Every method does nothing by default, so a subclass overrides only the ones it needs.

    Pass one as ``Agent(reporter=...)``; ``reporter=None`` shows nothing. A Reporter only watches: it does not
    change the State or the flow. Exceptions it raises are not swallowed.

    Methods may be called from several threads at once (a turn's tools run in parallel), so an implementation
    must be thread-safe.

    Example:
        ```python
        class Log(Reporter):
            def on_tool_start(self, state, call):
                logging.info("turn %d: %s", state.turn, call.name)
        ```
    """

    def on_run_start(self, state: State) -> None:
        """When ``run()`` starts."""

    def on_think_start(self, state: State) -> None:
        """Right before a request to the model, from ``think`` and ``ask``. In ``think``, ``state.turn`` is already
        this turn's number."""

    def on_text(self, state: State, chunk: str) -> None:
        """As model text arrives, chunk by chunk."""

    def on_think_end(self, state: State, reply: Reply) -> None:
        """Right after the reply ends. In ``think`` the reply is already recorded; ``ask`` records nothing to the
        context but calls this too."""

    def on_tool_start(self, state: State, call: ToolCall) -> None:
        """Right before a tool runs. Every call that gets this gets exactly one ``on_tool_end``."""

    def on_tool_end(self, state: State, call: ToolCall, result: ToolResultContent, outcome: ToolOutcome) -> None:
        """Right after a tool call ends. A call a permission denied or cancelled did not run, so it gets this
        without ``on_tool_start``.

        Args:
            state: The State.
            call: The tool call.
            result: What was sent to the model: a string, or a tuple of ``TextBlock`` and ``Image`` when the tool
                returned an image (``result_text(result)`` makes it one string). For a denied call, the denial
                reason.
            outcome: How the call ended. Branch on ``outcome.kind`` instead of reading ``result``, because result
                strings are written for the model and may change. For a denied or cancelled call,
                ``outcome.decided_by`` names the permission that decided.
        """

    def on_context_change(self, state: State, change: ContextChange) -> None:
        """When the messages change: ``compact`` (by the model or ``state.compact``), ``clear_tool_results`` or
        ``restore``. Not for the import of ``State(messages=...)``, and not when a failed ``think`` is taken back."""

    def on_model_event(self, state: State, event: ModelEvent) -> None:
        """When the Model reports something outside the reply, such as falling back to another model. Called
        while waiting for the reply."""

    def on_run_end(self, state: State, error: BaseException | None) -> None:
        """When ``run()`` ends, always, even after an exception.

        Args:
            state: The State. ``state.stopped`` says why the run stopped (``None`` after an exception, or when
                a loop written without ``@loop`` ended on its own check).
            error: The exception that ended the run, or ``None`` on a normal finish.
        """
