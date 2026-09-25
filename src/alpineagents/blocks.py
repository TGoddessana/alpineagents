"""Common blocks. A block is a callable that takes ``(agent, state)``; in an async loop, an ``async def`` one."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent
    from .state import State

__all__ = ["CompactIfFull", "compact_if_full", "acompact_if_full"]


class CompactIfFull:
    """A block that compacts the context when it is fuller than ``at``.

    Call it at the start of a turn, before ``agent.think``. ``compact_if_full`` is one with the defaults.

    Example:
        ``CompactIfFull(at=0.5, instructions="Keep file paths and test results")(agent, state)``
    """

    def __init__(self, at: float = 0.6, instructions: str | None = None):
        """
        Args:
            at: The fraction of the model's context window (``state.context_used``) above which to compact.
            instructions: What the summary must keep, added to the default summary prompt.
        """
        self.at = at
        """The fraction of the context window above which to compact."""
        self.instructions = instructions
        """What the summary should keep, or ``None``."""

    def __call__(self, agent: Agent, state: State):
        if state.context_used > self.at:
            agent.compact(state, instructions=self.instructions)

    async def acall(self, agent: Agent, state: State):
        """The async version, for async loops: ``await CompactIfFull(at=0.5).acall(agent, state)``."""
        if state.context_used > self.at:
            await agent.acompact(state, instructions=self.instructions)

    def __repr__(self) -> str:
        return f"CompactIfFull(at={self.at!r}, instructions={self.instructions!r})"


compact_if_full = CompactIfFull()
"""Compacts the context when it is more than 60% full. Call it as ``compact_if_full(agent, state)`` at the start
of a turn. The same as ``CompactIfFull()``."""
acompact_if_full = compact_if_full.acall
"""The async version of ``compact_if_full``, for async loops: ``await acompact_if_full(agent, state)``."""
