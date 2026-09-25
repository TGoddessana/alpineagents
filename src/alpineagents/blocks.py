"""Common blocks. A block is a callable that takes ``(agent, state)``; in an async loop, an ``async def`` one."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent
    from .state import State

__all__ = ["CompactIfFull", "compact_if_full", "acompact_if_full"]


class CompactIfFull:
    """Compact when the context goes over ``at`` (a fraction). ``instructions`` decides what to keep."""

    def __init__(self, at: float = 0.6, instructions: str | None = None):
        self.at = at
        self.instructions = instructions

    def __call__(self, agent: Agent, state: State):
        if state.context_used > self.at:
            agent.compact(state, instructions=self.instructions)

    async def acall(self, agent: Agent, state: State):
        """The async version, for async loops: ``await acompact_if_full(agent, state)`` or
        ``await CompactIfFull(at=0.5).acall(agent, state)``."""
        if state.context_used > self.at:
            await agent.acompact(state, instructions=self.instructions)

    def __repr__(self) -> str:
        return f"CompactIfFull(at={self.at!r}, instructions={self.instructions!r})"


compact_if_full = CompactIfFull()
acompact_if_full = compact_if_full.acall
