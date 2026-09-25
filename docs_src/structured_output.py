from dataclasses import dataclass
from pathlib import Path

from alpineagents import Agent, State


@dataclass
class Review:
    approved: bool
    reason: str


agent = Agent(model="claude-sonnet-5", system="You review code changes.")

state = State("Review this change:\n" + Path("change.diff").read_text())
agent.run(state)
review = agent.ask(state, "Should this change be merged?", returns=Review)
print(review.approved, review.reason)
