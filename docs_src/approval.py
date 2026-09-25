from pathlib import Path
from typing import Literal

from alpineagents import Agent, State, loop, tool

RISKY = {"write_file"}


@tool
def write_file(path: str, content: str) -> None:
    """Create a file, or replace its content"""
    Path(path).write_text(content)


@loop(until=State.is_answered, limit=30)
def careful(agent: Agent, state: State):
    agent.think(state)
    for call in state.pending_calls:
        if call.name in RISKY:
            question = f"Run {call.name}({call.args})?"
            answer = agent.ask_human(state, question, returns=Literal["yes", "no"])
            if answer == "no":
                state.deny(call, "The user declined this call")
    if state.wants_tools():
        agent.use_tools(state)


agent = Agent(model="claude-sonnet-5", tools=[write_file], loop=careful)
print(agent.run("Write a short README for this folder"))
