from pathlib import Path
from typing import Literal

from alpineagents import Agent, State, loop, tool


@tool(read_only=True, open_world=False)
def read_file(path: str) -> str:
    """Read a file"""
    return Path(path).read_text()


@tool(open_world=False)
def write_file(path: str, content: str) -> None:
    """Create a file, or replace its content"""
    Path(path).write_text(content)


@loop(until=State.is_answered, limit=30)
def careful(agent: Agent, state: State):
    agent.think(state)
    for call in state.pending_calls:
        found = agent.tool_map.get(call.name)  # None for a name the model made up
        if found is None or found.read_only:
            continue
        question = f"Run {call.name}({call.args})?"
        answer = agent.ask_human(state, question, returns=Literal["yes", "no"])
        if answer == "no":
            state.deny(call, "The user declined this call")
    if state.wants_tools():
        agent.use_tools(state)


agent = Agent(model="claude-sonnet-5", tools=[read_file, write_file], loop=careful)
print(agent.run("Write a short README for this folder"))
