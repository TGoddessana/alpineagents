from pathlib import Path

from alpineagents import Agent, CompactIfFull, State, loop, tool, waiting_for_user

compact = CompactIfFull(at=0.5, instructions="Keep file paths and failing test names")


@tool
def read_file(path: str) -> str:
    """Read a text file"""
    return Path(path).read_text()


@loop(until=waiting_for_user, limit=100)
def long_task(agent: Agent, state: State):
    compact(agent, state)
    agent.think(state)
    if state.pending_calls:
        agent.use_tools(state)
        state.clear_tool_results(keep_last=10)


agent = Agent(model="claude-sonnet-5", tools=[read_file], loop=long_task)
print(agent.run("Read every log file in logs/ and list the errors"))
