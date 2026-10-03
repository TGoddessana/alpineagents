from pathlib import Path

from alpineagents import Agent, Message, State, StoppedByLimit, loop, tool, waiting_for_user


@tool
def read_file(path: str) -> str:
    """Read a text file"""
    return Path(path).read_text()


def spent_too_much(state: State) -> bool:
    return state.usage.output_tokens > 20_000


@loop(until=[waiting_for_user, spent_too_much], limit=30)
def careful(agent: Agent, state: State):
    agent.think(state)
    if state.pending_calls:
        agent.use_tools(state)


agent = Agent(model="claude-sonnet-5", tools=[read_file], loop=careful)

state = State(messages=[Message.user("Summarize every file in docs/")])
agent.run(state)
print(state.stopped)
if isinstance(state.stopped, StoppedByLimit):
    print(f"Stopped after {state.stopped.turns} turns. The task may be unfinished.")
