from pathlib import Path

from alpineagents import Agent, Message, State, loop, tool, waiting_for_user


@tool
def read_file(path: str) -> str:
    """Read a text file"""
    return Path(path).read_text()


@tool
def list_files() -> list[str]:
    """List the files in this folder"""
    return sorted(p.name for p in Path(".").iterdir())


@loop(until=waiting_for_user, limit=30)
def plan_then_act(agent: Agent, state: State):
    if state.turn == 0:
        agent.think(state, tools=[])
        state.add_message(Message.notice("Now carry out the plan."))
        return
    agent.think(state)
    if state.pending_calls:
        agent.use_tools(state)


agent = Agent(model="claude-sonnet-5", tools=[read_file, list_files], loop=plan_then_act)
print(agent.run("Find out what this folder is for"))
