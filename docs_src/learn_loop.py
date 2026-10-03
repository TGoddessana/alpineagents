from pathlib import Path

from alpineagents import Agent, Message, State, ToolError, compact_if_full, loop, tool


@tool
def read_file(path: str) -> str:
    """Read a text file"""
    file = Path(path)
    if not file.exists():
        raise ToolError(f"No such file: {path}")
    return file.read_text()


def waiting_for_user(state: State) -> bool:
    if state.pending_calls or not state.messages:
        return False
    last = state.messages[-1]
    return last.role == "assistant" and not last.tool_calls


def warn_when_long(agent: Agent, state: State):
    if state.turn == 20:
        state.add_message(Message.notice("You have used 20 turns. Finish soon."))


@loop(until=waiting_for_user, limit=30)
def coding(agent: Agent, state: State):
    compact_if_full(agent, state)
    warn_when_long(agent, state)
    agent.think(state)
    if state.pending_calls:
        agent.use_tools(state)


agent = Agent(model="claude-sonnet-5", tools=[read_file], loop=coding)
print(agent.run("Summarize README.md in three lines"))
