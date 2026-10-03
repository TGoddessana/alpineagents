from pathlib import Path

from alpineagents import Agent, Message, State, StoppedByPermission, tool
from alpineagents.permissions import AllowByReadOnly, DecideByHuman


@tool(read_only=True, open_world=False)
def read_file(path: str) -> str:
    """Read a file"""
    return Path(path).read_text()


@tool(open_world=False)
def write_file(path: str, content: str) -> None:
    """Create a file, or replace its content"""
    Path(path).write_text(content)


agent = Agent(
    model="claude-sonnet-5",
    tools=[read_file, write_file],
    permissions=[AllowByReadOnly(), DecideByHuman()],
)

state = State(messages=[Message.user("Write a short README for this folder")])
agent.run(state)
while isinstance(state.stopped, StoppedByPermission):
    state.add_message(Message.user(input("What should it do instead? ")))
    agent.run(state)
print(state.answer)
