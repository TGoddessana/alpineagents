from pathlib import Path
from typing import Literal

from alpineagents import Agent, Terminal, tool
from alpineagents.permissions import AllowByReadOnly, Allowed, DecidePermission, Denied
from alpineagents.types import format_call


@tool(read_only=True, open_world=False)
def read_file(path: str) -> str:
    """Read a file"""
    return Path(path).read_text()


@tool(open_world=False)
def write_file(path: str, content: str) -> None:
    """Create a file, or replace its content"""
    Path(path).write_text(content)


class AskWithAlways(DecidePermission):
    """Asks before each call. "always" allows the tool for the rest of the task."""

    def __init__(self, human):
        self.human = human

    def check(self, state, call, tool):
        if call.name in state.extra_data.get("always", []):
            return Allowed()
        choices = Literal["yes", "no", "always"]
        answer = self.human.ask(state, f"Run {format_call(call)}?", choices)
        if answer == "no":
            return Denied("The user declined this call. Wait for their next message.", stop=True)
        if answer == "always":
            with state.edit_extra_data() as data:
                data.setdefault("always", []).append(call.name)
        return Allowed()


agent = Agent(
    model="claude-sonnet-5",
    tools=[read_file, write_file],
    permissions=[AllowByReadOnly(), AskWithAlways(Terminal())],
)
print(agent.run("Write a short README and a CONTRIBUTING file for this folder"))
