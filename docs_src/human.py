from pathlib import Path
from typing import Literal, get_args, get_origin

from alpineagents import Agent, Human, tool
from alpineagents.permissions import DecideByHuman


@tool(open_world=False)
def write_file(path: str, content: str) -> None:
    """Create a file, or replace its content"""
    Path(path).write_text(content)


class Unattended(Human):
    """For runs nobody watches. Says no to every yes/no question."""

    def ask(self, state, prompt, returns=str):
        if returns is bool:
            return False
        if get_origin(returns) is Literal and "no" in get_args(returns):
            return "no"
        raise RuntimeError(f"Nobody can answer this: {prompt}")


agent = Agent(
    model="claude-sonnet-5",
    tools=[write_file],
    permissions=[DecideByHuman()],
    human=Unattended(),
)
print(agent.run("Write a short README for this folder"))
