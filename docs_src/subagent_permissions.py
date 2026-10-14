from pathlib import Path

from alpineagents import Agent, tool
from alpineagents.permissions import AllowByName, AllowByReadOnly, DecideByHuman, DenyByName


@tool(read_only=True)
def read_file(path: str) -> str:
    """Read a text file"""
    return Path(path).read_text()


@tool
def write_file(path: str, content: str) -> None:
    """Create a file, or replace its content"""
    Path(path).write_text(content)


@tool
def delete_file(path: str) -> None:
    """Delete a file"""
    Path(path).unlink()


# No permissions= of its own: the lead's whole list decides its calls
editor = Agent(
    model="claude-sonnet-5",
    name="editor",
    description="Edits the files of the project",
    tools=[read_file, write_file, delete_file],
)

# A list of its own: it replaces the lead's allow and decide rules
drafter = Agent(
    model="claude-sonnet-5",
    name="drafter",
    description="Writes drafts into the drafts folder",
    tools=[write_file, delete_file],
    permissions=[AllowByName(["write_file"])],
)

lead = Agent(
    model="claude-sonnet-5",
    tools=[read_file, editor, drafter],
    permissions=[
        DenyByName(["delete_*"], reason="Nothing is deleted in this run."),
        AllowByName(["editor", "drafter"]),
        AllowByReadOnly(),
        DecideByHuman(),
    ],
)
