import subprocess

from alpineagents import Agent, Hints, MCPTool, tool
from alpineagents.permissions import AllowPermission, Allowed, DecideByHuman

LOOK = {"ls", "pwd", "cat", "grep"}  # change nothing
ADD = {"mkdir", "touch"}  # create things, never delete or overwrite


def bash_hints(args: dict) -> Hints | None:
    command = args.get("command")
    if not isinstance(command, str) or any(c in command for c in ";&|<>`$\n"):
        return None  # not one simple command: the tool's own hints
    program = command.split()[0] if command.split() else ""
    if program in LOOK:
        return Hints(read_only=True, open_world=False)
    if program in ADD:
        return Hints(destructive=False, open_world=False)
    return None


@tool(open_world=False, hints_for=bash_hints)
def bash(command: str) -> str:
    """Run a shell command in the project folder"""
    done = subprocess.run(command, shell=True, capture_output=True, text=True)
    return done.stdout + done.stderr


class AllowNotDestructive(AllowPermission):
    """Allows a call the tool says deletes and overwrites nothing."""

    def check(self, state, call, tool):
        if isinstance(tool, MCPTool):
            return None  # an MCP server describes its own tools: do not believe it
        return Allowed() if not tool.hints_for(call.args).destructive else None


agent = Agent(
    model="claude-sonnet-5",
    tools=[bash],
    permissions=[AllowNotDestructive(), DecideByHuman()],
)
