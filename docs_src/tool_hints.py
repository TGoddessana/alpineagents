import subprocess

from alpineagents import Hints, tool

READ_ONLY_COMMANDS = {"ls", "pwd", "cat", "git status"}


def bash_hints(args: dict) -> Hints | None:
    command = args.get("command")
    if isinstance(command, str) and command.strip() in READ_ONLY_COMMANDS:
        return Hints(read_only=True, open_world=False)
    return None  # not sure: the tool's own hints


@tool(open_world=False, hints_for=bash_hints)
def bash(command: str) -> str:
    """Run a shell command"""
    return subprocess.run(command, shell=True, capture_output=True, text=True).stdout
