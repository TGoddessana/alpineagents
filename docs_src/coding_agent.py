from pathlib import Path

from alpineagents import Agent, State, ToolError, compact_if_full, loop, tool, waiting_for_user


@tool
def list_files(folder: str = ".") -> list[str]:
    """List the files in a folder"""
    return sorted(p.name for p in Path(folder).iterdir())


@tool
def read_file(path: str) -> str:
    """Read a file"""
    file = Path(path)
    if not file.exists():
        raise ToolError(f"No such file: {path}")
    return file.read_text()


@tool
def write_file(path: str, content: str) -> None:
    """Create a file, or replace its content"""
    Path(path).write_text(content)


@loop(until=waiting_for_user, limit=30)
def coding(agent: Agent, state: State):
    compact_if_full(agent, state)
    agent.think(state)
    if state.pending_calls:
        agent.use_tools(state)


agent = Agent(
    model="claude-sonnet-5",
    system="You are a coding assistant. Read a file before you change it.",
    tools=[list_files, read_file, write_file],
    loop=coding,
)
print(agent.run("Add a test for the add() function in calc.py"))
