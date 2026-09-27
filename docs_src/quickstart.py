from pathlib import Path

from alpineagents import Agent, ToolError, tool


@tool
def read_file(path: str) -> str:
    """Read a text file"""
    file = Path(path)
    if not file.exists():
        raise ToolError(f"No such file: {path}")
    return file.read_text()


agent = Agent(model="claude-sonnet-5", tools=[read_file])
print(agent.run("Summarize README.md in three lines"))
