from pathlib import Path

from alpineagents import Agent, tool


@tool
def read_file(path: str) -> str:
    """Read a text file"""
    file = Path(path)
    return file.read_text() if file.exists() else f"No such file: {path}"


agent = Agent(model="claude-sonnet-5", tools=[read_file])
print(agent.run("Summarize README.md in three lines"))
