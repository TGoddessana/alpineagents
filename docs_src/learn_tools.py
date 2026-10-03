from pathlib import Path

from alpineagents import Agent, ToolError, tool


@tool
def read_file(path: str, max_lines: int = 200) -> str:
    """Read a text file

    Args:
        path: Path of the file
        max_lines: How many lines to return
    """
    file = Path(path)
    if not file.exists():
        raise ToolError(f"No such file: {path}")
    return "\n".join(file.read_text().splitlines()[:max_lines])


print(read_file.spec)

agent = Agent(model="claude-sonnet-5", tools=[read_file])
print(agent.run("What is on the first line of README.md?"))
