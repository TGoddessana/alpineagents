from pathlib import Path

from alpineagents import Agent, ToolError, tool
from alpineagents.testing import FakeModel, tool_call


@tool
def read_file(path: str) -> str:
    """Read a text file"""
    file = Path(path)
    if not file.exists():
        raise ToolError(f"No such file: {path}")
    return file.read_text()


fake = FakeModel([
    tool_call("read_file", path="README.md"),
    "README.md describes a Python agent framework.",
])
agent = Agent(model=fake, tools=[read_file])
print(agent.run("Summarize README.md"))
