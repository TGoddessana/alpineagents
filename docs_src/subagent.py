from pathlib import Path

from alpineagents import Agent, tool


@tool
def read_file(path: str) -> str:
    """Read a text file"""
    return Path(path).read_text()


researcher = Agent(
    model="claude-sonnet-5",
    name="researcher",
    description="Reads files and reports what they say",
    system="Read what you need, then answer with the facts only.",
    tools=[read_file],
)
lead = Agent(model="claude-sonnet-5", tools=[researcher])
print(lead.run("Which Python versions does this project support? Ask the researcher to read pyproject.toml"))
