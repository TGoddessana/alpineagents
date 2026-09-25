from pathlib import Path

from alpineagents import Agent, tool


class Workspace:
    def __init__(self, root: str):
        self.root = Path(root)

    @tool
    def read_file(self, path: str) -> str:
        """Read a file in the workspace"""
        return (self.root / path).read_text()

    @tool
    def list_files(self) -> list[str]:
        """List the files in the workspace"""
        return sorted(p.name for p in self.root.iterdir())


workspace = Workspace("my-repo")
agent = Agent(model="claude-sonnet-5", tools=[workspace])
reader = Agent(model="claude-sonnet-5", tools=[workspace.read_file])
