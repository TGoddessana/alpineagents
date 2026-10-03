from pathlib import Path

from alpineagents import Agent, tool
from alpineagents.permissions import AllowByReadOnly, DecideByHuman, Denied, DenyPermission


@tool(read_only=True, open_world=False)
def read_file(path: str) -> str:
    """Read a file"""
    return Path(path).read_text()


@tool(open_world=False)
def write_file(path: str, content: str) -> None:
    """Create a file, or replace its content"""
    Path(path).write_text(content)


class DenyOutsideFolder(DenyPermission):
    """Refuses a call whose path= argument is outside a folder."""

    def __init__(self, folder):
        self.folder = Path(folder).resolve()

    def check(self, state, call, tool):
        path = call.args.get("path")
        if path is None:
            return None  # no path: no opinion
        if not isinstance(path, str) or not (self.folder / path).resolve().is_relative_to(self.folder):
            return Denied(f"{path!r} is outside {self.folder}. Use a path inside it.")
        return None

    def __repr__(self):
        return f"DenyOutsideFolder({str(self.folder)!r})"


agent = Agent(
    model="claude-sonnet-5",
    tools=[read_file, write_file],
    permissions=[DenyOutsideFolder("."), AllowByReadOnly(), DecideByHuman()],
)
print(agent.run("Read notes.md and write a summary next to it"))
