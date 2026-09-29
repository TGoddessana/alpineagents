from pathlib import Path

from alpineagents.permissions import AllowByReadOnly, DecideByHuman, Denied, DenyPermission


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


agent = agent.copy(permissions=[DenyOutsideFolder("."), AllowByReadOnly(), DecideByHuman()])
