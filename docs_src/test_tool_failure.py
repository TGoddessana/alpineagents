from pathlib import Path

from alpineagents import Agent, Message, State, StoppedByUntil, ToolError, tool
from alpineagents.testing import FakeModel, tool_call


@tool
def read_file(path: str) -> str:
    """Read a file"""
    file = Path(path)
    if not file.exists():
        raise ToolError(f"No such file: {path}")
    return file.read_text()


agent = Agent(model="claude-sonnet-5", tools=[read_file])


def test_a_missing_file_is_an_error_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake = FakeModel([
        tool_call("read_file", path="missing.py"),
        "There is no missing.py",
    ])
    state = State(messages=[Message.user("Read missing.py")])

    agent.copy(model=fake, reporter=None).run(state)

    result = next(h for h in state.history if h.kind == "tool_result")
    assert (result.content, result.is_error) == ("No such file: missing.py", True)
    assert state.stopped == StoppedByUntil("is_answered")
