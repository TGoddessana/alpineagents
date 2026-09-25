from pathlib import Path

from alpineagents import Agent, State, tool
from alpineagents.testing import FakeModel, tool_call


@tool
def read_file(path: str) -> str:
    """Read a file"""
    return Path(path).read_text()


agent = Agent(model="claude-sonnet-5", tools=[read_file])


def test_reads_the_file_then_answers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("main.py").write_text("x = 1 +\n")
    fake = FakeModel([
        tool_call("read_file", path="main.py"),
        "Line 1 has a syntax error",
    ])
    state = State("Find the bug in main.py")

    agent.copy(model=fake, reporter=None).run(state)

    assert state.answer == "Line 1 has a syntax error"
    assert state.stopped_by == "is_answered"
    assert fake.remaining == 0
