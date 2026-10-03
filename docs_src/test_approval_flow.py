from pathlib import Path

from alpineagents import Agent, Message, State, StoppedByPermission, tool
from alpineagents.permissions import DecideByHuman
from alpineagents.testing import FakeHuman, FakeModel, tool_call


@tool(open_world=False)
def write_file(path: str, content: str) -> None:
    """Create a file, or replace its content"""
    Path(path).write_text(content)


agent = Agent(model="claude-sonnet-5", tools=[write_file], permissions=[DecideByHuman()])


def test_no_stops_the_run_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake = FakeModel([tool_call("write_file", path="a.md", content="x")])
    human = FakeHuman(["no"])
    state = State(messages=[Message.user("Write a.md")])

    agent.copy(model=fake, human=human, reporter=None).run(state)

    assert human.remaining == 0
    assert isinstance(state.stopped, StoppedByPermission)
    assert not Path("a.md").exists()
