from alpineagents import Agent, Message, State, StoppedByUntil
from alpineagents.testing import FakeModel, tool_call

researcher = Agent(model="claude-sonnet-5", name="researcher", description="Answers questions about the project")
lead = Agent(model="claude-sonnet-5", tools=[researcher])


def test_the_lead_asks_the_researcher_and_uses_its_answer():
    fake_researcher = FakeModel(["Python 3.11 and later"])
    fake_lead = FakeModel([
        tool_call("researcher", prompt="Which Python versions are supported?"),
        "It supports Python 3.11 and later",
    ])
    agent = lead.copy(model=fake_lead, tools=[researcher.copy(model=fake_researcher)], reporter=None)
    state = State(messages=[Message.user("Which Python versions are supported?")])

    agent.run(state)

    assert state.answer == "It supports Python 3.11 and later"
    (result,) = [entry for entry in state.history if entry.kind == "tool_result"]
    assert result.outcome == "done" and result.content == "Python 3.11 and later"
    child = state.child(result.child_ids[0])
    assert child.parent_id == state.id and child.parent_call_id == result.call.id
    assert child.messages[0].text == "Which Python versions are supported?"  # it saw the prompt only
    assert child.stopped == StoppedByUntil("waiting_for_user")
    assert state.usage.requests == 3  # two requests of the lead and one of the researcher
    assert fake_lead.remaining == 0 and fake_researcher.remaining == 0
