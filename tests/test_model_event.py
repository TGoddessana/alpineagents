"""What the Model reported through on_event: history (model_event) and Reporter.on_model_event."""

import io

import pytest

from alpineagents import Agent, ModelEvent, Reporter, State, Terminal
from alpineagents.testing import FakeModel

SWITCH = ModelEvent("fallback", "a rate limited → switched to b", {"from": "a", "to": "b"})


class Noisy(FakeModel):
    """A FakeModel that reports SWITCH before every reply."""

    def respond(self, request, on_text=None, on_event=None):
        if on_event is not None:
            on_event(SWITCH)
        return super().respond(request, on_text, on_event)


class Recorder(Reporter):
    def __init__(self):
        self.events = []

    def on_model_event(self, state, event):
        self.events.append((state, event))


def test_think_records_the_event_and_tells_the_reporter():
    reporter = Recorder()
    agent = Agent(model=Noisy(["Answer"]), reporter=reporter, human=None)
    state = State("Question")
    agent.run(state)

    entries = [e for e in state.history if e.kind == "model_event"]
    assert [e.content for e in entries] == [SWITCH]
    assert entries[0].turn == 1
    assert reporter.events == [(state, SWITCH)]


def test_event_is_recorded_without_a_reporter():
    state = State("Question")
    Agent(model=Noisy(["Answer"]), reporter=None, human=None).run(state)
    assert [e.kind for e in state.history] == ["user", "model_event", "reply"]


def test_ask_and_compact_pass_on_event_too():
    agent = Agent(model=Noisy(["First answer", "Yes", "Summary"]), reporter=None, human=None)
    state = State("Question")
    agent.run(state)
    agent.ask(state, "Is that right?")
    agent.compact(state)

    assert [e.kind for e in state.history if e.kind == "model_event"] == ["model_event"] * 3


def test_event_survives_a_failed_think():
    agent = Agent(model=Noisy([RuntimeError("disconnected")]), reporter=None, human=None)
    state = State("Question")
    with pytest.raises(RuntimeError):
        agent.run(state)
    kinds = [e.kind for e in state.history]
    assert kinds[:2] == ["user", "model_event"]  # history is not rolled back
    assert state.turn == 0


def test_on_event_rejects_anything_but_a_model_event():
    class Wrong(FakeModel):
        def respond(self, request, on_text=None, on_event=None):
            on_event("switched to another model")

    with pytest.raises(TypeError, match="ModelEvent"):
        Agent(model=Wrong([]), reporter=None, human=None).run("Question")


def test_terminal_shows_the_event_on_its_own_line():
    out = io.StringIO()
    terminal = Terminal(output=out)
    Agent(model=Noisy(["Answer"]), reporter=terminal, human=None).run("Question")

    assert out.getvalue().splitlines() == [
        "[turn 1] thinking",
        "  model: a rate limited → switched to b",
        "Answer",
        "done: is_answered (1 turn)",
    ]
