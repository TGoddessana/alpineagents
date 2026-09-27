"""Resume tests: ``Agent._summary`` (what a saved State records about the Agent that saved it) and the
``ResumeWarning`` a run gives when a different Agent resumes that State. No network.

``state.save``/``State.load`` are not implemented yet, so these tests stand in for the loader with
``state._set_saved_agent(...)``.
"""

from __future__ import annotations

import hashlib
import json
import warnings

import pytest

from alpineagents import MCP, Agent, ResumeWarning, State, loop, tool
from alpineagents.testing import FakeModel


@tool
def search_web(query: str) -> str:
    """Search the web"""
    return f"web: {query}"


@tool
def search_docs(query: str) -> str:
    """Search the docs"""
    return f"docs: {query}"


def make_agent(replies=("Done",), **settings) -> Agent:
    settings.setdefault("reporter", None)
    settings.setdefault("human", None)
    settings.setdefault("model", FakeModel(list(replies)))
    return Agent(**settings)


def saved_state(saved_by: Agent, task: str = "Find the bug") -> State:
    """A State as the Store loader will rebuild it: with the summary of the Agent that saved it."""
    state = State(task)
    state._set_saved_agent(saved_by._summary())
    return state


def resume_warnings(caught: list[warnings.WarningMessage]) -> list[ResumeWarning]:
    return [w.message for w in caught if isinstance(w.message, ResumeWarning)]


# ================================================================ _summary


def test_summary_values():
    agent = make_agent(system="You are careful", tools=[search_web, search_docs], name="researcher")
    assert agent._summary() == {
        "name": "researcher",
        "model": "fake/fake",
        "system_sha256": hashlib.sha256(b"You are careful").hexdigest(),
        "tools": ["search_docs", "search_web"],
        "mcp_servers": [],
    }


def test_summary_hashes_the_system_prompt():
    agent = make_agent(system="Secret internal rules")
    summary = agent._summary()
    assert summary["system_sha256"] == hashlib.sha256("Secret internal rules".encode()).hexdigest()
    assert "Secret internal rules" not in json.dumps(summary)
    assert make_agent()._summary()["system_sha256"] is None


def test_summary_model_without_provider():
    fake = FakeModel(["Done"], name="local-model")
    fake.provider = ""
    assert make_agent(model=fake)._summary()["model"] == "local-model"


def test_summary_is_json():
    agent = make_agent(system="s", tools=[search_web], name="a", description="d")
    summary = agent._summary()
    assert json.loads(json.dumps(summary)) == summary


def test_summary_lists_mcp_servers_without_connecting():
    github = MCP("npx -y @modelcontextprotocol/server-github", name="github")
    docs = MCP(url="https://example.com/mcp", name="docs")
    agent = make_agent(tools=[search_web, github, docs.search])
    summary = agent._summary()
    assert summary["tools"] == ["search_web"]
    assert summary["mcp_servers"] == ["docs", "github"]
    assert agent._mcp_ready is None and agent._mcp_users == 0
    assert json.loads(json.dumps(summary)) == summary


# ================================================================ no warning


def test_same_agent_does_not_warn():
    agent = make_agent(system="s", tools=[search_web])
    state = saved_state(agent)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert agent.run(state) == "Done"


def test_equal_agent_does_not_warn():
    """A new Agent object with the same settings (the next process after a restart) is not a change."""
    state = saved_state(make_agent(system="s", tools=[search_web]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_agent(system="s", tools=[search_web]).run(state)


def test_new_state_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_agent().run(State("Find the bug"))


def test_missing_keys_do_not_warn():
    """An older or partial summary compares only the keys it has."""
    agent = make_agent(system="new prompt", tools=[search_docs], name="b")
    state = State("Find the bug")
    state._set_saved_agent({"model": "fake/fake", "extra": 1})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        agent.run(state)


def test_tool_order_is_not_a_change():
    agent = make_agent(tools=[search_web, search_docs])
    state = State("Find the bug")
    state._set_saved_agent({**agent._summary(), "tools": ["search_web", "search_docs"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        agent.run(state)


# ================================================================ warnings


def test_tools_changed():
    state = saved_state(make_agent(tools=[search_web]))
    with pytest.warns(ResumeWarning) as caught:
        make_agent(tools=[search_docs]).run(state)
    (warning,) = resume_warnings(caught.list)
    assert warning.changes == {"tools": (["search_web"], ["search_docs"])}
    assert "tools: removed search_web, added search_docs" in str(warning)


def test_tool_removed_only():
    state = saved_state(make_agent(tools=[search_web, search_docs]))
    with pytest.warns(ResumeWarning) as caught:
        make_agent(tools=[search_docs]).run(state)
    (warning,) = resume_warnings(caught.list)
    assert warning.changes == {"tools": (["search_docs", "search_web"], ["search_docs"])}
    assert "tools: removed search_web)" in str(warning)
    assert "added" not in str(warning)


def test_model_changed():
    state = State("Find the bug")
    state._set_saved_agent({**make_agent()._summary(), "model": "anthropic/claude-sonnet-5"})
    agent = make_agent(model=FakeModel(["Done"], name="gpt-5"))
    with pytest.warns(ResumeWarning) as caught:
        agent.run(state)
    (warning,) = resume_warnings(caught.list)
    assert warning.changes == {"model": ("anthropic/claude-sonnet-5", "fake/gpt-5")}
    assert "model: anthropic/claude-sonnet-5 -> fake/gpt-5" in str(warning)


def test_system_changed_does_not_show_the_prompt():
    state = saved_state(make_agent(system="Old secret rules"))
    with pytest.warns(ResumeWarning) as caught:
        make_agent(system="New secret rules").run(state)
    (warning,) = resume_warnings(caught.list)
    assert set(warning.changes) == {"system_sha256"}
    message = str(warning)
    assert "system prompt changed" in message
    assert "secret" not in message
    for digest in warning.changes["system_sha256"]:
        assert digest not in message


def test_name_and_mcp_servers_changed():
    state = saved_state(make_agent(name="a", tools=[MCP("npx server", name="github")]))
    with pytest.warns(ResumeWarning) as caught:
        make_agent(name="b").run(state)
    (warning,) = resume_warnings(caught.list)
    assert warning.changes == {"name": ("a", "b"), "mcp_servers": (["github"], [])}
    assert "name: a -> b" in str(warning)
    assert "mcp_servers: removed github" in str(warning)


def test_several_changes_in_one_warning():
    state = saved_state(make_agent(system="s", tools=[search_web]))
    agent = make_agent(model=FakeModel(["Done"], name="other"), tools=[search_docs])
    with pytest.warns(ResumeWarning) as caught:
        agent.run(state)
    (warning,) = resume_warnings(caught.list)
    assert set(warning.changes) == {"model", "system_sha256", "tools"}
    assert "; " in str(warning)


def test_warns_only_once():
    state = saved_state(make_agent(tools=[search_web]))
    agent = make_agent(["First", "Second"], tools=[search_docs])
    with pytest.warns(ResumeWarning):
        agent.run(state)
    state.add_user_message("And now?")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert agent.run(state) == "Second"


def test_points_at_the_run_call():
    state = saved_state(make_agent(tools=[search_web]))
    with pytest.warns(ResumeWarning) as caught:
        make_agent().run(state)
    assert caught.list[0].filename == __file__


async def test_arun_warns():
    state = saved_state(make_agent(tools=[search_web]))
    with pytest.warns(ResumeWarning) as caught:
        assert await make_agent().arun(state) == "Done"
    (warning,) = resume_warnings(caught.list)
    assert warning.changes == {"tools": (["search_web"], [])}
    assert caught.list[0].filename == __file__


async def test_async_loop_warns():
    @loop(until=State.is_answered, limit=5)
    async def my_loop(agent, state):
        await agent.athink(state)

    state = saved_state(make_agent(system="a"))
    with pytest.warns(ResumeWarning):
        await make_agent(system="b", loop=my_loop).arun(state)


def test_filter_error_raises_from_run():
    state = saved_state(make_agent(tools=[search_web]))
    fake = FakeModel(["Done"])
    agent = make_agent(model=fake)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ResumeWarning)
        with pytest.raises(ResumeWarning) as raised:
            agent.run(state)
        assert raised.value.changes == {"tools": (["search_web"], [])}
        # The run did not start, so the next run refuses again.
        with pytest.raises(ResumeWarning):
            agent.run(state)
    assert fake.requests == []
    assert state.turn == 0
