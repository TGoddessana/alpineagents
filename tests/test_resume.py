"""Resume tests: ``AgentInfo`` (what a run records about its Agent in ``RunStartEntry``) and the ``ResumeWarning`` a
run gives when a different Agent resumes a State loaded from a store. No network.

A State loaded from a store remembers the Agent of its last ``RunStartEntry``. The first run after ``store.load``
compares it with the running Agent. Nothing else does: not a State built in this process, not a later run, and not
switching models with ``agent.copy(model=...)``.
"""

from __future__ import annotations

import hashlib
import json
import warnings

import pytest

from alpineagents import (
    MCP,
    Agent,
    AgentInfo,
    Message,
    MessageEntry,
    ResumeWarning,
    RunStartEntry,
    State,
    Store,
    loop,
    tool,
)
from alpineagents import _serial
from alpineagents.store import Record
from alpineagents.testing import FakeModel


@tool
def search_web(query: str) -> str:
    """Search the web"""
    return f"web: {query}"


@tool
def search_docs(query: str) -> str:
    """Search the docs"""
    return f"docs: {query}"


class MemoryStore(Store):
    """The smallest Store that follows the contract: a dict of records."""

    def __init__(self) -> None:
        self.states: dict[str, Record] = {}

    def write(self, state_id, entries, info, *, create=False):
        record = self.states.setdefault(state_id, Record([], {}))
        have = len(record.entries)
        record.entries.extend(json.loads(json.dumps(e)) for e in entries if e["seq"] >= have)
        self.states[state_id] = Record(record.entries, json.loads(json.dumps(info)))

    def read(self, state_id):
        return self.states.get(state_id)


def make_agent(replies=("Done",), **settings) -> Agent:
    settings.setdefault("reporter", None)
    settings.setdefault("human", None)
    settings.setdefault("model", FakeModel(list(replies)))
    return Agent(**settings)


def info_of(agent: Agent) -> AgentInfo:
    return agent._agent_info_now()


def saved_state(saved_by: Agent | AgentInfo, task: str = "Find the bug") -> State:
    """A State as ``store.load`` rebuilds it, last run by ``saved_by``: saved with a ``RunStartEntry`` for it, then
    loaded."""
    info = saved_by if isinstance(saved_by, AgentInfo) else info_of(saved_by)
    state = State(
        history=[
            MessageEntry(kind="user", content=task, turn=0),
            RunStartEntry(content=info, turn=0),
        ]
    )
    store = MemoryStore()
    store.save(state)
    return store.load(state.id)


def resume_warnings(caught: list[warnings.WarningMessage]) -> list[ResumeWarning]:
    return [w.message for w in caught if isinstance(w.message, ResumeWarning)]


def run_starts(state: State) -> list[AgentInfo]:
    return [e.content for e in state.history if isinstance(e, RunStartEntry)]


# ================================================================ AgentInfo


def test_info_values():
    agent = make_agent(system="You are careful", tools=[search_web, search_docs], name="researcher")
    assert info_of(agent) == AgentInfo(
        name="researcher",
        model="fake/fake",
        system_sha256=hashlib.sha256(b"You are careful").hexdigest(),
        tools=("search_docs", "search_web"),
        mcp_servers=(),
    )


def test_info_hashes_the_system_prompt():
    agent = make_agent(system="Secret internal rules")
    info = info_of(agent)
    assert info.system_sha256 == hashlib.sha256("Secret internal rules".encode()).hexdigest()
    assert "Secret internal rules" not in json.dumps(_serial.agent_info_to_dict(info))
    assert info_of(make_agent()).system_sha256 is None


def test_info_model_without_provider():
    fake = FakeModel(["Done"], name="local-model")
    fake.provider = ""
    assert info_of(make_agent(model=fake)).model == "local-model"


def test_info_is_json():
    agent = make_agent(system="s", tools=[search_web], name="a", description="d")
    info = info_of(agent)
    as_dict = _serial.agent_info_to_dict(info)
    assert json.loads(json.dumps(as_dict)) == as_dict
    assert _serial.agent_info_from_dict(json.loads(json.dumps(as_dict))) == info


def test_info_lists_mcp_servers_without_connecting():
    github = MCP("npx -y @modelcontextprotocol/server-github", name="github")
    docs = MCP(url="https://example.com/mcp", name="docs")
    agent = make_agent(tools=[search_web, github, docs.search])
    info = info_of(agent)
    assert info.tools == ("search_web",)
    assert info.mcp_servers == ("docs", "github")
    assert agent._mcp_ready is None and agent._mcp_users == 0


def test_a_run_records_its_agent_in_history():
    agent = make_agent(system="s", tools=[search_web], name="a")
    state = State(messages=[Message.user("Go")])
    agent.run(state)
    assert run_starts(state) == [info_of(agent)]
    [entry] = [e for e in state.history if e.kind == "run_start"]
    assert entry.turn == 0


def test_every_run_records_one_run_start():
    agent = make_agent(["First", "Second"])
    state = State(messages=[Message.user("Go")])
    agent.run(state)
    state.add_message(Message.user("And now?"))
    agent.run(state)
    assert len(run_starts(state)) == 2


def test_run_start_survives_a_store_round_trip():
    store = MemoryStore()
    agent = make_agent(system="s", tools=[search_web, search_docs], name="a", store=store)
    state = State(id="rt", messages=[Message.user("Go")])
    agent.run(state)
    loaded = store.load("rt")
    assert run_starts(loaded) == run_starts(state) == [info_of(agent)]


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
        make_agent().run(State(messages=[Message.user("Find the bug")]))
        make_agent().run("Find the bug")


def test_state_not_loaded_from_a_store_does_not_warn():
    """Only ``store.load`` remembers the Agent. A State built from a history in this process does not."""
    other = make_agent(tools=[search_web], name="other")
    state = State(
        history=[
            MessageEntry(kind="user", content="Find the bug", turn=0),
            RunStartEntry(content=info_of(other), turn=0),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_agent().run(state)


def test_loaded_state_that_never_ran_does_not_warn():
    """No ``RunStartEntry``, nothing to compare with."""
    store = MemoryStore()
    store.save(State(id="nr", messages=[Message.user("Find the bug")]))
    state = store.load("nr")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_agent(tools=[search_docs], name="anyone").run(state)


def test_tool_order_is_not_a_change():
    agent = make_agent(tools=[search_web, search_docs])
    saved = AgentInfo(model="fake/fake", tools=("search_web", "search_docs"))
    state = saved_state(saved)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        agent.run(state)


def test_a_fork_of_a_loaded_state_does_not_warn():
    """A fork is a new State that no store knows, so it is not compared."""
    state = saved_state(make_agent(tools=[search_web]))
    attempt = state.fork()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_agent(tools=[search_docs]).run(attempt)


def test_warning_survives_pickle_and_copy():
    import copy
    import pickle

    warning = ResumeWarning("tools changed", {"tools": (("a",), ("b",))})
    for clone in (pickle.loads(pickle.dumps(warning)), copy.copy(warning)):
        assert str(clone) == "tools changed"
        assert clone.changes == {"tools": (("a",), ("b",))}


# ================================================================ warnings


def test_tools_changed():
    state = saved_state(make_agent(tools=[search_web]))
    with pytest.warns(ResumeWarning) as caught:
        make_agent(tools=[search_docs]).run(state)
    (warning,) = resume_warnings(caught.list)
    assert warning.changes == {"tools": (("search_web",), ("search_docs",))}
    assert "tools: removed search_web, added search_docs" in str(warning)


def test_tool_removed_only():
    state = saved_state(make_agent(tools=[search_web, search_docs]))
    with pytest.warns(ResumeWarning) as caught:
        make_agent(tools=[search_docs]).run(state)
    (warning,) = resume_warnings(caught.list)
    assert warning.changes == {"tools": (("search_docs", "search_web"), ("search_docs",))}
    assert "tools: removed search_web)" in str(warning)
    assert "added" not in str(warning)


def test_model_changed():
    state = saved_state(AgentInfo(model="anthropic/claude-sonnet-5"))
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
    assert warning.changes == {"name": ("a", "b"), "mcp_servers": (("github",), ())}
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


def test_the_last_run_start_is_the_one_compared():
    """A State run by several Agents before it was saved is compared with the last of them."""
    first, second = make_agent(name="first"), make_agent(name="second")
    state = State(
        history=[
            MessageEntry(kind="user", content="Go", turn=0),
            RunStartEntry(content=info_of(first), turn=0),
            RunStartEntry(content=info_of(second), turn=0),
        ]
    )
    store = MemoryStore()
    store.save(state)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        second.run(store.load(state.id))
    with pytest.warns(ResumeWarning) as caught:
        first.run(store.load(state.id))
    (warning,) = resume_warnings(caught.list)
    assert warning.changes == {"name": ("second", "first")}


def test_a_real_run_saved_then_resumed_by_a_different_agent():
    store = MemoryStore()
    make_agent(store=store, name="old", tools=[search_web]).run(State(id="real", messages=[Message.user("Go")]))
    state = store.load("real")
    state.add_message(Message.user("More"))
    with pytest.warns(ResumeWarning) as caught:
        make_agent(store=store, name="new", tools=[search_web]).run(state)
    (warning,) = resume_warnings(caught.list)
    assert warning.changes == {"name": ("old", "new")}
    assert run_starts(state)[-1].name == "new"


# ================================================================ only the first run after load


def test_warns_only_once():
    state = saved_state(make_agent(tools=[search_web]))
    agent = make_agent(["First", "Second"], tools=[search_docs])
    with pytest.warns(ResumeWarning):
        agent.run(state)
    state.add_message(Message.user("And now?"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert agent.run(state) == "Second"


def test_a_second_different_agent_after_the_first_run_does_not_warn():
    state = saved_state(make_agent(tools=[search_web], name="saved"))
    with pytest.warns(ResumeWarning):
        make_agent(tools=[search_web], name="first").run(state)
    state.add_message(Message.user("And now?"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_agent(tools=[search_docs], name="second").run(state)


def test_no_warning_when_the_first_run_matches_and_a_later_one_differs():
    agent = make_agent(["First", "Second"], system="s")
    state = saved_state(agent)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        agent.run(state)
        state.add_message(Message.user("Again"))
        agent.copy(system="other").run(state)


def test_switching_models_with_copy_never_warns():
    """``agent.copy(model=...)`` in one process is ordinary: the run records which model ran, and nothing is
    compared."""
    cheap = make_agent(["One", "Two"], name="worker")
    strong = cheap.copy(model=FakeModel(["Three"], name="strong"))
    state = State(messages=[Message.user("Go")])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cheap.run(state)
        state.add_message(Message.user("Harder one"))
        strong.run(state)
        state.add_message(Message.user("Back"))
        cheap.run(state)
    assert [info.model for info in run_starts(state)] == ["fake/fake", "fake/strong", "fake/fake"]


def test_switching_models_with_copy_after_a_load_warns_once_at_most():
    store = MemoryStore()
    cheap = make_agent(["One"], name="worker", store=store)
    cheap.run(State(id="sw", messages=[Message.user("Go")]))
    state = store.load("sw")
    strong = cheap.copy(model=FakeModel(["Two", "Three"], name="strong"))

    # The first run after the load is the one compared, and it does differ.
    state.add_message(Message.user("Harder one"))
    with pytest.warns(ResumeWarning) as caught:
        strong.run(state)
    (warning,) = resume_warnings(caught.list)
    assert warning.changes == {"model": ("fake/fake", "fake/strong")}

    # Switching back and forth after that is never compared.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        state.add_message(Message.user("Back"))
        cheap.copy(model=FakeModel(["Four"])).run(state)
        state.add_message(Message.user("Again"))
        strong.run(state)


def test_switching_models_with_copy_on_a_loaded_state_that_matches_does_not_warn():
    store = MemoryStore()
    agent = make_agent(["One"], name="worker", store=store)
    agent.run(State(id="same", messages=[Message.user("Go")]))
    state = store.load("same")
    state.add_message(Message.user("More"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        agent.copy(model=FakeModel(["Two"])).run(state)  # same name "fake": the same model string


def test_other_calls_on_a_loaded_state_do_not_warn():
    """Only ``run``/``arun`` compare. ``think``, ``ask`` and ``compact`` with another Agent are not a resume."""
    state = saved_state(make_agent(tools=[search_web], name="saved"))
    other = make_agent(["A reply", "An answer"], tools=[search_docs], name="other")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        other.think(state)
        assert other.ask(state, "Anything else?") == "An answer"


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
    assert warning.changes == {"tools": (("search_web",), ())}
    assert caught.list[0].filename == __file__


async def test_arun_warns_only_once():
    state = saved_state(make_agent(tools=[search_web]))
    agent = make_agent(["First", "Second"])
    with pytest.warns(ResumeWarning):
        await agent.arun(state)
    state.add_message(Message.user("And now?"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert await agent.arun(state) == "Second"


def waiting_for_user(state: State) -> bool:
    if state.pending_calls or not state.messages:
        return False
    last = state.messages[-1]
    return last.role == "assistant" and not last.tool_calls


async def test_async_loop_warns():
    @loop(until=waiting_for_user, limit=5)
    async def my_loop(agent, state):
        await agent.athink(state)

    state = saved_state(make_agent(system="a"))
    with pytest.warns(ResumeWarning):
        await make_agent(system="b", loop=my_loop).arun(state)


def test_filter_error_raises_from_run():
    state = saved_state(make_agent(tools=[search_web]))
    fake = FakeModel(["Done"])
    agent = make_agent(model=fake)
    starts = len(state.history)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ResumeWarning)
        with pytest.raises(ResumeWarning) as raised:
            agent.run(state)
        assert raised.value.changes == {"tools": (("search_web",), ())}
        # The run did not start, so the next run refuses again.
        with pytest.raises(ResumeWarning):
            agent.run(state)
    assert fake.requests == []
    assert state.turn == 0
    assert len(state.history) == starts  # nothing was recorded, not even a run start
