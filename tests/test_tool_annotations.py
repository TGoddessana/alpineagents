"""Tool hints (read_only, destructive, idempotent, open_world) and agent.tool_map."""

from __future__ import annotations

import pytest
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from alpineagents import MCP, Agent, MCPTool, State, loop, tool
from alpineagents.testing import FakeModel, tool_call


def hints(t):
    return (t.read_only, t.destructive, t.idempotent, t.open_world)


def make_agent(replies, tools, **settings):
    settings.setdefault("reporter", None)
    settings.setdefault("human", None)
    return Agent(model=FakeModel(replies), tools=tools, **settings)


@tool(read_only=True, open_world=False)
def read_file(path: str) -> str:
    """Read a file"""
    return f"contents of {path}"


@tool(open_world=False)
def write_file(path: str, content: str) -> None:
    """Write a file"""


@tool
def bash(command: str) -> str:
    """Run a command"""
    return "ran"


# ---------------------------------------------------------------------------
# @tool hints
# ---------------------------------------------------------------------------


def test_hints_left_out_assume_the_worst():
    assert hints(bash) == (False, True, False, True)


def test_read_only_makes_a_tool_not_destructive_and_idempotent():
    assert hints(read_file) == (True, False, True, False)


def test_hints_as_given():
    @tool(destructive=False, idempotent=True)
    def add_label(issue: int, label: str) -> None:
        """Add a label"""

    assert hints(add_label) == (False, False, True, True)
    assert hints(write_file) == (False, True, False, False)


def test_read_only_with_consistent_hints_is_fine():
    @tool(read_only=True, destructive=False, idempotent=True)
    def look(x: str) -> str:
        """Look"""
        return x

    assert hints(look) == (True, False, True, True)


@pytest.mark.parametrize("hint", ["read_only", "destructive", "idempotent", "open_world"])
def test_a_hint_must_be_a_bool(hint):
    with pytest.raises(TypeError, match=f"{hint}= takes True or False") as exc:

        @tool(**{hint: "yes"})
        def f(x: str) -> str:
            """F"""
            return x

    assert "Fix:" in str(exc.value)


@pytest.mark.parametrize("contradiction", [{"destructive": True}, {"idempotent": False}])
def test_read_only_cannot_be_destructive_or_not_idempotent(contradiction):
    with pytest.raises(ValueError, match="contradict"):

        @tool(read_only=True, **contradiction)
        def f(x: str) -> str:
            """F"""
            return x


def test_hints_on_methods_stay_on_the_bound_tool():
    class Files:
        @tool(read_only=True)
        def read(self, path: str) -> str:
            """Read"""
            return path

    assert Files().read.read_only


def test_copy_changes_hints_and_fills_in_the_rest_again():
    safe = bash.copy(open_world=False)
    assert hints(safe) == (False, True, False, False)
    assert hints(bash) == (False, True, False, True)
    # Not given in @tool, so read_only=True sets them.
    assert hints(bash.copy(read_only=True)) == (True, False, True, True)
    with pytest.raises(ValueError, match="contradict"):
        bash.copy(read_only=True, destructive=True)


def test_the_model_does_not_see_the_hints():
    seen = []

    def reply(request):
        seen.extend(request.tools)
        return "done"

    make_agent([reply], [read_file]).run("Task")
    assert seen[0].input_schema == read_file.input_schema
    assert "read_only" not in str(seen[0])


# ---------------------------------------------------------------------------
# agent.tool_map
# ---------------------------------------------------------------------------


class Files:
    @tool(read_only=True)
    def list_files(self) -> list[str]:
        """List files"""
        return []

    @tool
    def delete_file(self, path: str) -> None:
        """Delete a file"""


def test_tool_map_has_every_tool_by_name():
    files = Files()
    agent = make_agent([], [read_file, files, bash])
    assert list(agent.tool_map) == ["read_file", "list_files", "delete_file", "bash"]
    assert agent.tool_map["read_file"] is read_file
    assert agent.tool_map["list_files"].bound_to is files
    assert len(agent.tool_map) == 4 and "bash" in agent.tool_map


def test_tool_map_is_read_only():
    agent = make_agent([], [read_file])
    with pytest.raises(TypeError):
        agent.tool_map["x"] = bash  # type: ignore[index]


def test_a_missing_name_is_a_plain_key_error_without_mcp():
    agent = make_agent([], [read_file])
    assert agent.tool_map.get("made_up") is None
    with pytest.raises(KeyError):
        agent.tool_map["made_up"]


def test_approval_loop_reads_the_hints():
    asked = []

    @loop(until=State.is_answered, limit=10)
    def careful(agent: Agent, state: State):
        agent.think(state)
        for call in state.pending_calls:
            tool_ = agent.tool_map.get(call.name)
            if tool_ is not None and not tool_.read_only:
                asked.append(call.name)
                state.deny(call, "The user declined")
        if state.wants_tools():
            agent.use_tools(state)

    fake = FakeModel([[tool_call("read_file", path="a"), tool_call("bash", command="rm -rf /")], "done"])
    state = State("Task")
    Agent(model=fake, tools=[read_file, bash], loop=careful, reporter=None, human=None).run(state)

    assert asked == ["bash"]
    results = [(e.kind, e.call.name) for e in state.history if e.call is not None]
    assert results == [("denied", "bash"), ("tool_result", "read_file")]  # denied first, then use_tools runs


def test_think_with_the_read_only_tools_of_the_map():
    seen = []

    def reply(request):
        seen.append([spec.name for spec in request.tools])
        return "done"

    agent = make_agent([reply], [read_file, Files(), bash])
    state = State("Task")
    agent.think(state, tools=[t for t in agent.tool_map.values() if t.read_only])
    assert seen == [["read_file", "list_files"]]


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


def make_server() -> MCPServer:
    server = MCPServer("demo")

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
    def get_issue(number: int) -> str:
        """Get an issue"""
        return f"issue {number}"

    @server.tool(annotations=ToolAnnotations(destructive_hint=False, idempotent_hint=True))
    def add_label(number: int, label: str) -> str:
        """Add a label"""
        return "ok"

    @server.tool()
    def delete_repo(name: str) -> str:
        """Delete a repository"""
        return "gone"

    return server


@pytest.fixture
def github() -> MCP:
    return MCP(server=make_server(), name="github")


def test_mcp_tools_get_the_servers_hints(github):
    agent = make_agent([], [read_file, github])
    with agent:
        tools = agent.tool_map
        assert list(tools) == ["read_file", "github__get_issue", "github__add_label", "github__delete_repo"]
        assert isinstance(tools["github__get_issue"], MCPTool)
        assert hints(tools["github__get_issue"]) == (True, False, True, False)
        assert hints(tools["github__add_label"]) == (False, False, True, True)
        assert hints(tools["github__delete_repo"]) == (False, True, False, True)
        assert tools["github__get_issue"].server is github
        assert tools["github__get_issue"].remote_name == "get_issue"


def test_mcp_tools_are_listed_only_while_connected(github):
    agent = make_agent([], [read_file, github])
    assert list(agent.tool_map) == ["read_file"]
    assert "github__get_issue" not in agent.tool_map
    assert agent.tool_map.get("github__get_issue") is None
    with pytest.raises(KeyError) as exc:
        agent.tool_map["github__get_issue"]
    message = str(exc.value)
    assert "MCP servers (github) are not connected" in message and "with agent:" in message
    # A map read while connected does not change after disconnecting, but a new read does.
    with agent:
        connected = agent.tool_map
    assert "github__get_issue" in connected
    assert "github__get_issue" not in agent.tool_map


def test_tool_map_during_a_run_has_the_mcp_tools(github):
    seen = []

    @loop(until=State.is_answered, limit=5)
    def look(agent: Agent, state: State):
        seen.append(agent.tool_map["github__get_issue"].read_only)
        agent.think(state)

    Agent(model=FakeModel(["done"]), tools=[github], loop=look, reporter=None, human=None).run("Task")
    assert seen == [True]


def test_think_takes_mcp_tools_from_the_map(github):
    seen = []

    def reply(request):
        seen.append([spec.name for spec in request.tools])
        return "done"

    agent = make_agent([reply], [read_file, github])
    with agent:
        readers = [t for t in agent.tool_map.values() if t.read_only]
        agent.think(State("Task"), tools=readers)
    assert seen == [["read_file", "github__get_issue"]]


def test_think_rejects_an_mcp_tool_of_another_agent(github):
    one = make_agent([], [github])
    other = make_agent(["done"], [github])
    with one, other:
        foreign = one.tool_map["github__get_issue"]
        assert foreign is not other.tool_map["github__get_issue"]
        with pytest.raises(ValueError, match="not a connected MCP tool of this Agent"):
            other.think(State("Task"), tools=[foreign])
