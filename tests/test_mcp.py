"""MCP tests: ``MCP(...)`` in ``tools=``, connection lifetime, tool names, error results, and the stdio and HTTP
transports (a local demo server, no network)."""

from __future__ import annotations

import asyncio
import logging
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from alpineagents import MCP, Agent, MCPConnectionError, Reporter, State, tool
from alpineagents.testing import FakeModel, tool_call
from alpineagents.types import ToolResultBlock

logging.getLogger("mcp").setLevel(logging.CRITICAL)

DEMO_SERVER = Path(__file__).with_name("mcp_demo_server.py")


def make_server() -> MCPServer:
    server = MCPServer("demo")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    @server.tool()
    def fail(message: str) -> str:
        """Always fails"""
        raise ToolError(message)

    @server.tool(name="search-code")
    def search_code(query: str) -> str:
        """Search code"""
        return f"found {query}"

    return server


@pytest.fixture
def demo() -> MCP:
    return MCP(server=make_server(), name="demo")


def make_agent(replies, tools, **settings) -> Agent:
    settings.setdefault("reporter", None)
    settings.setdefault("human", None)
    return Agent(model=FakeModel(replies), tools=tools, **settings)


def results(state: State) -> list[tuple[str, str]]:
    return [(e.call.name, e.content) for e in state.history if e.kind == "tool_result"]


class Outcomes(Reporter):
    def __init__(self) -> None:
        self.ends: list[tuple[str, str]] = []

    def on_tool_end(self, state, call, result, outcome):
        self.ends.append((call.name, outcome.kind))


# ================================================================ MCP(...) mistakes


def test_mcp_needs_exactly_one_server():
    with pytest.raises(TypeError, match="exactly one server"):
        MCP(name="x")
    with pytest.raises(TypeError, match="got: command, url"):
        MCP("npx server", url="https://example.com/mcp", name="x")


def test_mcp_needs_a_name():
    with pytest.raises(ValueError, match="needs name="):
        MCP("npx -y @modelcontextprotocol/server-github")


@pytest.mark.parametrize("name", ["a__b", "_x", "has space", "", "dot.name"])
def test_mcp_name_rule(name):
    with pytest.raises(ValueError, match="not allowed"):
        MCP("npx server", name=name)


def test_env_and_headers_belong_to_their_transport():
    with pytest.raises(TypeError, match="env="):
        MCP(url="https://example.com/mcp", name="x", env={"A": "1"})
    with pytest.raises(TypeError, match="headers="):
        MCP("npx server", name="x", headers={"A": "1"})


def test_two_servers_with_the_same_name_are_an_error_at_agent_creation():
    with pytest.raises(ValueError, match="Two MCP servers are named 'github'"):
        make_agent([], [MCP("a", name="github"), MCP("b", name="github")])


def test_creating_an_agent_does_not_connect(demo):
    make_agent([], [demo])
    assert demo._conn is None


# ================================================================ calling tools


def test_run_calls_prefixed_tools_and_closes_the_connection(demo):
    outcomes = Outcomes()
    agent = make_agent(
        [[tool_call("demo__add", a=1, b=2), tool_call("demo__fail", message="nope")], "It is 3"],
        [demo],
        reporter=outcomes,
    )
    state = State("1 + 2?")
    assert agent.run(state) == "It is 3"
    assert sorted(results(state)) == [("demo__add", "3"), ("demo__fail", "Error executing tool fail: nope")]
    blocks = {b.name: b for m in state.context for b in m.content if isinstance(b, ToolResultBlock)}
    assert blocks["demo__fail"].is_error and not blocks["demo__add"].is_error
    assert sorted(outcomes.ends) == [("demo__add", "done"), ("demo__fail", "error")]
    assert [s.name for s in agent.model.requests[0].tools] == ["demo__add", "demo__fail", "demo__search-code"]
    assert demo._conn is None


def test_pick_tools_with_attribute_and_item(demo):
    agent = make_agent([tool_call("demo__search-code", query="x"), "done"], [demo.add, demo["search-code"]])
    state = State("Find x")
    agent.run(state)
    assert [s.name for s in agent.model.requests[0].tools] == ["demo__add", "demo__search-code"]
    assert results(state) == [("demo__search-code", "found x")]


def test_a_picked_tool_that_does_not_exist_is_an_error_on_connect(demo):
    state = State("Task")
    with pytest.raises(ValueError, match="has no tool 'serch'"):
        make_agent(["x"], [demo.serch]).run(state)
    assert state.history[-1].kind == "error"
    assert demo._conn is None


def test_a_name_collision_with_another_tool_is_an_error_on_connect(demo):
    @tool(name="demo__add")
    def other_add(a: int) -> int:
        """Another add"""
        return a

    with pytest.raises(ValueError, match="Duplicate tool name 'demo__add'"):
        make_agent(["x"], [demo, other_add]).run("Task")
    assert demo._conn is None


def test_missing_required_argument_is_an_input_error(demo):
    state = State("Task")
    make_agent([tool_call("demo__add", a=1), "done"], [demo]).run(state)
    assert results(state) == [("demo__add", "(input error: missing required arguments: b)")]


def test_mixed_with_regular_tools(demo):
    @tool
    def double(n: int) -> int:
        """Double a number"""
        return n * 2

    state = State("Task")
    make_agent([[tool_call("double", n=2), tool_call("demo__add", a=2, b=2)], "4"], [double, demo]).run(state)
    assert sorted(results(state)) == [("demo__add", "4"), ("double", "4")]


def test_think_can_show_only_some_mcp_tools(demo):
    agent = make_agent(["ok"], [demo])
    state = State("Task")
    with agent:
        agent.think(state, tools=[demo.add])
    assert [s.name for s in agent.model.requests[0].tools] == ["demo__add"]


def test_repr_lists_servers(demo):
    assert "MCP(" in repr(make_agent([], [demo]))


# ================================================================ connection lifetime


def test_with_agent_keeps_one_connection_across_runs(demo):
    agent = make_agent([tool_call("demo__add", a=1, b=1), "2", tool_call("demo__add", a=2, b=2), "4"], [demo])
    with agent:
        conn = demo._conn
        assert conn is not None
        agent.run("First")
        agent.run("Second")
        assert demo._conn is conn
    assert demo._conn is None


def test_think_outside_run_connects_on_first_use(demo):
    agent = make_agent([tool_call("demo__add", a=1, b=1)], [demo])
    state = State("Task")
    agent.think(state)
    agent.use_tools(state)
    assert results(state) == [("demo__add", "2")]
    assert demo._conn is not None
    agent._mcp_close()  # the sticky use
    assert demo._conn is None


def test_two_agents_share_one_connection(demo):
    first = make_agent(["a"], [demo])
    second = make_agent(["b"], [demo.add])
    with first:
        conn = demo._conn
        with second:
            assert demo._conn is conn
        assert demo._conn is conn
    assert demo._conn is None


async def test_arun_and_async_with(demo):
    agent = make_agent([tool_call("demo__add", a=3, b=4), "7"], [demo])
    async with agent:
        state = State("3 + 4?")
        assert await agent.arun(state) == "7"
        assert demo._conn is not None
    assert results(state) == [("demo__add", "7")]
    assert demo._conn is None


async def test_concurrent_aruns_share_the_connection(demo):
    agent = Agent(
        model=FakeModel([tool_call("demo__add", a=1, b=1), tool_call("demo__add", a=1, b=1), "2", "2"]),
        tools=[demo],
        reporter=None,
        human=None,
    )
    states = [State("A"), State("B")]
    await asyncio.gather(*(agent.arun(s) for s in states))
    assert [results(s) for s in states] == [[("demo__add", "2")], [("demo__add", "2")]]
    assert demo._conn is None


async def test_cancelled_arun_closes_the_connection():
    server = MCPServer("slow")

    @server.tool()
    async def wait() -> str:
        """Waits"""
        await asyncio.sleep(10)
        return "never"

    slow = MCP(server=server, name="slow")
    task = asyncio.create_task(make_agent([tool_call("slow__wait")], [slow]).arun("Task"))
    for _ in range(100):
        if slow._conn is not None and slow._conn.client is not None:
            break
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert slow._conn is None


async def test_cancelling_one_agent_s_connect_does_not_break_another_sharing_the_server():
    slow = MCP(f"{sys.executable} {DEMO_SERVER}", name="slow", env={"START_DELAY": "0.5"})
    first = make_agent(["a"], [slow])
    second = make_agent([tool_call("slow__add", a=1, b=2), "3"], [slow])
    first_task = asyncio.create_task(first.arun("A"))
    second_task = asyncio.create_task(second.arun("B"))
    await asyncio.sleep(0.2)
    first_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_task
    assert await second_task == "3"
    assert slow._conn is None


def test_a_result_the_sdk_rejects_goes_to_the_model_and_keeps_the_connection():
    from mcp_types import CallToolResult, TextContent

    server = MCPServer("strict")

    @server.tool()
    def count() -> int:
        """Returns a number (but does not)"""
        return CallToolResult(content=[TextContent(type="text", text="many")], structured_content={"result": "many"})

    @server.tool()
    def one() -> int:
        """Returns 1"""
        return 1

    # The server stops checking its own output (the schema is still advertised), so the client SDK rejects it.
    server._tool_manager._tools["count"].fn_metadata.output_model = None
    strict = MCP(server=server, name="strict")
    state = State("Task")
    make_agent([tool_call("strict__count"), tool_call("strict__one"), "done"], [strict]).run(state)
    (first, second) = results(state)
    assert first[1].startswith("(error from MCP server 'strict'")
    assert second == ("strict__one", "1")


# ================================================================ stdio and HTTP transports


def test_stdio_server_with_env():
    local = MCP(f"{sys.executable} {DEMO_SERVER}", name="local", env={"GREETING": "hi"})
    state = State("Task")
    make_agent([tool_call("local__greeting"), "done"], [local]).run(state)
    assert results(state) == [("local__greeting", "hi")]


def test_lost_connection_raises():
    local = MCP(f"{sys.executable} {DEMO_SERVER}", name="local")
    state = State("Task")
    with pytest.raises(MCPConnectionError, match="Lost the connection"):
        make_agent([tool_call("local__crash"), "never"], [local]).run(state)
    assert not state.pending_calls
    assert local._conn is None


def test_a_server_that_cannot_start_raises_on_run():
    state = State("Task")
    with pytest.raises(MCPConnectionError, match="Could not connect to MCP server 'bad'"):
        make_agent(["x"], [MCP("definitely-not-a-command-alpineagents", name="bad")]).run(state)
    assert state.history[-1].kind == "error"


def test_streamable_http_server_with_headers():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, str(DEMO_SERVER), "http", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                socket.create_connection(("127.0.0.1", port), 0.1).close()
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)
        remote = MCP(url=f"http://127.0.0.1:{port}/mcp", name="remote", headers={"X-Test": "1"})
        state = State("Task")
        make_agent([tool_call("remote__add", a=4, b=5), "9"], [remote]).run(state)
        assert results(state) == [("remote__add", "9")]
    finally:
        proc.terminate()
        proc.wait()


def test_importing_alpineagents_does_not_import_mcp():
    code = "import sys, alpineagents; print('mcp' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"
