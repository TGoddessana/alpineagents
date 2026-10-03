"""Tool subclasses: tools that are not Python functions."""

from __future__ import annotations

import pytest
from mcp.server.mcpserver import MCPServer

from alpineagents import (
    MCP,
    Agent,
    FunctionTool,
    MCPTool,
    Message,
    Reporter,
    State,
    StoppedByUntil,
    Tool,
    ToolError,
    ToolInputError,
    tool,
)
from alpineagents.testing import FakeModel, tool_call
from alpineagents.types import INVALID_ARGS_KEY, ToolCall, ToolResultBlock

TICKET_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "priority": {"type": "integer"}},
    "required": ["title"],
}


class Ticket(Tool):
    """Stands in for a tool built from a row in a database."""

    def __init__(self, name: str = "create_ticket", **options) -> None:
        super().__init__(name=name, description="Create a ticket", input_schema=TICKET_SCHEMA, **options)
        self.calls: list[tuple[dict, State]] = []

    def run(self, args: dict, state: State) -> dict:
        self.calls.append((args, state))
        if "title" not in args:
            raise ToolInputError("title is required")
        if args["title"] == "down":
            raise ToolError("the ticket service is down")
        if args["title"] == "bug":
            raise KeyError("a bug in the tool")
        return {"id": 7, "title": args["title"]}


class AsyncTicket(Ticket):
    async def run(self, args: dict, state: State) -> str:
        return f"async {args['title']}"


def make_agent(replies, tools, **settings):
    settings.setdefault("reporter", None)
    settings.setdefault("human", None)
    return Agent(model=FakeModel(replies), tools=tools, **settings)


def run_once(t, **args):
    state = State(messages=[Message.user("Task")])
    make_agent([tool_call(t.name, **args), "done"], [t]).run(state)
    return state


def block(state) -> ToolResultBlock:
    return next(b for m in state.messages for b in m.content if isinstance(b, ToolResultBlock))


# ---------------------------------------------------------------------------
# Making one
# ---------------------------------------------------------------------------


def test_what_the_model_sees():
    ticket = Ticket()
    assert (ticket.spec.name, ticket.spec.description, ticket.spec.input_schema) == (
        "create_ticket", "Create a ticket", TICKET_SCHEMA,
    )
    assert ticket.parallel and repr(ticket) == "Ticket('create_ticket')"
    assert (ticket.read_only, ticket.destructive, ticket.idempotent, ticket.open_world) == (False, True, False, True)


def test_hints_and_parallel():
    ticket = Ticket(read_only=True, open_world=False, parallel=False)
    assert (ticket.read_only, ticket.destructive, ticket.idempotent, ticket.open_world) == (True, False, True, False)
    assert not ticket.parallel


def test_a_tool_without_arguments():
    class Now(Tool):
        def __init__(self) -> None:
            super().__init__(name="now", description="The time")

        def run(self, args: dict, state: State) -> str:
            return "noon"

    assert Now().input_schema == {"type": "object", "properties": {}}
    assert block(run_once(Now())).content == "noon"


def test_every_tool_is_a_tool():
    @tool
    def f(x: str) -> str:
        """F"""
        return x

    assert isinstance(f, FunctionTool) and isinstance(f, Tool)
    assert issubclass(MCPTool, Tool)
    assert repr(f) == "FunctionTool('f')"


def test_is_async_is_a_read_only_property_on_every_tool():
    # It was a method in 0.4: ``tool.is_async()`` became ``tool.is_async``.
    @tool
    def sync_fn(x: str) -> str:
        """Sync"""
        return x

    @tool
    async def async_fn(x: str) -> str:
        """Async"""
        return x

    for cls in (Tool, FunctionTool, MCPTool):
        assert isinstance(cls.is_async, property)
        assert cls.is_async.fset is None
    assert sync_fn.is_async is False and async_fn.is_async is True
    assert Ticket().is_async is False and AsyncTicket().is_async is True
    with pytest.raises(AttributeError):
        sync_fn.is_async = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Running one
# ---------------------------------------------------------------------------


def test_the_result_is_converted_like_a_tool_function():
    ticket = Ticket()
    state = run_once(ticket, title="hello")
    assert (block(state).content, block(state).is_error) == ('{"id": 7, "title": "hello"}', False)
    args, seen_state = ticket.calls[0]
    assert args == {"title": "hello"} and seen_state is state


def test_run_gets_a_copy_of_the_arguments():
    class Mutating(Ticket):
        def run(self, args: dict, state: State) -> str:
            args["title"] = "changed"
            return "ok"

    state = run_once(Mutating(), title="hello")
    assert state.messages[1].tool_calls[0].args == {"title": "hello"}


@pytest.mark.parametrize(
    "args, content, kind",
    [
        ({}, "(input error: title is required)", "input_error"),
        ({"title": "down"}, "the ticket service is down", "error"),
    ],
)
def test_tool_input_error_and_tool_error(args, content, kind):
    kinds = []

    class Outcomes(Reporter):
        def on_tool_end(self, state, call, result, outcome):
            kinds.append(outcome.kind)

    state = State(messages=[Message.user("Task")])
    make_agent([tool_call("create_ticket", **args), "done"], [Ticket()], reporter=Outcomes()).run(state)
    assert (block(state).content, block(state).is_error) == (content, True)
    assert kinds == [kind] and state.stopped == StoppedByUntil("is_answered")


def test_other_exceptions_stop_the_run():
    state = State(messages=[Message.user("Task")])
    with pytest.raises(KeyError):
        make_agent([tool_call("create_ticket", title="bug"), "done"], [Ticket()]).run(state)
    assert block(state).content == "(aborted: KeyError)"


def test_arguments_that_are_not_json_never_reach_run():
    ticket = Ticket()
    broken = ToolCall(name="create_ticket", args={INVALID_ARGS_KEY: '{"title": '}, id="c1")
    state = State(messages=[Message.user("Task")])
    make_agent([broken, "done"], [ticket]).run(state)
    assert block(state).content == '(input error: arguments are not valid JSON: {"title": )'
    assert ticket.calls == []


def test_async_run_in_a_sync_run():
    assert block(run_once(AsyncTicket(), title="x")).content == "async x"


async def test_async_run_in_arun():
    state = State(messages=[Message.user("Task")])
    await make_agent([tool_call("create_ticket", title="x"), "done"], [AsyncTicket()]).arun(state)
    assert block(state).content == "async x"


async def test_sync_run_in_arun():
    state = State(messages=[Message.user("Task")])
    await make_agent([tool_call("create_ticket", title="x"), "done"], [Ticket()]).arun(state)
    assert block(state).content == '{"id": 7, "title": "x"}'


def test_tool_map_think_and_hints_work_the_same():
    @tool(read_only=True)
    def search(q: str) -> str:
        """Search"""
        return q

    ticket = Ticket()
    agent = make_agent([], [search, ticket])
    assert agent.tool_map["create_ticket"] is ticket
    assert [t.name for t in agent.tool_map.values() if not t.read_only] == ["create_ticket"]

    seen = []
    agent = make_agent([lambda request: seen.append([s.name for s in request.tools]) or "done"], [search, ticket])
    agent.think(State(messages=[Message.user("Task")]), tools=[ticket])
    assert seen == [["create_ticket"]]


# ---------------------------------------------------------------------------
# Mistakes, caught before any model request
# ---------------------------------------------------------------------------


def test_forgetting_super_init():
    class NoInit(Tool):
        def __init__(self) -> None:
            pass

        def run(self, args: dict, state: State) -> str:
            return ""

    with pytest.raises(TypeError, match=r"did not call Tool.__init__") as exc:
        make_agent([], [NoInit()])
    assert "super().__init__" in str(exc.value)


def test_forgetting_run():
    class NoRun(Tool):
        def __init__(self) -> None:
            super().__init__(name="no_run")

    with pytest.raises(TypeError, match="does not implement run"):
        make_agent([], [NoRun()])


def test_passing_the_class():
    with pytest.raises(TypeError, match="got the class Ticket"):
        make_agent([], [Ticket])


@pytest.mark.parametrize(
    "options, error, match",
    [
        ({"name": "has space"}, TypeError, "cannot be used as a tool name"),
        ({"name": "x", "input_schema": {"type": "array"}}, TypeError, "must be a JSON Schema object"),
        ({"name": "x", "input_schema": "{}"}, TypeError, "must be a JSON Schema object"),
        ({"name": "x", "description": None}, TypeError, "description= takes a string"),
        ({"name": "x", "parallel": "no"}, TypeError, "parallel= takes True or False"),
        ({"name": "x", "read_only": True, "destructive": True}, ValueError, "contradict"),
    ],
)
def test_bad_options(options, error, match):
    with pytest.raises(error, match=match):
        Tool(**options)


def test_duplicate_names_across_kinds():
    @tool
    def create_ticket(title: str) -> str:
        """Create"""
        return title

    with pytest.raises(ValueError, match="Duplicate tool name 'create_ticket'"):
        make_agent([], [create_ticket, Ticket()])


def test_tool_decorator_on_a_tool_object():
    with pytest.raises(TypeError, match="applied twice"):
        tool(Ticket())


def test_an_mcp_tool_of_another_agent_is_not_a_tool_to_pass():
    server = MCPServer("demo")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add"""
        return a + b

    demo = MCP(server=server, name="demo")
    one = make_agent([], [demo])
    with one:
        connected = one.tool_map["demo__add"]
    with pytest.raises(TypeError, match="Pass the server, or pick the tool from it"):
        make_agent([], [connected])
