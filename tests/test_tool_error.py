"""ToolError and @tool(exception_handler=...): failures the model sees as error results while the run continues.

Like test_tool.py, this file does not use ``from __future__ import annotations``, so handlers defined inside test
functions can use exception classes defined there in their type hints.
"""

import pytest

from alpineagents import Agent, FileStore, Reporter, State, ToolError, tool
from alpineagents.models.openai_compatible import OpenAICompatible
from alpineagents.testing import FakeModel, tool_call
from alpineagents.types import Message, Request, ToolCall, ToolResultBlock


class HTTPError(Exception):
    """Stands in for a library's exception, such as httpx.HTTPError."""


class NotFound(HTTPError):
    pass


def http_errors(error: HTTPError) -> str:
    return f"request failed: {error}"


def make_agent(fake, tools, **settings):
    return Agent(model=fake, tools=tools, reporter=None, human=None, **settings)


def run_once(t, **args):
    """The model calls ``t`` once and then answers. Returns the State."""
    state = State("Task")
    make_agent(FakeModel([tool_call(t.name, **args), "done"]), [t]).run(state)
    return state


def result_entry(state):
    return next(entry for entry in state.history if entry.kind == "tool_result")


def result_block(state):
    return next(
        block
        for message in state.context
        for block in message.content
        if isinstance(block, ToolResultBlock)
    )


# ---------------------------------------------------------------------------
# ToolError
# ---------------------------------------------------------------------------


def test_tool_error_becomes_an_error_result_and_the_run_continues():
    @tool
    def fetch(url: str) -> str:
        """Fetch"""
        raise ToolError(f"HTTP 404: {url}")

    state = run_once(fetch, url="https://x.dev/missing")

    block = result_block(state)
    assert (block.content, block.is_error) == ("HTTP 404: https://x.dev/missing", True)
    assert state.stopped_by == "is_answered"
    assert state.answer == "done"
    assert not [entry for entry in state.history if entry.kind == "error"]


def test_the_history_entry_keeps_the_tool_error():
    @tool
    def fetch(url: str) -> str:
        """Fetch"""
        raise ToolError("HTTP 404")

    entry = result_entry(run_once(fetch, url="u"))
    assert entry.is_error
    assert isinstance(entry.error, ToolError) and str(entry.error) == "HTTP 404"


async def test_tool_error_from_an_async_tool():
    @tool
    async def fetch(url: str) -> str:
        """Fetch"""
        raise ToolError("HTTP 500")

    state = State("Task")
    await make_agent(FakeModel([tool_call("fetch", url="u"), "done"]), [fetch]).arun(state)
    assert (result_block(state).content, result_block(state).is_error) == ("HTTP 500", True)
    assert state.stopped_by == "is_answered"


async def test_tool_error_from_a_sync_tool_in_arun():
    @tool
    def fetch(url: str) -> str:
        """Fetch"""
        raise ToolError("HTTP 500")

    state = State("Task")
    await make_agent(FakeModel([tool_call("fetch", url="u"), "done"]), [fetch]).arun(state)
    assert (result_block(state).content, result_block(state).is_error) == ("HTTP 500", True)


def test_other_calls_of_the_turn_are_unaffected():
    @tool
    def fetch(url: str) -> str:
        """Fetch"""
        if url == "bad":
            raise ToolError("HTTP 404")
        return "page"

    fake = FakeModel([[tool_call("fetch", url="bad"), tool_call("fetch", url="good")], "done"])
    state = State("Task")
    make_agent(fake, [fetch]).run(state)
    results = [block for block in state.context[2].content]
    assert [(b.content, b.is_error) for b in results] == [("HTTP 404", True), ("page", False)]


def test_reporter_gets_the_error_outcome():
    seen = []

    class Recorder(Reporter):
        def on_tool_end(self, state, call, result, outcome):
            seen.append((result, outcome.kind))

    @tool
    def fetch(url: str) -> str:
        """Fetch"""
        raise ToolError("HTTP 404")

    agent = Agent(model=FakeModel([tool_call("fetch", url="u"), "done"]), tools=[fetch], reporter=Recorder())
    agent.run("Task")
    assert seen == [("HTTP 404", "error")]


def test_state_str_marks_error_results():
    @tool
    def fetch(url: str) -> str:
        """Fetch"""
        raise ToolError("HTTP 404: https://x.dev/missing")

    assert "[turn 1] tool_result fetch (error): HTTP 404: https://x.dev/missing" in str(run_once(fetch, url="u"))


@pytest.mark.parametrize("message, error", [("", ValueError), ("   ", ValueError), (404, TypeError)])
def test_tool_error_needs_a_message(message, error):
    with pytest.raises(error, match="ToolError"):
        ToolError(message)


# ---------------------------------------------------------------------------
# exception_handler
# ---------------------------------------------------------------------------


def test_handler_turns_its_exceptions_into_error_results():
    @tool(exception_handler=http_errors)
    def fetch(url: str) -> str:
        """Fetch"""
        raise NotFound(f"404 {url}")  # a subclass of the hinted HTTPError

    state = run_once(fetch, url="u")
    assert (result_block(state).content, result_block(state).is_error) == ("request failed: 404 u", True)
    assert state.stopped_by == "is_answered"
    entry = result_entry(state)
    assert isinstance(entry.error, ToolError) and isinstance(entry.error.__cause__, NotFound)


def test_handler_with_a_union_hint():
    def file_errors(error: FileNotFoundError | PermissionError) -> str:
        return f"cannot read: {error}"

    @tool(exception_handler=file_errors)
    def read(path: str) -> str:
        """Read"""
        raise PermissionError("denied")

    assert result_block(run_once(read, path="p")).content == "cannot read: denied"


def test_other_exceptions_still_stop_the_run():
    @tool(exception_handler=http_errors)
    def fetch(url: str) -> str:
        """Fetch"""
        raise KeyError("bug")

    state = State("Task")
    with pytest.raises(KeyError):
        make_agent(FakeModel([tool_call("fetch", url="u"), "done"]), [fetch]).run(state)
    assert result_block(state).content == "(aborted: KeyError)"


def test_handler_can_let_an_exception_stop_the_run_by_raising_it():
    def only_client_errors(error: HTTPError) -> str:
        if "500" in str(error):
            raise error
        return str(error)

    @tool(exception_handler=only_client_errors)
    def fetch(url: str) -> str:
        """Fetch"""
        raise HTTPError(url)

    assert result_block(run_once(fetch, url="404")).content == "404"
    with pytest.raises(HTTPError, match="500"):
        run_once(fetch, url="500")


def test_tool_error_is_not_passed_to_the_handler():
    calls = []

    def everything(error: Exception) -> str:
        calls.append(error)
        return "handled"

    @tool(exception_handler=everything)
    def fetch(url: str) -> str:
        """Fetch"""
        raise ToolError("from the tool")

    assert result_block(run_once(fetch, url="u")).content == "from the tool"
    assert calls == []


async def test_handler_on_an_async_tool():
    @tool(exception_handler=http_errors)
    async def fetch(url: str) -> str:
        """Fetch"""
        raise HTTPError("timeout")

    state = State("Task")
    await make_agent(FakeModel([tool_call("fetch", url="u"), "done"]), [fetch]).arun(state)
    assert (result_block(state).content, result_block(state).is_error) == ("request failed: timeout", True)


def test_handler_on_a_method_tool():
    class Web:
        def __init__(self, base: str) -> None:
            self.base = base

        @tool(exception_handler=http_errors)
        def fetch(self, path: str) -> str:
            """Fetch"""
            raise HTTPError(self.base + path)

    state = State("Task")
    make_agent(FakeModel([tool_call("fetch", path="/a"), "done"]), [Web("https://x.dev")]).run(state)
    assert result_block(state).content == "request failed: https://x.dev/a"


@pytest.mark.parametrize("returned", [None, "", "  ", 404])
def test_handler_must_return_a_message(returned):
    def bad(error: HTTPError):
        return returned

    @tool(exception_handler=bad)
    def fetch(url: str) -> str:
        """Fetch"""
        raise HTTPError("x")

    with pytest.raises(TypeError, match="returned") as exc:
        run_once(fetch, url="u")
    assert isinstance(exc.value.__cause__, HTTPError)


# ---------------------------------------------------------------------------
# exception_handler: mistakes caught when @tool runs
# ---------------------------------------------------------------------------


def no_hint(error):
    return "x"


def not_an_exception(error: str) -> str:
    return "x"


def interrupts(error: KeyboardInterrupt) -> str:
    return "x"


def mixed(error: HTTPError | KeyboardInterrupt) -> str:
    return "x"


async def async_handler(error: HTTPError) -> str:
    return "x"


def two_params(error: HTTPError, other: str) -> str:
    return "x"


def keyword_only(*, error: HTTPError) -> str:
    return "x"


@pytest.mark.parametrize(
    "handler, match",
    [
        (no_hint, "no type hint"),
        (not_an_exception, "must name exception classes"),
        (interrupts, "cannot be handled"),
        (mixed, "cannot be handled"),
        (async_handler, "is async"),
        (two_params, "exactly one argument"),
        (keyword_only, "exactly one argument"),
        ("http_errors", "is not a function"),
        (HTTPError, "is not a function"),
    ],
)
def test_bad_handlers_raise_when_tool_is_applied(handler, match):
    with pytest.raises(TypeError, match=match) as exc:

        @tool(exception_handler=handler)
        def fetch(url: str) -> str:
            """Fetch"""
            return url

    assert "Fix:" in str(exc.value)


def test_handler_with_extra_defaulted_parameters_and_callable_objects():
    def with_default(error: HTTPError, prefix: str = "failed") -> str:
        return f"{prefix}: {error}"

    class Handler:
        def __call__(self, error: HTTPError) -> str:
            return f"object: {error}"

    for handler, expected in ((with_default, "failed: x"), (Handler(), "object: x")):

        @tool(exception_handler=handler)
        def fetch(url: str) -> str:
            """Fetch"""
            raise HTTPError("x")

        assert result_block(run_once(fetch, url="u")).content == expected


# ---------------------------------------------------------------------------
# Tool.copy
# ---------------------------------------------------------------------------


def test_copy_without_the_handler_lets_exceptions_stop_the_run():
    @tool(exception_handler=http_errors)
    def fetch(url: str) -> str:
        """Fetch"""
        raise HTTPError("x")

    strict = fetch.copy(exception_handler=None)
    assert strict.exception_handler is None and fetch.exception_handler is http_errors
    assert (strict.name, strict.input_schema) == (fetch.name, fetch.input_schema)
    with pytest.raises(HTTPError):
        run_once(strict, url="u")
    assert result_block(run_once(fetch, url="u")).is_error


def test_copy_keeps_a_bound_tool_bound():
    class Web:
        @tool
        def fetch(self, url: str) -> str:
            """Fetch"""
            raise HTTPError(url)

    web = Web()
    handled = web.fetch.copy(exception_handler=http_errors, name="get")
    assert handled.bound_to is web and handled.name == "get"
    state = State("Task")
    make_agent(FakeModel([tool_call("get", url="u"), "done"]), [handled]).run(state)
    assert result_block(state).content == "request failed: u"


def test_copy_rejects_unknown_options():
    @tool
    def fetch(url: str) -> str:
        """Fetch"""
        return url

    with pytest.raises(TypeError, match="unknown options: handler"):
        fetch.copy(handler=http_errors)


# ---------------------------------------------------------------------------
# What the model and the store get
# ---------------------------------------------------------------------------


def test_openai_compatible_marks_error_results():
    # OpenAI's tool message has no error flag, so the text says it.
    call = ToolCall(name="fetch", args={"url": "u"}, id="c1")
    other = ToolCall(name="fetch", args={"url": "v"}, id="c2")
    request = Request(
        None,
        (
            Message.user("Task"),
            Message("assistant", (call, other)),
            Message("user", (ToolResultBlock("c1", "HTTP 404", "fetch", is_error=True), ToolResultBlock("c2", "ok"))),
        ),
        (),
    )
    messages = OpenAICompatible("gpt-5", api_key="k")._request_kwargs(request)["messages"]
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert [m["content"] for m in tool_messages] == ["Error: HTTP 404", "ok"]


def test_store_keeps_the_error_result(tmp_path):
    @tool
    def fetch(url: str) -> str:
        """Fetch"""
        raise ToolError("HTTP 404")

    store = FileStore(tmp_path)
    state = State("Task")
    make_agent(FakeModel([tool_call("fetch", url="u"), "done"]), [fetch], store=store).run(state)

    loaded = store.load(state.id)
    entry = result_entry(loaded)
    assert (entry.content, entry.is_error, entry.error) == ("HTTP 404", True, None)
    assert result_block(loaded).is_error
