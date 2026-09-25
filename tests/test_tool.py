"""Black-box pytest: checks ARCHITECTURE.md "Tool contract" (creating tools, returning and failing, concurrency
and State) and the @tool rows of "Mistake-proofing errors".

Each test carries a comment quoting the spec sentence it checks, and observes behavior only through ARCHITECTURE.md
and the public API of src/alpineagents (``Agent``, ``State``, ``@tool``, ``alpineagents.testing``).
It does not look at implementation internals. Things outside the MVP (Image/File, subagents, MCP, skills) are not
covered, and no real model API is ever called (FakeModel only, reporter=None).

Note: this file deliberately does not use ``from __future__ import annotations``. That lets tests use
dataclass/TypedDict/Pydantic models defined locally inside test functions directly as @tool type hints
(string hints would be resolved later and fail to find the local names).
"""

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import pytest
from pydantic import BaseModel

# On Python 3.11 (the minimum version), Pydantic does not accept typing.TypedDict.
from typing_extensions import TypedDict

from alpineagents import Agent, State, tool
from alpineagents.testing import FakeModel, tool_call
from alpineagents.types import ToolResultBlock


def make_agent(fake, tools):
    """A test Agent that prints nothing to the terminal (reporter=None, human=None)."""
    return Agent(model=fake, tools=tools, reporter=None, human=None)


def single_call_result(t, **args):
    """Calls one tool once and returns the result string the model receives."""
    fake = FakeModel([tool_call(t.name, **args), "done"])
    agent = make_agent(fake, [t])
    state = State("Task")
    agent.think(state)
    agent.use_tools(state)
    return state.context[-1].content[0].content


# ---------------------------------------------------------------------------
# Tool: creating tools
# ---------------------------------------------------------------------------


def test_tool_missing_type_hint_raises_type_error():
    # "Every parameter must have a type hint. If one is missing or the type is unsupported, @tool errors right away."
    # Mistake-proofing error: "tool parameter has no type hint -> when @tool is applied -> add a type hint.
    # To receive State, use state: State"
    with pytest.raises(TypeError) as exc:

        @tool
        def broken(x):
            """A tool with a parameter that has no type hint"""
            return x

    message = str(exc.value)
    assert "type hint" in message
    assert "Fix:" in message
    assert "state: State" in message


def test_tool_unsupported_parameter_type_raises_type_error():
    # "Supported types: str, int, float, bool, Literal[...], Enum, list[T], dict[str, T], T | None,
    # dataclass, TypedDict, Pydantic model." Any other type is an error.
    # Mistake-proofing error: "tool parameter type is unsupported -> when @tool is applied -> list of supported types"
    with pytest.raises(TypeError) as exc:

        @tool
        def broken(x: complex) -> str:
            """A tool that takes an unsupported type (complex)"""
            return str(x)

    message = str(exc.value)
    assert "unsupported" in message
    assert "dataclass" in message  # the list of supported types shows in the message
    assert "Pydantic" in message


def test_tool_hides_state_parameter_from_schema_regardless_of_name():
    # "Parameters whose type hint is State are hidden from the model... the parameter name does not matter."
    @tool
    def peek(msg: str, ctx: State) -> str:
        """A State parameter is left out of the schema whatever its name"""
        return msg

    props = peek.input_schema.get("properties", {})
    assert set(props) == {"msg"}


def test_tool_injects_current_state_regardless_of_parameter_name():
    # "The framework passes in the current State."
    seen = {}

    @tool
    def peek(anything_i_want: State) -> str:
        """The current State is injected regardless of the name"""
        seen["state"] = anything_i_want
        return "ok"

    fake = FakeModel([tool_call("peek"), "done"])
    agent = make_agent(fake, [peek])
    state = State("Task")
    agent.think(state)
    agent.use_tools(state)

    assert seen["state"] is state


def test_tool_description_and_arg_docs_come_from_docstring():
    # "The docstring's first paragraph becomes the description the model sees, the Args: section becomes
    # the parameter descriptions, and the type hints become the input schema."
    @tool
    def greet(name: str) -> str:
        """Greets someone.

        Args:
            name: name of the person to greet
        """
        return f"Hello {name}"

    assert greet.description == "Greets someone."
    assert greet.input_schema["properties"]["name"]["description"] == "name of the person to greet"


def test_tool_parameter_with_default_is_not_required():
    # "The model can omit parameters that have a default value."
    @tool
    def add(a: int, b: int = 10) -> int:
        """With a default, the model can omit it"""
        return a + b

    required = add.input_schema.get("required", [])
    assert "a" in required
    assert "b" not in required


def test_tool_uses_default_value_when_argument_is_omitted():
    # When the model omits a parameter that has a default, the function's default is used as is.
    @tool
    def add(a: int, b: int = 10) -> int:
        """If b is omitted, the default 10 is used"""
        return a + b

    assert single_call_result(add, a=5) == "15"


def test_tool_supports_the_documented_type_repertoire():
    # "Supported types: str, int, float, bool, Literal[...], Enum, list[T], dict[str, T], T | None,
    # dataclass, TypedDict, Pydantic model." Using the types in this list as parameters raises no error.
    class Color(Enum):
        RED = "red"
        BLUE = "blue"

    @dataclass
    class Point:
        x: int
        y: int

    class Address(TypedDict):
        city: str
        zip: str

    class Profile(BaseModel):
        age: int
        nickname: str | None = None

    @tool
    def rich(
        text: str,
        count: int,
        ratio: float,
        active: bool,
        mode: Literal["fast", "slow"],
        color: Color,
        tags: list[str],
        meta: dict[str, int],
        note: str | None,
        point: Point,
        address: Address,
        profile: Profile,
    ) -> str:
        """Takes every type in the supported list as a parameter"""
        return "ok"

    props = rich.input_schema["properties"]
    assert set(props) == {
        "text", "count", "ratio", "active", "mode", "color",
        "tags", "meta", "note", "point", "address", "profile",
    }


def test_tool_works_the_same_on_a_plain_function_and_a_method():
    # "Tools always get @tool. The same goes for functions and object methods."
    @tool
    def standalone(x: int) -> int:
        """Adds one.

        Args:
            x: value
        """
        return x + 1

    class Box:
        @tool
        def standalone(self, x: int) -> int:
            """Adds one.

            Args:
                x: value
            """
            return x + 1

    bound = Box().standalone
    assert bound.input_schema["properties"].keys() == standalone.input_schema["properties"].keys()
    assert bound.description == standalone.description


def test_object_with_multiple_tool_methods_registers_all_of_them():
    # "Putting an object in tools=[...] makes every @tool method on it a tool..."
    class Calc:
        @tool
        def add(self, a: int, b: int) -> int:
            """Adds"""
            return a + b

        @tool
        def sub(self, a: int, b: int) -> int:
            """Subtracts"""
            return a - b

    calc = Calc()
    fake = FakeModel([[tool_call("add", a=2, b=3), tool_call("sub", a=5, b=1)], "done"])
    agent = make_agent(fake, [calc])
    state = State("Calculate")
    agent.think(state)
    agent.use_tools(state)

    results = {b.name: b.content for b in state.context[-1].content}
    assert results["add"] == "5"
    assert results["sub"] == "4"


def test_object_single_bound_method_registers_only_that_tool():
    # "...or you can pass just one with obj.method."
    class Calc:
        @tool
        def add(self, a: int, b: int) -> int:
            """Adds"""
            return a + b

        @tool
        def sub(self, a: int, b: int) -> int:
            """Subtracts"""
            return a - b

    calc = Calc()
    fake = FakeModel([[tool_call("add", a=1, b=1), tool_call("sub", a=1, b=1)], "done"])
    agent = make_agent(fake, [calc.add])
    state = State("Calculate")
    agent.think(state)
    agent.use_tools(state)

    results = {b.name: b.content for b in state.context[-1].content}
    assert results["add"] == "2"
    # sub was not put in tools=, so it is not in the tool list
    assert results["sub"].startswith("(input error:")


# ---------------------------------------------------------------------------
# Tool: returning and failing
# ---------------------------------------------------------------------------


def test_tool_return_none_becomes_done_marker():
    # "If it is None, "(done)" is sent."
    @tool
    def noop() -> None:
        """Returns nothing"""
        return None

    assert single_call_result(noop) == "(done)"


def test_tool_return_string_passed_through_unchanged():
    # "A string is sent as is..."
    @tool
    def echo(text: str) -> str:
        """A string is passed through as is"""
        return text

    assert single_call_result(echo, text="passed as is") == "passed as is"


def test_tool_return_dict_and_list_become_json():
    # "Everything else (dict, list, dataclass, Pydantic model) is converted to JSON and sent."
    @tool
    def info() -> dict:
        """Returns a dict"""
        return {"a": 1, "b": [1, 2, 3]}

    content = single_call_result(info)
    assert json.loads(content) == {"a": 1, "b": [1, 2, 3]}


def test_tool_return_dataclass_becomes_json():
    # "Everything else (dict, list, dataclass, Pydantic model) is converted to JSON and sent."
    @dataclass
    class Point:
        x: int
        y: int

    @tool
    def make_point() -> Point:
        """Returns a dataclass"""
        return Point(1, 2)

    content = single_call_result(make_point)
    assert json.loads(content) == {"x": 1, "y": 2}


def test_tool_return_pydantic_model_becomes_json():
    # "Everything else (dict, list, dataclass, Pydantic model) is converted to JSON and sent."
    class Profile(BaseModel):
        age: int
        name: str

    @tool
    def make_profile() -> Profile:
        """Returns a Pydantic model"""
        return Profile(age=9, name="Marble")

    content = single_call_result(make_profile)
    assert json.loads(content) == {"age": 9, "name": "Marble"}


def test_tool_return_value_that_cannot_be_jsoned_raises_error():
    # "If it cannot be converted to JSON, it is an error."
    class Unserializable:
        pass

    @tool
    def bad():
        """Returns an object that cannot be converted to JSON"""
        return Unserializable()

    fake = FakeModel([tool_call("bad"), "done"])
    agent = make_agent(fake, [bad])
    state = State("Task")
    agent.think(state)
    with pytest.raises(TypeError, match="JSON"):
        agent.use_tools(state)


def test_tool_exception_propagates_and_execution_stops():
    # "Failures the model should know about and handle are caught inside the tool and returned as a string.
    # Raising an exception stops the run (see the behavior policy)."
    @tool
    def boom() -> str:
        """A tool that raises an exception"""
        raise RuntimeError("pipe burst")

    fake = FakeModel([tool_call("boom"), "later reply"])
    agent = make_agent(fake, [boom])
    state = State("Task")
    agent.think(state)

    with pytest.raises(RuntimeError, match="pipe burst"):
        agent.use_tools(state)

    # The run stopped, so the failed call stays with no result
    assert len(state.pending_calls) == 1


def test_tool_catches_its_own_failure_and_returns_string_to_model():
    # "Failures the model should know about and handle are caught inside the tool and returned as a string."
    @tool
    def search(query: str) -> str:
        """Catches the failure inside the tool and returns a string the model can understand"""
        try:
            raise TimeoutError()
        except TimeoutError:
            return "Search timed out. Shorten the query and try again."

    content = single_call_result(search, query="a very long search query")
    assert content == "Search timed out. Shorten the query and try again."


def test_invalid_arguments_return_input_error_without_calling_the_tool():
    # "If the arguments the model gave do not match the schema, the tool is not called and "(input error: ...)" is
    # returned as the result. It is the model's mistake, so it is information for the model, not an exception."
    called = []

    @tool
    def strict(n: int) -> str:
        """A tool that takes only an integer"""
        called.append(n)
        return "ok"

    fake = FakeModel([tool_call("strict", n="not a number"), "done"])
    agent = make_agent(fake, [strict])
    state = State("Task")
    agent.think(state)
    agent.use_tools(state)

    content = state.context[-1].content[0].content
    assert content.startswith("(input error:")
    assert called == []  # if the arguments do not match the schema, the tool is not called at all


# ---------------------------------------------------------------------------
# Tool: concurrency and State
# ---------------------------------------------------------------------------


def test_tool_calls_in_one_turn_run_concurrently():
    # "The calls of one turn run concurrently."
    @tool
    def slow_a() -> str:
        """Slow tool A"""
        time.sleep(0.3)
        return "a"

    @tool
    def slow_b() -> str:
        """Slow tool B"""
        time.sleep(0.3)
        return "b"

    fake = FakeModel([[tool_call("slow_a"), tool_call("slow_b")], "done"])
    agent = make_agent(fake, [slow_a, slow_b])
    state = State("Task")
    agent.think(state)

    start = time.monotonic()
    agent.use_tools(state)
    elapsed = time.monotonic() - start

    # Sequential runs take 0.6s or more. Concurrent runs should take about 0.3-0.4s.
    assert elapsed < 0.5


def test_parallel_false_tool_runs_only_after_the_rest_finish():
    # "Mark tools that must not run alongside others (e.g. writing a file) with @tool(parallel=False).
    # They then run one at a time after the rest finish."
    order: list[str] = []
    lock = threading.Lock()

    @tool
    def fast_a() -> str:
        """A tool that runs with the default (parallel=True)"""
        time.sleep(0.2)
        with lock:
            order.append("a")
        return "a"

    @tool
    def fast_b() -> str:
        """A tool that runs with the default (parallel=True)"""
        time.sleep(0.2)
        with lock:
            order.append("b")
        return "b"

    @tool(parallel=False)
    def writer() -> str:
        """A tool that must not run alongside others"""
        with lock:
            order.append("writer")
        return "written"

    fake = FakeModel([[tool_call("fast_a"), tool_call("fast_b"), tool_call("writer")], "done"])
    agent = make_agent(fake, [fast_a, fast_b, writer])
    state = State("Task")
    agent.think(state)
    agent.use_tools(state)

    assert order[-1] == "writer"
    assert set(order[:2]) == {"a", "b"}


def test_async_def_tool_is_supported():
    # "async def tools work too."
    @tool
    async def fetch() -> str:
        """An async tool"""
        await asyncio.sleep(0.01)
        return "finished"

    assert fetch.is_async is True
    assert single_call_result(fetch) == "finished"


def test_tool_can_finish_the_state():
    # "E.g. a submit(answer: str, state: State) tool the model calls to end the task
    # calls state.finish(answer)."
    @tool
    def submit(answer: str, state: State) -> None:
        """A tool called to end the task"""
        state.finish(answer)

    fake = FakeModel([tool_call("submit", answer="This is the answer"), "done"])
    agent = make_agent(fake, [submit])
    state = State("Task")
    agent.think(state)
    agent.use_tools(state)

    assert state.is_finished() is True
    assert state.answer == "This is the answer"


def test_tool_can_add_notice_and_write_to_data():
    # "A tool can read the injected State and use finish(), add_notice() and data."
    @tool
    def record(state: State) -> str:
        """A tool that uses State's add_notice and data"""
        state.data["found"] = 42
        state.add_notice("Recorded it")
        return "ok"

    fake = FakeModel([tool_call("record"), "done"])
    agent = make_agent(fake, [record])
    state = State("Task")
    agent.think(state)
    agent.use_tools(state)

    assert state.data["found"] == 42
    notice_entries = [h.content for h in state.history if h.kind == "notice"]
    assert notice_entries == ["[notice] Recorded it"]


def test_notice_added_during_tool_run_is_deferred_until_turn_closes():
    # "A notice added while a tool runs goes in after all of that turn's results are recorded."
    @tool
    def note_then_return(state: State) -> str:
        """A tool that adds a notice while it runs"""
        state.add_notice("notice first")
        return "result next"

    fake = FakeModel([tool_call("note_then_return"), "done"])
    agent = make_agent(fake, [note_then_return])
    state = State("Task")
    agent.think(state)
    agent.use_tools(state)

    ctx = state.context
    result_message, notice_message = ctx[-2], ctx[-1]
    assert isinstance(result_message.content[0], ToolResultBlock)
    assert result_message.content[0].content == "result next"
    assert notice_message.role == "user"
    assert notice_message.text == "[notice] notice first"


def test_concurrent_data_mutation_is_protected_by_state_lock():
    # "Code that changes data in several steps (e.g. appending to a list) goes inside with state.lock:.
    # It may run at the same time as other tools in the same turn. Parallel execution is kept."
    @tool
    def collect(i: int, state: State) -> None:
        """Several tools append to state.data at the same time"""
        with state.lock:
            state.data.setdefault("items", []).append(i)

    calls = [tool_call("collect", i=i) for i in range(20)]
    fake = FakeModel([calls, "done"])
    agent = make_agent(fake, [collect])
    state = State("Task")
    agent.think(state)
    agent.use_tools(state)

    # Concurrent appends without a lock can lose values. If the lock held, all 20 remain.
    assert sorted(state.data["items"]) == list(range(20))


def test_changing_context_while_calls_are_pending_raises_error():
    # "Methods that change the context raise an error while there are pending calls."
    @tool
    def anything() -> str:
        """An ordinary tool"""
        return "ok"

    fake = FakeModel([tool_call("anything"), "done"])
    agent = make_agent(fake, [anything])
    state = State("Task")
    agent.think(state)  # this creates a pending call, so state.pending_calls is not empty

    assert state.wants_tools() is True
    with pytest.raises(ValueError, match="calls are pending"):
        agent.compact(state)
