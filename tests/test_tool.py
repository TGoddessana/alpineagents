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
import dataclasses
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

from alpineagents import Agent, FileStore, Message, Reporter, State, Tool, ToolError, ToolOutcomeKind, tool
from alpineagents.permissions import AllowByDefault, DenyPermission, Denied
from alpineagents.testing import FakeModel, tool_call
from alpineagents._frozen import FrozenDict, FrozenList
from alpineagents.types import ToolResultBlock, ToolResultEntry


def make_agent(fake, tools):
    """A test Agent that prints nothing to the terminal (reporter=None, human=None)."""
    return Agent(model=fake, tools=tools, reporter=None, human=None)


def single_call_result(t, **args):
    """Calls one tool once and returns the result string the model receives."""
    fake = FakeModel([tool_call(t.name, **args), "done"])
    agent = make_agent(fake, [t])
    state = State(messages=[Message.user("Task")])
    agent.think(state)
    agent.use_tools(state)
    return state.messages[-1].content[0].content


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
    state = State(messages=[Message.user("Task")])
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
        "text",
        "count",
        "ratio",
        "active",
        "mode",
        "color",
        "tags",
        "meta",
        "note",
        "point",
        "address",
        "profile",
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
    state = State(messages=[Message.user("Calculate")])
    agent.think(state)
    agent.use_tools(state)

    results = {b.name: b.content for b in state.messages[-1].content}
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
    state = State(messages=[Message.user("Calculate")])
    agent.think(state)
    agent.use_tools(state)

    results = {b.name: b.content for b in state.messages[-1].content}
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
    state = State(messages=[Message.user("Task")])
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
    state = State(messages=[Message.user("Task")])
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
    state = State(messages=[Message.user("Task")])
    agent.think(state)
    agent.use_tools(state)

    content = state.messages[-1].content[0].content
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
    state = State(messages=[Message.user("Task")])
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
    state = State(messages=[Message.user("Task")])
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
    state = State(messages=[Message.user("Task")])
    agent.think(state)
    agent.use_tools(state)

    assert state.finished is True
    assert state.answer == "This is the answer"


def test_tool_can_add_notice_and_write_to_extra_data():
    # "A tool can read the injected State and use finish(), add_message(Message.notice(...)) and
    # edit_extra_data()."
    @tool
    def record(state: State) -> str:
        """A tool that uses State's add_message and extra_data"""
        with state.edit_extra_data() as data:
            data["found"] = 42
        state.add_message(Message.notice("Recorded it"))
        return "ok"

    fake = FakeModel([tool_call("record"), "done"])
    agent = make_agent(fake, [record])
    state = State(messages=[Message.user("Task")])
    agent.think(state)
    agent.use_tools(state)

    assert state.extra_data["found"] == 42
    notice_entries = [h.content for h in state.history if h.kind == "notice"]
    assert notice_entries == ["[notice] Recorded it"]


def test_notice_added_during_tool_run_is_deferred_until_turn_closes():
    # "A notice added while a tool runs goes in after all of that turn's results are recorded."
    @tool
    def note_then_return(state: State) -> str:
        """A tool that adds a notice while it runs"""
        state.add_message(Message.notice("notice first"))
        return "result next"

    fake = FakeModel([tool_call("note_then_return"), "done"])
    agent = make_agent(fake, [note_then_return])
    state = State(messages=[Message.user("Task")])
    agent.think(state)
    agent.use_tools(state)

    ctx = state.messages
    result_message, notice_message = ctx[-2], ctx[-1]
    assert isinstance(result_message.content[0], ToolResultBlock)
    assert result_message.content[0].content == "result next"
    assert notice_message.role == "user"
    assert notice_message.text == "[notice] notice first"


def test_concurrent_extra_data_edits_are_serialized_by_edit_extra_data():
    # "edit_extra_data() holds a lock for the whole block, so a read-modify-write (e.g. appending to a list)
    # is atomic. Tools may run at the same time in the same turn. Parallel execution is kept."
    @tool
    def collect(i: int, state: State) -> None:
        """Several tools append to state.extra_data at the same time"""
        with state.edit_extra_data() as data:
            data.setdefault("items", []).append(i)

    calls = [tool_call("collect", i=i) for i in range(20)]
    fake = FakeModel([calls, "done"])
    agent = make_agent(fake, [collect])
    state = State(messages=[Message.user("Task")])
    agent.think(state)
    agent.use_tools(state)

    # Concurrent appends without a lock can lose values. If the lock held, all 20 remain.
    assert sorted(state.extra_data["items"]) == list(range(20))


def test_changing_context_while_calls_are_pending_raises_error():
    # "Methods that change the context raise an error while there are pending calls."
    @tool
    def anything() -> str:
        """An ordinary tool"""
        return "ok"

    fake = FakeModel([tool_call("anything"), "done"])
    agent = make_agent(fake, [anything])
    state = State(messages=[Message.user("Task")])
    agent.think(state)  # this creates a pending call, so state.pending_calls is not empty

    assert state.pending_calls
    with pytest.raises(ValueError, match="calls are pending"):
        agent.compact(state)


# ---------------------------------------------------------------------------
# ToolResultEntry.outcome: one entry kind for every way a call can end
# ---------------------------------------------------------------------------


@tool
def fine(x: int) -> str:
    """Works"""
    return "fine"


@tool
def failing() -> str:
    """Raises a ToolError"""
    raise ToolError("nope")


@tool
def boom() -> str:
    """Raises a plain exception"""
    raise RuntimeError("boom")


@tool
def ctrl_c() -> str:
    """Stops right away, imitating Ctrl+C while it runs"""
    raise KeyboardInterrupt()


class DenyFailing(DenyPermission):
    def check(self, state, call, tool_):
        return Denied("not allowed") if call.name == "failing" else None


class StopOnFailing(DenyPermission):
    def check(self, state, call, tool_):
        return Denied("not allowed", stop=True) if call.name == "failing" else None


class EndSpy(Reporter):
    def __init__(self):
        self.ends = []

    def on_tool_end(self, state, call, result, outcome):
        self.ends.append((call.name, outcome.kind))


class Ran:
    """What a run of one turn of calls left: the State, the Reporter's view, and the exception that ended the run."""

    def __init__(self, state, spy, raised):
        self.state, self.spy, self.raised = state, spy, raised

    @property
    def results(self):
        return [e for e in self.state.history if isinstance(e, ToolResultEntry)]


def run_calls(calls, tools, *, store=None, state_id=None, **settings):
    """Runs one turn of calls (the second model reply is a plain answer). A run that a plain exception or Ctrl+C
    ends does not raise here: the exception is in ``.raised``."""
    spy = EndSpy()
    settings.setdefault("human", None)
    agent = Agent(model=FakeModel([calls, "done"]), tools=tools, reporter=spy, store=store, **settings)
    state = State(messages=[Message.user("Task")], id=state_id)
    raised = None
    try:
        agent.run(state)
    except (RuntimeError, KeyboardInterrupt) as e:
        raised = e
    return Ran(state, spy, raised)


def test_done_outcome():
    (entry,) = run_calls([tool_call("fine", x=1)], [fine]).results
    assert (entry.outcome, entry.content, entry.is_error, entry.late) == (ToolOutcomeKind.DONE, "fine", False, False)
    assert entry.error is None and entry.call.name == "fine"


def test_error_outcome_keeps_the_tool_error():
    ran = run_calls([tool_call("failing")], [failing])
    (entry,) = ran.results
    assert (entry.outcome, entry.content, entry.is_error) == (ToolOutcomeKind.ERROR, "nope", True)
    assert isinstance(entry.error, ToolError)
    assert ran.raised is None


def test_input_error_outcome_for_bad_arguments_and_for_an_unknown_tool():
    ran = run_calls([tool_call("fine", x="not a number"), tool_call("missing_tool")], [fine])
    by_name = {e.call.name: e for e in ran.results}
    assert [e.outcome for e in by_name.values()] == [ToolOutcomeKind.INPUT_ERROR] * 2
    assert all(e.is_error and e.error is None for e in by_name.values())
    assert by_name["fine"].content.startswith("(input error:")
    assert "unknown tool" in by_name["missing_tool"].content


def test_aborted_outcome_when_a_tool_raises_and_the_run_stops():
    ran = run_calls([tool_call("boom")], [boom])
    assert isinstance(ran.raised, RuntimeError)
    (entry,) = ran.results
    assert (entry.outcome, entry.content, entry.is_error) == (ToolOutcomeKind.ABORTED, "(aborted: RuntimeError)", True)
    assert entry.error is ran.raised
    assert ran.state.pending_calls == ()


def test_interrupted_outcome_for_ctrl_c():
    ran = run_calls([tool_call("ctrl_c")], [ctrl_c])
    assert isinstance(ran.raised, KeyboardInterrupt)
    (entry,) = ran.results
    assert entry.outcome == ToolOutcomeKind.INTERRUPTED
    assert (entry.content, entry.is_error) == ("(interrupted by user)", True)
    assert entry.error is ran.raised
    assert ran.state.pending_calls == ()


def test_denied_outcome_for_a_permission_that_denies():
    ran = run_calls(
        [tool_call("fine", x=1), tool_call("failing")], [fine, failing], permissions=[DenyFailing(), AllowByDefault()]
    )
    by_name = {e.call.name: e for e in ran.results}
    assert by_name["failing"].outcome == ToolOutcomeKind.DENIED
    assert (by_name["failing"].content, by_name["failing"].is_error) == ("not allowed", True)
    assert by_name["fine"].outcome == ToolOutcomeKind.DONE and not by_name["fine"].is_error  # the other call ran
    assert str(ran.state.stopped) == "stopped by waiting_for_user"  # a plain denial does not stop the run


def test_cancelled_outcome_for_the_calls_of_a_stopped_turn():
    calls = [tool_call("fine", x=1), tool_call("failing"), tool_call("fine", x=2)]
    ran = run_calls(calls, [fine, failing], permissions=[StopOnFailing(), AllowByDefault()])
    assert [(e.call.name, e.outcome) for e in ran.results] == [
        ("failing", ToolOutcomeKind.DENIED),
        ("fine", ToolOutcomeKind.CANCELLED),
        ("fine", ToolOutcomeKind.CANCELLED),
    ]
    assert all(e.is_error for e in ran.results)


def test_is_error_is_a_read_only_property_of_the_outcome():
    (entry,) = run_calls([tool_call("fine", x=1)], [fine]).results
    assert isinstance(ToolResultEntry.is_error, property)
    assert "is_error" not in {f.name for f in dataclasses.fields(entry)}
    with pytest.raises(AttributeError):
        entry.is_error = True  # type: ignore[misc]
    for kind in ToolOutcomeKind:
        made = ToolResultEntry(content="x", call=entry.call, outcome=kind, turn=1)
        assert made.is_error is (kind != ToolOutcomeKind.DONE)
    assert ToolResultEntry(content="x", call=entry.call, outcome="denied", turn=1).outcome is ToolOutcomeKind.DENIED
    with pytest.raises(ValueError):
        ToolResultEntry(content="x", call=entry.call, outcome="exploded", turn=1)


ALL_PATHS = {
    "done": ([tool_call("fine", x=1)], [fine], {}),
    "error": ([tool_call("failing")], [failing], {}),
    "input_error": ([tool_call("missing_tool")], [fine], {}),
    "aborted": ([tool_call("boom")], [boom], {}),
    "interrupted": ([tool_call("ctrl_c")], [ctrl_c], {}),
    "denied": ([tool_call("failing")], [failing], {"permissions": [DenyFailing(), AllowByDefault()]}),
    "cancelled": (
        [tool_call("failing"), tool_call("fine", x=1)],
        [failing, fine],
        {"permissions": [StopOnFailing(), AllowByDefault()]},
    ),
}


@pytest.mark.parametrize("path", ALL_PATHS)
def test_the_reporter_outcome_matches_the_entry_outcome_on_every_path(path):
    calls, tools, settings = ALL_PATHS[path]
    ran = run_calls(calls, tools, **settings)
    assert sorted((e.call.name, e.outcome) for e in ran.results) == sorted(ran.spy.ends)
    assert path in {str(e.outcome) for e in ran.results}  # every outcome kind is reachable from a run


def test_every_outcome_kind_is_covered_by_the_paths():
    assert set(ALL_PATHS) == {str(kind) for kind in ToolOutcomeKind}


@pytest.mark.parametrize("path", ALL_PATHS)
def test_every_outcome_survives_a_store_round_trip(tmp_path, path):
    calls, tools, settings = ALL_PATHS[path]
    store = FileStore(tmp_path)
    ran = run_calls(calls, tools, store=store, state_id="s1", **settings)
    store.save(ran.state)
    loaded = store.load("s1")
    loaded_results = [e for e in loaded.history if isinstance(e, ToolResultEntry)]
    assert [(e.call.name, e.outcome, e.content, e.is_error) for e in loaded_results] == [
        (e.call.name, e.outcome, e.content, e.is_error) for e in ran.results
    ]
    assert path in {str(e.outcome) for e in loaded_results}
    assert all(e.error is None for e in loaded_results)  # a live exception is never saved
    assert loaded.snapshot() == ran.state.snapshot()
    assert State(history=ran.state.history).snapshot() == ran.state.snapshot()


# ---------------------------------------------------------------------------
# Plain mutable arguments for tools; ToolCall.args stays frozen
# ---------------------------------------------------------------------------


def test_a_tool_function_gets_plain_mutable_arguments_and_the_recorded_call_stays_frozen():
    seen = {}

    @tool
    def plan(steps: list[str], options: dict[str, list[int]]) -> str:
        """Takes containers and edits them"""
        seen["types"] = (type(steps), type(options), type(options["sizes"]))
        steps.append("extra")
        options["sizes"].append(99)
        options["new"] = [0]
        return f"{len(steps)} steps"

    ran = run_calls([tool_call("plan", steps=["a", "b"], options={"sizes": [1, 2]})], [plan])
    assert seen["types"] == (list, dict, list)
    (entry,) = ran.results
    assert entry.content == "3 steps"
    args = entry.call.args
    assert isinstance(args, FrozenDict) and isinstance(args["steps"], FrozenList)
    assert args == {"steps": ["a", "b"], "options": {"sizes": [1, 2]}}  # what the model sent, unchanged
    with pytest.raises(TypeError):
        args["steps"].append("x")
    assert ran.state.messages[1].tool_calls[0].args == {"steps": ["a", "b"], "options": {"sizes": [1, 2]}}


def test_a_tool_class_gets_plain_nested_arguments():
    seen = {}

    class Edit(Tool):
        def __init__(self):
            super().__init__(name="edit", description="Edit", input_schema={"type": "object", "properties": {}})

        def run(self, args, state):
            seen["types"] = (type(args), type(args["lines"]), type(args["opts"]), type(args["opts"]["tags"]))
            args["lines"].append(3)
            args["opts"]["tags"].append("z")
            args.pop("lines")
            return "edited"

    (entry,) = run_calls([tool_call("edit", lines=[1, 2], opts={"tags": ["a"]})], [Edit()]).results
    assert seen["types"] == (dict, list, dict, list)
    assert entry.content == "edited"
    assert entry.call.args == {"lines": [1, 2], "opts": {"tags": ["a"]}}


def test_each_run_of_a_tool_gets_its_own_copy_of_the_arguments():
    seen = []

    @tool
    def take(items: list[int]) -> str:
        """Appends to its argument"""
        items.append(len(seen))
        seen.append(items)
        return "ok"

    state = State(messages=[Message.user("Task")])
    model = FakeModel([tool_call("take", items=[0]), tool_call("take", items=[0]), "done"])
    Agent(model=model, tools=[take], reporter=None).run(state)
    assert seen == [[0, 0], [0, 1]]  # the second call did not see the first call's append
    results = [e for e in state.history if isinstance(e, ToolResultEntry)]
    assert [e.call.args for e in results] == [{"items": [0]}, {"items": [0]}]


def test_permissions_and_reporters_see_the_frozen_arguments():
    seen = {}

    class Look(DenyPermission):
        def check(self, state, call, tool_):
            seen["permission"] = (type(call.args), type(call.args["items"]))
            return None

    class Watch(Reporter):
        def on_tool_start(self, state, call):
            seen["reporter"] = type(call.args)

    @tool
    def take(items: list[int]) -> str:
        """Takes a list"""
        return "ok"

    agent = Agent(
        model=FakeModel([tool_call("take", items=[1]), "done"]),
        tools=[take],
        reporter=Watch(),
        human=None,
        permissions=[Look(), AllowByDefault()],
    )
    agent.run("Task")
    assert seen == {"permission": (FrozenDict, FrozenList), "reporter": FrozenDict}
