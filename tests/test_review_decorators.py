"""Regression tests for @tool and @loop defects found in review.

Hints use types defined inside functions, so this file does not use ``from __future__ import annotations``.
"""

import dataclasses
import sys
import typing
import warnings

import pytest
import typing_extensions

from alpineagents.loop import Loop, default_loop, loop
from alpineagents.state import State
from alpineagents.tool import _parse_docstring, tool

# --- TypedDict: typing_extensions.TypedDict works on every version, typing.TypedDict on 3.12+ ---


def test_typing_extensions_typeddict_param_is_accepted():
    # TypedDict is a supported type (ARCHITECTURE.md "Tool contract"), and the minimum version is 3.11.
    class Q(typing_extensions.TypedDict):
        x: int

    @tool
    def f(q: Q) -> str:
        return "ok"

    assert "q" in f.input_schema["properties"]
    assert f.prepare({"q": {"x": 1}}) == {"q": {"x": 1}}


def test_typing_typeddict_param():
    class Q(typing.TypedDict):
        x: int

    if sys.version_info >= (3, 12):

        @tool
        def f(q: Q) -> str:
            return "ok"

        assert "q" in f.input_schema["properties"]
    else:
        # @tool's own TypeError with a fix, instead of Pydantic's PydanticUserError.
        with pytest.raises(TypeError, match="typing_extensions.TypedDict"):

            @tool
            def g(q: Q) -> str:
                return "ok"


def test_nested_typing_typeddict_on_old_python_is_a_tool_error():
    class Inner(typing.TypedDict):
        x: int

    @dataclasses.dataclass
    class Outer:
        items: list[Inner]

    if sys.version_info >= (3, 12):
        pytest.skip("typing.TypedDict works on 3.12+")
    with pytest.raises(TypeError, match="typing_extensions.TypedDict"):

        @tool
        def f(o: Outer) -> str:
            return "ok"


# --- Tool names: only names providers accept pass at @tool time ---


def test_non_ascii_function_name_is_rejected_at_tool_time():
    with pytest.raises(TypeError, match="Fix:"):

        @tool
        def café(requête: str) -> str:
            """Search."""
            return requête


def test_non_ascii_function_name_can_be_fixed_with_name():
    @tool(name="search")
    def café(requête: str) -> str:
        """Search."""
        return requête

    assert café.name == "search"


@pytest.mark.parametrize("bad", ["web search", "a" * 65, "", "outil_é", "a.b"])
def test_bad_explicit_name_is_rejected(bad):
    with pytest.raises(TypeError, match="1-64 characters"):

        @tool(name=bad)
        def f(x: str) -> str:
            """f."""
            return x


@pytest.mark.parametrize("good", ["a" * 64, "web-search", "Web_Search2"])
def test_good_names_are_accepted(good):
    @tool(name=good)
    def f(x: str) -> str:
        return x

    assert f.name == good


# --- @loop: keeps the outer until/limit even when the body is a Loop ---


def test_nested_loop_keeps_outer_until_and_limit():
    def over_budget(state: State) -> bool:
        return False

    outer = loop(until=over_budget, limit=3)(default_loop)
    assert isinstance(outer, Loop)
    assert outer.body is default_loop
    assert outer.until == (over_budget,)
    assert outer.limit == 3
    assert outer.__wrapped__ is default_loop
    assert outer.__name__ == "default_loop"
    # The inner loop is unchanged
    assert default_loop.limit == 50


def test_body_with_colliding_attributes_does_not_override_loop():
    def body(agent, state):
        return None

    body.limit = 999  # type: ignore[attr-defined]
    body.until = "junk"  # type: ignore[attr-defined]

    def done(state: State) -> bool:
        return True

    lp = loop(until=done, limit=2)(body)
    assert lp.limit == 2
    assert lp.until == (done,)
    assert lp.body is body


def test_nested_loop_runs_with_outer_limit():
    turns = []

    @loop(until=State.is_finished, limit=5)
    def inner(agent, state):
        turns.append("inner")

    outer = loop(until=State.is_finished, limit=2)(inner)
    state = State("hi")
    outer(None, state)
    # 2 outer turns x 5 inner turns
    assert len(turns) == 10
    assert state.stopped_by == "limit"


# --- State injection: TYPE_CHECKING-only import, State subclass ---


def test_tool_with_typechecking_only_state_import(tmp_path):
    mod_path = tmp_path / "review_tool_mod.py"
    mod_path.write_text(
        "from __future__ import annotations\n"
        "from typing import TYPE_CHECKING\n"
        "from alpineagents import tool\n"
        "if TYPE_CHECKING:\n"
        "    from alpineagents import State\n"
        "\n"
        "@tool\n"
        "def submit(answer: str, state: State):\n"
        '    """Submits."""\n'
        "    return answer\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        import review_tool_mod

        submit = review_tool_mod.submit
        assert "state" not in submit.input_schema.get("properties", {})
        assert submit.input_schema["required"] == ["answer"]
        s = State("q")
        assert submit.invoke(submit.prepare({"answer": "a"}), s) == "a"
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("review_tool_mod", None)


def test_unresolvable_other_hint_still_errors(tmp_path):
    mod_path = tmp_path / "review_tool_mod2.py"
    mod_path.write_text(
        "from __future__ import annotations\n"
        "from alpineagents import tool\n"
        "\n"
        "@tool\n"
        "def f(x: Missing):\n"
        "    return x\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        with pytest.raises(TypeError, match="Missing"):
            import review_tool_mod2  # noqa: F401
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("review_tool_mod2", None)


def test_state_subclass_is_injected():
    class MyState(State):
        pass

    @tool
    def f(x: str, st: MyState) -> str:
        return x

    assert list(f.input_schema["properties"]) == ["x"]
    s = MyState("q")
    seen = []

    @tool
    def g(st: MyState) -> None:
        seen.append(st)

    g.invoke(g.prepare({}), s)
    assert seen == [s]


# --- Parameter names that collide with BaseModel attributes ---


def test_model_config_param_is_in_schema_and_roundtrips():
    @tool
    def f(model_config: str):
        return model_config

    assert "model_config" in f.input_schema["properties"]
    assert f.input_schema["required"] == ["model_config"]
    assert f.prepare({"model_config": "a"}) == {"model_config": "a"}


def test_missing_model_config_arg_is_reported_by_name():
    from alpineagents.errors import ToolInputError

    @tool
    def f(model_config: str):
        return model_config

    with pytest.raises(ToolInputError, match="model_config: required argument is missing"):
        f.prepare({})


def test_underscore_param_works():
    @tool
    def f(_x: int):
        return _x

    assert "_x" in f.input_schema["properties"]
    assert f.prepare({"_x": 3}) == {"_x": 3}


def test_model_dump_param_works():
    @tool
    def f(model_dump: str):
        return model_dump

    assert f.prepare({"model_dump": "v"}) == {"model_dump": "v"}


@pytest.mark.parametrize("pname", ["json", "schema", "copy", "validate"])
def test_basemodel_attribute_names_do_not_warn(pname):
    ns: dict = {}
    exec(f"def f({pname}: str):\n    return {pname}\n", ns)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = tool(ns["f"])
    assert t.prepare({pname: "v"}) == {pname: "v"}


def test_internal_field_names_are_not_accepted_as_args():
    from alpineagents.errors import ToolInputError

    @tool
    def f(x: str):
        return x

    with pytest.raises(ToolInputError, match="unknown argument"):
        f.prepare({"a0": "v"})


def test_param_descriptions_and_defaults_still_work():
    @tool
    def f(q: str, n: int = 3):
        """Finds.

        Args:
            q: the query
            n: count
        """
        return q

    props = f.input_schema["properties"]
    assert props["q"]["description"] == "the query"
    assert props["n"]["description"] == "count"
    assert f.input_schema["required"] == ["q"]
    assert f.prepare({"q": "x"}) == {"q": "x", "n": 3}


# --- Parsing the docstring Args: section ---


def test_args_only_docstring_has_no_description_leak():
    doc = """Args:
    q: the query
"""
    description, param_docs = _parse_docstring(doc)
    assert description == ""
    assert param_docs == {"q": "the query"}


def test_summary_directly_followed_by_args_no_blank_line():
    doc = """Search the web.
Args:
    q: the query
"""
    description, param_docs = _parse_docstring(doc)
    assert description == "Search the web."
    assert param_docs == {"q": "the query"}


def test_args_section_stops_at_next_section():
    doc = """Search.

    Args:
        q: the query
            continued
        n: count

    Returns:
        results
    """
    description, param_docs = _parse_docstring(doc)
    assert description == "Search."
    assert param_docs == {"q": "the query continued", "n": "count"}


def test_args_section_stops_at_header_without_blank_line():
    doc = """Search.

    Args:
        q: the query
    Returns:
        results
    """
    assert _parse_docstring(doc) == ("Search.", {"q": "the query"})


def test_args_entries_separated_by_blank_lines():
    doc = """Search.

    Args:
        q: the query

        n: count
    """
    assert _parse_docstring(doc) == ("Search.", {"q": "the query", "n": "count"})


def test_word_args_inside_summary_is_not_a_header():
    doc = """Parses args
    from the command line.
    """
    assert _parse_docstring(doc) == ("Parses args from the command line.", {})
