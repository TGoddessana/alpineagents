"""Hints of one call: ``Hints``, ``Tool.hints_for(args)``, ``@tool(hints_for=...)`` and ``tool.copy(hints_for=...)``."""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass
from typing import Any

import pytest

import alpineagents
from alpineagents import FunctionTool, Hints, State, Tool, tool

# --- Hints ---


def test_hints_left_out_assume_the_worst():
    hints = Hints()
    assert (hints.read_only, hints.destructive, hints.idempotent, hints.open_world) == (False, True, False, True)


def test_read_only_sets_destructive_and_idempotent():
    hints = Hints(read_only=True, open_world=False)
    assert hints == Hints(read_only=True, destructive=False, idempotent=True, open_world=False)
    assert all(isinstance(getattr(hints, f.name), bool) for f in dataclasses.fields(hints))


def test_hints_are_frozen_and_keyword_only():
    hints = Hints(read_only=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        hints.read_only = False  # type: ignore[misc]
    with pytest.raises(TypeError):
        Hints(True)  # type: ignore[misc]


@pytest.mark.parametrize("hint", ["destructive", "idempotent"])
def test_contradicting_hints_raise_value_error(hint):
    contradiction = {"destructive": True, "idempotent": False}[hint]
    with pytest.raises(ValueError, match=rf"Hints: read_only=True and {hint}={contradiction} contradict") as e:
        Hints(read_only=True, **{hint: contradiction})
    assert "Hints(read_only=True)" in str(e.value)


def test_a_hint_that_is_not_a_bool_raises_type_error():
    with pytest.raises(TypeError, match=r"Hints: destructive= takes True or False \(got: 'yes'\)") as e:
        Hints(destructive="yes")  # type: ignore[arg-type]
    assert "Hints(read_only=True, open_world=False)" in str(e.value)


def test_same_rules_as_tool_hints():
    for given in [{}, {"read_only": True}, {"destructive": False}, {"idempotent": True, "open_world": False}]:

        @tool(**given)
        def noop() -> None: ...

        hints = Hints(**given)
        assert (hints.read_only, hints.destructive, hints.idempotent, hints.open_world) == (
            noop.read_only, noop.destructive, noop.idempotent, noop.open_world,
        )


@dataclass(frozen=True)
class FileHints(Hints):
    paths: tuple[str, ...] = ()


def test_a_subclass_carries_more_facts():
    hints = FileHints(read_only=True, paths=("a.txt",))
    assert hints.paths == ("a.txt",)
    assert hints.destructive is False and hints.idempotent is True
    assert isinstance(hints, Hints)
    with pytest.raises(ValueError, match="FileHints: read_only=True and destructive=True"):
        FileHints(read_only=True, destructive=True)


def test_hints_is_exported():
    assert "Hints" in alpineagents.__all__
    assert alpineagents.Hints is Hints
    assert "Hints" in sys.modules["alpineagents.tool"].__all__


# --- Tool.hints_for ---


def test_default_hints_for_returns_the_tools_own_hints():
    @tool(read_only=True, open_world=False)
    def read(path: str) -> str:
        return path

    assert read.hints_for({"path": "x"}) == Hints(read_only=True, open_world=False)

    @tool
    def anything() -> None: ...

    assert anything.hints_for({}) == Hints()


class Echo(Tool):
    def __init__(self) -> None:
        super().__init__(name="echo", destructive=False, open_world=False)

    def run(self, args: dict, state: State) -> Any:
        return args


def test_a_tool_subclass_inherits_the_default():
    assert Echo().hints_for({"anything": 1}) == Hints(destructive=False, open_world=False)


class Shell(Tool):
    def __init__(self) -> None:
        super().__init__(name="shell")

    def hints_for(self, args):
        if args.get("command") == "ls":
            return Hints(read_only=True)
        return super().hints_for(args)

    def run(self, args: dict, state: State) -> Any:
        return None


def test_a_tool_subclass_can_override_hints_for():
    shell = Shell()
    assert shell.hints_for({"command": "ls"}).read_only is True
    assert shell.hints_for({"command": "rm -rf /"}) == Hints()
    assert shell.read_only is False  # the static hints stay the worst case


# --- @tool(hints_for=...) ---


def bash_hints(args):
    if args.get("command") in ("ls", "pwd"):
        return Hints(read_only=True)
    return None


@tool(hints_for=bash_hints)
def bash(command: str) -> str:
    return command


def test_hints_for_function_gives_the_hints_of_one_call():
    assert bash.hints_for({"command": "ls"}) == Hints(read_only=True)
    assert bash.read_only is False


def test_none_means_the_tools_own_hints():
    assert bash.hints_for({"command": "rm x"}) == Hints()

    @tool(destructive=False, hints_for=lambda args: None)
    def add(item: str) -> None: ...

    assert add.hints_for({"item": "a"}) == Hints(destructive=False)


def test_the_function_gets_the_raw_arguments():
    seen = []

    def spy(args):
        seen.append(args)
        return None

    @tool(hints_for=spy)
    def count(n: int) -> int:
        return n

    raw = {"n": "not a number", "extra": True}
    count.hints_for(raw)
    assert seen == [raw]


def test_a_subclass_of_hints_is_returned_as_is():
    def file_hints(args):
        return FileHints(read_only=True, paths=(args.get("path", ""),))

    @tool(hints_for=file_hints)
    def read(path: str) -> str:
        return path

    hints = read.hints_for({"path": "a.txt"})
    assert isinstance(hints, FileHints) and hints.paths == ("a.txt",)


@pytest.mark.parametrize("returned", [True, "read_only", {"read_only": True}, Hints])
def test_a_return_value_that_is_not_hints_raises_type_error(returned):
    @tool(hints_for=lambda args: returned)
    def odd() -> None: ...

    with pytest.raises(TypeError, match=r"Tool odd: hints_for .* returned .*, not Hints") as e:
        odd.hints_for({})
    assert "return None" in str(e.value)


def test_an_exception_in_the_function_propagates():
    def broken(args):
        raise KeyError("command")

    @tool(hints_for=broken)
    def run_it(command: str) -> None: ...

    with pytest.raises(KeyError):
        run_it.hints_for({})


def test_hints_for_on_a_method_tool_stays_when_bound():
    class Files:
        @tool(hints_for=lambda args: Hints(read_only=True))
        def look(self, path: str) -> str:
            return path

    bound = Files().look
    assert isinstance(bound, FunctionTool) and bound.bound_to is not None
    assert bound.hints_for({"path": "x"}) == Hints(read_only=True)


async def _async_hints(args):
    return None


@pytest.mark.parametrize(
    ("hints_for", "match"),
    [
        ("read_only", "is not a function"),
        (Hints, "is not a function"),
        (_async_hints, "is async"),
        (lambda: None, "must take exactly one argument"),
        (lambda args, state: None, "must take exactly one argument"),
    ],
)
def test_a_bad_hints_function_raises_when_the_tool_is_made(hints_for, match):
    with pytest.raises(TypeError, match=match) as e:

        @tool(hints_for=hints_for)
        def t() -> None: ...

    assert "Tool t: hints_for=" in str(e.value)
    assert "@tool(hints_for=bash_hints)" in str(e.value)


def test_a_callable_object_is_accepted():
    class ByCommand:
        def __call__(self, args):
            return Hints(read_only=True) if args.get("command") == "ls" else None

    @tool(hints_for=ByCommand())
    def sh(command: str) -> str:
        return command

    assert sh.hints_for({"command": "ls"}).read_only is True
    assert sh.hints_for({"command": "rm"}) == Hints()


# --- copy(hints_for=...) ---


def test_copy_adds_replaces_and_removes_hints_for():
    @tool
    def plain(command: str) -> str:
        return command

    with_hints = plain.copy(hints_for=bash_hints)
    assert with_hints.hints_for({"command": "ls"}) == Hints(read_only=True)
    assert plain.hints_for({"command": "ls"}) == Hints()  # the original does not change

    everything_read_only = with_hints.copy(hints_for=lambda args: Hints(read_only=True))
    assert everything_read_only.hints_for({"command": "rm"}) == Hints(read_only=True)

    without = with_hints.copy(hints_for=None)
    assert without.hints_for({"command": "ls"}) == Hints()


def test_copy_keeps_hints_for_when_other_options_change():
    renamed = bash.copy(name="shell", open_world=False)
    assert renamed.hints_for({"command": "pwd"}) == Hints(read_only=True)
    assert renamed.hints_for({"command": "rm"}) == Hints(open_world=False)


def test_copy_checks_the_new_hints_function():
    with pytest.raises(TypeError, match="is not a function"):
        bash.copy(hints_for="no")


# --- MCPTool ---


def test_mcp_tool_inherits_the_default_from_server_annotations():
    from types import SimpleNamespace

    from alpineagents import MCPTool

    server = SimpleNamespace(name="files")
    remote = SimpleNamespace(
        name="read",
        description="",
        input_schema={"type": "object", "properties": {}},
        annotations=SimpleNamespace(read_only_hint=True, open_world_hint=False),
    )
    mcp_tool = MCPTool(server, remote)  # type: ignore[arg-type]
    assert type(mcp_tool).hints_for is Tool.hints_for
    assert mcp_tool.hints_for({}) == Hints(read_only=True, open_world=False)
