"""Permissions (``alpineagents.permissions``, ``Agent(permissions=...)``): the check phase of ``use_tools``."""

from __future__ import annotations

import asyncio
import threading
import warnings
from types import SimpleNamespace
from typing import Any

import pytest
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

import alpineagents
from alpineagents import (
    MCP,
    Agent,
    FileStore,
    Hints,
    Human,
    MCPTool,
    NoHumanError,
    PermissionWarning,
    Reporter,
    State,
    StoppedByFinish,
    StoppedByPermission,
    StoppedByUntil,
    ToolError,
    ToolOutcome,
    ToolOutcomeKind,
    loop,
    tool,
)
from alpineagents.agent import SETTINGS
from alpineagents.permissions import (
    AllowByDefault,
    AllowByName,
    AllowByReadOnly,
    Allowed,
    AllowPermission,
    DecideByHuman,
    DecidePermission,
    Denied,
    DenyByName,
    DenyPermission,
    Permission,
)
from alpineagents.testing import FakeHuman, FakeModel, tool_call
from alpineagents.types import Exchange, ToolResultBlock

STOPPED_TURN = "(not run: the user stopped this turn)"
DECLINED = "The user declined this call. Wait for their next message."
NO_PERMISSION = "No permission allowed this call."

ran: list[str] = []


@tool(read_only=True, open_world=False)
def read_file(path: str) -> str:
    """Read a file"""
    ran.append(f"read_file {path}")
    return f"contents of {path}"


@tool(open_world=False)
def write_file(path: str, content: str) -> str:
    """Write a file"""
    ran.append(f"write_file {path}")
    return "written"


def _bash_hints(args: dict[str, Any]) -> Hints | None:
    return Hints(read_only=True) if args.get("command") == "ls" else None


@tool(hints_for=_bash_hints)
def bash(command: str) -> str:
    """Run a command"""
    ran.append(f"bash {command}")
    return "ran"


@tool(parallel=False)
def deploy(target: str) -> str:
    """Deploy (one at a time)"""
    ran.append(f"deploy {target}")
    return "deployed"


@pytest.fixture(autouse=True)
def _clear_ran():
    ran.clear()


def make_agent(replies, permissions, **settings) -> Agent:
    settings.setdefault("reporter", None)
    settings.setdefault("human", None)
    settings.setdefault("tools", [read_file, write_file, bash, deploy])
    return Agent(model=FakeModel(replies), permissions=permissions, **settings)


def results(state: State) -> list[ToolResultBlock]:
    """The tool results of the last turn, as the model sees them."""
    for message in reversed(state.context):
        blocks = [b for b in message.content if isinstance(b, ToolResultBlock)]
        if blocks:
            return blocks
    return []


def kinds(state: State) -> list[tuple[str, str]]:
    return [(e.kind, e.call.name) for e in state.history if e.call is not None and e.kind != "error"]


class Spy(Reporter):
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def on_tool_start(self, state, call):
        self.events.append(("start", call.name))

    def on_tool_end(self, state, call, result, outcome):
        self.events.append(("end", call.name, result, outcome))


class Recorder:
    """Records the order permissions were asked in."""

    def __init__(self) -> None:
        self.asked: list[str] = []


def recording(kind: type, label: str, verdict: Any, recorder: Recorder) -> Permission:
    class Recording(kind):  # type: ignore[misc, valid-type]
        def check(self, state, call, tool):
            recorder.asked.append(label)
            return verdict

        def __repr__(self) -> str:
            return label

    return Recording()


# ================================================================ verdicts


def test_allowed_and_denied_are_frozen_values():
    assert Allowed() == Allowed()
    assert Denied("no") == Denied("no", stop=False)
    assert Denied("no", stop=True).stop is True
    with pytest.raises(AttributeError):
        Denied("no").reason = "yes"  # type: ignore[misc]


@pytest.mark.parametrize("reason", ["", "   \n"])
def test_denied_needs_a_reason(reason):
    with pytest.raises(ValueError, match="Denied got an empty reason") as e:
        Denied(reason)
    assert "Fix:" in str(e.value)


def test_denied_reason_and_stop_types():
    with pytest.raises(TypeError, match="Denied takes a reason string"):
        Denied(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="stop= takes True or False"):
        Denied("no", stop="yes")  # type: ignore[arg-type]


# ================================================================ kinds


def test_subclassing_permission_directly_is_a_type_error():
    class Direct(Permission):
        def check(self, state, call, tool):
            return None

    with pytest.raises(TypeError, match="Direct subclasses Permission directly") as e:
        Direct()
    assert "DenyPermission" in str(e.value) and "Fix:" in str(e.value)


def test_subclassing_two_kinds_is_a_type_error():
    class Both(DenyPermission, AllowPermission):
        def check(self, state, call, tool):
            return None

    with pytest.raises(TypeError, match=r"more than one kind of permission \(DenyPermission, AllowPermission\)"):
        Both()


def test_neither_check_nor_acheck_is_a_type_error_at_instantiation():
    class Nothing(DecidePermission):
        pass  # defining it is fine (an abstract middle class)

    with pytest.raises(TypeError, match="Nothing implements neither check nor acheck"):
        Nothing()
    with pytest.raises(TypeError, match="implements neither check nor acheck"):
        DenyPermission()


def test_default_repr_names_the_class():
    class Mine(AllowPermission):
        def check(self, state, call, tool):
            return None

    assert repr(Mine()) == "Mine()"


@pytest.mark.parametrize(
    ("kind", "verdict", "may"),
    [
        (DenyPermission, Allowed(), r"a DenyPermission may return only Denied\(\.\.\.\) or None"),
        (AllowPermission, Denied("no"), r"an AllowPermission may return only Allowed\(\) or None"),
        (DecidePermission, True, r"a DecidePermission may return only Allowed\(\) or Denied\(\.\.\.\) or None"),
        (AllowPermission, "yes", r"may return only Allowed\(\) or None"),
    ],
)
def test_a_wrong_verdict_is_a_type_error_naming_the_permission(kind, verdict, may):
    permission = recording(kind, "Wrong()", verdict, Recorder())
    agent = make_agent([tool_call("read_file", path="a")], [permission])
    state = State("Task")
    agent.think(state)
    with pytest.raises(TypeError, match=may) as e:
        agent.use_tools(state)
    assert f"Wrong() returned {verdict!r}" in str(e.value)
    example = str(e.value).split("Example:")[-1]
    if kind is AllowPermission:  # the example shows what it may return, not the rejected Denied
        assert "Allowed()" in example and "Denied" not in example
    else:
        assert "Denied(" in example
    assert [c.name for c in state.pending_calls] == ["read_file"]  # not run, not closed
    assert ran == []
    errors = [h for h in state.history if h.kind == "error"]
    assert len(errors) == 1 and errors[0].call is not None and errors[0].call.name == "read_file"


# ================================================================ order


def test_deny_permissions_are_asked_first_wherever_they_are():
    recorder = Recorder()
    permissions = [
        recording(AllowPermission, "allow1", None, recorder),
        recording(DenyPermission, "deny1", None, recorder),
        recording(DecidePermission, "decide", None, recorder),
        recording(DenyPermission, "deny2", None, recorder),
        recording(AllowPermission, "allow2", Allowed(), recorder),
    ]
    make_agent([tool_call("read_file", path="a"), "done"], permissions).run("Task")
    assert recorder.asked == ["deny1", "deny2", "allow1", "decide", "allow2"]
    assert ran == ["read_file a"]


def test_a_deny_listed_last_still_wins_over_allow_by_default():
    fake = [[tool_call("bash", command="rm -rf /"), tool_call("read_file", path="a")], "done"]
    state = State("Task")
    make_agent(fake, [AllowByDefault(), DenyByName(["bash"])]).run(state)
    assert ran == ["read_file a"]
    assert kinds(state) == [("denied", "bash"), ("tool_result", "read_file")]
    first = results(state)[0]
    assert first == ToolResultBlock(first.call_id, "bash is not allowed.", name="bash", is_error=True)


def test_the_first_deny_decides_and_later_denies_are_not_asked():
    recorder = Recorder()
    permissions = [
        recording(DenyPermission, "deny1", Denied("first"), recorder),
        recording(DenyPermission, "deny2", Denied("second"), recorder),
        recording(AllowPermission, "allow", Allowed(), recorder),
    ]
    state = State("Task")
    make_agent([tool_call("read_file", path="a"), "done"], permissions).run(state)
    assert recorder.asked == ["deny1"]
    assert results(state)[0].content == "first"


def test_the_first_non_none_verdict_wins():
    recorder = Recorder()
    spy = Spy()
    permissions = [
        recording(AllowPermission, "abstain", None, recorder),
        recording(DecidePermission, "decider", Denied("not today"), recorder),
        recording(AllowPermission, "late", Allowed(), recorder),
    ]
    state = State("Task")
    make_agent([tool_call("read_file", path="a"), "done"], permissions, reporter=spy).run(state)
    assert recorder.asked == ["abstain", "decider"]
    assert ran == []
    assert results(state)[0].content == "not today" and results(state)[0].is_error
    ends = [e for e in spy.events if e[0] == "end"]
    assert ends == [("end", "read_file", "not today", ToolOutcome(ToolOutcomeKind.DENIED, decided_by="decider"))]
    assert ("start", "read_file") not in spy.events


def test_every_call_is_checked_in_request_order_before_any_tool_runs():
    seen: list[tuple[str, list[str], list[str]]] = []

    class Look(AllowPermission):
        def check(self, state, call, tool):
            seen.append((call.name, [c.name for c in state.pending_calls], list(ran)))
            return None if call.name == "bash" else Allowed()

    calls = [tool_call("read_file", path="a"), tool_call("bash", command="rm"), tool_call("write_file", path="b",
                                                                                           content="x")]
    with pytest.warns(PermissionWarning):
        make_agent([calls, "done"], [Look()]).run("Task")
    assert seen == [
        ("read_file", ["read_file", "bash", "write_file"], []),
        ("bash", ["read_file", "bash", "write_file"], []),  # allowed calls stay pending until they run
        ("write_file", ["read_file", "write_file"], []),  # the denied call dropped out
    ]
    assert sorted(ran) == ["read_file a", "write_file b"]


def test_input_errors_come_before_permissions():
    recorder = Recorder()
    permissions = [recording(AllowPermission, "allow", Allowed(), recorder)]
    broken = tool_call("read_file")
    broken = type(broken)(name="read_file", args={"__invalid_json__": "{oops"}, id="b1")
    state = State("Task")
    make_agent([[tool_call("made_up"), broken], "done"], permissions).run(state)
    assert recorder.asked == []
    assert all(b.content.startswith("(input error:") for b in results(state))


def test_permissions_get_the_raw_arguments_and_the_tool():
    got: list[tuple[dict, Any]] = []

    class Look(AllowPermission):
        def check(self, state, call, tool):
            got.append((dict(call.args), tool))
            return Allowed()

    state = State("Task")
    make_agent([tool_call("read_file", path=42), "done"], [Look()]).run(state)
    assert got == [({"path": 42}, read_file)]  # not validated yet: the tool then reports the input error
    assert results(state)[0].content.startswith("(input error:")


# ================================================================ nobody decides


def test_nobody_decides_denies_with_a_warning():
    spy = Spy()
    state = State("Task")
    with pytest.warns(PermissionWarning, match="read_file") as record:
        make_agent([tool_call("read_file", path="a"), "done"], [DenyByName(["bash"])], reporter=spy).run(state)
    assert "AllowByDefault()" in str(record[0].message)
    assert ran == []
    assert results(state)[0].content == NO_PERMISSION and results(state)[0].is_error
    assert kinds(state) == [("denied", "read_file")]
    assert [e for e in spy.events if e[0] == "end"] == [
        ("end", "read_file", NO_PERMISSION, ToolOutcome(ToolOutcomeKind.DENIED))
    ]
    assert state.stopped == StoppedByUntil("is_answered")  # the run went on


def test_permission_warning_is_a_top_level_user_warning():
    assert issubclass(PermissionWarning, UserWarning)
    assert alpineagents.PermissionWarning is PermissionWarning
    assert "PermissionWarning" in alpineagents.__all__


def test_permission_names_are_not_top_level():
    for name in ["Permission", "DenyByName", "AllowByDefault", "DecideByHuman", "Allowed", "Denied"]:
        assert not hasattr(alpineagents, name), name


def test_allow_by_default_allows_everything():
    calls = [tool_call("read_file", path="a"), tool_call("bash", command="rm"), tool_call("deploy", target="x")]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_agent([calls, "done"], [AllowByDefault()]).run("Task")
    assert sorted(ran) == ["bash rm", "deploy x", "read_file a"]
    assert repr(AllowByDefault()) == "AllowByDefault()"


def test_no_permissions_means_no_checks():
    agent = make_agent([tool_call("bash", command="rm"), "done"], None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        agent.run("Task")
    assert agent.permissions == ()
    assert ran == ["bash rm"]


@pytest.mark.parametrize("use_async", [False, True])
async def test_an_empty_permissions_list_denies_every_call(use_async):
    # Only None means no checks. A list, even an empty one (e.g. every rule of a built list turned off), is asked,
    # and a call nobody allows is denied: failing closed.
    agent = make_agent([tool_call("write_file", path="b", content="x"), "done"], [])
    state = State("Task")
    with pytest.warns(PermissionWarning, match="write_file"):
        if use_async:
            await agent.arun(state)
        else:
            agent.run(state)
    assert ran == []
    assert results(state)[0].content == NO_PERMISSION and results(state)[0].is_error
    assert kinds(state) == [("denied", "write_file")]
    assert agent.permissions == ()
    # copy keeps the difference
    with pytest.warns(PermissionWarning):
        agent.copy(model=FakeModel([tool_call("bash", command="rm"), "done"])).run("Task")
    assert ran == []
    agent.copy(model=FakeModel([tool_call("bash", command="rm"), "done"]), permissions=None).run("Task")
    assert ran == ["bash rm"]


# ================================================================ by name


def check(permission: Permission, name: str, tool_: Any = None, **args: Any) -> Any:
    return permission.check(State("Task"), tool_call(name, **args), tool_ or read_file)


def test_allow_and_deny_by_name_use_globs_on_the_model_visible_name():
    allow = AllowByName(["read_*", "github__get_*"])
    assert check(allow, "read_file") == Allowed()
    assert check(allow, "github__get_issue") == Allowed()
    assert check(allow, "github__delete_repo") is None
    assert check(allow, "write_file") is None
    deny = DenyByName(["github__*", "rm"])
    assert check(deny, "github__delete_repo") == Denied("github__delete_repo is not allowed.")
    assert check(deny, "rm") == Denied("rm is not allowed.")
    assert check(deny, "rmdir") is None
    assert check(DenyByName(["rm"], reason="Use trash instead"), "rm") == Denied("Use trash instead")


def test_by_name_reprs():
    assert repr(AllowByName(["read_*"])) == "AllowByName(['read_*'])"
    assert repr(DenyByName(("rm",))) == "DenyByName(['rm'])"
    assert repr(DenyByName(["rm"], reason="No")) == "DenyByName(['rm'], reason='No')"


def test_built_in_reprs_name_the_subclass():
    class DenyShell(DenyByName):
        pass

    class AllowReads(AllowByName):
        pass

    class ReadOnly(AllowByReadOnly):
        pass

    class AskInSlack(DecideByHuman):
        def question(self, call, tool):
            return "?"

    assert repr(DenyShell(["bash"], reason="No")) == "DenyShell(['bash'], reason='No')"
    assert repr(AllowReads(["read_*"])) == "AllowReads(['read_*'])"
    assert repr(ReadOnly(trust_mcp=True)) == "ReadOnly(trust_mcp=True)" and repr(ReadOnly()) == "ReadOnly()"
    assert repr(AskInSlack()) == "AskInSlack()"


def test_decide_by_human_repr_is_the_same_in_every_process():
    class Approvals(Human):
        def ask(self, state, prompt, returns=str):
            return "yes"

    class Named(Approvals):
        def __repr__(self) -> str:
            return "Named('#ops')"

    # object's default repr has a memory address; decided_by and StoppedByPermission are saved and compared
    assert repr(DecideByHuman(human=Approvals())) == "DecideByHuman(human=Approvals())"
    assert repr(DecideByHuman(human=Named())) == "DecideByHuman(human=Named('#ops'))"


def test_deny_by_name_reason_errors_name_deny_by_name():
    for reason, error in (("", ValueError), (" ", ValueError), (3, TypeError)):
        with pytest.raises(error, match=r"DenyByName") as e:
            DenyByName(["rm"], reason=reason)  # type: ignore[arg-type]
        text = str(e.value)
        assert "reason=" in text and 'DenyByName(["delete_*"], reason="Deleting files is not allowed")' in text
        assert "Denied" not in text.replace("DenyByName", "")


def test_by_name_argument_checks():
    with pytest.raises(TypeError, match="takes a list of name patterns") as e:
        DenyByName("rm")  # type: ignore[arg-type]
    assert "square brackets" in str(e.value)
    with pytest.raises(TypeError, match="as strings"):
        AllowByName([1])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="empty reason"):
        DenyByName(["rm"], reason=" ")


def make_server() -> MCPServer:
    server = MCPServer("demo")

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
    def get_issue(number: int) -> str:
        """Get an issue"""
        return f"issue {number}"

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def delete_repo(name: str) -> str:
        """Delete a repository (and claims to be read-only)"""
        return "gone"

    @server.tool()
    def add_label(number: int, label: str) -> str:
        """Add a label"""
        return "ok"

    return server


def test_mcp_tools_by_name_and_read_only_trust():
    github = MCP(server=make_server(), name="github")
    calls = [
        tool_call("github__get_issue", number=1),
        tool_call("github__delete_repo", name="x"),
        tool_call("github__add_label", number=1, label="bug"),
    ]
    state = State("Task")
    agent = Agent(
        model=FakeModel([calls, "done"]),
        tools=[github],
        reporter=None,
        human=None,
        permissions=[AllowByReadOnly(trust_mcp=True), DenyByName(["github__delete_*"]), AllowByName(["*label"])],
    )
    agent.run(state)
    found = kinds(state)
    assert found[0] == ("denied", "github__delete_repo")  # the deny wins over the server's read-only claim
    assert sorted(found[1:]) == [("tool_result", "github__add_label"), ("tool_result", "github__get_issue")]
    assert [b.content for b in results(state)] == ["issue 1", "github__delete_repo is not allowed.", "ok"]


# ================================================================ read-only


def test_allow_by_read_only_uses_the_static_hints():
    assert check(AllowByReadOnly(), "read_file", read_file, path="a") == Allowed()
    assert check(AllowByReadOnly(), "write_file", write_file, path="a", content="x") is None


def test_allow_by_read_only_uses_hints_for_per_call():
    assert check(AllowByReadOnly(), "bash", bash, command="ls") == Allowed()
    assert check(AllowByReadOnly(), "bash", bash, command="rm -rf /") is None
    state = State("Task")
    with pytest.warns(PermissionWarning):
        make_agent(
            [[tool_call("bash", command="ls"), tool_call("bash", command="rm")], "done"], [AllowByReadOnly()]
        ).run(state)
    assert ran == ["bash ls"]
    assert [b.content for b in results(state)] == ["ran", NO_PERMISSION]


def _mcp_tool(read_only: bool) -> MCPTool:
    server = SimpleNamespace(name="files")
    remote = SimpleNamespace(
        name="read",
        description="",
        input_schema={"type": "object", "properties": {}},
        annotations=SimpleNamespace(read_only_hint=read_only),
    )
    return MCPTool(server, remote)  # type: ignore[arg-type]


def test_allow_by_read_only_does_not_trust_mcp_servers_unless_told():
    assert check(AllowByReadOnly(), "files__read", _mcp_tool(True)) is None
    assert check(AllowByReadOnly(trust_mcp=True), "files__read", _mcp_tool(True)) == Allowed()
    assert check(AllowByReadOnly(trust_mcp=True), "files__read", _mcp_tool(False)) is None
    assert repr(AllowByReadOnly()) == "AllowByReadOnly()"
    assert repr(AllowByReadOnly(trust_mcp=True)) == "AllowByReadOnly(trust_mcp=True)"
    with pytest.raises(TypeError, match="trust_mcp= takes True or False"):
        AllowByReadOnly(trust_mcp="yes")  # type: ignore[arg-type]


def test_an_exception_from_hints_for_follows_the_permission_rules():
    def broken_hints(args):
        raise ValueError("cannot tell")

    def refusing_hints(args):
        raise ToolError("cannot tell what this command does")

    @tool(hints_for=broken_hints)
    def shell(command: str) -> str:
        """Run a command"""
        ran.append("shell")
        return "ran"

    with pytest.raises(ValueError, match="cannot tell") as e:
        make_agent([tool_call("shell", command="ls")], [AllowByReadOnly()], tools=[shell]).run("Task")
    assert "exception raised in permission AllowByReadOnly() checking shell(command=\"ls\")" in e.value.__notes__

    state = State("Task")
    agent = make_agent([tool_call("shell", command="ls"), "ok"], [AllowByReadOnly()],
                       tools=[shell.copy(hints_for=refusing_hints)])
    agent.run(state)
    assert results(state)[0].content == "cannot tell what this command does"
    assert ran == []


# ================================================================ asking a person


def test_decide_by_human_yes_runs_and_is_recorded_like_ask_human():
    human = FakeHuman(["yes"])
    state = State("Task")
    make_agent(
        [tool_call("write_file", path="a", content="x"), "done"], [DecideByHuman()], human=human
    ).run(state)
    question = 'Run write_file(path="a", content="x")?'
    assert human.questions == [question]
    assert ran == ["write_file a"]
    entries = [h for h in state.history if h.kind == "human"]
    assert len(entries) == 1 and entries[0].content == Exchange(question, "yes")


def test_decide_by_human_no_denies_and_stops():
    human = FakeHuman(["no"])
    fake = FakeModel([tool_call("write_file", path="a", content="x"), "never"])
    state = State("Task")
    agent = Agent(model=fake, tools=[write_file], permissions=[DecideByHuman()], human=human, reporter=None)
    assert agent.run(state) is None
    call = state.history[1].content.tool_calls[0]
    assert state.stopped == StoppedByPermission(call, "DecideByHuman()")
    assert results(state)[0].content == DECLINED
    assert len(fake.requests) == 1
    assert [h.content for h in state.history if h.kind == "human"] == [
        Exchange('Run write_file(path="a", content="x")?', "no")
    ]


def test_decide_by_human_question_can_be_overridden():
    class Short(DecideByHuman):
        def question(self, call, tool):
            return f"{call.name}? ({'read-only' if tool.read_only else 'changes things'})"

    human = FakeHuman(["yes"])
    make_agent([tool_call("write_file", path="a", content="x"), "done"], [Short()], human=human).run("Task")
    assert human.questions == ["write_file? (changes things)"]


def test_decide_by_human_without_a_human_fails_at_run_start():
    fake = FakeModel(["never"])
    agent = Agent(model=fake, tools=[write_file], permissions=[DecideByHuman()], human=None, reporter=None)
    with pytest.raises(NoHumanError, match=r"DecideByHuman\(\) in permissions= asks the Agent's human"):
        agent.run("Task")
    assert fake.requests == []


async def test_decide_by_human_without_a_human_fails_at_arun_start():
    fake = FakeModel(["never"])
    agent = Agent(model=fake, tools=[write_file], permissions=[DecideByHuman()], human=None, reporter=None)
    with pytest.raises(NoHumanError):
        await agent.arun("Task")
    assert fake.requests == []


def test_decide_by_human_with_its_own_human():
    own = FakeHuman(["yes"])
    agent_human = FakeHuman([])
    state = State("Task")
    make_agent(
        [tool_call("write_file", path="a", content="x"), "done"], [DecideByHuman(human=own)], human=agent_human
    ).run(state)
    assert own.questions == ['Run write_file(path="a", content="x")?'] and agent_human.questions == []
    assert [h.kind for h in state.history].count("human") == 1  # still recorded
    # works with human=None on the Agent too
    ran.clear()
    make_agent([tool_call("write_file", path="b", content="x"), "done"], [DecideByHuman(human=FakeHuman(["yes"]))],
               human=None).run("Task")
    assert ran == ["write_file b"]
    assert repr(DecideByHuman(human=own)).startswith("DecideByHuman(human=")


async def test_a_decide_by_human_subclass_that_changes_check_decides_the_same_in_arun():
    class TrustReads(DecideByHuman):
        def check(self, state, call, tool):
            return Allowed() if tool.read_only else super().check(state, call, tool)

    human = FakeHuman(["yes"])
    calls = [tool_call("read_file", path="a"), tool_call("write_file", path="b", content="x")]
    await make_agent([calls, "done"], [TrustReads()], human=human).arun("Task")
    assert human.questions == ['Run write_file(path="b", content="x")?']
    assert sorted(ran) == ["read_file a", "write_file b"]


def test_decide_by_human_takes_only_a_human_object():
    with pytest.raises(TypeError, match="is not a Human object"):
        DecideByHuman(human=FakeHuman)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="is not a Human object"):
        DecideByHuman(human=object())  # type: ignore[arg-type]


def test_use_tools_outside_run_raises_no_human_error_and_keeps_the_calls():
    agent = make_agent([tool_call("write_file", path="a", content="x")], [DecideByHuman()])
    state = State("Task")
    agent.think(state)
    with pytest.raises(NoHumanError):
        agent.use_tools(state)
    assert [c.name for c in state.pending_calls] == ["write_file"]


class AsyncHuman(Human):
    def __init__(self, answers):
        self.answers = list(answers)
        self.threads: list[int] = []

    async def aask(self, state, prompt, returns=str):
        self.threads.append(threading.get_ident())
        await asyncio.sleep(0)
        return self.answers.pop(0)


async def test_decide_by_human_asks_asynchronously_in_arun():
    human = AsyncHuman(["yes"])
    state = State("Task")
    await make_agent(
        [tool_call("write_file", path="a", content="x"), "done"], [DecideByHuman()], human=human
    ).arun(state)
    assert human.threads == [threading.get_ident()]  # awaited on the event loop, not on a worker thread
    assert ran == ["write_file a"]
    assert [h.content.answer for h in state.history if h.kind == "human"] == ["yes"]


# ================================================================ async-only permissions


class AsyncOnly(DecidePermission):
    def __init__(self, verdict: Any = None) -> None:
        self.verdict = verdict if verdict is not None else Allowed()

    async def acheck(self, state, call, tool):
        await asyncio.sleep(0)
        return self.verdict


def test_sync_run_with_an_acheck_only_permission_fails_at_run_start():
    fake = FakeModel(["never"])
    agent = Agent(model=fake, tools=[read_file], permissions=[AsyncOnly()], reporter=None, human=None)
    with pytest.raises(TypeError, match=r"AsyncOnly\(\) in permissions= checks calls only asynchronously") as e:
        agent.run("Task")
    assert "arun()" in str(e.value)
    assert fake.requests == []


@pytest.mark.parametrize("own", [True, False])
def test_sync_run_with_decide_by_human_asking_an_async_only_human_fails_at_run_start(own):
    fake = FakeModel(["never"])
    human = AsyncHuman(["yes"])
    permission = DecideByHuman(human=human) if own else DecideByHuman()
    agent = Agent(
        model=fake, tools=[write_file], permissions=[permission], reporter=None, human=None if own else human
    )
    with pytest.raises(TypeError, match="asks AsyncHuman, which answers only asynchronously") as e:
        agent.run("Task")
    assert "await agent.arun(" in str(e.value) and "aask_human" not in str(e.value)
    assert fake.requests == []  # before the first model call


def test_sync_run_with_a_decide_by_human_subclass_that_does_not_ask_is_not_checked():
    class NeverAsks(DecideByHuman):
        def check(self, state, call, tool):
            return Allowed()

    make_agent([tool_call("write_file", path="a", content="x"), "done"], [NeverAsks()],
               human=AsyncHuman([])).run("Task")
    assert ran == ["write_file a"]


class AsyncApprover(DecideByHuman):
    """Changes only acheck: it decides without asking, so run() must not ask the human instead."""

    async def acheck(self, state, call, tool):
        return Allowed()


def test_sync_run_with_a_decide_by_human_subclass_that_changed_only_acheck_fails_at_run_start():
    human = FakeHuman(["no"])
    agent = make_agent(["never"], [AsyncApprover()], human=human)
    with pytest.raises(TypeError, match=r"AsyncApprover\(\) in permissions= checks calls only asynchronously"):
        agent.run("Task")
    assert agent.model.requests == [] and human.questions == []


def test_sync_use_tools_with_a_decide_by_human_subclass_that_changed_only_acheck_does_not_ask():
    human = FakeHuman(["no"])
    agent = make_agent([tool_call("write_file", path="a", content="x")], [AsyncApprover()], human=human)
    state = State("Task")
    agent.think(state)
    with pytest.raises(TypeError, match="checks calls only asynchronously"):
        agent.use_tools(state)
    assert ran == [] and human.questions == [] and len(state.pending_calls) == 1


async def test_arun_with_a_decide_by_human_subclass_that_changed_only_acheck_uses_it():
    human = FakeHuman(["no"])
    state = State("Task")
    await make_agent([tool_call("write_file", path="a", content="x"), "done"], [AsyncApprover()], human=human).arun(
        state
    )
    assert ran == ["write_file a"] and human.questions == []
    assert state.stopped == StoppedByUntil("is_answered")


def test_a_decide_by_human_subclass_that_changed_both_methods_still_asks_through_check():
    class Both(DecideByHuman):
        def check(self, state, call, tool):
            return super().check(state, call, tool)

        async def acheck(self, state, call, tool):
            return await super().acheck(state, call, tool)

    make_agent([tool_call("write_file", path="a", content="x"), "done"], [Both()], human=FakeHuman(["yes"])).run(
        "Task"
    )
    assert ran == ["write_file a"]


def test_sync_use_tools_with_an_acheck_only_permission_raises_type_error():
    agent = make_agent([tool_call("read_file", path="a")], [AsyncOnly()])
    state = State("Task")
    agent.think(state)
    with pytest.raises(TypeError, match="checks calls only asynchronously"):
        agent.use_tools(state)
    assert ran == [] and len(state.pending_calls) == 1


async def test_arun_uses_acheck_and_runs_check_on_a_worker_thread():
    threads: list[int] = []

    class SyncAllow(AllowPermission):
        def check(self, state, call, tool):
            threads.append(threading.get_ident())
            return Allowed() if call.name == "read_file" else None

    state = State("Task")
    calls = [tool_call("read_file", path="a"), tool_call("write_file", path="b", content="x")]
    agent = make_agent([calls, "done"], [SyncAllow(), AsyncOnly(Denied("no writes"))])
    await agent.arun(state)
    assert threads and all(t != threading.get_ident() for t in threads)
    assert ran == ["read_file a"]
    assert [b.content for b in results(state)] == ["contents of a", "no writes"]


# ================================================================ stopping the turn


class Stopper(DecidePermission):
    """Denies ``write_file`` with stop=True, allows the rest."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def check(self, state, call, tool):
        self.asked.append(call.name)
        return Denied("Stop right there", stop=True) if call.name == "write_file" else Allowed()


STOP_CALLS = [
    tool_call("read_file", path="a"),
    tool_call("write_file", path="b", content="x"),
    tool_call("bash", command="ls"),
    tool_call("deploy", target="prod"),
]


def _assert_stopped_turn(state: State, spy: Spy, stopper: Stopper) -> None:
    assert ran == []  # no tool of the turn ran, not even the one allowed before the stop
    assert stopper.asked == ["read_file", "write_file"]  # the calls after the stop were not asked about
    write_call = state.history[1].content.tool_calls[1]
    assert state.stopped == StoppedByPermission(write_call, "Stopper()")
    assert not state.is_finished()
    assert state.pending_calls == ()
    assert [(b.content, b.is_error) for b in results(state)] == [
        (STOPPED_TURN, True), ("Stop right there", True), (STOPPED_TURN, True), (STOPPED_TURN, True),
    ]
    assert kinds(state) == [
        ("denied", "write_file"), ("cancelled", "read_file"), ("cancelled", "bash"), ("cancelled", "deploy"),
    ]
    assert not any(e[0] == "start" for e in spy.events)
    denied = ToolOutcome(ToolOutcomeKind.DENIED, decided_by="Stopper()")
    cancelled = ToolOutcome(ToolOutcomeKind.CANCELLED, decided_by="Stopper()")
    assert spy.events == [
        ("end", "write_file", "Stop right there", denied),
        ("end", "read_file", STOPPED_TURN, cancelled),
        ("end", "bash", STOPPED_TURN, cancelled),
        ("end", "deploy", STOPPED_TURN, cancelled),
    ]
    assert all(isinstance(e[3].kind, ToolOutcomeKind) for e in spy.events)


def test_denied_with_stop_cancels_the_turn_and_stops_the_run():
    spy, stopper = Spy(), Stopper()
    fake = FakeModel([STOP_CALLS, "Done"])
    agent = Agent(
        model=fake, tools=[read_file, write_file, bash, deploy], permissions=[stopper], reporter=spy, human=None
    )
    state = State("Task")
    assert agent.run(state) is None  # ends normally; no answer yet
    _assert_stopped_turn(state, spy, stopper)
    assert len(fake.requests) == 1

    state.add_user_message("Only read the file")
    assert agent.run(state) == "Done"
    assert state.stopped == StoppedByUntil("is_answered")
    last = fake.requests[-1].messages
    assert last[-1].text == "Only read the file"
    assert [b.content for b in last[-2].content] == [STOPPED_TURN, "Stop right there", STOPPED_TURN, STOPPED_TURN]


async def test_denied_with_stop_async():
    spy, stopper = Spy(), Stopper()
    fake = FakeModel([STOP_CALLS, "Done"])
    agent = Agent(
        model=fake, tools=[read_file, write_file, bash, deploy], permissions=[stopper], reporter=spy, human=None
    )
    state = State("Task")
    assert await agent.arun(state) is None
    _assert_stopped_turn(state, spy, stopper)
    state.add_user_message("Only read the file")
    assert await agent.arun(state) == "Done"


def test_a_deny_permission_can_stop_too():
    class Guard(DenyPermission):
        def check(self, state, call, tool):
            return Denied("Never deploy", stop=True) if call.name == "deploy" else None

    state = State("Task")
    make_agent([[tool_call("deploy", target="prod"), tool_call("read_file", path="a")]], [Guard(),
                                                                                           AllowByDefault()]).run(state)
    assert ran == []
    assert isinstance(state.stopped, StoppedByPermission) and state.stopped.permission == "Guard()"
    assert kinds(state) == [("denied", "deploy"), ("cancelled", "read_file")]


def test_denied_without_stop_lets_the_other_calls_run():
    class NoWrites(DenyPermission):
        def check(self, state, call, tool):
            return Denied("No writes") if call.name == "write_file" else None

    state = State("Task")
    make_agent([STOP_CALLS, "done"], [NoWrites(), AllowByDefault()]).run(state)
    assert sorted(ran) == ["bash ls", "deploy prod", "read_file a"]
    assert [b.content for b in results(state)] == ["contents of a", "No writes", "ran", "deployed"]


def test_a_loop_without_at_loop_stops_on_a_permission_stop():
    turns = []

    def plain(agent, state):
        while state.stopped is None and not state.is_answered():
            turns.append(state.turn)
            agent.think(state)
            if state.wants_tools():
                agent.use_tools(state)
        return state.answer

    fake = FakeModel([tool_call("write_file", path="a", content="x"), "ok"])
    agent = Agent(
        model=fake, tools=[write_file], loop=plain, permissions=[DecideByHuman()], human=FakeHuman(["no"]),
        reporter=None,
    )
    state = State("Task")
    assert state.stopped is None
    agent.run(state)
    call = state.history[1].content.tool_calls[0]
    assert state.stopped == StoppedByPermission(call, "DecideByHuman()")
    assert turns == [0] and len(fake.requests) == 1 and ran == []
    state.add_user_message("Never mind")
    assert agent.run(state) == "ok"
    # The next run started from None; a plain loop sets nothing when it ends on is_answered.
    assert state.stopped is None


@pytest.mark.parametrize("use_async", [False, True])
async def test_a_permission_stop_is_visible_right_after_use_tools(use_async):
    seen = []

    if use_async:

        @loop(until=State.is_answered, limit=5)
        async def body(agent, state):
            await agent.athink(state)
            if state.wants_tools():
                await agent.ause_tools(state)
                seen.append(state.stopped)

    else:

        @loop(until=State.is_answered, limit=5)
        def body(agent, state):
            agent.think(state)
            if state.wants_tools():
                agent.use_tools(state)
                seen.append(state.stopped)

    agent = make_agent([STOP_CALLS, "Done"], [Stopper()], loop=body)
    state = State("Task")
    if use_async:
        await agent.arun(state)
    else:
        agent.run(state)
    write_call = state.history[1].content.tool_calls[1]
    assert seen == [StoppedByPermission(write_call, "Stopper()")]
    assert state.stopped == seen[0] and state.turn == 1


def test_finish_before_a_permission_stop_in_the_same_turn_wins():
    class FinishThenStop(DecidePermission):
        def check(self, state, call, tool):
            state.finish("done early")
            return Denied("Stop", stop=True)

    state = State("Task")
    make_agent([tool_call("bash", command="ls")], [FinishThenStop()]).run(state)
    assert state.stopped == StoppedByFinish()
    assert state.answer == "done early"


def test_a_permission_stop_is_cleared_by_the_next_run():
    agent = make_agent([STOP_CALLS, "Done"], [Stopper()])
    state = State("Task")
    agent.run(state)
    assert isinstance(state.stopped, StoppedByPermission)
    assert agent.run(state) == "Done"  # no new user message needed: the model answers the results
    assert state.stopped == StoppedByUntil("is_answered")


# ================================================================ parallel and serial tools


def test_allowed_parallel_and_serial_tools_run_denied_serial_does_not():
    class NoProd(DenyPermission):
        def check(self, state, call, tool):
            return Denied("Not prod") if call.args.get("target") == "prod" else None

    calls = [
        tool_call("deploy", target="prod"),
        tool_call("read_file", path="a"),
        tool_call("deploy", target="staging"),
        tool_call("bash", command="ls"),
    ]
    state = State("Task")
    make_agent([calls, "done"], [NoProd(), AllowByDefault()]).run(state)
    assert sorted(ran[:2]) == ["bash ls", "read_file a"]  # the parallel group first
    assert ran[2:] == ["deploy staging"]  # then the serial group
    assert [b.content for b in results(state)] == ["Not prod", "contents of a", "deployed", "ran"]


# ================================================================ exceptions from a permission


class Raising(DecidePermission):
    def __init__(self, error: BaseException, name: str = "bash") -> None:
        self.error = error
        self.name = name

    def check(self, state, call, tool):
        if call.name == self.name:
            raise self.error
        return None

    async def acheck(self, state, call, tool):
        return self.check(state, call, tool)

    def __repr__(self) -> str:
        return "Raising()"


@pytest.mark.parametrize("use_async", [False, True])
async def test_tool_error_from_a_permission_denies_the_call_and_the_run_continues(use_async):
    error = ToolError("Could not check this command")
    spy = Spy()
    state = State("Task")
    calls = [tool_call("bash", command="ls"), tool_call("read_file", path="a")]
    agent = make_agent([calls, "done"], [Raising(error), AllowByDefault()], reporter=spy)
    if use_async:
        await agent.arun(state)
    else:
        agent.run(state)
    assert ran == ["read_file a"]
    assert [b.content for b in results(state)] == ["Could not check this command", "contents of a"]
    denied = [h for h in state.history if h.kind == "denied"]
    assert len(denied) == 1 and denied[0].error is error and denied[0].is_error
    end = next(e for e in spy.events if e[0] == "end" and e[1] == "bash")
    assert end[3] == ToolOutcome(ToolOutcomeKind.DENIED, error, "Raising()")
    assert state.stopped == StoppedByUntil("is_answered")


@pytest.mark.parametrize("use_async", [False, True])
async def test_another_exception_from_a_permission_propagates_and_closes_the_calls(use_async):
    calls = [tool_call("write_file", path="a", content="x"), tool_call("bash", command="ls"),
             tool_call("read_file", path="b")]
    permissions = [DenyByName(["write_file"]), Raising(RuntimeError("checker broke")), AllowByDefault()]
    agent = make_agent([calls, "never"], permissions)
    state = State("Task")
    with pytest.raises(RuntimeError, match="checker broke") as e:
        if use_async:
            await agent.arun(state)
        else:
            agent.run(state)
    assert 'exception raised in permission Raising() checking bash(command="ls")' in e.value.__notes__
    assert ran == []  # never treated as allowed, and nothing ran after it
    assert [b.content for b in results(state)] == [
        "write_file is not allowed.", "(aborted: RuntimeError)", "(aborted: RuntimeError)"
    ]
    errors = [h for h in state.history if h.kind == "error"]
    assert len(errors) == 1 and errors[0].call is not None and errors[0].call.name == "bash"
    assert state.stopped is None


def test_keyboard_interrupt_from_a_permission_closes_calls_as_interrupted():
    agent = make_agent([tool_call("bash", command="ls")], [Raising(KeyboardInterrupt())])
    state = State("Task")
    with pytest.raises(KeyboardInterrupt):
        agent.run(state)
    assert [b.content for b in results(state)] == ["(interrupted by user)"]
    assert ran == []


# ================================================================ async parity


async def test_ause_tools_matches_use_tools():
    def run_sync() -> list:
        state = State("Task")
        agent = make_agent([STOP_CALLS], [DenyByName(["bash"]), AllowByReadOnly(), AllowByName(["deploy"])])
        agent.think(state)
        with pytest.warns(PermissionWarning):
            agent.use_tools(state)
        return [(e.kind, e.content, e.call.name) for e in state.history if e.call is not None]

    sync_history = run_sync()  # no async run is going on, so the sync API may be called here
    sync_ran = sorted(ran)
    ran.clear()
    state = State("Task")
    agent = make_agent([STOP_CALLS], [DenyByName(["bash"]), AllowByReadOnly(), AllowByName(["deploy"])])
    await agent.athink(state)
    with pytest.warns(PermissionWarning):
        await agent.ause_tools(state)
    assert [(e.kind, e.content, e.call.name) for e in state.history if e.call is not None] == sync_history
    assert sorted(ran) == sync_ran == ["deploy prod", "read_file a"]


# ================================================================ store


class Writes(FileStore):
    """Remembers the history kinds of each write."""

    def __init__(self, path) -> None:
        super().__init__(path)
        self.written: list[str] = []

    def write(self, state_id, entries, snapshot, *, create=False):
        super().write(state_id, entries, snapshot, create=create)
        self.written.extend(entry["kind"] for entry in entries)


def test_the_check_phase_is_saved_before_the_tools_run(tmp_path):
    store = Writes(tmp_path)
    seen: list[list[str]] = []

    @tool
    def look() -> str:
        """Look"""
        seen.append(list(store.written))
        return "looked"

    agent = Agent(
        model=FakeModel([[tool_call("bash", command="rm"), tool_call("look")], "done"]),
        tools=[bash, look],
        reporter=None,
        human=None,
        store=store,
        permissions=[DenyByName(["bash"]), AllowByDefault()],
    )
    agent.run(State("Task", id="t1"))
    assert seen == [["user", "reply", "denied"]]


@pytest.mark.parametrize("use_async", [False, True])
async def test_a_decide_by_human_yes_is_saved_before_the_tool_runs(tmp_path, use_async):
    store = Writes(tmp_path)
    seen: list[list[str]] = []

    @tool
    def deploy_now(env: str) -> str:
        """Deploy"""
        seen.append(list(store.written))
        return "deployed"

    agent = Agent(
        model=FakeModel([tool_call("deploy_now", env="prod"), "done"]),
        tools=[deploy_now],
        reporter=None,
        human=FakeHuman(["yes"]),
        store=store,
        permissions=[DecideByHuman()],
    )
    state = State("Task", id="t1")
    if use_async:
        await agent.arun(state)
    else:
        agent.run(state)
    assert seen == [["user", "reply", "human"]]  # the approval reached the store before the tool started


def test_resume_after_a_stop_keeps_denied_and_cancelled(tmp_path):
    store = FileStore(tmp_path)
    fake = FakeModel([STOP_CALLS])
    agent = Agent(
        model=fake, tools=[read_file, write_file, bash, deploy], permissions=[Stopper()], reporter=None, human=None,
        store=store,
    )
    live = State("Task", id="s1")
    agent.run(live)

    loaded = store.load("s1")
    assert [(e.kind, e.content) for e in loaded.history] == [(e.kind, e.content) for e in live.history]
    assert {e.kind for e in loaded.history} >= {"denied", "cancelled"}
    assert loaded.context == live.context
    assert loaded.stopped == live.stopped
    assert loaded.pending_calls == ()

    loaded.add_user_message("Only read the file")
    assert agent.copy(model=FakeModel(["Done"])).run(loaded) == "Done"


class SnapshotOnce(FileStore):
    """Keeps only the first snapshot, so loading replays everything after it."""

    def write(self, state_id, entries, snapshot, *, create=False):
        super().write(state_id, entries, snapshot if create else None, create=create)


def test_replay_restores_cancelled_like_denied(tmp_path):
    store = SnapshotOnce(tmp_path)
    agent = Agent(
        model=FakeModel([STOP_CALLS]),
        tools=[read_file, write_file, bash, deploy],
        permissions=[Stopper()],
        reporter=None,
        human=None,
        store=store,
    )
    live = State("Task", id="r1")
    agent.run(live)
    loaded = store.load("r1")
    assert loaded.context == live.context  # replayed from the first snapshot
    assert loaded.pending_calls == ()
    assert [e.kind for e in loaded.history].count("cancelled") == 3


# ================================================================ Agent settings


def test_agent_permissions_setting_and_copy():
    deny, allow = DenyByName(["bash"]), AllowByDefault()
    agent = make_agent([], [deny, allow])
    assert agent.permissions == (deny, allow)
    assert "permissions" in SETTINGS
    assert agent.copy().permissions == (deny, allow)
    other = agent.copy(permissions=[allow])
    assert other.permissions == (allow,) and agent.permissions == (deny, allow)
    assert agent.copy(permissions=None).permissions == ()


def test_agent_permissions_argument_checks():
    with pytest.raises(TypeError, match="AllowByDefault passed in permissions= is not a Permission object") as e:
        make_agent([], [AllowByDefault])
    assert "(it is the class)" in str(e.value)
    with pytest.raises(TypeError, match="is not a Permission object"):
        make_agent([], ["allow everything"])
    with pytest.raises(TypeError, match="permissions= takes a list"):
        make_agent([], AllowByDefault())
