"""The code in the documentation runs as written.

Each file in ``docs_src/`` is included in a docs page as is. These tests execute those files with the model
strings replaced by ``FakeModel``s and ``input()`` replaced by prepared answers, so no network is used.
"""

from __future__ import annotations

import builtins
import logging
import re
import signal
import sys
import types
from pathlib import Path

import pytest

import alpineagents.agent as agent_module
from alpineagents import (
    Agent,
    AuthError,
    ContextTooLongError,
    Message,
    Model,
    ProviderError,
    Reply,
    State,
    StoppedByPermission,
    StoppedByUntil,
    Usage,
    tool,
)
from alpineagents.testing import FakeModel, tool_call

ROOT = Path(__file__).resolve().parent.parent
DOCS_SRC = ROOT / "docs_src"
DOC_FILES = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


def run(*names: str, ns: dict | None = None) -> dict:
    """Executes docs_src files in order, in one namespace, and returns the namespace."""
    if ns is None:
        module = types.ModuleType("docs_example")
        sys.modules["docs_example"] = module
        ns = module.__dict__
    for name in names:
        path = DOCS_SRC / f"{name}.py"
        exec(compile(path.read_text(), str(path), "exec"), ns)
    return ns


@pytest.fixture
def models(monkeypatch):
    """A list of FakeModels. Each model string given to Agent(...) takes the next one."""
    queue: list[FakeModel] = []

    def resolve(model):
        return model if isinstance(model, Model) else queue.pop(0)

    monkeypatch.setattr(agent_module, "resolve_model", resolve)
    return queue


@pytest.fixture
def answers(monkeypatch):
    """A list of answers that input() returns in order. An exception instance is raised instead."""
    queue: list = []

    def fake_input(prompt=""):
        answer = queue.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(builtins, "input", fake_input)
    return queue


def last_request_text(fake: FakeModel) -> str:
    return "\n".join(str(m.content) for m in fake.requests[-1].messages)


# ---------------------------------------------------------------- README


def test_quickstart(models, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    fake = FakeModel([tool_call("read_file", path="README.md"), "Three lines of summary"])
    models.append(fake)
    run("quickstart")
    assert "Three lines of summary" in capsys.readouterr().out
    assert "A Python agent framework" in last_request_text(fake)


def test_no_api_key(models, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    models.append(FakeModel(["unused"]))
    ns = run()
    exec(compile(_without_last_line("quickstart"), "quickstart", "exec"), ns)
    run("no_api_key", ns=ns)
    assert "README.md describes a Python agent framework." in capsys.readouterr().out


def test_coding_agent(models, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    Path("calc.py").write_text("def add(a, b):\n    return a + b\n")
    models.append(FakeModel([
        tool_call("list_files"),
        tool_call("read_file", path="calc.py"),
        tool_call("write_file", path="test_calc.py", content="def test_add(): ..."),
        "Added test_calc.py",
    ]))
    run("coding_agent")
    assert Path("test_calc.py").exists()
    assert "Added test_calc.py" in capsys.readouterr().out


def test_readme_code_matches_docs_src():
    readme = (ROOT / "README.md").read_text()
    for name in ["quickstart", "no_api_key", "coding_agent"]:
        assert (DOCS_SRC / f"{name}.py").read_text().strip() in readme, name


def _without_last_line(name: str) -> str:
    return "".join((DOCS_SRC / f"{name}.py").read_text().splitlines(keepends=True)[:-1])


# ---------------------------------------------------------------- guides


def test_stop_conditions_guide_lists_every_way_a_loop_stops():
    from typing import get_args

    from alpineagents.types import Stopped

    guide = (ROOT / "docs" / "guides" / "stop-conditions.md").read_text()
    ways = get_args(Stopped)
    assert f"one of {['zero', 'one', 'two', 'three', 'four', 'five'][len(ways)]} ways" in guide
    for way in ways:
        assert f"`{way.__name__}(" in guide, way.__name__


def test_stop_conditions():
    ns = run("stop_conditions")
    big = Reply(
        message=Message("assistant", (tool_call("unknown"),)),
        usage=Usage(output_tokens=30_000, requests=1),
    )
    state = State("Work")
    Agent(model=FakeModel([big]), loop=ns["careful"], reporter=None).run(state)
    assert state.stopped == StoppedByUntil("spent_too_much")


def test_submit_tool(models, capsys):
    models.append(FakeModel([tool_call("submit", summary="Renamed", files_changed=["utils.py"])]))
    run("submit_tool")
    assert "'files_changed': ['utils.py']" in capsys.readouterr().out


def test_approval_no(models, answers, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    fake = FakeModel([tool_call("write_file", path="README.md", content="# Hi"), "I did not write it"])
    models.append(fake)
    answers.append("no")
    run("approval")
    assert not Path("README.md").exists()
    assert len(fake.requests) == 1  # the run stopped instead of asking the model again
    assert capsys.readouterr().out.splitlines()[-1] == "None"


def test_approval_stop(models, answers, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    fake = FakeModel([
        tool_call("write_file", path="README.md", content="# Hi"),
        tool_call("write_file", path="README.txt", content="Hi"),
        "Wrote README.txt",
    ])
    models.append(FakeModel(["Nothing to write"]))  # the run at the end of approval.py
    ns = run("approval")
    ns["agent"] = ns["agent"].copy(model=fake)
    answers.extend(["no", "Use README.txt instead", "yes"])
    run("approval_stop", ns=ns)
    assert Path("README.txt").read_text() == "Hi" and not Path("README.md").exists()
    assert "The user declined this call. Wait for their next message." in last_request_text(fake)
    assert "Use README.txt instead" in last_request_text(fake)
    assert capsys.readouterr().out.splitlines()[-1] == "Wrote README.txt"
    assert answers == []


def test_approval_yes(models, answers, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    models.append(FakeModel([tool_call("write_file", path="README.md", content="# Hi"), "Done"]))
    answers.append("yes")
    run("approval")
    assert Path("README.md").read_text() == "# Hi"


def test_approval_read_only_runs_without_asking(models, answers, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path("notes.md").write_text("hello")
    fake = FakeModel([tool_call("read_file", path="notes.md"), tool_call("made_up"), "It says hello"])
    models.append(fake)
    run("approval")  # no answers queued: asking would fail
    assert "hello" in last_request_text(fake)


def test_approval_always(models, answers, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    models.append(FakeModel([tool_call("write_file", path="a.md", content="a"), "Done"]))
    answers.append("yes")
    fake = FakeModel([
        tool_call("write_file", path="b.md", content="b"),
        tool_call("write_file", path="c.md", content="c"),
        "Done",
    ])
    models.append(fake)  # for the agent.copy(...) in approval_always.py
    ns = run("approval", "approval_always")
    answers.append("always")
    state = State("Write two files")
    ns["agent"].run(state)
    assert Path("b.md").exists() and Path("c.md").exists()
    assert answers == []
    assert state.root.data["always"] == ["write_file"]


def test_approval_hints(models, answers, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    models.append(FakeModel([
        tool_call("bash", command="mkdir build"),
        tool_call("bash", command="ls"),
        tool_call("bash", command="rm -rf build"),
        "unused",
    ]))
    ns = run("approval_hints")
    answers.append("no")  # only rm -rf build is asked about
    state = State("Clean up")
    ns["agent"].run(state)
    assert Path("build").is_dir()
    assert answers == []
    assert state.stopped == StoppedByPermission(state.history[-1].call, "DecideByHuman()")
    bash = ns["bash"]
    assert bash.hints_for({"command": "ls -la"}).read_only
    assert not bash.hints_for({"command": "mkdir build"}).destructive
    assert bash.hints_for({"command": "ls; rm -rf build"}).destructive
    assert bash.hints_for({"command": 5}).destructive


def test_tool_hints():
    bash = run("tool_hints")["bash"]
    assert bash.hints_for({"command": "ls"}).read_only
    assert not bash.hints_for({"command": "rm -rf build"}).read_only
    assert not bash.read_only


def test_approval_deny(models, answers, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path("notes.md").write_text("hello")
    models.append(FakeModel(["Nothing to write"]))  # the run at the end of approval.py
    fake = FakeModel([
        tool_call("read_file", path="../secret.txt"),
        tool_call("read_file", path="notes.md"),
        tool_call("write_file", path="/tmp/out.md", content="x"),
        "It says hello",
    ])
    models.append(fake)  # for the agent.copy(...) in approval_deny.py
    ns = run("approval", "approval_deny")
    state = State("Read the notes")
    ns["agent"].run(state)  # no answers queued: asking would fail
    denied = [h for h in state.history if h.kind == "denied"]
    assert [h.call.args["path"] for h in denied] == ["../secret.txt", "/tmp/out.md"]
    assert "is outside" in denied[0].content and "Use a path inside it." in denied[0].content
    assert "hello" in last_request_text(fake)
    assert repr(ns["agent"].permissions[0]) == f"DenyOutsideFolder({str(tmp_path.resolve())!r})"


def test_verify():
    ns = run("verify")
    results = iter(["1 failed: test_add", None])
    ns["failing_tests"] = lambda: next(results)
    state = State("Fix the tests")
    Agent(model=FakeModel(["Done", "Fixed"]), loop=ns["fix_until_green"], reporter=None).run(state)
    assert state.answer == "Fixed"
    assert state.turn == 2
    assert any("The tests still fail" in m.text for m in state.context)


def test_chat(models, answers, capsys):
    models.append(FakeModel(["Hello!", "More detail."]))
    answers.extend(["Hi", "Tell me more", EOFError()])
    with pytest.raises(EOFError):
        run("chat")
    out = capsys.readouterr().out
    assert "Hello!" in out and "More detail." in out


def test_resume(models, answers, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["chat.py"])
    models.append(FakeModel(["Hello!"]))
    answers.extend(["Hi", EOFError()])
    with pytest.raises(EOFError):
        run("resume")

    # Another process: --resume loads the conversation and continues it.
    monkeypatch.setattr(sys, "argv", ["chat.py", "--resume"])
    fake = FakeModel(["Welcome back."])
    models.append(fake)
    answers.extend(["Where were we?", EOFError()])
    with pytest.raises(EOFError):
        run("resume")
    assert [m.text for m in fake.requests[0].messages] == ["Hi", "Hello!", "Where were we?"]
    assert "Welcome back." in capsys.readouterr().out


def test_sigterm(monkeypatch):
    handlers = {}
    monkeypatch.setattr(signal, "signal", lambda signum, handler: handlers.__setitem__(signum, handler))
    run("sigterm")
    with pytest.raises(SystemExit) as info:
        handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert info.value.code == 128 + signal.SIGTERM


def test_structured_output(models, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    Path("change.diff").write_text("- a\n+ b\n")
    models.append(FakeModel(["Looks fine.", '{"approved": true, "reason": "Small and tested"}']))
    run("structured_output")
    assert "True Small and tested" in capsys.readouterr().out


def test_plan_first():
    ns = run("plan_first", "tool_state")
    seen = []

    def plan(request):
        seen.append(len(request.tools))
        return "1. Save a note"

    fake = FakeModel([plan, tool_call("remember", key="a", value="1"), "Done"])
    state = State("Save a note")
    agent = Agent(model=fake, tools=[ns["remember"]], loop=ns["plan_then_act"], reporter=None)
    agent.run(state)
    assert seen == [0]
    assert state.turn == 3
    assert state.stopped == StoppedByUntil("is_answered")


def test_nested_plain_loop_goes_on_after_the_inner_loop_reaches_its_limit():
    ns = run("nested_plain_loop")

    @tool
    def look() -> str:
        """Look."""
        return "seen"

    replies = [tool_call("look")] * 6 + ["Done"]
    state = State("Research")
    Agent(model=FakeModel(replies), tools=[look], loop=ns["three_rounds"], reporter=None).run(state)
    assert state.answer == "Done"
    assert state.stopped == StoppedByUntil("is_answered")


def test_nested_plain_loop_stops_on_a_permission_stop(answers):
    from alpineagents.permissions import DecideByHuman

    ns = run("nested_plain_loop")

    @tool
    def look() -> str:
        """Look."""
        return "seen"

    answers.append("no")
    state = State("Research")
    fake = FakeModel([tool_call("look"), "Done"])
    agent = Agent(model=fake, tools=[look], loop=ns["three_rounds"], permissions=[DecideByHuman()], reporter=None)
    agent.run(state)
    assert isinstance(state.stopped, StoppedByPermission)
    assert len(fake.requests) == 1


def test_tool_state():
    ns = run("tool_state")
    fake = FakeModel([tool_call("remember", key="a", value="1"), tool_call("recall", key="a"), "Done"])
    state = State("Remember a")
    Agent(model=fake, tools=[ns["remember"], ns["recall"]], reporter=None).run(state)
    results = [h.content for h in state.history if h.kind == "tool_result"]
    assert results == ["Saved a", "1"]
    assert state.data["notes"] == {"a": "1"}


def test_context_size():
    ns = run("context_size")

    @tool
    def read_log() -> str:
        """Read the log"""
        return "line\n" * 2000

    fake = FakeModel([tool_call("read_log"), "Summary of the work", "Done"], context_window=6000)
    state = State("Read the log")
    Agent(model=fake, tools=[read_log], loop=ns["long_task"], reporter=None).run(state)
    assert state.answer == "Done"
    assert [h.content.kind for h in state.history if h.kind == "context_change"] == ["compact"]


def test_async_agent(models, capsys):
    models.append(FakeModel([tool_call("run_command", command="echo 3.13"), "Python 3.13"]))
    run("async_agent")
    assert "Python 3.13" in capsys.readouterr().out


def test_reporter(caplog):
    ns = run("reporter")
    caplog.set_level(logging.INFO, logger="agent")
    tools = run("tool_state")
    fake = FakeModel([tool_call("recall", key="a"), "Done"])
    Agent(model=fake, tools=[tools["recall"]], reporter=ns["LogReporter"]()).run("Recall a")
    assert "turn 1: recall({'key': 'a'})" in caplog.text
    assert "stopped by is_answered after 2 turns" in caplog.text


def test_human(models, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    models.append(FakeModel(["Nothing to write"]))
    ns = run("approval", "human")
    fake = FakeModel([tool_call("write_file", path="x.md", content="x"), "Skipped"])
    agent = ns["agent"].copy(model=fake, human=ns["Unattended"](), reporter=None)
    state = State("Write x.md")
    agent.run(state)
    assert not Path("x.md").exists()
    assert isinstance(state.stopped, StoppedByPermission)
    assert state.stopped.permission == "DecideByHuman()"


class _Drops(Model):
    """Wraps a FakeModel. The first requests fail with the given errors, after streaming "partial " if asked."""

    name = "drops"
    provider = "fake"

    def __init__(self, fake: FakeModel, errors: list[Exception], *, after_text: bool = True):
        self.fake = fake
        self.errors = list(errors)
        self.after_text = after_text
        self.calls = 0

    @property
    def context_window(self) -> int:
        return self.fake.context_window

    def respond(self, request, on_text=None, on_event=None):
        self.calls += 1
        if self.errors:
            if self.after_text and on_text:
                on_text("partial ")
            raise self.errors.pop(0)
        return self.fake.respond(request, on_text, on_event)


def test_retry_stream():
    ns = run("retry_stream", "retry_reporter")
    assert ns["agent"].model.name == "claude-sonnet-5"
    inner = _Drops(FakeModel(["The answer"]), [ProviderError("connection reset")])
    wrapped = ns["RetryDroppedStream"](inner)
    assert (wrapped.name, wrapped.provider, wrapped.context_window) == ("drops", "fake", 200_000)
    reporter = ns["LiveText"]()
    state = State("Question")
    assert Agent(model=wrapped, reporter=reporter).run(state) == "The answer"
    assert inner.calls == 2
    assert reporter.text == "The answer"
    events = [h.content for h in state.history if h.kind == "model_event"]
    assert [e.kind for e in events] == ["retry"]
    assert "connection reset" in events[0].message


def test_retry_stream_async():
    import asyncio

    ns = run("retry_stream")
    inner = _Drops(FakeModel(["The answer"]), [ProviderError("dropped"), ProviderError("dropped")])
    agent = Agent(model=ns["RetryDroppedStream"](inner), reporter=None)
    assert asyncio.run(agent.arun("Question")) == "The answer"
    assert inner.calls == 3


@pytest.mark.parametrize(
    ("errors", "after_text", "calls"),
    [
        ([ProviderError("down")], False, 1),  # before the stream: the SDK already retried
        ([AuthError("bad key")], True, 1),
        ([ContextTooLongError("too long")], True, 1),
        ([ProviderError("dropped")] * 3, True, 3),  # every attempt used
    ],
)
def test_retry_stream_raises(errors, after_text, calls):
    ns = run("retry_stream")
    inner = _Drops(FakeModel(["The answer"]), errors, after_text=after_text)
    with pytest.raises(type(errors[-1])):
        Agent(model=ns["RetryDroppedStream"](inner), reporter=None).run("Question")
    assert inner.calls == calls


def test_nudge():
    ns = run("nudge")

    @tool
    def look() -> str:
        """Look around"""
        return "a file"

    fake = FakeModel(["I will look now", tool_call("look"), "Done", "Done", "Done"])
    state = State("Look")
    Agent(model=fake, tools=[look], loop=ns["nudging"], reporter=None).run(state)
    assert state.answer == "Done"
    assert fake.remaining == 0
    notices = [h.content for h in state.history if h.kind == "notice"]
    assert len(notices) == 3 and notices[0] == "[notice] " + ns["NUDGE"]
    assert state.stopped == StoppedByUntil("is_answered")


def test_nudge_gives_up():
    ns = run("nudge")
    fake = FakeModel(["Maybe", "Maybe", "Maybe", "unused"])
    state = State("Look")
    Agent(model=fake, loop=ns["nudging"], reporter=None).run(state)
    assert fake.remaining == 1
    assert len([h for h in state.history if h.kind == "notice"]) == ns["NUDGES"]


def test_repeating():
    ns = run("repeating")

    @tool
    def look(path: str) -> str:
        """Look at a path"""
        return "nothing"

    same = [tool_call("look", path="a") for _ in range(3)]
    fake = FakeModel([tool_call("look", path="b"), *same, "unused"])
    state = State("Look")
    Agent(model=fake, tools=[look], loop=ns["watched"], reporter=None).run(state)
    assert state.stopped == StoppedByUntil("repeating")
    assert fake.remaining == 1


def test_custom_model():
    ns = run("custom_model")
    state = State("hello")
    assert Agent(model=ns["Echo"](), reporter=None).run(state) == "hello"
    assert [h.content.model for h in state.history if h.kind == "reply"] == ["echo"]


def test_testing_example(tmp_path, monkeypatch):
    ns = run("test_example")
    ns["test_reads_the_file_then_answers"](tmp_path, monkeypatch)


def test_testing_tool_failure_example(tmp_path, monkeypatch):
    ns = run("test_tool_failure")
    ns["test_a_missing_file_is_an_error_result"](tmp_path, monkeypatch)


def test_tool_object(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path("my-repo").mkdir()
    Path("my-repo/a.txt").write_text("hello")
    ns = run("tool_object")
    fake = FakeModel([tool_call("list_files"), tool_call("read_file", path="a.txt"), "It says hello"])
    state = State("What is in the repo?")
    ns["agent"].copy(model=fake, reporter=None).run(state)
    assert [h.content for h in state.history if h.kind == "tool_result"] == ['["a.txt"]', "hello"]
    assert [t.name for t in fake.requests[0].tools] == ["read_file", "list_files"]
    reader_fake = FakeModel(["Done"])
    ns["reader"].copy(model=reader_fake, reporter=None).run("Read a.txt")
    assert [t.name for t in reader_fake.requests[0].tools] == ["read_file"]


def test_model_settings(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    ns = run("model_settings")
    assert ns["ollama"].context_window == 32_000
    assert ns["agent"].model is ns["claude"]


def test_mcp_servers(monkeypatch):
    pytest.importorskip("mcp")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    monkeypatch.setenv("LINEAR_API_KEY", "lin_test")
    ns = run("mcp_servers")
    assert len(ns["agent"].tools) == 2


# ---------------------------------------------------------------- the pages themselves


def test_every_docs_src_file_is_used():
    pages = "\n".join(p.read_text() for p in DOC_FILES)
    for path in DOCS_SRC.glob("*.py"):
        assert f"docs_src/{path.name}" in pages or path.read_text().strip() in pages, path.name


def test_default_loop_on_the_concepts_page_matches_the_source():
    page = (ROOT / "docs" / "concepts" / "overview.md").read_text()
    source = (ROOT / "src" / "alpineagents" / "loop.py").read_text()
    for line in ["@loop(until=State.is_answered, limit=50)", "def default_loop(agent: Agent, state: State):",
                 "    compact_if_full(agent, state)", "    agent.think(state)", "    if state.wants_tools():",
                 "        agent.use_tools(state)"]:
        assert line in page and line in source, line


@pytest.mark.parametrize("path", DOC_FILES + sorted(DOCS_SRC.glob("*.py")), ids=lambda p: p.name)
def test_no_em_dash(path):
    assert "—" not in path.read_text()


def test_internal_links_point_to_pages():
    for page in (ROOT / "docs").rglob("*.md"):
        for target in re.findall(r"\]\(([^)#:]+\.md)(?:#[^)]*)?\)", page.read_text()):
            assert (page.parent / target).resolve().exists(), f"{page.name} -> {target}"
