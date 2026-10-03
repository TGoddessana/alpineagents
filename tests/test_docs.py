"""The code in the documentation runs as written.

Each file in ``docs_src/`` is included in a docs page as is. These tests execute those files with the model
strings replaced by ``FakeModel``s and ``input()`` replaced by prepared answers, so no network is used.
"""

from __future__ import annotations

import builtins
import logging
import re
import signal
import subprocess
import sys
import types
from pathlib import Path

import pytest

import alpineagents.agent as agent_module
from alpineagents import (
    Agent,
    AuthError,
    ContextTooLongError,
    FileStore,
    Image,
    Message,
    Model,
    ProviderError,
    Reply,
    State,
    StateSnapshot,
    StoppedByPermission,
    StoppedByUntil,
    ToolInputError,
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


def user(text: str) -> State:
    """A State that starts with one user message, the 0.5 way to write user("text")."""
    return State(messages=[Message.user(text)])


# A 1x1 PNG, so the image examples need no file.
PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa7\x1c\xa1\xd5\x00\x00\x00\x00IEND\xaeB`\x82"
)


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


def test_no_api_key(monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    run("no_api_key")
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


# ---------------------------------------------------------------- guides


def test_stop_conditions_guide_lists_every_way_a_loop_stops():
    from typing import get_args

    from alpineagents.types import Stopped

    guide = (ROOT / "docs" / "guides" / "stop-conditions.md").read_text()
    ways = get_args(Stopped)
    assert f"one of {['zero', 'one', 'two', 'three', 'four', 'five'][len(ways)]} ways" in guide
    for way in ways:
        assert f"`{way.__name__}(" in guide, way.__name__


def test_stop_conditions(models, capsys):
    models.append(FakeModel([tool_call("read_file", path="README.md"), "Done"]))
    ns = run("stop_conditions")
    assert "stopped by waiting_for_user" in capsys.readouterr().out.splitlines()
    big = Reply(
        message=Message("assistant", (tool_call("unknown"),)),
        usage=Usage(output_tokens=30_000, requests=1),
    )
    state = user("Work")
    Agent(model=FakeModel([big]), loop=ns["careful"], reporter=None).run(state)
    assert state.stopped == StoppedByUntil("spent_too_much")


def test_stop_conditions_reports_the_limit(models, capsys):
    models.append(FakeModel([tool_call("read_file", path="README.md")] * 40))
    run("stop_conditions")
    assert "Stopped after 30 turns. The task may be unfinished." in capsys.readouterr().out


def test_submit_tool(models, capsys):
    models.append(FakeModel([tool_call("submit", summary="Renamed", files_changed=["utils.py"])]))
    run("submit_tool")
    assert "'files_changed': ['utils.py']" in capsys.readouterr().out


def test_approval_stop(models, answers, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    fake = FakeModel([
        tool_call("write_file", path="README.md", content="# Hi"),
        tool_call("write_file", path="README.txt", content="Hi"),
        "Wrote README.txt",
    ])
    models.append(fake)
    answers.extend(["no", "Use README.txt instead", "yes"])
    run("approval_stop")
    assert Path("README.txt").read_text() == "Hi" and not Path("README.md").exists()
    assert "The user declined this call. Wait for their next message." in last_request_text(fake)
    assert "Use README.txt instead" in last_request_text(fake)
    assert capsys.readouterr().out.splitlines()[-1] == "Wrote README.txt"
    assert answers == []


def test_approval_no_stops_instead_of_asking_the_model_again(models, answers, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    fake = FakeModel([tool_call("write_file", path="README.md", content="# Hi"), "Not written"])
    models.append(fake)
    answers.extend(["no", EOFError()])
    with pytest.raises(EOFError):  # the example asks what to do instead
        run("approval_stop")
    assert not Path("README.md").exists()
    assert len(fake.requests) == 1


def test_approval_yes(models, answers, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    models.append(FakeModel([tool_call("write_file", path="README.md", content="# Hi"), "Done"]))
    answers.append("yes")
    run("approval_stop")
    assert Path("README.md").read_text() == "# Hi"


def test_approval_read_only_runs_without_asking(models, answers, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path("notes.md").write_text("hello")
    fake = FakeModel([tool_call("read_file", path="notes.md"), tool_call("made_up"), "It says hello"])
    models.append(fake)
    run("approval_stop")  # no answers queued: asking would fail
    assert "hello" in last_request_text(fake)


def test_approval_always(models, answers, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    models.append(FakeModel([
        tool_call("write_file", path="README.md", content="a"),
        tool_call("write_file", path="CONTRIBUTING.md", content="b"),
        "Done",
    ]))
    answers.append("always")
    ns = run("approval_always")  # asked once: "always" covers the second call
    assert Path("README.md").exists() and Path("CONTRIBUTING.md").exists()
    assert answers == []

    answers.append("always")
    fake = FakeModel([tool_call("write_file", path="x.md", content="x"), "Done"])
    state = user("Write x")
    ns["agent"].copy(model=fake, reporter=None).run(state)
    assert state.extra_data["always"] == ["write_file"]


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
    state = user("Clean up")
    ns["agent"].run(state)
    assert Path("build").is_dir()
    assert answers == []
    last_result = [h for h in state.history if h.kind == "tool_result"][-1]
    assert last_result.outcome == "denied"
    assert state.stopped == StoppedByPermission(last_result.call, "DecideByHuman()")
    bash = ns["bash"]
    assert bash.hints_for({"command": "ls -la"}).read_only
    assert not bash.hints_for({"command": "mkdir build"}).destructive
    assert bash.hints_for({"command": "ls; rm -rf build"}).destructive
    assert bash.hints_for({"command": 5}).destructive



def test_approval_deny(models, answers, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path("notes.md").write_text("hello")
    fake = FakeModel([
        tool_call("read_file", path="../secret.txt"),
        tool_call("read_file", path="notes.md"),
        tool_call("write_file", path="/tmp/out.md", content="x"),
        "It says hello",
    ])
    models.append(fake)
    ns = run("approval_deny")  # no answers queued: asking would fail
    text = last_request_text(fake)
    assert "is outside" in text and "Use a path inside it." in text and "hello" in text
    assert repr(ns["agent"].permissions[0]) == f"DenyOutsideFolder({str(tmp_path.resolve())!r})"


def test_verify(models, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    outputs = iter([types.SimpleNamespace(returncode=1, stdout="1 failed: test_add"),
                    types.SimpleNamespace(returncode=0, stdout="")])
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: next(outputs))
    fake = FakeModel([tool_call("write_file", path="calc.py", content="x = 1"), "Done", "Fixed"])
    models.append(fake)
    run("verify")
    assert capsys.readouterr().out.splitlines()[-1] == "Fixed"
    assert fake.remaining == 0
    assert "The tests still fail" in last_request_text(fake) and "1 failed: test_add" in last_request_text(fake)


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


def test_sigterm(models, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    models.append(FakeModel(["Report"]))
    handlers = {}
    monkeypatch.setattr(signal, "signal", lambda signum, handler: handlers.__setitem__(signum, handler))
    run("sigterm")
    assert Path(".agent-runs/nightly-report").is_dir()
    with pytest.raises(SystemExit) as info:
        handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert info.value.code == 128 + signal.SIGTERM


def test_structured_output(models, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    Path("change.diff").write_text("- a\n+ b\n")
    models.append(FakeModel(["Looks fine.", '{"approved": true, "reason": "Small and tested"}']))
    run("structured_output")
    assert "True Small and tested" in capsys.readouterr().out


def test_plan_first(models, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    seen = []

    def plan(request):
        seen.append(len(request.tools))
        return "1. List the files"

    fake = FakeModel([plan, tool_call("list_files"), "A docs folder"])
    models.append(fake)
    run("plan_first")
    assert seen == [0]
    assert [len(r.tools) for r in fake.requests] == [0, 2, 2]
    assert "[notice] Now carry out the plan." in last_request_text(fake)
    assert capsys.readouterr().out.splitlines()[-1] == "A docs folder"


def test_nested_plain_loop_goes_on_after_the_inner_loop_reaches_its_limit(models):
    models.append(FakeModel(["Done"]))
    ns = run("nested_plain_loop")
    replies = [tool_call("look", topic="cache")] * 6 + ["Done"]
    state = user("Research")
    Agent(model=FakeModel(replies), tools=[ns["look"]], loop=ns["three_rounds"], reporter=None).run(state)
    assert state.answer == "Done"
    assert state.stopped == StoppedByUntil("waiting_for_user")


def test_nested_plain_loop_stops_on_a_permission_stop(models, answers):
    from alpineagents.permissions import DecideByHuman

    models.append(FakeModel(["Done"]))
    ns = run("nested_plain_loop")
    answers.append("no")
    state = user("Research")
    fake = FakeModel([tool_call("look", topic="cache"), "Done"])
    agent = Agent(model=fake, tools=[ns["look"]], loop=ns["three_rounds"], permissions=[DecideByHuman()], reporter=None)
    agent.run(state)
    assert isinstance(state.stopped, StoppedByPermission)
    assert len(fake.requests) == 1


def test_tool_state(models, capsys):
    fake = FakeModel([tool_call("remember", key="a", value="1"), tool_call("recall", key="a"), "Done"])
    models.append(fake)
    ns = run("tool_state")
    assert capsys.readouterr().out.splitlines()[-1] == "Done"
    assert "Saved a" in last_request_text(fake)
    state = user("Remember a")
    fake = FakeModel([tool_call("remember", key="a", value="1"), tool_call("recall", key="a"), "Done"])
    ns["agent"].copy(model=fake, reporter=None).run(state)
    results = [h.content for h in state.history if h.kind == "tool_result"]
    assert results == ["Saved a", "1"]
    assert state.extra_data["notes"] == {"a": "1"}


def test_context_size(models, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path("big.log").write_text("line\n" * 2000)
    replies = [tool_call("read_file", path="big.log"), "Summary of the work", "Done"]
    models.append(FakeModel(replies, context_window=6000))
    ns = run("context_size")
    fake = FakeModel(replies, context_window=6000)
    state = user("Read the log")
    ns["agent"].copy(model=fake, reporter=None).run(state)
    assert state.answer == "Done"
    kinds = [h.content.kind for h in state.history if h.kind == "context_change"]
    assert kinds == ["import", "compact", "clear_tool_results"][:len(kinds)] and "compact" in kinds


def test_async_agent(models, capsys):
    models.append(FakeModel([tool_call("run_command", command="echo 3.13"), "Python 3.13"]))
    run("async_agent")
    assert "Python 3.13" in capsys.readouterr().out


def test_reporter(models, caplog):
    models.append(FakeModel(["unused"]))
    ns = run("reporter")
    caplog.set_level(logging.INFO, logger="agent")

    @tool
    def recall(key: str) -> str:
        """Read a note"""
        return "1"

    ns["agent"].copy(model=FakeModel([tool_call("recall", key="a"), "Done"]), tools=[recall]).run("Recall a")
    assert "turn 1: recall({'key': 'a'})" in caplog.text
    assert "stopped by waiting_for_user after 2 turns" in caplog.text


def test_human(models, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    models.append(FakeModel([tool_call("write_file", path="x.md", content="x"), "unused"]))
    ns = run("human")
    assert not Path("x.md").exists()
    fake = FakeModel([tool_call("write_file", path="x.md", content="x"), "Skipped"])
    state = user("Write x.md")
    ns["agent"].copy(model=fake, reporter=None).run(state)
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
    ns = run("retry_stream")
    assert ns["agent"].model.name == "claude-sonnet-5"
    inner = _Drops(FakeModel(["The answer"]), [ProviderError("connection reset")])
    wrapped = ns["RetryDroppedStream"](inner)
    assert (wrapped.name, wrapped.provider, wrapped.context_window) == ("drops", "fake", 200_000)
    reporter = run("retry_reporter")["LiveText"]()
    state = user("Question")
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


def test_nudge(models):
    models.append(FakeModel(["Still going", "Still going", "Done"]))
    ns = run("nudge")

    fake = FakeModel(["I will look now", tool_call("look"), "Done", "Done", "Done"])
    state = user("Look")
    Agent(model=fake, tools=[ns["look"]], loop=ns["nudging"], reporter=None).run(state)
    assert state.answer == "Done"
    assert fake.remaining == 0
    notices = [h.content for h in state.history if h.kind == "notice"]
    assert len(notices) == 3 and notices[0] == "[notice] " + ns["NUDGE"]
    assert state.stopped == StoppedByUntil("waiting_for_user")


def test_nudge_gives_up(models):
    models.append(FakeModel(["Maybe"] * 3))
    ns = run("nudge")
    fake = FakeModel(["Maybe", "Maybe", "Maybe", "unused"])
    state = user("Look")
    Agent(model=fake, loop=ns["nudging"], reporter=None).run(state)
    assert fake.remaining == 1
    assert len([h for h in state.history if h.kind == "notice"]) == ns["NUDGES"]


def test_repeating(models):
    models.append(FakeModel(["Done"]))
    ns = run("repeating")
    same = [tool_call("look", path="a") for _ in range(3)]
    fake = FakeModel([tool_call("look", path="b"), *same, "unused"])
    state = user("Look")
    Agent(model=fake, tools=[ns["look"]], loop=ns["watched"], reporter=None).run(state)
    assert state.stopped == StoppedByUntil("repeating")
    assert fake.remaining == 1


def test_custom_model(capsys):
    ns = run("custom_model")
    assert capsys.readouterr().out.splitlines()[-1] == "hello"
    state = user("hello")
    assert Agent(model=ns["Echo"](), reporter=None).run(state) == "hello"
    assert [h.content.model for h in state.history if h.kind == "model_reply"] == ["echo"]


def test_testing_example(tmp_path, monkeypatch):
    ns = run("test_example")
    ns["test_reads_the_file_then_answers"](tmp_path, monkeypatch)


def test_testing_tool_failure_example(tmp_path, monkeypatch):
    ns = run("test_tool_failure")
    ns["test_a_missing_file_is_an_error_result"](tmp_path, monkeypatch)


def test_tool_failure_abort(models, capsys):
    models.append(FakeModel([tool_call("get_json", path="/legacy")]))
    run("tool_failure_abort")
    out = capsys.readouterr().out
    assert "exception raised in tool get_json" in out
    assert "tool_result get_json (error): (aborted: JSONDecodeError)" in out


def test_tool_object(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path("my-repo").mkdir()
    Path("my-repo/a.txt").write_text("hello")
    ns = run("tool_object")
    fake = FakeModel([tool_call("list_files"), tool_call("read_file", path="a.txt"), "It says hello"])
    state = user("What is in the repo?")
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


# ---------------------------------------------------------------- 0.5 sections of the guides


def test_switch_model(models, capsys):
    fast, strong = FakeModel(["Plan: three steps"]), FakeModel(["Backfill in batches"])
    models.extend([fast, strong])
    state_ns = run("switch_model")
    out = capsys.readouterr().out
    assert "Plan: three steps" in out and "Backfill in batches" in out
    assert [m.text for m in strong.requests[0].messages] == [
        "Outline a plan to move the users table to a new schema",
        "Plan: three steps",
        "Now write the data backfill. It must not lock the table",
    ]
    assert state_ns["fast"].model is not state_ns["strong"].model


def test_switch_model_is_recorded_and_does_not_warn(models, recwarn):
    from alpineagents import ResumeWarning

    models.extend([FakeModel(["one"], name="fast"), FakeModel(["two"], name="strong")])
    ns = run("switch_model")
    assert not [w for w in recwarn if issubclass(w.category, ResumeWarning)]
    state = ns["state"]
    requested = [h.content for h in state.history if h.kind == "model_request"]
    assert requested == ["fake/fast", "fake/strong"]
    assert state.usage.requests == 2


def test_snapshot(models, capsys):
    models.append(FakeModel(["Fixed calc.py", "Rewritten with a function"]))
    ns = run("snapshot")
    state, before = ns["state"], ns["before"]
    assert isinstance(before, StateSnapshot)
    assert "True" in capsys.readouterr().out.splitlines()
    assert state.answer == "Fixed calc.py"
    assert state.messages == before.messages
    assert state.usage.requests == 2  # the tokens of the undone run stay counted
    assert state.history[-1].kind == "context_change" and state.history[-1].content.kind == "restore"
    assert state.history[: len(before.history)] == before.history


def test_snapshot_is_frozen_and_equal_to_a_replay():
    import dataclasses

    state = user("Hi")
    snap = state.snapshot()
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.turn = 5  # type: ignore[misc]
    state.add_message(Message.user("More"))
    assert snap != state.snapshot() and len(snap.messages) == 1
    assert State(history=state.history).snapshot() == state.snapshot()


def test_fork(models, capsys):
    models.append(FakeModel(["Fixed calc.py", "As a class"]))
    ns = run("fork")
    state, attempt = ns["state"], ns["attempt"]
    out = capsys.readouterr().out.splitlines()
    assert out[-2:] == ["Fixed calc.py", "As a class"]
    assert attempt.id != state.id
    assert len(attempt.history) > len(state.history)
    assert state.answer == "Fixed calc.py"


def test_user_image(models, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    Path("screen.png").write_bytes(PNG)
    Path("screen-2.png").write_bytes(PNG)
    fake = FakeModel(["A button is missing", "Nothing is wrong"])
    models.append(fake)
    run("user_image")
    first = fake.requests[0].messages[0]
    assert [type(block).__name__ for block in first.content] == ["TextBlock", "Image"]
    last = fake.requests[1].messages[-1]
    assert last.text == "And in this one?" and any(isinstance(block, Image) for block in last.content)
    assert "Nothing is wrong" in capsys.readouterr().out


def test_chat_with_an_image_is_saved_and_loaded(tmp_path):
    store = FileStore(tmp_path)
    state = State(id="shot", messages=[Message.user("What is this?", Image(PNG))])
    Agent(model=FakeModel(["A pixel"]), reporter=None, store=store).run(state)
    loaded = store.load("shot")
    assert loaded.snapshot() == state.snapshot()
    assert any(isinstance(b, Image) for b in loaded.messages[0].content)


def test_resume_guide_invariants(tmp_path):
    """What the resume guide promises: a loaded State equals the saved one, and States list by first message."""
    store = FileStore(tmp_path)
    state = State(id="r1", messages=[Message.user("Hello there")])
    Agent(model=FakeModel(["Hi"]), reporter=None, store=store).run(state)
    state.add_message(Message.user("Next"))
    store.save(state)
    loaded = store.load("r1")
    assert loaded.snapshot() == state.snapshot()
    assert State(history=loaded.history).snapshot() == loaded.snapshot()
    assert [(i.id, i.first_message) for i in store.list()] == [("r1", "Hello there")]


def test_tool_state_edits_are_atomic_across_threads():
    from concurrent.futures import ThreadPoolExecutor

    state = State()

    def bump(_):
        with state.edit_extra_data() as data:
            data["n"] = data.get("n", 0) + 1

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(bump, range(50)))
    assert state.extra_data["n"] == 50
    with pytest.raises(TypeError, match="must be JSON"):
        with state.edit_extra_data() as data:
            data["seen"] = {1, 2}
    with pytest.raises(TypeError):
        state.extra_data["n"] = 0  # type: ignore[index]


def test_tool_subclass(models, monkeypatch, capsys):
    class Response:
        status_code = 200

        def json(self):
            return {"id": 7}

    posted = []
    fake_httpx = types.SimpleNamespace(post=lambda url, json: posted.append((url, json)) or Response())
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    fake = FakeModel([tool_call("create_ticket", title="Login down"), "Ticket 7 is open"])
    models.append(fake)
    ns = run("tool_subclass")
    assert posted == [("https://example.com/hooks/tickets", {"title": "Login down"})]
    assert "Ticket 7 is open" in capsys.readouterr().out
    webhook = ns["agent"].tool_map["create_ticket"]
    with pytest.raises(ToolInputError):
        webhook.run({}, State())


def test_tool_image(models, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    Path("sales.png").write_bytes(PNG)
    fake = FakeModel([tool_call("show_chart", path="sales.png"), "Sales go up"])
    models.append(fake)
    run("tool_image")
    result = [m for m in fake.requests[1].messages if m.role == "user"][-1]
    assert any(isinstance(block, Image) or any(isinstance(b, Image) for b in getattr(block, "content", ()))
               for block in result.content)
    assert "Sales go up" in capsys.readouterr().out


def test_own_store(models, capsys):
    models.append(FakeModel(["Hello"]))
    ns = run("own_store")
    store = ns["store"]
    assert capsys.readouterr().out.splitlines()[-1] == "Hello"
    assert [info.id for info in store.list()] == ["demo"]
    saved = store.load("demo")
    assert [m.text for m in saved.messages] == ["Say hello", "Hello"]
    store.delete("demo")
    assert store.list() == [] and store.read("demo") is None


def test_own_store_refuses_a_second_create(models):
    models.append(FakeModel(["Hello"]))
    ns = run("own_store")
    store = ns["store"]
    with pytest.raises(ValueError, match="already exists"):
        store.write("demo", [], store.records["demo"].info, create=True)
    with pytest.raises(LookupError):
        store.write("other", [], store.records["demo"].info)


# ---------------------------------------------------------------- the Learn pages


def test_learn_tools(models, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    Path("README.md").write_text("# Title\nSecond line\n")
    fake = FakeModel([tool_call("read_file", path="README.md", max_lines=1), "# Title"])
    models.append(fake)
    run("learn_tools")
    out = capsys.readouterr().out
    assert "ToolSpec(name='read_file', description='Read a text file'" in out
    assert "'description': 'How many lines to return'" in out
    assert out.splitlines()[-1] == "# Title"
    assert "# Title" in last_request_text(fake) and "Second line" not in last_request_text(fake)


def test_learn_loop(models, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    Path("README.md").write_text("hello")
    models.append(FakeModel([tool_call("read_file", path="README.md"), "It says hello"]))
    ns = run("learn_loop")
    assert "It says hello" in capsys.readouterr().out
    assert ns["coding"].until == (ns["waiting_for_user"],)


def test_learn_conversation(models, capsys):
    models.append(FakeModel(["Flask, Django and FastAPI.", "Flask."]))
    run("learn_conversation")
    lines = capsys.readouterr().out.splitlines()
    assert "stopped by waiting_for_user" in lines
    assert "[turn 1] model_reply: Flask, Django and FastAPI." in lines
    assert lines[-1] == "Flask."


def test_learn_conversation_notice_continues_the_run():
    state = user("Hi")
    fake = FakeModel(["Hello", "Fixing it"])
    agent = Agent(model=fake, reporter=None)
    agent.run(state)
    state.add_message(Message.notice("The tests failed"))
    assert agent.run(state) == "Fixing it"
    assert fake.requests[1].messages[-1].text == "[notice] The tests failed"


def test_testing_approval_flow_example(tmp_path, monkeypatch):
    ns = run("test_approval_flow")
    ns["test_no_stops_the_run_and_writes_nothing"](tmp_path, monkeypatch)


def test_testing_resume_example(tmp_path):
    ns = run("test_resume")
    ns["test_a_saved_conversation_loads_as_it_was"](tmp_path)


# ---------------------------------------------------------------- the pages themselves


def test_every_docs_src_file_is_used():
    pages = "\n".join(p.read_text() for p in DOC_FILES)
    for path in DOCS_SRC.glob("*.py"):
        assert f"docs_src/{path.name}" in pages or path.read_text().strip() in pages, path.name


def test_default_loop_on_the_learn_page_matches_the_source():
    page = (ROOT / "docs" / "learn" / "loop.md").read_text()
    source = (ROOT / "src" / "alpineagents" / "loop.py").read_text()
    for line in ["@loop(until=waiting_for_user, limit=50)", "def default_loop(agent: Agent, state: State):",
                 "    compact_if_full(agent, state)", "    agent.think(state)", "    if state.pending_calls:",
                 "        agent.use_tools(state)"]:
        assert line in page and line in source, line


@pytest.mark.parametrize("path", DOC_FILES + sorted(DOCS_SRC.glob("*.py")), ids=lambda p: p.name)
def test_no_em_dash(path):
    assert "—" not in path.read_text()


def test_internal_links_point_to_pages():
    for page in (ROOT / "docs").rglob("*.md"):
        for target in re.findall(r"\]\(([^)#:]+\.md)(?:#[^)]*)?\)", page.read_text()):
            assert (page.parent / target).resolve().exists(), f"{page.name} -> {target}"


def test_every_page_in_the_nav_exists_and_every_page_is_in_the_nav():
    nav = {(ROOT / "docs" / p).resolve() for p in re.findall(r":\s+([\w/-]+\.md)\s*$", (ROOT / "mkdocs.yml").read_text(), re.M)}
    pages = {p.resolve() for p in (ROOT / "docs").rglob("*.md")}
    assert nav == pages


def test_inline_python_blocks_are_valid_python():
    import ast

    for page in DOC_FILES:
        for block in re.findall(r"```python\n(.*?)```", page.read_text(), re.S):
            if "--8<--" not in block:
                ast.parse(block)
