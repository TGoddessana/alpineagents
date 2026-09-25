"""The code in the documentation runs as written.

Each file in ``docs_src/`` is included in a docs page as is. These tests execute those files with the model
strings replaced by ``FakeModel``s and ``input()`` replaced by prepared answers, so no network is used.
"""

from __future__ import annotations

import builtins
import logging
import re
import sys
import types
from pathlib import Path

import pytest

import alpineagents.agent as agent_module
from alpineagents import Agent, Message, Model, Reply, State, Usage, tool
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


def test_stop_conditions():
    ns = run("stop_conditions")
    big = Reply(
        message=Message("assistant", (tool_call("unknown"),)),
        usage=Usage(output_tokens=30_000, requests=1),
    )
    state = State("Work")
    Agent(model=FakeModel([big]), loop=ns["careful"], reporter=None).run(state)
    assert state.stopped_by == "spent_too_much"


def test_submit_tool(models, capsys):
    models.append(FakeModel([tool_call("submit", summary="Renamed", files_changed=["utils.py"])]))
    run("submit_tool")
    assert "'files_changed': ['utils.py']" in capsys.readouterr().out


def test_approval_no(models, answers, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    fake = FakeModel([tool_call("write_file", path="README.md", content="# Hi"), "I did not write it"])
    models.append(fake)
    answers.append("no")
    run("approval")
    assert not Path("README.md").exists()
    assert "The user declined this call" in last_request_text(fake)


def test_approval_yes(models, answers, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    models.append(FakeModel([tool_call("write_file", path="README.md", content="# Hi"), "Done"]))
    answers.append("yes")
    run("approval")
    assert Path("README.md").read_text() == "# Hi"


def test_approval_always(models, answers, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    models.append(FakeModel([tool_call("write_file", path="a.md", content="a"), "Done"]))
    answers.append("yes")
    ns = run("approval", "approval_always")
    fake = FakeModel([
        tool_call("write_file", path="b.md", content="b"),
        tool_call("write_file", path="c.md", content="c"),
        "Done",
    ])
    answers.append("always")
    ns["agent"].copy(model=fake, loop=ns["careful"], reporter=None).run("Write two files")
    assert Path("b.md").exists() and Path("c.md").exists()
    assert answers == []


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
    assert state.stopped_by == "is_answered"


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
    assert agent.run("Write x.md") == "Skipped"
    assert not Path("x.md").exists()


def test_custom_model():
    ns = run("custom_model")
    assert Agent(model=ns["Echo"](), reporter=None).run("hello") == "hello"


def test_testing_example(tmp_path, monkeypatch):
    ns = run("test_example")
    ns["test_reads_the_file_then_answers"](tmp_path, monkeypatch)


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
