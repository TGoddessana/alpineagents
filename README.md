# alpineagents

[![CI](https://github.com/TGoddessana/alpineagents/actions/workflows/ci.yml/badge.svg)](https://github.com/TGoddessana/alpineagents/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/TGoddessana/alpineagents/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-tgoddessana.github.io-blue.svg)](https://tgoddessana.github.io/alpineagents/)

A Python agent framework where the agent loop is a function you write.

- `Agent` holds the settings: the model, the system prompt and the tools.
- `State` holds one task: the messages so far and the answer.
- A loop takes both and repeats one turn until a stop condition is true.

## Why

In most frameworks the loop lives inside the library. LangGraph has you declare it as a graph. The OpenAI Agents SDK
and PydanticAI run it for you.

In alpineagents the loop is a few lines of Python in your own code:

- `until=` decides when it stops.
- `limit=` sets the maximum number of turns.
- Compaction runs only where you call it.

The default loop, `alpineagents.default_loop`, is written the same way. Read it, or copy it as a starting point.
Misuse raises an error that says how to fix it.

## Install

Requires Python 3.11 or later.

```bash
pip install alpineagents     # or: uv add alpineagents
```

Anthropic models read `ANTHROPIC_API_KEY`. OpenAI-compatible servers read `OPENAI_API_KEY`.

## Quick start

Define a tool, create an Agent, run a task:

```python
from pathlib import Path

from alpineagents import Agent, tool


@tool
def read_file(path: str) -> str:
    """Read a text file"""
    file = Path(path)
    return file.read_text() if file.exists() else f"No such file: {path}"


agent = Agent(model="claude-sonnet-5", tools=[read_file])
print(agent.run("Summarize README.md in three lines"))
```

- `@tool` turns the function into a tool. The type hints and the docstring tell the model how to call it.
- `agent.run` repeats turns until the model answers, then returns the answer.
- Progress is printed to the terminal while it runs.

### Without an API key

`FakeModel` returns prepared replies in order. It needs no API key and no network. Replace the `print` line above with:

```python
from alpineagents.testing import FakeModel, tool_call

fake = FakeModel([
    tool_call("read_file", path="README.md"),
    "README.md describes a Python agent framework.",
])
print(agent.copy(model=fake).run("Summarize README.md"))
```

The first reply asks for `read_file`. The Agent runs the tool, and the second reply is the answer.

## Write your own loop

The Agent above uses `default_loop`. To change what happens in a turn, write the loop yourself:

```python
from pathlib import Path

from alpineagents import Agent, State, compact_if_full, loop, tool


@tool
def list_files(folder: str = ".") -> list[str]:
    """List the files in a folder"""
    return sorted(p.name for p in Path(folder).iterdir())


@tool
def read_file(path: str) -> str:
    """Read a file"""
    file = Path(path)
    return file.read_text() if file.exists() else f"No such file: {path}"


@tool
def write_file(path: str, content: str) -> None:
    """Create a file, or replace its content"""
    Path(path).write_text(content)


@loop(until=State.is_answered, limit=30)
def coding(agent: Agent, state: State):
    compact_if_full(agent, state)
    agent.think(state)
    if state.wants_tools():
        agent.use_tools(state)


agent = Agent(
    model="claude-sonnet-5",
    system="You are a coding assistant. Read a file before you change it.",
    tools=[list_files, read_file, write_file],
    loop=coding,
)
print(agent.run("Add a test for the add() function in calc.py"))
```

- `coding` is one turn: summarize the context if it is more than 60% full, ask the model, run the tools it asked for.
- `@loop` repeats the turn. It stops when `State.is_answered` is true, or after 30 turns.
- To change the agent, add, remove or reorder lines in `coding`.

## Documentation

[tgoddessana.github.io/alpineagents](https://tgoddessana.github.io/alpineagents/). Read it in this order:

1. [Concepts](https://tgoddessana.github.io/alpineagents/concepts/overview/): how a run works, then Agent, State,
   loops and tools. About 15 minutes.
2. [Guides](https://tgoddessana.github.io/alpineagents/guides/stop-conditions/): one task per page, for example
   [asking before a tool runs](https://tgoddessana.github.io/alpineagents/guides/approval/) or
   [testing an agent](https://tgoddessana.github.io/alpineagents/guides/testing/).
3. [API reference](https://tgoddessana.github.io/alpineagents/api/agent/): every class and method.

## Roadmap

- OpenAI adapter, LiteLLM adapter
- Subagents (`tools=[researcher]`). Passing an Agent in `tools=` raises `NotImplementedError` for now
- Skills (`skills=`). Passing `skills=` raises `NotImplementedError` for now
- `agent.run_tool`, `agent.load_skill`
- `state.save()` / `State.load()`
- Multimodal tool results (`Image`, `File`)
- `alpineagents[prices]` (cost calculation with genai-prices; for now `usage.cost` is filled only when you pass
  `price=Price(...)`)
- `alpineagents add` CLI (copies the default loop and block sources into your project)
- Per-adapter server-side compaction optimizations

## Development

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

## License

MIT. See [LICENSE](https://github.com/TGoddessana/alpineagents/blob/main/LICENSE).
