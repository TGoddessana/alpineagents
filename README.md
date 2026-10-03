# alpineagents

[![CI](https://github.com/TGoddessana/alpineagents/actions/workflows/ci.yml/badge.svg)](https://github.com/TGoddessana/alpineagents/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/TGoddessana/alpineagents/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-tgoddessana.github.io-blue.svg)](https://tgoddessana.github.io/alpineagents/)

A Python agent framework where the agent loop is a function you write.

- `Agent` holds the settings: the model, the system prompt and the tools.
- `State` holds one conversation: its history, the messages the model sees next, and the answer. You change it only
  through its methods.
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

from alpineagents import Agent, ToolError, tool


@tool
def read_file(path: str) -> str:
    """Read a text file"""
    file = Path(path)
    if not file.exists():
        raise ToolError(f"No such file: {path}")
    return file.read_text()


agent = Agent(model="claude-sonnet-5", tools=[read_file])
print(agent.run("Summarize README.md in three lines"))
```

- `@tool` turns the function into a tool. The type hints and the docstring tell the model how to call it.
- `ToolError` tells the model the call failed, and the run continues. Any other exception stops the run.
- `agent.run` repeats turns until the model answers, then returns the answer.
- Progress is printed to the terminal while it runs.

### Without an API key

`FakeModel` returns prepared replies in order. It needs no API key and no network:

```python
from pathlib import Path

from alpineagents import Agent, ToolError, tool
from alpineagents.testing import FakeModel, tool_call


@tool
def read_file(path: str) -> str:
    """Read a text file"""
    file = Path(path)
    if not file.exists():
        raise ToolError(f"No such file: {path}")
    return file.read_text()


fake = FakeModel([
    tool_call("read_file", path="README.md"),
    "README.md describes a Python agent framework.",
])
agent = Agent(model=fake, tools=[read_file])
print(agent.run("Summarize README.md"))
```

The first reply asks for `read_file`. The Agent runs the tool, and the second reply is the answer.

## Write your own loop

The Agent above uses `default_loop`. To change what happens in a turn, write the loop yourself:

```python
from pathlib import Path

from alpineagents import Agent, State, ToolError, compact_if_full, loop, tool, waiting_for_user


@tool
def list_files(folder: str = ".") -> list[str]:
    """List the files in a folder"""
    return sorted(p.name for p in Path(folder).iterdir())


@tool
def read_file(path: str) -> str:
    """Read a file"""
    file = Path(path)
    if not file.exists():
        raise ToolError(f"No such file: {path}")
    return file.read_text()


@tool
def write_file(path: str, content: str) -> None:
    """Create a file, or replace its content"""
    Path(path).write_text(content)


@loop(until=waiting_for_user, limit=30)
def coding(agent: Agent, state: State):
    compact_if_full(agent, state)
    agent.think(state)
    if state.pending_calls:
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
- `@loop` repeats the turn. It stops when `waiting_for_user` is true (the model answered and waits for the user), or
  after 30 turns. Any function from State to `bool` works as `until`.
- To change the agent, add, remove or reorder lines in `coding`.

[Write your own loop](https://tgoddessana.github.io/alpineagents/learn/loop/) explains each part.

## Keep and inspect the State

`agent.run("text")` creates a State and throws it away. To keep it, create it yourself:

```python
from alpineagents import Message, State

state = State(messages=[Message.user("Add a test for the add() function in calc.py")])
agent.run(state)

print(state.answer, state.stopped, state.usage.cost)
state.add_message(Message.user("Now run the tests"))   # the one way to talk to it
agent.run(state)                                       # continues the same conversation
```

Every change to a State adds one entry to `state.history`, and everything else is computed from it. A snapshot goes
back, a fork tries another path, and `FileStore` with `Agent(store=...)` saves the history to continue it in another
process.

## Documentation

[tgoddessana.github.io/alpineagents](https://tgoddessana.github.io/alpineagents/)

1. [Learn](https://tgoddessana.github.io/alpineagents/learn/first-agent/): seven short pages, read in order.
   [Your first agent](https://tgoddessana.github.io/alpineagents/learn/first-agent/),
   [tools](https://tgoddessana.github.io/alpineagents/learn/tools/),
   [your own loop](https://tgoddessana.github.io/alpineagents/learn/loop/),
   [the conversation](https://tgoddessana.github.io/alpineagents/learn/conversation/),
   [approval](https://tgoddessana.github.io/alpineagents/learn/approval/),
   [save and resume](https://tgoddessana.github.io/alpineagents/learn/save-resume/) and
   [testing](https://tgoddessana.github.io/alpineagents/learn/testing/).
2. [Guides](https://tgoddessana.github.io/alpineagents/guides/stop-conditions/): one task per page, in any order, for
   example [writing permission rules](https://tgoddessana.github.io/alpineagents/guides/permissions/).
3. [Concepts](https://tgoddessana.github.io/alpineagents/concepts/overview/): how it works, with the precise rules.
4. [API reference](https://tgoddessana.github.io/alpineagents/api/agent/): every class and method.

## Roadmap

- OpenAI adapter, LiteLLM adapter
- Subagents (`tools=[researcher]`). Passing an Agent in `tools=` raises `NotImplementedError` for now
- Skills (`skills=`). Passing `skills=` raises `NotImplementedError` for now
- `agent.run_tool`, `agent.load_skill`
- More tool result types (`File`, image URLs)
- `alpineagents add` CLI (copies the default loop and block sources into your project)
- Per-adapter server-side compaction optimizations

## Development

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

## License

MIT. See [LICENSE](https://github.com/TGoddessana/alpineagents/blob/main/LICENSE).
