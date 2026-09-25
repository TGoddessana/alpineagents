# alpineagents

[![CI](https://github.com/TGoddessana/alpineagents/actions/workflows/ci.yml/badge.svg)](https://github.com/TGoddessana/alpineagents/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Python agent framework where the agent loop is a function you write.

`Agent` holds the configuration, `State` holds the history, and a loop takes both and returns the answer.

> The agent looks at the state and thinks, and uses tools when it needs them. Repeat until there is an answer.

## Why

In most frameworks the loop lives inside the library. LangGraph has you declare it as a graph, and the OpenAI Agents
SDK and PydanticAI run it for you. In alpineagents the loop is a few lines of Python in your own code: `until=` decides
when it stops, `limit=` caps the turns, and compaction runs only where you call it. The default loop,
`alpineagents.default_loop`, is written the same way, so you can read it or copy it as a starting point. Misuse raises
an error that says how to fix it.

## Install

Requires Python 3.11 or later.

```bash
uv add alpineagents        # or
pip install alpineagents
```

To install from the repository for development:

```bash
uv venv && uv pip install -e ".[dev]"     # or python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Anthropic models use `ANTHROPIC_API_KEY`; OpenAI-compatible servers use `OPENAI_API_KEY` (or `api_key=`).

## At a glance

The smallest agent. Without a loop, `default_loop` runs.

```python
from alpineagents import Agent, tool

@tool
def web_search(query: str) -> str:
    """Search the web"""
    ...

agent = Agent(model="claude-sonnet-5", tools=[web_search])
print(agent.run("Find out whether a Python package called alpineagents already exists"))
```

Add pieces one at a time, only as needed. Here with a tool object, a hand-written loop, a compaction block and a look
at the result:

```python
from pathlib import Path
from alpineagents import Agent, State, loop, tool
from alpineagents.blocks import compact_if_full

class FileSystem:
    def __init__(self, root: str = "."):
        self.root = Path(root)

    @tool
    def read_file(self, path: str) -> str:
        """Read a file's contents"""
        file = self.root / path
        if not file.exists():
            return f"No such file: {path}"
        return file.read_text()

@loop(until=State.is_answered, limit=50)
def coding(agent: Agent, state: State):
    compact_if_full(agent, state)
    agent.think(state)
    if state.wants_tools():
        agent.use_tools(state)

agent = Agent(
    model="claude-sonnet-5",
    system="You are a coding assistant",
    tools=[FileSystem(), web_search],
    loop=coding,
)

state = State("Find the bug in this repo")
answer = agent.run(state)
print(state.stopped_by, state.usage.cost)
```

## Core concepts

| Object | One-line responsibility |
| --- | --- |
| `Agent` | Holds the configuration and performs the actions that need a model, tools or a human: `run`, `think`, `use_tools`, `ask`, `ask_human`, `compact`, `copy` (and `arun`, `athink`, ... for async code) |
| `State` | Holds this run's record (`history`) and, separately, the context the model sees (`context`). Thread-safe (`state.lock`) |
| `@loop` | Turns a one-turn function into an `(agent, state)` loop. `until` and `limit` are required; `copy(limit=...)` changes only the settings |
| `@tool` | Type hints become the input schema and the docstring becomes the description. A parameter typed `State` receives the current State |
| `Terminal` | The default Reporter (shows progress) and Human (asks the person) |

- Models: pass a string such as `"claude-sonnet-5"`, `"anthropic/claude-sonnet-5"` or `"ollama/llama3:8b"`, or, when
  you need settings, `Anthropic("claude-sonnet-5", thinking=2048)` or `OpenAICompatible("llama3", base_url=...)`.
- Stopping is quiet: reaching `limit` stops without an exception and leaves `state.stopped_by == "limit"`.
- Failure is honest: tool exceptions propagate as is; only provider exceptions are wrapped in `ProviderError`
  (`RateLimitError`, `ContextTooLongError`, `AuthError`). The original is in `__cause__`.
- After stopping with `Ctrl+C` you can `run` again with the same State. Unfinished tool calls are closed with
  `(interrupted by user)`.
- What a Model goes through outside its reply (falling back to another model, a retry, ...) it reports via
  `on_event(ModelEvent(...))`. The Agent records it in history as `model_event` and shows it via
  `Reporter.on_model_event` (`Terminal` prints `  model: {message}`).

## MCP servers

Put an MCP server in `tools=` like any other tool (`pip install "alpineagents[mcp]"`):

```python
from alpineagents import MCP

github = MCP("npx -y @modelcontextprotocol/server-github", name="github", env={"GITHUB_TOKEN": token})
linear = MCP(url="https://mcp.linear.app/mcp", name="linear", headers={"Authorization": f"Bearer {key}"})

agent = Agent(model="claude-sonnet-5", tools=[FileSystem(), github, linear.list_issues])
```

- The model sees a server's tools as `github__create_issue`. `name=` is required, so two servers never clash.
- `linear.list_issues` (or `linear["list-issues"]`) picks one tool, like passing one `@tool` method of an object.
- `run()` connects and disconnects. `with agent:` (or `async with agent:`) keeps the connections open across runs.
  Concurrent runs of one Agent share its connections.
- A tool name that collides with another tool, or a picked tool the server does not have, is reported when the
  server connects, before the first `think`.
- An error the server returns for a call (`isError`) goes to the model as an error result. A lost connection
  raises `MCPConnectionError`.

## Async

Every Agent method that waits on a model, a tool or a person has an async version with an `a` prefix: `arun`,
`athink`, `ause_tools`, `aask`, `aask_human`, `acompact`. Nothing else changes.

```python
answer = await agent.arun("Find out whether a Python package called alpineagents already exists")
```

Without a loop, `arun` uses `adefault_loop`, the async version of `default_loop`. A loop of your own is an
`async def` body under the same `@loop`:

```python
from alpineagents.blocks import acompact_if_full

@loop(until=State.is_answered, limit=50)
async def coding(agent: Agent, state: State):
    await acompact_if_full(agent, state)
    await agent.athink(state)
    if state.wants_tools():
        await agent.ause_tools(state)
```

- `async def` tools run as tasks on your event loop; regular tools run on worker threads. A turn's calls still run
  concurrently.
- Cancelling the task (a client disconnect, `asyncio.timeout`) follows the same rules as `Ctrl+C`: the model step is
  rolled back, unfinished tool calls are closed with `(interrupted by user)`, and `async def` tools are cancelled.
- `Anthropic` and `OpenAICompatible` use the providers' async clients, so cancelling also closes the HTTP stream. A
  `Model` of your own works as is (its `respond` runs on a worker thread) or can override `arespond`.
- A `Human` for a web page implements `async def aask(...)` instead of `ask`.
- Calling a sync method such as `agent.think` inside an async loop raises `TypeError`, because it would block the
  event loop. The error names the async method to use.

## Testing

`FakeModel` returns prepared replies in order, and `tool_call()` builds a tool call reply. No API key or network needed.
Code that asks a human is tested with `FakeHuman(["yes", "always"])`.

```python
from alpineagents import State
from alpineagents.testing import FakeModel, tool_call

def test_reads_file_then_answers():
    fake = FakeModel([
        tool_call("read_file", path="main.py"),
        "The bug is on line 3",
    ])
    state = State("Find the bug")
    agent.copy(model=fake, reporter=None).run(state)

    assert state.answer == "The bug is on line 3"
    assert state.turn == 2
    assert state.stopped_by == "is_answered"
```

Run this repository's tests like this:

```bash
.venv/bin/python -m pytest -q
```

## Roadmap

- OpenAI adapter, LiteLLM adapter
- Subagents (`tools=[researcher]`, `state.root`, `substate`). Passing an Agent in `tools=` raises `NotImplementedError` for now
- Skills (`skills=`, `load_skill`). Passing `skills=` raises `NotImplementedError` for now
- `agent.run_tool`, `agent.load_skill`
- `state.save()` / `State.load()`
- Multimodal tool results (`Image`, `File`)
- `alpineagents[prices]` (cost calculation with genai-prices; for now `usage.cost` is filled only when you pass
  `price=Price(...)`)
- `alpineagents add` CLI (copies the default loop and block sources into your project)
- Per-adapter server-side compaction optimizations

## License

MIT. See [LICENSE](LICENSE).
