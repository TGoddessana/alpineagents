# Agent

An Agent holds settings and performs actions. The data of a run is in the [State](state.md), not in the Agent.

## Settings

`read_file`, `write_file` and `coding` are defined in [Write your own loop](../index.md#write-your-own-loop).

```python
from alpineagents import Agent

agent = Agent(
    model="claude-sonnet-5",
    system="You are a coding assistant.",
    tools=[read_file, write_file],
    loop=coding,
)
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `model` | required | A model name such as `"claude-sonnet-5"`, or a `Model` object. See [Models](../guides/models.md) |
| `system` | `None` | The system prompt |
| `tools` | none | `@tool` functions, objects with `@tool` methods, MCP servers. See [Tools](tools.md) |
| `loop` | `default_loop` | The loop `run` calls. See [Loops](loops.md) |
| `reporter` | `Terminal` | Receives progress. `None` shows nothing. See [Progress and questions](../guides/progress.md) |
| `human` | `Terminal` | Answers `ask_human`. `None` means nobody can answer |
| `name`, `description` | `None` | Labels for your own use |

- Creating an Agent uses no network and needs no API key.
- A wrong setting raises `TypeError` or `ValueError` here, with how to fix it.
- Settings do not change after creation. `copy` returns a new Agent with some settings changed:

```python
quiet = agent.copy(reporter=None)
```

## Actions

Every action except `run` takes the State it works on.

| Action | What it does | Changes the context |
| --- | --- | --- |
| `run(task)` | Runs the loop on a task string or a State, and returns the answer | Yes |
| `think(state)` | Sends the context to the model once, and records the reply | Yes |
| `use_tools(state)` | Runs the pending calls, and records the results | Yes |
| `compact(state)` | Replaces the context with the task and a summary written by the model | Yes |
| `ask(state, prompt, returns=...)` | Asks the model a side question, and returns the answer | No |
| `ask_human(state, prompt, returns=...)` | Asks the person, and returns the answer | No |

You call `run`. The loop body calls `think`, `use_tools` and `compact`. `ask` and `ask_human` work in the loop body
and after a run.

Each action has an async version with an `a` prefix: `arun`, `athink`, `ause_tools`, `acompact`, `aask`,
`aask_human`. See [Async](../guides/async.md).

## think and use_tools

A turn usually looks like this:

```python
agent.think(state)
if state.wants_tools():
    agent.use_tools(state)
```

1. `think` sends one request to the model. The reply is either an answer, or a request to call tools.
2. Requested tool calls become pending calls in `state.pending_calls`. `state.wants_tools()` is true while any are
   pending.
3. `use_tools` runs every pending call. The results go into the context, so the model sees them at the next `think`.

Rules:

- `think` raises `ValueError` while calls are pending. Run them with `use_tools`, or refuse them with `state.deny`.
- `think(state, tools=[...])` limits the tools the model sees in this request. `tools=[]` shows none.
- If `think` fails, the context goes back to how it was before the call, and the turn does not count.

## ask compared with think

|  | `think` | `ask` |
| --- | --- | --- |
| Adds the reply to the context | Yes | No |
| The model can call tools | Yes | No |
| Returns | Nothing. The reply is recorded in the State | The answer as `str`, a dataclass or a Pydantic model |
| Adds to `state.turn` | Yes | No |

Use `ask` to get a decision or a typed result from the run so far without changing the run. See
[Structured output](../guides/structured-output.md).

## One State belongs to one Agent

The first Agent that calls `think`, `use_tools` or `compact` on a State owns it. The same calls from another Agent
raise `ValueError`. `ask` works from any Agent.

`agent.copy(...)` returns another Agent. It cannot continue a State that the original Agent already ran.

## After finish

`state.finish()` ends the State for good. After it, `think`, `use_tools`, `ask` and `run` raise `ValueError`.
