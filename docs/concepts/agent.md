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
| `tools` | none | `@tool` functions, objects with `@tool` methods, objects of `Tool` subclasses, MCP servers. See [Tools](tools.md) |
| `loop` | `default_loop` | The loop `run` calls. See [Loops](loops.md) |
| `reporter` | `Terminal` | Receives progress. `None` shows nothing. See [Progress and questions](../guides/progress.md) |
| `human` | `Terminal` | Answers `ask_human`. `None` means nobody can answer |
| `permissions` | `None` | Permissions that decide, before a tool call runs, whether it may run. `None` runs every call without checks. See [Ask before a tool runs](../guides/approval.md) |
| `store` | `None` | Saves each State as it runs, so it can be continued later. See [Save and resume](../guides/resume.md) |
| `name`, `description` | `None` | Labels for your own use |

- Creating an Agent uses no network and needs no API key.
- A wrong setting raises `TypeError` or `ValueError` here, with how to fix it.
- Settings do not change after creation. `copy` returns a new Agent with some settings changed:

```python
quiet = agent.copy(reporter=None)
```

## Actions

Every action except `run` takes the State it works on.

| Action | What it does | Changes `state.messages` |
| --- | --- | --- |
| `run(prompt_or_state)` | Runs the loop on a prompt string or a State, and returns the answer | Yes |
| `think(state)` | Sends the messages to the model once, and records the request and the reply | Yes |
| `use_tools(state)` | Runs the pending calls, and records the results | Yes |
| `compact(state)` | Replaces the messages with the first message and a summary written by the model | Yes |
| `ask(state, prompt, returns=...)` | Asks the model a side question, and returns the answer | No |
| `ask_human(state, prompt, returns=...)` | Asks the person, and returns the answer | No |
| `context_tokens(state)` | Estimates the size of what the model would receive next, in tokens | No |
| `context_used(state)` | The fraction of the model's context window that fills | No |

You call `run`. The loop body calls `think`, `use_tools` and `compact`. `ask` and `ask_human` work in the loop body
and after a run. Every action records what it did in the State's history, and with a store, saves by itself. To save
at a time you choose, use the store: `store.save(state)`.

Each action has an async version with an `a` prefix: `arun`, `athink`, `ause_tools`, `acompact`, `aask`,
`aask_human`. See [Async](../guides/async.md).

## think and use_tools

A turn usually looks like this:

```python
agent.think(state)
if state.pending_calls:
    agent.use_tools(state)
```

1. `think` records a `ModelRequestEntry`, sends one request to the model, and records the `ModelReplyEntry`. The reply
   is either an answer, or a request to call tools.
2. Requested tool calls become pending calls in `state.pending_calls`.
3. `use_tools` runs every pending call and records a `ToolResultEntry` for each. The results go into
   `state.messages`, so the model sees them at the next `think`.

Rules:

- `think` raises `ValueError` while calls are pending, and on a State with no messages. Run them with `use_tools`. With `Agent(permissions=[...])`,
  `use_tools` first asks the permissions, and a call they refuse does not run. See
  [Ask before a tool runs](../guides/approval.md).
- `think(state, tools=[...])` limits the tools the model sees in this request. `tools=[]` shows none.
- `agent.tool_map` has every tool the model can call, by name. Use it to find the tool of a pending call, for example
  to read its hints. See
  [Tools: describe what a tool does](tools.md#describe-what-a-tool-does).
- If `think` fails, the request is taken back: `state.messages` is as it was before the call, and the turn does not count.
  History keeps the request and an `ErrorEntry`.

## ask compared with think

|  | `think` | `ask` |
| --- | --- | --- |
| Adds the reply to `state.messages` | Yes | No |
| The model can call tools | Yes | No |
| Returns | Nothing. The reply is recorded in the State | The answer as `str`, a dataclass or a Pydantic model |
| Adds to `state.turn` | Yes | No |

Use `ask` to get a decision or a typed result from the run so far without changing the run. See
[Structured output](../guides/structured-output.md).

## Any Agent can continue a State

A State does not belong to an Agent. Any Agent can run, think, use tools and compact on any State, so
`agent.copy(model=...)` continues a conversation with another model:

```python
agent.run(state)
cheaper = agent.copy(model="claude-haiku-4-5")
cheaper.run(state)
```

The State records which Agent started each run (`RunStartEntry`, with an `AgentInfo`), and which model every `think` request
went to (`ModelRequestEntry`). One run at a time: a State that another `run` is running raises `ValueError`.

A State loaded with `store.load(id)` warns with `ResumeWarning` the first time an Agent that differs from its last
one runs it. See [State](state.md#a-state-saved-by-another-agent) and [Save and resume](../guides/resume.md).

## After finish

`state.finish()` ends the State for good. After it, `think`, `use_tools`, `ask` and `run` raise `ValueError`.
