# Glossary

Terms in the order you meet them.

| Term | Meaning |
| --- | --- |
| Run | One call of `agent.run`. It runs the loop on one State. A State can go through several runs |
| Turn | One call of the loop body. `limit` counts turns |
| State | One conversation and its record. The one object you change, through its methods. See [State and history](state.md) |
| Messages | `state.messages`: what the model sees at the next `think`. Computed from history. It can shrink |
| History | `state.history`: everything that happened in a State, as entries. It only grows, and it is the only source of the State's values. See [History](state.md#history) |
| History entry | One item of `state.history`. There are 11 classes, told apart by `kind` |
| `state.turn` | How many model replies there were on the State. With one `think` per loop body, it equals the number of turns |
| `extra_data` | Your own JSON values on the State. The model never sees them. See [extra_data](state.md#extra_data-your-notepad) |
| `StateSnapshot` | The frozen value of a State at one moment: `state.snapshot()` |
| `AgentInfo` | What an Agent was when a run started: name, model, system prompt hash, tool and MCP server names |
| `StateInfo` | One saved State as `store.list()` shows it |
| Reply | What the model returns for one request: text, tool calls or both |
| Answer | `state.answer`: the value given to `finish`, or the text of the latest reply without tool calls |
| Stop condition | A function from State to `bool` in `until=`. The loop stops when one returns `True`. The default loop's is named `is_answered` |
| Stop value | Why a run is set to stop: `state.stopped`. See [Loops and stop rules](loops.md#when-a-loop-stops) |
| Block | A function that takes `(agent, state)` and does one step of a turn, such as `compact_if_full`. See [Write your own loop](../learn/loop.md#your-loop) |
| Content block | A part of a message: text (`TextBlock`), an `Image`, a tool call (`ToolCall`) or a tool result. Not the same as a block |
| Tool | Something the model can call. See [Tool calls](tools.md) |
| Tool call | A request in the model's reply to run one tool with arguments |
| Pending call | A tool call that has no result yet. Listed in `state.pending_calls` |
| Outcome | How a tool call ended, as `ToolResultEntry.outcome`: `done`, `error`, `input_error`, `aborted`, `interrupted`, `denied` or `cancelled` |
| Error result | A tool result the model is told is a failure. The run continues. See [When a call goes wrong](tools.md#when-a-call-goes-wrong) |
| Tool hint | What a tool does, for your code: `read_only`, `destructive`, `idempotent`, `open_world`. The model does not see it. See [Write permission rules](../guides/permissions.md) |
| Permission | An object in `Agent(permissions=[...])` that decides, before a call runs, whether it may run. See [Write permission rules](../guides/permissions.md) |
| Verdict | What a permission returns about a call: `Allowed()`, `Denied(reason)`, or `None` for no opinion |
| Denied call | A call a permission refused. It does not run, and the model gets the reason as an error result |
| Cancelled call | A call that did not run because a permission refused another call of the same turn with `stop=True` |
| Notice | A message from your code to the model: `Message.notice(text)`. It starts with `[notice] ` |
| Compaction | Replacing `state.messages` with the first message and a summary. History keeps everything. See [Keep the context small](../guides/context-size.md) |
| Restore | `state.restore(snapshot)`: going back to an earlier snapshot of the same State. See [Go back or try another path](../guides/undo-and-fork.md) |
| Fork | `state.fork()`: a new State with the same history and a new id |
| Model | An adapter that sends a request to a provider and returns a `Reply`. See [Use other models](../guides/models.md) |
| Reporter, Human | Objects that receive progress events and answer `ask_human`. See [Show progress and ask the person](../guides/progress.md) |
| Store | Where States are saved so another process can continue them, such as `FileStore`. See [Survive crashes and restarts](../guides/production.md) |
