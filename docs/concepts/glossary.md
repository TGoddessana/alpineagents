# Glossary

Terms in the order you meet them.

| Term | Meaning |
| --- | --- |
| State | One conversation and its record. The one object you change, through its methods. A `State` object. See [State](state.md) |
| `StateSnapshot` | The frozen value of a State at one moment: `state.snapshot()`. It never changes, and `state.restore(snapshot)` goes back to it |
| Run | One call of `agent.run`. It runs the loop on one State. A State can go through several runs |
| Turn | One call of the loop body. `limit` counts turns |
| `state.turn` | How many model replies there were on the State, one more while a request waits on the model. With one `think` per loop body, it equals the number of turns |
| Messages | `state.messages`: what the model sees at the next `think`. Computed from history. It can shrink |
| History | `state.history`: everything that happened in a State, as entries. It only grows, and it is the only source of the State's values |
| History entry | One item of `state.history`. There are 11 classes, told apart by `kind`: `MessageEntry`, `ModelRequestEntry`, `ModelReplyEntry`, `ModelEventEntry`, `ToolResultEntry`, `ExchangeEntry`, `ContextChangeEntry`, `RunStartEntry`, `StopEntry`, `ExtraDataEntry` and `ErrorEntry`. `Model...` entries are about the model, `ToolResultEntry` about a tool call. See [State](state.md#history) |
| `extra_data` | Your own values on the State, a JSON notepad the model never sees. Change it with `state.edit_extra_data()` |
| `AgentInfo` | What an Agent was when a run started: name, model, system prompt hash, tool and MCP server names. It is the content of a `RunStartEntry` |
| `StateInfo` | One saved State as `store.list()` shows it: id, first message, times, turn, `stopped`, `finished` |
| Reply | What the model returns for one request: text, tool calls or both. A `Reply` object |
| Answer | `state.answer`: the value given to `finish` (stored as JSON), or the text of the latest reply without tool calls |
| Stop condition | A function from State to `bool` in `until=`. The loop stops when one returns `True` |
| Stop value | Why a run is set to stop: `state.stopped`, one of `StoppedByUntil(name)`, `StoppedByLimit(turns)`, `StoppedByFinish(answer)` and `StoppedByPermission(call, permission)`. `None` until something decides. See [Stop conditions](../guides/stop-conditions.md) |
| Tool | Something the model can call. A `Tool` object: `@tool` makes a `FunctionTool`, an MCP server's tool is an `MCPTool`, and you can subclass `Tool` |
| Tool call | A request in the model's reply to run one tool with arguments. A `ToolCall` object |
| Pending call | A tool call that has no result yet. Listed in `state.pending_calls` |
| Outcome | How a tool call ended, as `ToolResultEntry.outcome`: `done`, `error`, `input_error`, `aborted`, `interrupted`, `denied` or `cancelled` |
| Tool hint | What a tool does, for your code: `read_only`, `destructive`, `idempotent`, `open_world`. The model does not see it. `tool.read_only` and the others hold for every call, so they are the worst case |
| `Hints` | The four hints for one call: `tool.hints_for(call.args)`. A tool gives them with `@tool(hints_for=...)` |
| Permission | An object in `Agent(permissions=[...])` that decides, before a call runs, whether it may run. See [Ask before a tool runs](../guides/approval.md) |
| Verdict | What a permission returns about a call: `Allowed()`, `Denied(reason)`, or `None` for no opinion, which lets the next permission decide |
| `DenyPermission`, `AllowPermission`, `DecidePermission` | The three kinds of permission. A deny permission may only refuse, and is asked before the others. An allow permission may only allow. A decide permission may do either |
| Denied call | A call a permission refused. It does not run, and the model gets the reason as an error result. `outcome.kind` is `"denied"` |
| Cancelled call | A call that did not run because a permission refused another call of the same turn with `stop=True`. `outcome.kind` is `"cancelled"` (`ToolOutcomeKind.CANCELLED`) |
| Tool result | What the model gets for a tool call: a string, or text and images when the tool returned an `Image` |
| Error result | A tool result the model is told is a failure: from `ToolError`, an `exception_handler`, invalid arguments, a call a permission refused, or a call closed by an exception. The run continues |
| Notice | A message from your code to the model, made with `Message.notice(text)` and added with `state.add_message`. It starts with `[notice] ` |
| Compaction | Replacing `state.messages` with the first message and a summary, to make it smaller: `state.compact(summary)`, or `agent.compact(state)` to have the model write the summary. History keeps everything |
| Restore | `state.restore(snapshot)`: going back to an earlier snapshot of the same State. History keeps everything and records the restore |
| Fork | `state.fork()`: a new State with the same history and a new id |
| Block | A function that takes `(agent, state)` and does one step of a turn, such as `compact_if_full` |
| Content block | A part of a message: text (`TextBlock`), an `Image`, a tool call (`ToolCall`) or a tool result. A tool result can hold `TextBlock`s and `Image`s. Not the same as a block |
| Model | An adapter that sends a request to a provider and returns a `Reply`. See [Models](../guides/models.md) |
| Reporter | An object that receives progress events. See [Progress and questions](../guides/progress.md) |
| Human | An object that answers `ask_human`. See [Progress and questions](../guides/progress.md) |
| Store | Where States are saved so another process can continue them. A `Store` object, such as `FileStore`; `store.save(state)` saves, `store.load(id)` continues. See [Save and resume](../guides/resume.md) |
| State id | The name a store saves a State under: `state.id` |
