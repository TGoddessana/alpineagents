# Glossary

Terms in the order you meet them.

| Term | Meaning |
| --- | --- |
| State | One task: its history, context and answer. A `State` object |
| Run | One call of `agent.run`. It runs the loop on one State. A State can go through several runs |
| Turn | One call of the loop body. `limit` counts turns |
| `state.turn` | How many times `think` succeeded on the State. With one `think` per loop body, it equals the number of turns |
| Context | The messages the model sees at the next `think`. It can shrink |
| History | Everything that happened in a State. It only grows |
| Reply | What the model returns for one request: text, tool calls or both. A `Reply` object |
| Answer | `state.answer`: the value given to `finish`, or the text of the latest reply without tool calls |
| Stop condition | A function from State to `bool` in `until=`. The loop stops when one returns `True` |
| Tool | Something the model can call. A `Tool` object: `@tool` makes a `FunctionTool`, an MCP server's tool is an `MCPTool`, and you can subclass `Tool` |
| Tool call | A request in the model's reply to run one tool with arguments. A `ToolCall` object |
| Pending call | A tool call that has no result yet. Listed in `state.pending_calls` |
| Tool hint | What a tool does, for your code: `read_only`, `destructive`, `idempotent`, `open_world`. The model does not see it |
| Tool result | What the model gets for a tool call: a string, or text and images when the tool returned an `Image` |
| Error result | A tool result the model is told is a failure: from `ToolError`, an `exception_handler`, invalid arguments, `state.deny`, or a call closed by an exception. The run continues |
| Notice | A message from your code to the model, added with `add_notice`. It starts with `[notice] ` |
| Compaction | Replacing the context with the task and a summary, to make it smaller. History keeps everything |
| Block | A function that takes `(agent, state)` and does one step of a turn, such as `compact_if_full` |
| Content block | A part of a message: text (`TextBlock`), a tool call (`ToolCall`) or a tool result. A tool result can hold `TextBlock`s and `Image`s. Not the same as a block |
| Model | An adapter that sends a request to a provider and returns a `Reply`. See [Models](../guides/models.md) |
| Reporter | An object that receives progress events. See [Progress and questions](../guides/progress.md) |
| Human | An object that answers `ask_human`. See [Progress and questions](../guides/progress.md) |
| Owner | The first Agent that called `think`, `use_tools` or `compact` on a State. Only it can make those calls on that State |
| Store | Where an Agent saves States so another process can continue them. A `Store` object, such as `FileStore`. See [Save and resume](../guides/resume.md) |
| State id | The name a store saves a State under: `state.id` |
