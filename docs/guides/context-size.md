# Context size

A long run fills the model's context window. When a request is larger than the window, the provider rejects it and
`think` raises `ContextTooLongError`. Shrink the context before that happens.

## Four ways to shrink the context

| Method | Makes a model request | The context afterwards |
| --- | --- | --- |
| `compact_if_full(agent, state)` | Only when the context is more than 60% full | The first user message and a summary |
| `agent.compact(state)` | Yes | The first user message and a summary |
| `state.clear_tool_results(keep_last=5)` | No | Every message, with old tool results (images included) replaced by `(cleared: kept in history)` |
| `state.compact(summary)` | No | The first user message and your summary |

All four change only `state.messages`, what the model sees next. `state.history` keeps everything, including a
`context_change` entry that says what was shrunk.

## Example

```python
--8<-- "docs_src/context_size.py"
```

1. `CompactIfFull(at=0.5, ...)` makes a block that compacts at 50% instead of 60%.
2. `instructions` tells the model what the summary must keep.
3. After the tools run, `clear_tool_results(keep_last=10)` blanks every tool result except the latest ten.

`clear_tool_results` and `compact` raise `ValueError` while calls are pending, so call them after `use_tools`.

## Watch the size

| Call | Value |
| --- | --- |
| `agent.context_tokens(state)` | Estimated size of the context, in tokens, with the Agent's system prompt and tools. Each image counts as 1,600 |
| `agent.context_used(state)` | `context_tokens` divided by the context window of the Agent's model |

They are on the Agent, not the State, because the size depends on the model that will read the context. The same
State is a different fraction of a small window and a large one.
The terminal shows each compaction under the next turn header:

```text
[turn 12] thinking
  context compacted: 121k → 18k tokens (cache rebuilds)
```

## Related

- [State: history and context](../concepts/state.md#history-and-context)
