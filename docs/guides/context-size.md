# Context size

A long run fills the model's context window. When a request is larger than the window, the provider rejects it and
`think` raises `ContextTooLongError`. Shrink the context before that happens.

## Four ways to shrink the context

| Method | Makes a model request | The context afterwards |
| --- | --- | --- |
| `compact_if_full(agent, state)` | Only when the context is more than 60% full | The task and a summary |
| `agent.compact(state)` | Yes | The task and a summary |
| `state.clear_tool_results(keep_last=5)` | No | Every message, with old tool results replaced by `(cleared: kept in history)` |
| `state.start_from(summary)` | No | The task and your summary |

All four change only the context. `state.history` keeps everything.

## Example

```python
--8<-- "docs_src/context_size.py"
```

1. `CompactIfFull(at=0.5, ...)` makes a block that compacts at 50% instead of 60%.
2. `instructions` tells the model what the summary must keep.
3. After the tools run, `clear_tool_results(keep_last=10)` blanks every tool result except the latest ten.

`clear_tool_results` raises `ValueError` while calls are pending, so call it after `use_tools`.

## Watch the size

| Property | Value |
| --- | --- |
| `state.context_tokens` | Estimated size of the context, in tokens |
| `state.context_used` | `context_tokens` divided by the model's context window |

The terminal shows each compaction under the next turn header:

```text
[turn 12] thinking
  context compacted: 121k → 18k tokens (cache rebuilds)
```

## Related

- [State: history and context](../concepts/state.md#history-and-context)
