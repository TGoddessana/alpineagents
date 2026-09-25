# Tools that use the State

A tool can read or change the run it is part of.

```python
--8<-- "docs_src/tool_state.py"
```

1. A parameter typed `State` is hidden from the model. The model sees `remember(key, value)` and `recall(key)`.
2. When the tool runs, that parameter receives the current State.
3. `state.data` keeps values across turns. The model never sees it.

## What a tool can do with the State

| Call | Effect |
| --- | --- |
| `state.data[...]` | Read or keep your own values |
| `state.finish(answer)` | End the run after this turn. See [Stop conditions](stop-conditions.md#stop-from-a-tool) |
| `state.add_notice(text)` | Tell the model something. It goes into the context right after this turn's results |
| `state.answer`, `state.turn`, `state.usage`, ... | Read the run |

## Thread safety

Tools in one turn can run at the same time on worker threads. Every State method is thread-safe. For an update to
`state.data` in several steps, hold `state.lock`. See [State: your own data](../concepts/state.md#your-own-data).

## Related

- [State](../concepts/state.md)
- [Tools](../concepts/tools.md)
