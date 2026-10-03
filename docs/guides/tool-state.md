# Tools that use the State

A tool can read or change the run it is part of.

```python
--8<-- "docs_src/tool_state.py"
```

1. A parameter typed `State` is hidden from the model. The model sees `remember(key, value)` and `recall(key)`.
2. When the tool runs, that parameter receives the current State.
3. `state.extra_data` keeps values across turns. The model never sees it. `edit_extra_data()` changes it.

## What a tool can do with the State

| Call | Effect |
| --- | --- |
| `state.extra_data[...]` | Read your own values. It is read-only: change it with `with state.edit_extra_data() as data:` |
| `state.finish(answer)` | End the run after this turn. See [Stop conditions](stop-conditions.md#stop-from-a-tool) |
| `state.add_message(Message.notice(text))` | Tell the model something. It goes into the context right after this turn's results |
| `state.answer`, `state.turn`, `state.usage`, ... | Read the run |

## Thread safety

Tools in one turn can run at the same time on worker threads. Every State method is thread-safe. A
`with state.edit_extra_data() as data:` block is atomic: edits from several threads take turns, so
`data["n"] = data.get("n", 0) + 1` never loses an update. If the block raises, nothing changes. Values must be JSON
(a set raises `TypeError` naming the key: use a list). See [State](../concepts/state.md).

## Related

- [State](../concepts/state.md)
- [Tools](../concepts/tools.md)
