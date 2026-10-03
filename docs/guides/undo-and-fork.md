# Go back or try another path

Two tools: `snapshot` and `restore` go back to an earlier point. `fork` keeps the State as it is and tries something
else on a copy. Use them between turns, not while the model is being waited on or tool calls are pending.

## Go back

`state.snapshot()` returns a `StateSnapshot`: the State's value at that moment, frozen, so it never changes. Take one
before something you may want to undo, and `state.restore(snapshot)` goes back:

```python
--8<-- "docs_src/snapshot.py"
```

1. `before` is the State as it was after the first run.
2. The second run adds to `state`.
3. `restore(before)` puts the messages back. If `before` was taken from another State, or while calls were pending,
   `restore` raises `ValueError`.

What goes back and what does not:

| Part | After `restore` |
| --- | --- |
| `messages`, `turn`, `extra_data`, `finished`, `answer`, `stopped` | What they were |
| `usage` | Unchanged: the tokens were spent |
| `history` | Unchanged, plus one `context_change` entry with `kind == "restore"`. A store saves the going back like any other step |

A snapshot lives in memory. To keep one, keep what a store keeps, its history: `State(history=snapshot.history)`
rebuilds the same value, in this process or another.

## Try another path

`state.fork()` makes a new State with the same history and a new id, bound to no store and to no running Agent:

```python
--8<-- "docs_src/fork.py"
```

1. `attempt` starts as a copy of `state`. They share nothing.
2. The new message and the run change only `attempt`.
3. `state.answer` is still the first answer.

- With `Agent(store=...)`, the fork is saved under its own id the first time the Agent runs it.
- A fork of a finished State is finished too. To continue from its messages, start `State(messages=state.messages)`.
- To use a snapshot of another State, build one from it with `State(history=snapshot.history)`.
- `fork().snapshot() == state.snapshot()`: equal histories give equal snapshots.
- `fork()` raises `ValueError` while the model is being waited on. Fork before `think`, or after it returns, for
  example in `Reporter.on_think_end`.

## Related

- [State and history](../concepts/state.md#statesnapshot-a-value-you-can-keep): what a snapshot is
- [Survive crashes and restarts](production.md): what a store keeps
