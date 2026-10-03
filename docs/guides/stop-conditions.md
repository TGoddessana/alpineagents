# Stop a run

A loop stops in one of four ways. Pick the one that matches who decides.

| Who decides | How | `state.stopped` |
| --- | --- | --- |
| A check on the State, before every turn | An `until` function | `StoppedByUntil(name)`, the function's name |
| Your code or a tool, at a specific moment | `state.finish(answer)` | `StoppedByFinish(answer)` |
| A fixed maximum number of turns | `limit` | `StoppedByLimit(turns)`, the limit |
| A permission, when it refuses a call with `stop=True` | `Denied(reason, stop=True)`, for example `DecideByHuman`'s `no` ([Ask before a tool runs](../learn/approval.md)) | `StoppedByPermission(call, permission)` |

## Stop on a check

```python
--8<-- "docs_src/stop_conditions.py"
```

1. `waiting_for_user` is true when the model has answered: no call is pending, and the last message is the model's
   and asks for no tool. You write it, so you decide what "done" means. The default loop has the same check as
   `is_answered` ([Write your own loop](../learn/loop.md#the-name-is_answered)).
2. `spent_too_much` takes the State and returns `True` to stop.
3. `until` takes a list. The loop stops when any function in it returns `True`.
4. After the run, `state.stopped` is `StoppedByUntil("waiting_for_user")`, `StoppedByUntil("spent_too_much")` or
   `StoppedByLimit(30)`. The terminal shows the same name, for example `done: stopped by spent_too_much`.
5. Reaching `limit` does not raise. Check `state.stopped` when an unfinished run matters, as the last lines do.

Give stop conditions clear names. The name is the only record of why the run stopped.

## Stop from a tool

Let the model decide when the work is done, and return a structured answer:

```python
--8<-- "docs_src/submit_tool.py"
```

1. The model calls `submit` when it thinks the work is done.
2. `state.finish(...)` sets `state.answer` to the dict. The loop stops before the next turn.
3. `agent.run` returns the dict.
4. The State is now finished for good: see [State and history](../concepts/state.md#finish-ends-the-state).

The model fills the tool's typed parameters, so the answer has a known shape. It is JSON data: a dataclass or a
Pydantic model comes back as a read-only dict, and `Review(**state.answer)` rebuilds it.

## Stop when the person says no

A permission that refuses a call with `stop=True` stops the loop before its next turn. Unlike `finish`, the State is
not finished: add the person's next message and run again. See
[Ask before a tool runs](../learn/approval.md#continue-after-no).

## In a loop without @loop

`state.finish()` and a permission's `stop=True` set `state.stopped` right away, and `run` resets it to `None` when it
starts. A loop written without `@loop` has to check it itself:
see [Loops without `@loop`](../concepts/loops.md#loops-without-loop).

## Related

- [Loops and stop rules](../concepts/loops.md): the order of the checks
- [Let tools use the State](tool-state.md)
