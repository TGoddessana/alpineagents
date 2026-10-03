# Stop conditions

A loop stops in one of four ways. Pick the one that matches who decides.

| Who decides | How | `state.stopped` |
| --- | --- | --- |
| A check on the State, before every turn | An `until` function | `StoppedByUntil(name)`, the function's name |
| Your code or a tool, at a specific moment | `state.finish(answer)` | `StoppedByFinish()` |
| A fixed maximum number of turns | `limit` | `StoppedByLimit(turns)`, the limit |
| A permission, when it refuses a call with `stop=True` | `Denied(reason, stop=True)`, for example `DecideByHuman`'s `no` ([Ask before a tool runs](approval.md)) | `StoppedByPermission(call, permission)` |

## Stop on a check

```python
--8<-- "docs_src/stop_conditions.py"
```

1. `waiting_for_user` is true when the model has answered: no call is pending, and the last message is the model's
   and asks for no tool. You write it, so you decide what "done" means. The default loop has the same check built in.
2. `spent_too_much` takes the State and returns `True` to stop.
3. `until` takes a list. The loop stops when any function in it returns `True`.
4. After the run, `state.stopped` is `StoppedByUntil("waiting_for_user")`, `StoppedByUntil("spent_too_much")` or
   `StoppedByLimit(30)`. The terminal output shows the same name, for example `done: stopped by spent_too_much`.

Give stop conditions clear names. The name is the only record of why the run stopped.

## Stop from a tool

Let the model decide when the work is done, and return a structured answer:

```python
--8<-- "docs_src/submit_tool.py"
```

1. The model calls `submit` when it thinks the work is done.
2. `state.finish(...)` sets `state.answer` to the dict. The loop stops before the next turn.
3. `agent.run` returns the dict.
4. The State is now finished. Running it again or calling `ask` on it raises `ValueError`.

The answer can be any value. The model fills the tool's typed parameters, so the answer has a known shape.

## Stop when the person says no

A permission that refuses a call with `stop=True` stops the loop before its next turn. Unlike `finish`, the State is
not finished: add the person's next message with `state.add_message(Message.user(...))` and run again. See
[Ask before a tool runs](approval.md#when-the-person-says-no).

## In a loop without @loop

`state.finish()` and a permission's `stop=True` set `state.stopped` right away. A loop written without `@loop` stops
on them by checking it before each turn:

```python
def my_loop(agent: Agent, state: State):
    while state.stopped is None and not waiting_for_user(state):
        agent.think(state)
        if state.pending_calls:
            agent.use_tools(state)
```

`run` resets `state.stopped` to `None` when it starts, so the previous run's reason does not stop the next one.

A `@loop` called inside such a loop leaves its own `until` or `limit` reason in `state.stopped`. See
[Loops](../concepts/loops.md#when-a-loop-stops) for the check to use then.

## Check for the limit

Reaching `limit` stops the loop without an exception. Check it when an unfinished run matters. `agent` is the Agent
from the example above:

```python
from alpineagents import Message, State, StoppedByLimit

state = State(messages=[Message.user("Rename the helper functions in utils.py")])
agent.run(state)
if isinstance(state.stopped, StoppedByLimit):
    print(f"Stopped after {state.stopped.turns} turns. The task may be unfinished.")
```

## Related

- [Loops](../concepts/loops.md#when-a-loop-stops): the order of the checks
- [Tools that use the State](tool-state.md)
