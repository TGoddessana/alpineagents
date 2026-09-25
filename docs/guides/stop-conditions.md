# Stop conditions

A loop stops in one of three ways. Pick the one that matches who decides.

| Who decides | How | `state.stopped_by` |
| --- | --- | --- |
| A check on the State, before every turn | An `until` function | The function's name |
| Your code or a tool, at a specific moment | `state.finish(answer)` | `"finish"` |
| A fixed maximum number of turns | `limit` | `"limit"` |

## Stop on a check

```python
--8<-- "docs_src/stop_conditions.py"
```

1. `spent_too_much` takes the State and returns `True` to stop.
2. `until` takes a list. The loop stops when any function in it returns `True`.
3. After the run, `state.stopped_by` is `"is_answered"`, `"spent_too_much"` or `"limit"`. The terminal output shows the
   same name.

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

## Check for the limit

Reaching `limit` stops the loop without an exception. Check it when an unfinished run matters. `agent` is the Agent
from the example above:

```python
from alpineagents import State

state = State("Rename the helper functions in utils.py")
agent.run(state)
if state.stopped_by == "limit":
    print(f"Stopped after {state.stopped_limit} turns. The task may be unfinished.")
```

## Related

- [Loops](../concepts/loops.md#when-a-loop-stops): the order of the checks
- [Tools that use the State](tool-state.md)
