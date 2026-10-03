# Write your own loop

In alpineagents the loop is a function. The Agent in the last pages used a default one. This page shows it, and then
you write your own.

## The default loop

A loop does one turn, and a decorator repeats the turn until a stop condition is true. This is the full source of the
default loop:

```python
from alpineagents import Agent, State, compact_if_full, loop


def is_answered(state: State) -> bool:
    if state.pending_calls or not state.messages:
        return False
    last = state.messages[-1]
    return last.role == "assistant" and not last.tool_calls


@loop(until=is_answered, limit=50)
def default_loop(agent: Agent, state: State):
    compact_if_full(agent, state)
    agent.think(state)
    if state.pending_calls:
        agent.use_tools(state)
```

1. The body is one turn. The first step, `compact_if_full`, summarizes the messages when they fill more than 60% of
   the model's context window.
2. `agent.think(state)` sends the messages to the model and records its reply in the State.
3. If the reply asked for tools, `agent.use_tools(state)` runs them and records the results. The next turn sends
   them to the model.
4. `@loop` repeats the body. It stops when `until` is true, or after `limit` turns.

## The name `is_answered`

The `until` function says when the loop is done: the model answered, with no tool call waiting. The examples on
these pages define the same check as `waiting_for_user`. The default loop has it as `is_answered`. The check is the
same, only the name differs. A State has no such method, because what counts as done is your decision.

The name shows up wherever a stop is shown. A run stopped by `is_answered` prints `done: stopped by is_answered`,
and `state.stopped` is `StoppedByUntil("is_answered")`. With your own function it is `StoppedByUntil("waiting_for_user")`.
Tests assert on that name.

## Your loop

Add a step of your own to the turn. This loop warns the model when it has used 20 turns:

```python
--8<-- "docs_src/learn_loop.py"
```

1. `waiting_for_user` is the stop condition. It takes the State and returns `True` to stop.
2. `warn_when_long` is a block: a function that takes `(agent, state)` and does one step of a turn. It adds a notice
   to the messages. `compact_if_full` is a block that comes with the library.
3. `coding` is the loop. `@loop` needs both `until` and `limit`, and `limit=30` means 30 turns at most. Reaching
   it does not raise: the run just ends.
4. `Agent(loop=coding)` makes `agent.run` call it.

To change what the agent does, add, remove or reorder lines in the body. There is no hook to register. The order
of the lines is the order of the steps.

## Why the run stopped

`state.stopped` says why. After a run it is, for example:

| `state.stopped` | The run ended because |
| --- | --- |
| `StoppedByUntil("waiting_for_user")` | The `until` function returned `True` |
| `StoppedByLimit(30)` | The loop ran `limit` turns |
| `StoppedByFinish(answer)` | Something called `state.finish(answer)` |
| `StoppedByPermission(call, permission)` | A permission stopped the run. See [Ask before a tool runs](approval.md) |

## Next

[Keep the conversation](conversation.md). To go deeper on loops:

- [Stop a run](../guides/stop-conditions.md): more than one `until`, a budget, a submit tool
- [Loops and stop rules](../concepts/loops.md): the order of the checks, nested loops, loops without `@loop`
- [Check the work before finishing](../guides/verify.md) and [Plan before acting](../guides/plan-first.md): loops
  with a step added
