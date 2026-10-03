# Write your own loop

In alpineagents the loop is a function. The Agent in the last pages used a default one. This page shows it, and then
you write your own.

## The default loop

A loop does one turn, and a decorator repeats the turn until a stop condition is true. This is the full source of the
default loop:

```python
from alpineagents import Agent, State, compact_if_full, loop, waiting_for_user


@loop(until=waiting_for_user, limit=50)
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

## waiting_for_user

`waiting_for_user` is the stop condition of the default loop, and you can import it for yours. It is true when the
model has handed the turn back: nothing waits for a tool result, and the last message is the model's reply without
tool calls. It means "your turn", not "the task is done": a message added after the reply makes it false again, and
the loop goes on.

Its name is what a stop shows: `done: stopped by waiting_for_user` in the terminal, and
`StoppedByUntil("waiting_for_user")` in `state.stopped`. Any function from State to `bool` works as `until`; its
name is shown the same way.

## Your loop

Add a step of your own to the turn. This loop warns the model when it has used 20 turns:

```python
--8<-- "docs_src/learn_loop.py"
```

1. `waiting_for_user` is the stop condition, imported from the library.
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
