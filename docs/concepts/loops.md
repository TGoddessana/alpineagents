# Loops

A loop decides the steps of one turn and when to stop.

## Write a loop

`read_file` is the tool from the [quick start](../index.md#quick-start).

```python
from alpineagents import Agent, State, loop


@loop(until=State.is_answered, limit=30)
def coding(agent: Agent, state: State):
    agent.think(state)
    if state.wants_tools():
        agent.use_tools(state)


agent = Agent(model="claude-sonnet-5", tools=[read_file], loop=coding)
```

1. The function is the body of one turn. It takes `(agent, state)` and returns nothing.
2. `@loop` repeats the body. `until` and `limit` are both required.
3. `Agent(loop=...)` sets the loop. `agent.run` calls it.

## When a loop stops

Before every turn, the loop checks three things in this order:

| Order | Check | `state.stopped_by` |
| --- | --- | --- |
| 1 | `state.finish()` was called | `"finish"` |
| 2 | An `until` function returns `True` | The function's name, for example `"is_answered"` |
| 3 | `limit` turns have run in this call | `"limit"` |

- The checks run before each turn, so an `until` function sees the State after the previous turn.
- Reaching `limit` does not raise. Check `state.stopped_by == "limit"` when you need to know.

## until

`until` takes a function from State to `bool`, or a list of them. The loop stops when any of them returns `True`.

```python
def spent_too_much(state: State) -> bool:
    return state.usage.output_tokens > 20_000


@loop(until=[State.is_answered, spent_too_much], limit=30)
def careful(agent: Agent, state: State):
    ...
```

- Pass the function itself: `until=State.is_answered`. Not `state.is_answered()`, and not `state.is_answered`.
- Use a named function. Its name becomes `state.stopped_by`. A `lambda` works, but warns because it has no name.
- The names `finish` and `limit` are reserved. A function with one of these names raises `ValueError`.

## Stop from inside a turn

`state.finish(answer)` stops the loop before the next turn, with `stopped_by == "finish"`. The loop body and tools
can call it. See [Stop conditions](../guides/stop-conditions.md).

## Blocks

A block is a function that takes `(agent, state)` and does one step of a turn. `compact_if_full` is a block that
comes with the library. Write your own to reuse a step across loops:

```python
from alpineagents import Agent, State, compact_if_full, loop


def warn_when_long(agent: Agent, state: State):
    if state.turn == 20:
        state.add_notice("You have used 20 turns. Finish soon.")


@loop(until=State.is_answered, limit=30)
def coding(agent: Agent, state: State):
    compact_if_full(agent, state)
    warn_when_long(agent, state)
    agent.think(state)
    if state.wants_tools():
        agent.use_tools(state)
```

## Change until or limit

`copy` returns a new loop with different settings. The original does not change.

```python
quick = coding.copy(limit=5)
```

## Rules

| Rule | Detail |
| --- | --- |
| The body must return `None` | A returned value raises `TypeError`. Set the answer with `state.finish(answer)` |
| Each call of the loop counts turns from zero | `state.turn` keeps counting across calls, the loop's count does not |
| Exceptions from the body or an `until` function propagate | Nothing is swallowed. See [Errors and interruptions](errors.md) |
| An `async def` body makes an async loop | Run it with `agent.arun`. See [Async](../guides/async.md) |
