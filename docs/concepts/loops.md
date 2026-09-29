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

`state.stopped` says why the run is set to stop. It is `None` until something decides:

| Who decides | When `state.stopped` is set | `state.stopped` |
| --- | --- | --- |
| `state.finish()` | Right away, when it is called | `StoppedByFinish()` |
| A permission denies a tool call with `stop=True` | Right away, when `use_tools` records the results | `StoppedByPermission(call, permission)` |
| An `until` function returns `True` | When the loop checks it, before a turn | `StoppedByUntil(name)`, for example `StoppedByUntil("is_answered")` |
| `limit` turns have run in this call | When the loop checks it, before a turn | `StoppedByLimit(turns)`, where `turns` is the limit |

Before every turn, the loop stops if `state.stopped` is already set. Otherwise it checks the `until` functions, then
`limit`.

- The checks run before each turn, so an `until` function sees the State after the previous turn.
- Reaching `limit` does not raise. Check `isinstance(state.stopped, StoppedByLimit)` when you need to know.
- `run` resets `state.stopped` to `None` when it starts. After a run that ended with an exception, it is `None`.
- A loop inside another loop: its `until` or `limit` ends only the inner loop. The outer loop checks its own, and if
  it goes on, `state.stopped` goes back to `None`. `finish()` and a permission's stop end every loop.
- A loop written without `@loop` makes none of these checks. Check `state.stopped` yourself:

```python
def my_loop(agent: Agent, state: State):
    while state.stopped is None and not state.is_answered():
        agent.think(state)
        if state.wants_tools():
            agent.use_tools(state)
```

A loop written without `@loop` does not clear what a `@loop` inside it leaves behind. After that inner loop stops on
its own `until` or `limit`, `state.stopped` stays set, so `state.stopped is None` would end the outer loop too. When
it calls a `@loop`, check only the stops that end every loop, or write the outer loop with `@loop` as well:

```python
--8<-- "docs_src/nested_plain_loop.py"
```

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
- Use a named function. Its name becomes `state.stopped.name`. A `lambda` works, but warns because it has no useful
  name.

## Stop from inside a turn

`state.finish(answer)` sets `state.stopped` to `StoppedByFinish()` right away, and the loop stops before the next
turn. The loop body and tools can call it. See [Stop conditions](../guides/stop-conditions.md).

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
