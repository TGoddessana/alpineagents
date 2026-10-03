# Loops and stop rules

A loop decides the steps of one turn and when to stop. To write one, see [Write your own loop](../learn/loop.md). This
page has the rules.

## When a loop stops

`state.stopped` says why the run is set to stop. It is `None` until something decides:

| Who decides | When `state.stopped` is set | `state.stopped` |
| --- | --- | --- |
| `state.finish()` | Right away, when it is called | `StoppedByFinish(answer)` |
| A permission denies a tool call with `stop=True` | Right away, when `use_tools` records the results | `StoppedByPermission(call, permission)` |
| An `until` function returns `True` | When the loop checks it, before a turn | `StoppedByUntil(name)`, for example `StoppedByUntil("waiting_for_user")` |
| `limit` turns have run in this call | When the loop checks it, before a turn | `StoppedByLimit(turns)`, where `turns` is the limit |

The default loop's check has its own name: [`is_answered`](../learn/loop.md#the-name-is_answered).

Before every turn, the loop stops if `state.stopped` is already set. Otherwise it checks the `until` functions, then
`limit`. Each stop is a `StopEntry` in `state.history`.

- The checks run before each turn, so an `until` function sees the State after the previous turn.
- Reaching `limit` does not raise. Check `isinstance(state.stopped, StoppedByLimit)` when you need to know.
- `run` records a `RunStartEntry`, which resets `state.stopped` to `None`. After a run that ended with an exception,
  it is `None` too.
- `state.finish(answer)` ends every loop, and it can be called from the loop body or from a tool. It ends the State for
  good: see [State and history](state.md#finish-ends-the-state). A permission's `stop=True` ends every loop too.

## until

`until` takes a function from State to `bool`, or a list of them. The loop stops when any of them returns `True`.

```python
from alpineagents import Agent, State, loop


def waiting_for_user(state: State) -> bool:
    if state.pending_calls or not state.messages:
        return False
    last = state.messages[-1]
    return last.role == "assistant" and not last.tool_calls


def spent_too_much(state: State) -> bool:
    return state.usage.output_tokens > 20_000


@loop(until=[waiting_for_user, spent_too_much], limit=30)
def careful(agent: Agent, state: State):
    agent.think(state)
    if state.pending_calls:
        agent.use_tools(state)


agent = Agent(model="claude-sonnet-5", loop=careful)
quick = careful.copy(limit=5)
```

- Pass the function itself: `until=waiting_for_user`. Not `waiting_for_user(state)`.
- Use a named function. Its name becomes `state.stopped.name`. A `lambda` works, but warns because it has no useful
  name.
- An `until` function that reads several values can take `snap = state.snapshot()` first, so they all belong to one
  moment even if another thread adds a message in between.
- `copy` returns a new loop with different `until` or `limit`. The original does not change.

## Loops without `@loop`

A loop is any function that takes `(agent, state)`. One written without `@loop` makes none of the checks above, so
check `state.stopped` yourself:

```python
from alpineagents import Agent, State


def waiting_for_user(state: State) -> bool:
    if state.pending_calls or not state.messages:
        return False
    last = state.messages[-1]
    return last.role == "assistant" and not last.tool_calls


def my_loop(agent: Agent, state: State):
    while state.stopped is None and not waiting_for_user(state):
        agent.think(state)
        if state.pending_calls:
            agent.use_tools(state)


agent = Agent(model="claude-sonnet-5", loop=my_loop)
```

## Nested loops

A loop inside another loop: its `until` or `limit` ends only the inner loop. The outer `@loop` checks its own, and if
it goes on, `state.stopped` goes back to `None`. That is a `StopEntry` with `content=None`, which history shows as a
stop that was cleared. `finish()` and a permission's stop end every loop.

A loop written without `@loop` does not clear what a `@loop` inside it leaves behind. After the inner loop stops on
its own `until` or `limit`, `state.stopped` stays set, so `state.stopped is None` would end the outer loop too. When
it calls a `@loop`, check only the stops that end every loop, or write the outer loop with `@loop` as well:

```python
--8<-- "docs_src/nested_plain_loop.py"
```

## Rules

| Rule | Detail |
| --- | --- |
| The body must return `None` | A returned value raises `TypeError`. Set the answer with `state.finish(answer)` |
| Each call of the loop counts turns from zero | `state.turn` keeps counting across calls, the loop's count does not |
| Exceptions from the body or an `until` function propagate | Nothing is swallowed. See [Errors and interruptions](errors.md) |
| An `async def` body makes an async loop | Run it with `agent.arun`. See [Use async](../guides/async.md) |

A block is a function that takes `(agent, state)` and does one step of a turn, such as `compact_if_full`. See
[Write your own loop](../learn/loop.md#your-loop).
