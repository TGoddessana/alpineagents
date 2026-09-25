# How a run works

This page follows one call to `agent.run` from start to end. The next pages explain each part in detail. Terms are
defined in the [Glossary](glossary.md).

## Three parts

| Part | Holds | Changes during a run |
| --- | --- | --- |
| `Agent` | The model, the system prompt, the tools and the loop | No |
| `State` | What happened so far, what the model sees next, the answer | Yes |
| Loop | The steps of one turn, and when to stop | No |

The Agent does the work. The State records it. The loop decides the order.

## One run, step by step

`agent` is the Agent from the [quick start](../index.md#quick-start).

```python
answer = agent.run("Find the bug in main.py")
```

1. `run` creates `State("Find the bug in main.py")`.
2. `run` calls the Agent's loop with the Agent and the State.
3. Before each turn, the loop checks whether to stop.
4. Each turn runs the loop body. The default body does three steps:
    1. `compact_if_full(agent, state)`: if the context is more than 60% of the model's context window, it replaces
       the context with the task and a summary.
    2. `agent.think(state)`: sends the context to the model and records the reply in the State.
    3. `agent.use_tools(state)`: if the reply asked for tool calls, runs them and records the results.
5. When the loop stops, `run` returns `state.answer`.

## The default loop

An Agent created without `loop=` uses `default_loop`. This is its full source:

```python
@loop(until=State.is_answered, limit=50)
def default_loop(agent: Agent, state: State):
    compact_if_full(agent, state)
    agent.think(state)
    if state.wants_tools():
        agent.use_tools(state)
```

- The function body is one turn.
- `until=State.is_answered`: stop when the last message in the context is a reply without tool calls.
- `limit=50`: stop after 50 turns at most.

A loop you write has the same shape. [Loops](loops.md) explains how.

## Keep the State

`run` also accepts a State. Create it yourself to inspect the run afterwards:

```python
from alpineagents import State

state = State("Find the bug in main.py")
agent.run(state)

state.answer      # the answer
state.stopped_by  # why the loop stopped, for example "is_answered"
state.turn        # how many turns it took
state.usage       # tokens, requests and cost
print(state)      # one line per history entry
```

`print(state)` shows what happened, in order:

```text
[turn 0] user: Find the bug in main.py
[turn 1] reply: read_file(path="main.py")
[turn 1] tool_result read_file: 8B
[turn 2] reply: Line 1 is fine.
done: is_answered (2 turns)
```
