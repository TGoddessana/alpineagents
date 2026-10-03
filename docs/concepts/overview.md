# How a run works

This page follows one call to `agent.run` from start to end. The next pages explain each part in detail. Terms are
defined in the [Glossary](glossary.md).

## Three parts

| Part | Holds | Changes during a run |
| --- | --- | --- |
| `Agent` | The model, the system prompt, the tools and the loop | No |
| `State` | What happened so far (history), what the model sees next (messages), the answer | Yes |
| Loop | The steps of one turn, and when to stop | No |

The Agent does the work. The State records it. The loop decides the order.

## One run, step by step

`agent` is the Agent from the [quick start](../index.md#quick-start).

```python
answer = agent.run("Find the bug in main.py")
```

1. `run` creates `State(messages=[Message.user("Find the bug in main.py")])`, and records that a run started.
2. `run` calls the Agent's loop with the Agent and the State.
3. Before each turn, the loop checks whether to stop.
4. Each turn runs the loop body. The default body does three steps:
    1. `compact_if_full(agent, state)`: if the messages are more than 60% of the model's context window, it replaces
       them with the first message and a summary.
    2. `agent.think(state)`: sends the messages to the model and records the request and the reply in the State.
    3. `agent.use_tools(state)`: if the reply asked for tool calls, runs them and records the results. With
       `Agent(permissions=[...])`, it first asks the permissions about every call of the turn, before any tool runs.
       A refused call does not run, and the model gets the reason as its result. See
       [Ask before a tool runs](../guides/approval.md).
5. When the loop stops, `run` returns `state.answer`.

## The default loop

An Agent created without `loop=` uses `default_loop`. This is its full source:

```python
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

- The function body is one turn.
- `until=is_answered`: stop when the last message is a reply without tool calls and no call is pending.
- `limit=50`: stop after 50 turns at most.
- `is_answered` is a plain function inside the library, not part of the API. A `State` has no such method: a stop condition is a
  function of the State that you write. [Loops](loops.md#until) shows the same function as `waiting_for_user`.

A loop you write has the same shape. [Loops](loops.md) explains how.

## Keep the State

`run` also accepts a State. Create it yourself to inspect the run afterwards:

```python
from alpineagents import Message, State

state = State(messages=[Message.user("Find the bug in main.py")])
agent.run(state)

state.answer   # the answer
state.stopped  # why the loop stopped, for example StoppedByUntil("is_answered")
state.turn     # how many turns it took
state.usage    # tokens, requests and cost
print(state)   # one line per history entry
```

`print(state)` shows what happened, in order:

```text
[turn 0] context_change import: 1 messages
[turn 0] run_start: anthropic/claude-sonnet-5
[turn 1] model_request: anthropic/claude-sonnet-5
[turn 1] model_reply: read_file(path="main.py")
[turn 1] tool_result read_file: 8B
[turn 2] model_request: anthropic/claude-sonnet-5
[turn 2] model_reply: Line 1 is fine.
[turn 2] stop: stopped by is_answered
done: stopped by is_answered (2 turns)
```

That list is `state.history`, and it is all a State is: `state.messages`, `state.turn`, `state.answer` and the rest are
computed from it. See [State](state.md).
To keep a State after the process ends, and continue it later, give the Agent a store. See
[Save and resume](../guides/resume.md).
