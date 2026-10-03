# How a run works

This page follows one call to `agent.run` from start to end. The next pages explain each part. Terms are defined in the
[Glossary](glossary.md). If you have not written an agent yet, start with [Your first agent](../learn/first-agent.md).

## Three parts

| Part | Holds | Changes during a run |
| --- | --- | --- |
| `Agent` | The model, the system prompt, the tools and the loop | No |
| `State` | What happened so far (history), what the model sees next (messages), the answer | Yes |
| Loop | The steps of one turn, and when to stop | No |

The Agent does the work. The State records it. The loop decides the order.

## One run, step by step

```python
from pathlib import Path

from alpineagents import Agent, Message, State, tool


@tool
def read_file(path: str) -> str:
    """Read a text file"""
    return Path(path).read_text()


agent = Agent(model="claude-sonnet-5", tools=[read_file])
state = State(messages=[Message.user("Find the bug in main.py")])
answer = agent.run(state)
print(state)
```

1. `run` records that a run started. Given a string instead of a State, it first creates
   `State(messages=[Message.user(text)])`.
2. `run` calls the Agent's loop with the Agent and the State.
3. Before each turn, the loop checks whether to stop. See [Loops and stop rules](loops.md#when-a-loop-stops).
4. Each turn runs the loop body. The default body does three steps:
    1. `compact_if_full(agent, state)`: if the messages fill more than 60% of the model's context window, it replaces
       them with the first message and a summary.
    2. `agent.think(state)`: sends the messages to the model, and records the request and the reply.
    3. `agent.use_tools(state)`: if the reply asked for tool calls, runs them and records the results. With
       `Agent(permissions=[...])`, it first asks the permissions about every call of the turn, before any tool runs.
       A refused call does not run, and the model gets the reason as its result.
5. When the loop stops, `run` returns `state.answer`.

The source of the default loop is in [Write your own loop](../learn/loop.md#the-default-loop). A loop you write has
the same shape.

## What the run recorded

`print(state)` shows one line per history entry, in order:

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
computed from it. See [State and history](state.md).

## When a run ends

| How it ends | `run` | The State afterwards |
| --- | --- | --- |
| A stop rule fires, or `finish` is called | Returns `state.answer` | `state.stopped` says why |
| An exception leaves the loop | Raises it | `state.stopped` is `None`; pending calls are closed. See [Errors and interruptions](errors.md) |
| Ctrl+C | Raises `KeyboardInterrupt` | Pending calls are closed with `(interrupted by user)`. See [Errors and interruptions](errors.md#ctrlc-and-cancellation) |

With `Agent(store=...)`, each State is saved as it runs. See [Save and resume](../learn/save-resume.md).
