# Progress and questions

Two settings connect an Agent to the outside:

| Setting | Role | Default |
| --- | --- | --- |
| `reporter` | Receives progress events. Only watches, never changes the run | `Terminal` |
| `human` | Answers `agent.ask_human` | `Terminal` |

## The terminal

By default the Agent prints progress and asks questions in the terminal:

```text
[turn 1] thinking
Let me look at main.py first.
  tool read_file(path="main.py")
  done 1.2KB
[turn 2] thinking
The bug is on line 3.
done: is_answered (2 turns)
```

- `[turn N] thinking` starts each model request. The model's text follows as it streams in.
- Indented lines are tool calls and their results.
- The last line says why the run stopped and how many turns it took. When the Model has prices, it also shows the
  cost, for example `done: is_answered (5 turns, ~$0.42)`.

Other last lines:

```text
done: reached limit(50), the task may be unfinished (50 turns)
done: error TimeoutError: took too long (3 turns)
done: interrupted by user (3 turns)
```

| To | Use |
| --- | --- |
| Hide the model's streamed text, keep the other lines | `Agent(..., reporter=Terminal(show_text=False))` |
| Show nothing | `Agent(..., reporter=None)` |
| Allow no questions | `Agent(..., human=None)`. `ask_human` then raises `NoHumanError` |

## Write a Reporter

Subclass `Reporter` and override only the events you need. Every event does nothing by default.

```python
--8<-- "docs_src/reporter.py"
```

```python
from alpineagents import Agent

agent = Agent(model="claude-sonnet-5", reporter=LogReporter())
```

| Event | When |
| --- | --- |
| `on_run_start(state)` | `run` starts |
| `on_think_start(state)` | Right before `think` or `ask` sends a request |
| `on_text(state, chunk)` | Each piece of model text as it streams in |
| `on_think_end(state, reply)` | Right after the reply ends |
| `on_tool_start(state, call)` | Right before a tool runs |
| `on_tool_end(state, call, result, outcome)` | Right after a tool call ends. `outcome.kind` says how: `"done"`, `"error"`, `"denied"`, ... |
| `on_context_change(state, change)` | The context was compacted, restarted with `start_from`, cleared or rolled back |
| `on_model_event(state, event)` | The Model reported something outside the reply, such as a fallback |
| `on_run_end(state, error)` | `run` ends, always. `error` is `None` on a normal finish |

- Events can come from several threads. For example, a tool that calls `state.deny` runs on a worker thread, and
  Agents in different threads share the default Terminal. Make the Reporter thread-safe.
- Exceptions raised in a Reporter propagate.

## Write a Human

Subclass `Human` and implement `ask`:

```python
--8<-- "docs_src/human.py"
```

`write_file` and `careful` are from [Ask before a tool runs](approval.md):

```python
agent = Agent(
    model="claude-sonnet-5",
    tools=[write_file],
    loop=careful,
    human=Unattended(),
)
```

- `returns` is `str`, `bool` or a `Literal[...]` of choices. Return a value of that type.
- For a person reached asynchronously, such as on a web page, implement `async def aask(...)` instead and use
  `agent.aask_human`.

## Related

- [Ask before a tool runs](approval.md)
- [Reporter, Human, Terminal API](../api/io.md)
