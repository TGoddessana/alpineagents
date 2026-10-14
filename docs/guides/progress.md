# Show progress and ask the person

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
done: stopped by waiting_for_user (2 turns)
```

- `[turn N] thinking` starts each model request. The model's text follows as it streams in.
- Indented lines are tool calls and their results.
- The last line says why the run stopped and how many turns it took. When the Model has prices, it also shows the
  cost, for example `done: stopped by waiting_for_user (5 turns, ~$0.42)`.

A [subagent](subagents.md)'s lines are indented under the tool call that started it, and start with its name. Its
streamed text is not shown:

```text
  tool researcher(prompt="How does Redis evict keys?")
  researcher started
  [researcher turn 1] thinking
    tool web_search(query="Redis eviction")
    done 4.0KB
  researcher done: stopped by finish (2 turns)
  done 612B
```

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

Subclass `Reporter` and override only the events you need. Every event does nothing by default. Pass it with `reporter=`.

```python
--8<-- "docs_src/reporter.py"
```

| Event | When |
| --- | --- |
| `on_run_start(state)` | `run` starts |
| `on_think_start(state)` | Right before `think` or `ask` sends a request |
| `on_text(state, chunk)` | Each piece of model text as it streams in |
| `on_think_end(state, reply)` | Right after the reply ends |
| `on_tool_start(state, call)` | Right before a tool runs |
| `on_tool_end(state, call, result, outcome)` | Right after a tool call ends. `outcome.kind` says how: `"done"`, `"error"`, `"denied"`, ... `result` is a string, or a tuple of text and images for a tool that returned an [`Image`](more-tools.md#images-from-a-tool) (`result_text(result)` makes it one string) |
| `on_context_change(state, change)` | The messages were compacted (`state.compact` or `agent.compact`), cleared (`clear_tool_results`) or restored (`state.restore`). `change.kind` says which |
| `on_model_event(state, event)` | The Model reported something outside the reply, such as a fallback |
| `on_run_end(state, error)` | `run` ends, always. `error` is `None` on a normal finish |

- Events can come from several threads. For example, Agents in different threads share the default Terminal. Make
  the Reporter thread-safe.
- A call a [permission](permissions.md) refused gets `on_tool_end` without `on_tool_start`, with `outcome.kind`
  `"denied"`, or `"cancelled"` for the other calls of a turn it stopped. `outcome.decided_by` names the permission.
- A [subagent](subagents.md) with no `reporter=` reports to the Reporter of the run above it. Its events come with the
  subagent's State, so `state.parent_id` (the parent State's id, or `None`) tells them apart from the top-level run's.
  [Use subagents](subagents.md#show-the-tree) has a Reporter that prints a tree.
- Exceptions raised in a Reporter propagate.

## Write a Human

Subclass `Human` and implement `ask`:

```python
--8<-- "docs_src/human.py"
```

- `returns` is `str`, `bool` or a `Literal[...]` of choices. Return a value of that type.
- A subagent with no `human=` asks the person of the run above it.
- For a person reached asynchronously, such as on a web page, implement `async def aask(...)` instead and use
  `agent.aask_human`.

## Related

- [Ask before a tool runs](../learn/approval.md)
- [Reporter, Human, Terminal API](../api/io.md)
