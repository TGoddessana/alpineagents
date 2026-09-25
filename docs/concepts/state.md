# State

A State holds one task: its history, its context and its answer. Create one per task:

```python
from alpineagents import State

state = State("Find the bug in main.py")
```

The task becomes the first message the model sees. A State can go through several runs, for example one per message
in a chat.

## history and context

A State keeps two lists.

|  | `history` | `context` |
| --- | --- | --- |
| Contains | Everything that happened | The messages the model sees at the next `think` |
| Can shrink | No. Entries are only added | Yes, by compaction, `clear_tool_results` and `start_from` |
| Use it to | Inspect, log or debug | Know what the model knows right now |

After compaction, for example, the context holds only the task and a summary. The history still holds every message,
reply and tool result.

## Read the result

| Property | Value |
| --- | --- |
| `answer` | The value given to `finish(answer)`. Otherwise the text of the latest reply without tool calls. Otherwise `None` |
| `stopped_by` | Why the last loop stopped: `"finish"`, `"limit"`, or the name of an `until` function. `None` while running or after an exception |
| `turn` | How many times `think` succeeded |
| `usage` | Tokens, requests and cost of every model request on this State |

## Read the inside

| Property | Value |
| --- | --- |
| `task` | The task the State was created with |
| `pending_calls` | Tool calls the model asked for that have no result yet |
| `context` | See above |
| `history` | See above. `HistoryEntry.kind` values are listed in the [Data types API](../api/types.md) |
| `data` | A dict for your own values. The model never sees it |

The size of the context is in [Context size](../guides/context-size.md#watch-the-size). Every property is in the
[State API](../api/state.md).

## Check the run

| Method | True when |
| --- | --- |
| `is_answered()` | The last message in the context is a model reply without tool calls |
| `wants_tools()` | `pending_calls` is not empty |
| `is_finished()` | `finish()` was called |

These take only the State, so they work as stop conditions: `until=State.is_answered`.

## Add a message

| Method | Adds |
| --- | --- |
| `add_user_message(text)` | A message from the person |
| `add_notice(text)` | A message from your code. The model sees it with the prefix `[notice] ` |

A message added after the model's answer makes `is_answered()` false, so the loop continues. The guide
[Check the work before finishing](../guides/verify.md) uses this.

## Control the run

| Method | Effect |
| --- | --- |
| `deny(call, reason)` | Refuses one pending call. The model gets `reason` as that call's error result |
| `finish(answer=None)` | Stops the loop before the next turn. Sets `answer` if it is not `None` |

`finish` ends the State for good. After it, `think`, `use_tools`, `ask` and `run` raise `ValueError`. If a tool calls
`finish`, end the loop body without another `think`.

## Shrink the context

`clear_tool_results` and `start_from` shrink the context without a model request. See
[Context size](../guides/context-size.md).

## Pending calls

When a reply asks for tools, each call becomes a pending call in `state.pending_calls`. A call stops being pending
when:

- `use_tools` records its result,
- `deny` refuses it, or
- an exception ends the run and closes it.

Every tool call needs a result before the model is asked again. So while calls are pending:

| Method | Behavior |
| --- | --- |
| `think`, `compact`, `start_from`, `clear_tool_results` | Raise `ValueError` |
| `add_user_message`, `add_notice` | Wait, and go into the context right after the results |
| `finish`, `deny` | Work as usual |

## Your own data

`state.data` is a dict for your code and your tools. Use it to keep values across turns:

```python
allowed = state.data.setdefault("allowed", set())
```

A single read or write is thread-safe. For an update in several steps, hold `state.lock`:

```python
with state.lock:
    state.data["calls"] = state.data.get("calls", 0) + 1
```

## Run the same State again

`agent.run(state)` continues a State that already ran. Use the same Agent: another Agent, including one made with
`agent.copy(...)`, raises `ValueError`.

| The State | What `run` does |
| --- | --- |
| Stopped by an `until` function | Stops at once if the function is still true. To continue a conversation, add a message with `add_user_message` first. See [Chat](../guides/chat.md) |
| Stopped by `limit` | Runs up to `limit` more turns. The loop counts turns from zero on every call |
| Stopped by Ctrl+C or an exception | Continues. Pending calls were closed with a result such as `(interrupted by user)` |
| Ended with `finish()` | Raises `ValueError` |
