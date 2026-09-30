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
reply and tool result. A `"reply"` entry holds the whole `Reply`, including `reply.model`, the model that answered.

## Read the result

| Property | Value |
| --- | --- |
| `answer` | The value given to `finish(answer)`. Otherwise the text of the latest reply without tool calls. Otherwise `None` |
| `stopped` | Why this run is set to stop: `StoppedByFinish()`, `StoppedByLimit(turns)`, `StoppedByUntil(name)` or `StoppedByPermission(call, permission)`. `None` until something decides, and after a run that ended with an exception. See [Loops](loops.md#when-a-loop-stops) |
| `turn` | How many times `think` succeeded |
| `usage` | Tokens, requests and cost of every model request on this State |

## Read the inside

| Property | Value |
| --- | --- |
| `id` | The name a store saves the State under. `State(task, id=...)`, or random |
| `task` | The task the State was created with |
| `pending_calls` | Tool calls the model asked for that have no result yet |
| `context` | See above |
| `history` | See above. Each entry is a `ReplyEntry`, `ToolResultEntry` and so on, told apart by `kind`; they are listed in the [Data types API](../api/types.md#history-entries). Each entry's `at` is when it was recorded, in UTC |
| `created_at` | When the State was created, in UTC: the `at` of the first history entry |
| `updated_at` | When something was last added to history, in UTC. Changing `data` alone does not move it |
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
| `finish(answer=None)` | Sets `stopped` to `StoppedByFinish()` right away, so the loop stops before the next turn. Sets `answer` if it is not `None` |

To refuse tool calls before they run, give the Agent [permissions](../guides/approval.md).

`finish` ends the State for good. After it, `think`, `use_tools`, `ask` and `run` raise `ValueError`. If a tool calls
`finish`, end the loop body without another `think`.

## Shrink the context

`clear_tool_results` and `start_from` shrink the context without a model request. See
[Context size](../guides/context-size.md).

## Pending calls

When a reply asks for tools, each call becomes a pending call in `state.pending_calls`. A call stops being pending
when:

- `use_tools` records its result,
- a [permission](../guides/approval.md) refuses it, or cancels it because it stopped the turn, or
- an exception ends the run and closes it.

Every tool call needs a result before the model is asked again. So while calls are pending:

| Method | Behavior |
| --- | --- |
| `think`, `compact`, `start_from`, `clear_tool_results` | Raise `ValueError` |
| `add_user_message`, `add_notice` | Wait, and go into the context right after the results |
| `finish` | Works as usual |

## Your own data

`state.data` is a dict for your code and your tools. Use it to keep values across turns:

```python
allowed = state.data.setdefault("allowed", [])
```

If the Agent has a [store](../guides/resume.md), keep values JSON can hold there: dicts with string keys, lists,
strings, numbers, booleans and `None`. Saving a State whose `data` holds a set or another object raises `TypeError`.

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
| Loaded with `store.load(id)` | Continues where the saved steps end. It has no owner yet, so the Agent in your code today can run it. See [Save and resume](../guides/resume.md) |

### A State saved by another Agent

A State saved with a [store](../guides/resume.md) also records which Agent saved it: its name, model,
a hash of its system prompt, its tools and its MCP servers. The Agent's settings live in your code, so an Agent that
resumes the State after a deploy may differ. The run still works, but it may behave differently: the model gets an
error result when it calls a tool that was removed, and blocks only one provider understands are not sent to another
provider.

So the first `run` or `arun` of a loaded State compares the two Agents. If they differ, it warns with `ResumeWarning`,
for example `tools: removed search_web, added search_docs; model: anthropic/claude-sonnet-5 -> openai/gpt-5`. The
system prompt is reported only as changed. To refuse such a resume, make the warning an exception:

```python
import warnings

from alpineagents import ResumeWarning

warnings.filterwarnings("error", category=ResumeWarning)
```

`run` then raises `ResumeWarning` before the loop starts. Its `changes` attribute holds
`{key: (saved_value, current_value)}` for each difference, for example to show a notice in your UI.
