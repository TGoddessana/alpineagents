# State

A State is one conversation with its record: the messages the model sees, everything that happened, and the answer.
Create one per task:

```python
from alpineagents import Message, State

state = State(messages=[Message.user("Find the bug in main.py")])
```

A State can go through several runs, for example one per message in a chat. `agent.run("text")` creates one for you
when you do not need to keep it.

## The model in short

- A State is the one object you change. You change it only through its methods (`add_message`, `finish`, `compact`,
  `edit_extra_data`, `restore` and the rest), and the Agent changes it through the same path. Reading never changes
  anything.
- Every change adds one entry to `state.history`. History only grows, and it is the only source: `messages`,
  `turn`, `usage`, `extra_data`, `answer` and the other values are computed from it. A State built from the same
  history has the same values.
- `state.snapshot()` returns a `StateSnapshot`, the frozen value of the State at that moment. It never changes, so
  you can keep it, compare it, or go back to it with `state.restore(snapshot)`.
- A store saves only the history, and loading a State replays it.

So there is no hidden state to keep in sync. If you know the history, you know everything.

## Create a State

| Argument | Meaning |
| --- | --- |
| `messages` | The conversation to start from, as `Message` objects. A `str` is not accepted: use `Message.user("...")` |
| `extra_data` | Your own values. See [extra_data](#extra_data-your-notepad) |
| `id` | The name a store saves the State under. Letters, digits, `_` and `-`. A random id when left out |
| `history` | Rebuild a State from the history of another one. Cannot be combined with `messages` or `extra_data` |

Every argument is keyword-only: `State("text")` raises `TypeError` with the right form. A State may start empty and get
its messages with `add_message`. Calling `agent.think` on a State with no messages raises `ValueError`.

```python
state = State(
    id="bug-1",
    messages=[Message.user("Find the bug in main.py")],
    extra_data={"repo": "api"},
)
```

## Add a message

| Message | Made with | The model sees |
| --- | --- | --- |
| From the person | `Message.user(text, *images)` | The text, and the images |
| From your code | `Message.notice(text)` | The text with the prefix `[notice] ` |
| From the model | `Message.assistant(text)` | Only for `State(messages=...)`, to start from a conversation that has replies |

Add a message while the State is running, or between runs:

```python
state.add_message(Message.user("Now fix it"))
state.add_message(Message.notice("The tests failed"))
```

`add_message` takes user messages and notices only. A message with no text and no image raises `ValueError`. A message
added after the model's answer makes the loop's stop condition false again, so the loop continues. The guide
[Check the work before finishing](../guides/verify.md) uses this.

## Read the State

All of these are read-only properties. They read the current snapshot.

| Property | Value |
| --- | --- |
| `messages` | What the model sees at the next `think` |
| `history` | Everything that happened. See [History](#history) |
| `answer` | The value given to `finish(answer)`, in JSON form (a dataclass or Pydantic model becomes a read-only dict). Otherwise the text of the latest reply without tool calls. Otherwise `None` |
| `finished` | Whether `finish()` was called |
| `stopped` | Why this run is set to stop: `StoppedByFinish(answer)`, `StoppedByLimit(turns)`, `StoppedByUntil(name)` or `StoppedByPermission(call, permission)`. `None` until something decides, and after a run that ended with an exception. See [Loops](loops.md#when-a-loop-stops) |
| `pending_calls` | Tool calls the model asked for that have no result yet |
| `turn` | How many model replies there were. It is one more while a request waits on the model |
| `usage` | Tokens, requests and cost of every `think`, `ask` and `compact` on this State |
| `extra_data` | Your notepad. Read-only here: change it with `edit_extra_data()` |
| `created_at`, `updated_at` | When the first and the latest history entry were recorded, in UTC |
| `id` | The name a store saves the State under |

The size of the context is a question for the Agent, because it depends on the model and the tools:
`agent.context_used(state)` and `agent.context_tokens(state)`. See
[Context size](../guides/context-size.md#watch-the-size). Every property is in the [State API](../api/state.md).

`print(state)` shows the history, one line per entry.

## messages and history

<a id="history-and-context"></a>

A State keeps two lists, and `messages` is computed from `history`.

|  | `history` | `messages` |
| --- | --- | --- |
| Contains | Everything that happened | What the model sees at the next `think` |
| Can shrink | No. Entries are only added | Yes, by `compact` and `clear_tool_results` |
| Use it to | Inspect, log, debug, save | Know what the model knows right now |

After a compaction, `messages` holds the first message and a summary. The history still holds every message, reply
and tool result.

## History

Every entry has `kind`, `content`, `turn` (the State's turn when it was recorded) and `at` (when, in UTC). There are 11
entry classes. Check `kind`, or use `isinstance` or `match`, to know which one an entry is and so what its `content`
holds.

| Class | `kind` | Recorded when |
| --- | --- | --- |
| `MessageEntry` | `user`, `notice` | `add_message` |
| `ModelRequestEntry` | `model_request` | `think` sends a request. `content` is the model name |
| `ModelReplyEntry` | `model_reply` | The model answered. `content` is the `Reply`, including `reply.model` |
| `ModelEventEntry` | `model_event` | The model adapter reported something, such as a retry or a fallback to another model |
| `ToolResultEntry` | `tool_result` | A tool call got its result, whichever way it ended. `outcome` says how |
| `ExchangeEntry` | `ask`, `human` | `agent.ask` or `agent.ask_human` got an answer |
| `ContextChangeEntry` | `context_change` | `State(messages=...)`, `compact`, `clear_tool_results`, `restore` |
| `RunStartEntry` | `run_start` | A run started. `content` is the [`AgentInfo`](#agentinfo) of the Agent |
| `StopEntry` | `stop` | The run was set to stop: `finish`, a loop's `until` or `limit`, a permission |
| `ExtraDataEntry` | `extra_data` | `State(extra_data=...)` or `edit_extra_data()` changed something |
| `ErrorEntry` | `error` | An exception was raised in `think`, `use_tools` or `run` |

The names follow one rule. `Model...` entries are about the model: what was asked, what it answered, what happened to
the request. `ToolResultEntry` is about a tool call. The other names say what the entry records.

A tool call can end in seven ways, and each is a `ToolResultEntry` with an `outcome`: `done`, `error`,
`input_error`, `aborted`, `interrupted`, `denied` and `cancelled`. `entry.is_error` is true for every outcome except
`done`. Every class is in the [Data types API](../api/types.md#history-entries).

```python
for entry in state.history:
    if entry.kind == "model_reply":
        print(entry.content.tool_calls)
    elif entry.kind == "tool_result":
        print(entry.call.name, entry.outcome)
```

The values in history are read-only. A tool call's `args` and a recorded `ask` answer are frozen dicts and lists: they
work with `json.dumps`, indexing and `==`, but `args["x"] = 1` raises `TypeError`. Copy with `dict(args)` to change
one.

Values are frozen to their JSON form when they are recorded, on a live State as well as a loaded one. A tuple becomes
a list. The `answer` of `finish(answer)` and of an `ask` entry, if it is a Pydantic model or a dataclass, is stored as
a dict, so `state.answer` is a dict even before any save (`agent.ask(..., returns=Type)` still returns the object to
its caller). A value that is not JSON raises `TypeError`.

## extra_data: your notepad

`state.extra_data` holds your own values. The model never sees them. Use it to keep values across turns and runs, such
as which files a tool already read. It is read-only; change it inside `edit_extra_data()`:

```python
with state.edit_extra_data() as data:
    data.setdefault("notes", {})["bug"] = "b.py"
```

- The block gives you a plain dict that is a copy. When the block ends normally, the changes become one
  `ExtraDataEntry`. If nothing changed, nothing is recorded. If the block raises, nothing changes.
- Two threads editing at once take turns, so `data["calls"] = data.get("calls", 0) + 1` never loses an update. That
  is why there is no lock to hold.
- Values must be JSON: dicts with string keys, lists, strings, numbers, booleans and `None`. A set or another object
  raises `TypeError` that names the key, such as `state.extra_data['seen'] must be JSON (got set)`. Use a list.
- Because it is history, `extra_data` is saved with the State, and `restore` takes it back too.

`state.extra_data["x"] = 1` raises `TypeError`. A tool does the same thing with `state.edit_extra_data()`. See
[Tools that use the State](../guides/tool-state.md).

## Control the run

| Method | Effect |
| --- | --- |
| `finish(answer=None)` | Sets `stopped` to `StoppedByFinish(answer)` right away, so the loop stops before the next turn. Sets `answer` if it is not `None` |

To refuse tool calls before they run, give the Agent [permissions](../guides/approval.md).

`finish` ends the State for good. After it, `think`, `use_tools`, `ask` and `run` raise `ValueError`. If a tool calls
`finish`, end the loop body without another `think`. Only `restore` can take it back.

## Shrink the context

`compact(summary)` and `clear_tool_results()` shrink `messages` without a model request. `agent.compact(state)` asks the
model for the summary. See [Context size](../guides/context-size.md).

## StateSnapshot: a value you can keep

```python
before = state.snapshot()
agent.run(state)
state.restore(before)   # back to before the run
```

`state.snapshot()` returns a `StateSnapshot`. It has the fields of the properties above, `history`, `messages`,
`pending_calls`, `turn`, `usage`, `extra_data`, `finished`, `answer`, `stopped`, `created_at` and `updated_at`, and no
methods that change anything. It is frozen: `snapshot.turn = 1` raises, and so does `snapshot.extra_data["x"] = 1`.
Two snapshots are equal when their histories are equal.

Use a snapshot when:

- You read several values that must belong together while a run goes on in another thread. Each property reads the
  current snapshot on its own, so two reads can see two moments. One snapshot is one moment.
- You want to go back. `state.restore(snapshot)` makes every value what it was then: `messages`, `turn`,
  `extra_data`, `finished`, `answer` and `stopped`. Two things stay as they are. `usage` stays cumulative, because
  the tokens were spent, and `history` keeps everything and gets one `context_change` entry with `kind="restore"`.
- You compare two points of one State.

`restore` takes a snapshot of this State, taken between turns. It raises `ValueError` for a snapshot of another State
(build one from it with `State(history=snapshot.history)`), while calls are pending, and while the model is being
waited on.

## Copy a State

`state.fork()` returns a new State with the same history, a new id, no store and no running Agent. Run the copy
without touching the original:

```python
attempt = state.fork()
agent.run(attempt)
```

A fork is the same State at that point: a fork of a finished State is finished too, so to continue from its
messages start a new State: `State(messages=state.messages)`. Fork between turns: a fork taken while the model is being
waited on is waiting too.

## Pending calls

When a reply asks for tools, each call becomes a pending call in `state.pending_calls`. A call stops being pending
when:

- `use_tools` records its result,
- a [permission](../guides/approval.md) refuses it, or cancels it because it stopped the turn, or
- an exception ends the run and closes it.

Every tool call needs a result before the model is asked again. So while calls are pending:

| Method | Behavior |
| --- | --- |
| `think`, `compact`, `clear_tool_results`, `restore` | Raise `ValueError` |
| `add_message` | Waits, and goes into `messages` right after the results |
| `finish` | Works as usual |

`add_message` also waits while `think` is waiting on the model, and goes in right after the reply.

## Run the same State again

`agent.run(state)` continues a State that already ran, with any Agent, including `agent.copy(model=...)`. A State can
be run by one `run` at a time: starting a second one raises `ValueError`.

| The State | What `run` does |
| --- | --- |
| Stopped by an `until` function | Stops at once if the function is still true. To continue a conversation, add a message with `add_message` first. See [Chat](../guides/chat.md) |
| Stopped by `limit` | Runs up to `limit` more turns. The loop counts turns from zero on every call |
| Stopped by Ctrl+C or an exception | Continues. Pending calls were closed with a result such as `(interrupted by user)` |
| Ended with `finish()` | Raises `ValueError` |
| Loaded with `store.load(id)` | Continues where the saved history ends. See [Save and resume](../guides/resume.md) |

Because any Agent may continue a State, you can switch models between runs:

```python
agent.run(state)
cheaper = agent.copy(model="claude-haiku-4-5")
cheaper.run(state)
```

Each run records a `RunStartEntry`, so history tells which Agent did what. Switching the model this way never warns.

### AgentInfo

`RunStartEntry.content` is an `AgentInfo`: the Agent's `name`, `model` (`provider/name`), `system_sha256` (a hash of
the system prompt, not the text), `tools` (names) and `mcp_servers` (names). `ModelRequestEntry.content` holds the model
name of every request the run loop sends (`think`), so history also shows when the model changed inside one run.
`ask` and `compact` do not record one.

### A State saved by another Agent

The Agent's settings live in your code, so an Agent that resumes a saved State after a deploy may differ from the one
that ran it. The run still works, but it may behave differently: the model gets an error result when it calls a tool
that was removed, and blocks only one provider understands are not sent to another provider.

So the first `run` or `arun` of a State that a store loaded compares the `AgentInfo` of its last run with the Agent's.
If they differ, it warns with `ResumeWarning`, for example
`tools: removed search_web, added search_docs; model: anthropic/claude-sonnet-5 -> openai/gpt-5`. The system prompt is
reported only as changed. To refuse such a resume, make the warning an exception:

```python
import warnings

from alpineagents import ResumeWarning

warnings.filterwarnings("error", category=ResumeWarning)
```

`run` then raises `ResumeWarning` before the loop starts. Its `changes` attribute holds
`{key: (saved_value, current_value)}` for each difference, for example to show a notice in your UI.

## StateInfo

`store.list()` returns one `StateInfo` per saved State, without loading it: `id`, `first_message`, `created_at`,
`updated_at`, `turn`, `stopped` and `finished`. Use it to show a list of conversations, then `store.load(id)` the one
the person picks. See [Save and resume](../guides/resume.md).
