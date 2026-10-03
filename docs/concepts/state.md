# State and history

A State is one conversation with its record: the messages the model sees, everything that happened, and the answer. A
State can go through several runs, for example one per message in a chat. To create and use one, see
[Keep the conversation](../learn/conversation.md).

## The model in short

- A State is the one object you change. You change it only through its methods (`add_message`, `finish`, `compact`,
  `edit_extra_data`, `restore` and the rest), and the Agent changes it through the same path. Reading never changes
  anything.
- Every change adds one entry to `state.history`. History only grows, and it is the only source: `messages`,
  `turn`, `usage`, `extra_data`, `answer` and the other values are computed from it. A State built from the same
  history has the same values.
- `state.snapshot()` returns a `StateSnapshot`, the frozen value of the State at that moment.
- A store saves only the history, and loading a State replays it.

There is no hidden state to keep in sync.

## Create a State

```python
from alpineagents import Message, State

state = State(
    id="bug-1",
    messages=[Message.user("Find the bug in main.py")],
    extra_data={"repo": "api"},
)
```

All arguments are keyword-only and optional. `messages` takes `Message` objects, not a `str`. `id` is the name a store
saves the State under (letters, digits, `_` and `-`; random when left out). `history` rebuilds a State from the history
of another one, and cannot be combined with `messages` or `extra_data`. A State may start empty, but `think` needs at
least one message.

## Add a message

| Message | Made with | The model sees |
| --- | --- | --- |
| From the person | `Message.user(text, *images)` | The text, and the images |
| From your code | `Message.notice(text)` | The text with the prefix `[notice] ` |
| From the model | `Message.assistant(text)` | Only for `State(messages=...)`, to start from a conversation that has replies |

`state.add_message(...)` takes user messages and notices, while the State is running or between runs. A message added
after the model's answer makes the stop condition false again, so the loop continues
([Check the work before finishing](../guides/verify.md)).

## Read the State

All of these are read-only properties. They read the current snapshot.

| Property | Value |
| --- | --- |
| `messages` | What the model sees at the next `think` |
| `history` | Everything that happened. See [History](#history) |
| `answer` | The value given to `finish(answer)`, in JSON form (a dataclass or Pydantic model becomes a read-only dict). Otherwise the text of the latest reply without tool calls. Otherwise `None` |
| `finished` | Whether `finish()` was called |
| `stopped` | Why this run is set to stop, or `None`. See [Loops and stop rules](loops.md#when-a-loop-stops) |
| `pending_calls` | Tool calls the model asked for that have no result yet |
| `turn` | How many model replies there were. It is one more while a request waits on the model |
| `usage` | Tokens, requests and cost of every `think`, `ask` and `compact` on this State |
| `extra_data` | Your notepad. Read-only here: change it with `edit_extra_data()` |
| `created_at`, `updated_at` | When the first and the latest history entry were recorded, in UTC |
| `id` | The name a store saves the State under |

The size of the context is a question for the Agent: `agent.context_used(state)`. See
[Keep the context small](../guides/context-size.md).

## messages and history

<a id="history-and-context"></a>

|  | `history` | `messages` |
| --- | --- | --- |
| Contains | Everything that happened | What the model sees at the next `think` |
| Can shrink | No. Entries are only added | Yes, by `compact` and `clear_tool_results` |
| Use it to | Inspect, log, debug, save | Know what the model knows right now |

After a compaction, `messages` holds the first message and a summary. The history still holds every message, reply
and tool result.

## History

Every entry has `kind`, `content`, `turn` (the State's turn when it was recorded) and `at` (when, in UTC). There are 11
entry classes. Check `kind`, or use `isinstance` or `match`, to know what an entry's `content` holds.

| Class | `kind` | Recorded when |
| --- | --- | --- |
| `MessageEntry` | `user`, `notice` | `add_message` |
| `ModelRequestEntry` | `model_request` | `think` sends a request. `content` is the model name |
| `ModelReplyEntry` | `model_reply` | The model answered. `content` is the `Reply`, including `reply.model` |
| `ModelEventEntry` | `model_event` | The model adapter reported something, such as a retry or a fallback to another model |
| `ToolResultEntry` | `tool_result` | A tool call got its result, whichever way it ended. `outcome` says how |
| `ExchangeEntry` | `ask`, `human` | `agent.ask` or `agent.ask_human` got an answer. `content` is an `Exchange` |
| `ContextChangeEntry` | `context_change` | `State(messages=...)`, `compact`, `clear_tool_results`, `restore` |
| `RunStartEntry` | `run_start` | A run started. `content` is the [`AgentInfo`](#agentinfo) of the Agent |
| `StopEntry` | `stop` | The run was set to stop: `finish`, a loop's `until` or `limit`, a permission |
| `ExtraDataEntry` | `extra_data` | `State(extra_data=...)` or `edit_extra_data()` changed something |
| `ErrorEntry` | `error` | An exception was raised in `think`, `use_tools` or `run` |

A `ToolResultEntry` has an `outcome`: `done`, `error`, `input_error`, `aborted`, `interrupted`, `denied` or
`cancelled`. `entry.is_error` is true for every outcome except `done`. Every class is in the
[Data types API](../api/types.md#history-entries).

```python
from alpineagents import Message, State

state = State(messages=[Message.user("Find the bug in main.py")])
for entry in state.history:  # after a run, it also has replies and tool results
    if entry.kind == "model_reply":
        print(entry.content.tool_calls)
    elif entry.kind == "tool_result":
        print(entry.call.name, entry.outcome)
```

### Values are frozen to JSON

Values are frozen to their JSON form when they are recorded, on a live State as well as a loaded one. A tool call's
`args` and a recorded `ask` answer are read-only dicts and lists: they work with `json.dumps`, indexing and `==`, but
`args["x"] = 1` raises `TypeError`. Copy with `dict(args)` to change one. A tuple becomes a list. A dataclass or
Pydantic model, for example the `answer` of `finish(answer)`, becomes a dict, so `state.answer` is a dict even before
any save (`agent.ask(..., returns=Type)` still returns the object). A value that is not JSON raises `TypeError`.

## extra_data: your notepad

`state.extra_data` holds your own values. The model never sees them. It is read-only; change it inside `edit_extra_data()`:

```python
from alpineagents import State

state = State()
with state.edit_extra_data() as data:
    data.setdefault("notes", {})["bug"] = "b.py"
print(state.extra_data)
```

- The block gives you a plain dict that is a copy. When the block ends normally, the changes become one
  `ExtraDataEntry`. If the block raises, nothing changes. Two threads editing at once take turns.
- Values must be JSON. A set or another object raises `TypeError` that names the key. Use a list.
- Because it is history, `extra_data` is saved with the State, and `restore` takes it back too.

## finish ends the State

`state.finish(answer=None)` sets `stopped` to `StoppedByFinish(answer)` right away, so the loop stops before the next
turn. It sets `answer` if it is not `None`.

`finish` ends the State for good. After it, `think`, `use_tools`, `ask` and `run` raise `ValueError`. If a tool calls
`finish`, end the loop body without another `think`. Only `restore` can take it back. To refuse tool calls before they
run, give the Agent [permissions](../guides/permissions.md).

`compact(summary)` and `clear_tool_results()` shrink `messages` without a model request. `agent.compact(state)` asks
the model for the summary. See [Keep the context small](../guides/context-size.md).

## Pending calls

When a reply asks for tools, each call becomes a pending call in `state.pending_calls`. A call stops being pending
when `use_tools` records its result, when a [permission](../guides/permissions.md) refuses it or cancels it because it
stopped the turn, or when an exception ends the run and closes it.

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
| Stopped by an `until` function | Stops at once if the function is still true. To continue a conversation, add a message first |
| Stopped by `limit` | Runs up to `limit` more turns. The loop counts turns from zero on every call |
| Stopped by Ctrl+C or an exception | Continues. Pending calls were closed with a result such as `(interrupted by user)` |
| Ended with `finish()` | Raises `ValueError` |
| Loaded with `store.load(id)` | Continues where the saved history ends. See [Save and resume](../learn/save-resume.md) |

Switching the model between runs never warns. See
[Use other models](../guides/models.md#switch-models-in-one-conversation).

### AgentInfo

`RunStartEntry.content` is an `AgentInfo`: the Agent's `name`, `model` (`provider/name`), `system_sha256` (a hash of
the system prompt, not the text), `tools` (names) and `mcp_servers` (names). `ask` and `compact` record no
`ModelRequestEntry`.

### A State saved by another Agent

An Agent that resumes a saved State after a deploy may differ from the one that ran it. The run still works, but may
behave differently: the model gets an error result when it calls a tool that was removed.

So the first `run` of a State that a store loaded compares the `AgentInfo` of its last run with the Agent's. If they
differ, it warns with `ResumeWarning`, for example
`tools: removed search_web, added search_docs; model: anthropic/claude-sonnet-5 -> openai/gpt-5`. The system prompt is
reported only as changed. To refuse such a resume, make the warning an exception:

```python
import warnings

from alpineagents import ResumeWarning

warnings.filterwarnings("error", category=ResumeWarning)
```

`run` then raises `ResumeWarning` before the loop starts. Its `changes` attribute holds
`{key: (saved_value, current_value)}` for each difference.

## StateInfo

`store.list()` returns one `StateInfo` per saved State, without loading it: `id`, `first_message`, `created_at`,
`updated_at`, `turn`, `stopped` and `finished`.

## StateSnapshot: a value you can keep

`state.snapshot()` returns a `StateSnapshot`: the same fields as the properties above, and no methods that change
anything. It is frozen, and two snapshots are equal when their histories are equal. Each property of a State reads the
current snapshot on its own, so two reads can see two moments while a run goes on in another thread. One snapshot is
one moment.

`restore` and `fork` are between-turn operations. What they keep and what they refuse is in
[Go back or try another path](../guides/undo-and-fork.md).
