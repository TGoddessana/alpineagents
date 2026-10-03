# Changelog

All notable changes to alpineagents are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/)
(while the version is 0.x, a minor release may change the API).

## [Unreleased]

This release rebuilds the State on its history, and it is a breaking one: nothing removed keeps an alias. Read
**Breaking changes** and the migration table first. The short version: a State is made with
`State(messages=[Message.user("...")])`, changed only through its methods, and read through properties that are all
computed from `state.history`.

### Breaking changes

- **The State constructor takes keywords only.** `State("text")` is a `TypeError` that shows the new form:

  ```python
  # before
  state = State("Fix calc.py", id="bug-1")

  # now
  state = State(messages=[Message.user("Fix calc.py")], id="bug-1", extra_data={"repo": "api"})
  ```

  `messages=` takes `Message` objects (a `str` is a `TypeError`), `extra_data=` takes JSON values, and `history=`
  takes entries to rebuild a State from. `history=` cannot be combined with `messages=` or `extra_data=`. A State may
  start empty; `agent.think` on a State with no messages raises `ValueError`. `agent.run("text")` still works.
  `state.task` is removed: read the first message from `state.messages`, or `StateInfo.first_message` in a listing.

- **`state.data` is `state.extra_data`, and it is read-only.** Change it with `edit_extra_data()`, which records
  the change in the history:

  ```python
  # before
  with state.lock:
      state.data["calls"] = state.data.get("calls", 0) + 1

  # now
  with state.edit_extra_data() as data:
      data["calls"] = data.get("calls", 0) + 1
  ```

  `state.extra_data["x"] = 1` raises `TypeError` that points to `edit_extra_data()`. The block holds a lock of its
  own, so two threads' edits take turns and no update is lost. `state.lock` is removed: no State method needs you
  to hold a lock.

- **Renamed or replaced State members**, with no aliases (see the table below): `state.context` is
  `state.messages`; `add_user_message(text)` and `add_notice(text)` are `add_message(Message.user(text))` and
  `add_message(Message.notice(text))`; `start_from(summary)` is `compact(summary)`; `wants_tools()` is
  `state.pending_calls`; `is_finished()` is `state.finished`; `State.is_answered` is a function you write;
  `state.context_tokens` and `state.context_used` are now `agent.context_tokens(state)` and
  `agent.context_used(state)`, because they need the Agent's model and tools.

- **`State.is_answered` is gone.** An `until` function is now always one you write, or the default loop's. The
  default loops stop on a private function that is still named `is_answered`, so `StoppedByUntil("is_answered")` and
  `stopped by is_answered` are unchanged:

  ```python
  # before
  @loop(until=State.is_answered, limit=30)

  # now
  def waiting_for_user(state: State) -> bool:
      if state.pending_calls or not state.messages:
          return False
      last = state.messages[-1]
      return last.role == "assistant" and not last.tool_calls

  @loop(until=waiting_for_user, limit=30)
  ```

- **`agent.save(state)` and `agent.asave(state)` are `store.save(state)` and `await store.asave(state)`.** The
  Agent still saves at the same points when it has `store=`.

- **The owner rule is gone.** A State no longer belongs to the first Agent that thought on it: any Agent can run,
  think, use tools on and compact any State, so `agent.copy(model=...)` can carry on a State another model started.
  The `ValueError` for "a different Agent" is removed. What remains is one run at a time: a State that another
  `run` or `arun` is running right now raises `ValueError` (wait for that run to end).

- **History entries were renamed and merged.** There are 11 classes now, all frozen:

  | 0.4 | 0.5 |
  | --- | --- |
  | `ReplyEntry`, kind `"reply"` | `ModelReplyEntry`, kind `"model_reply"` |
  | `NotRunEntry`, kinds `"denied"` and `"cancelled"` | `ToolResultEntry`, kind `"tool_result"`, with `outcome` `DENIED` or `CANCELLED` |
  | `ToolResultEntry(is_error=...)` | `ToolResultEntry(outcome=...)`. `is_error` is a read-only property: `outcome != DONE` |
  | `ContextChange` kinds `"start_from"` and `"rollback"` | `"compact"` (a `compact` call, or the model's), and no entry: a failed `think` is taken back by an `ErrorEntry` |

  Code that checks `entry.kind in ("tool_result", "denied", "cancelled")` checks `entry.kind == "tool_result"` and
  reads `entry.outcome`. `ToolResultEntry.outcome` is a required keyword.

  `StoppedByFinish` has a new field `answer` that takes part in `==`, so `state.stopped == StoppedByFinish()` is
  `False` after `finish("x")`. Use `isinstance(state.stopped, StoppedByFinish)` or `state.finished` instead.

- **`Block` includes `Image`** (`TextBlock | Image | ToolCall | ToolResultBlock | RawBlock`), for images in the
  user's messages. A `Model` you wrote, or any code that walks `Message.content` with an exhaustive `match` or an
  `isinstance` chain, must handle `Image` blocks in user messages (`Message.user("", image).text` is `""`).
  See [Write a Model](https://tgoddessana.github.io/alpineagents/guides/models/#write-a-model).

- **`state.finish(answer)` and `ask` answers are stored as JSON.** A dataclass or a Pydantic model passed to
  `finish()` comes back from `state.answer` and `agent.run` as a read-only dict, not the object (rebuild it with
  `Review(**answer)`). A value that is not JSON (a set, `Path`, `datetime`, `Enum`, `bytes`) raises `TypeError` from
  `finish()`. `agent.ask(..., returns=Review)` still returns the object.

- **`SavedState` is `StateInfo`**, what `store.list()` returns. It has `first_message` where `SavedState` had `task`,
  and `created_at` and `updated_at` can be `None` for a State saved with no history. Implementers of a Store build
  one with `StateInfo.from_info(state_id, info)`, which replaces `SavedState.from_snapshot`.

- **Stores save in format version 4, and 0.5 cannot load what 0.4 saved.** `store.load` raises `ValueError` ("saved
  by an older alpineagents (format 3, 0.4.x); 0.5 cannot load it"), and there is no converter. A State saved by a
  newer version is refused the same way. `FileStore` keeps `log.jsonl` and replaces `snapshot.json` with
  `info.json`; `store.list()` skips the folders of 0.4 States. Finish or export what you need with 0.4 first.

- **`Store.write` takes `info`, not a snapshot.** A Store you wrote changes `write(state_id, entries, snapshot, *,
  create=False)` to `write(state_id, entries, info, *, create=False)`, and `Record(entries, snapshot)` to
  `Record(entries, info)`. `info` is a small JSON dict with the fields of `StateInfo` and `"v": 4`, to be stored as
  is and given back in `Record.info`. It has no messages and no history. `load` no longer builds a State from a
  snapshot and the entries after it: it replays all the entries (see Changed).

- **`state.extra_data`, `ToolCall.args`, `RawBlock.data`, `ModelEvent.data`, `Exchange.answer` and
  the `finish` answer are read-only** (`FrozenDict` and `FrozenList`, subclasses of `dict` and `list`). Reading,
  `json.dumps`, `**args` and `==` with plain values work as before; assigning or calling `append`, `pop`, `update`
  and the like raises `TypeError`. A tool still gets its own plain, editable copy of its arguments. Code that edits
  `call.args` (in a permission or a Reporter) copies first: `dict(call.args)`.

- **`ResumeWarning` fires only on the first run after `store.load`**, and compares with the last `RunStartEntry`.
  Switching the model of a State in memory never warns. The values in `ResumeWarning.changes` are tuples.

- **`state.created_at` and `state.updated_at` can be `None`**, for a State with no history.

### Added

- **`StateSnapshot`**, the frozen value of a State. `state.snapshot()` returns it, and every property of the State
  (`messages`, `history`, `pending_calls`, `turn`, `usage`, `extra_data`, `finished`, `answer`, `stopped`,
  `created_at`, `updated_at`) reads the current one. Take a snapshot when several values must belong together. See
  the [State](https://tgoddessana.github.io/alpineagents/concepts/state/) concept page.
  - Every value of a snapshot is computed from its `history`, so `State(history=state.history).snapshot() ==
    state.snapshot()` holds at any moment, also in the middle of a turn.
  - `state.fork()` makes a State with the same history, a new id, and no store or running Agent.
    `state.restore(snapshot)` goes back to a snapshot of the same State: `messages`, `turn`, `extra_data`, `finished`,
    `answer` and `stopped` go back, `usage` stays (the tokens were spent), and `history` keeps everything and gains
    a `context_change` entry of kind `"restore"`.
- **`state.edit_extra_data()`**, described above. A block that changes nothing records nothing; one that raises
  changes nothing. A value that is not JSON raises `TypeError` naming the key.
- **`Message.notice(text)` and `Message.assistant(text)`**, next to `Message.user(text)`. `Message.notice` adds the
  `"[notice] "` prefix if it is missing, and `message.is_notice` tells whether a message is one. `add_message`
  takes user messages and notices; to import a conversation that has assistant messages, use `State(messages=[...])`.
- **Images in the user's messages.** `Message.user("What is wrong in this screenshot?", Image.from_path("shot.png"))`.
  `Anthropic` sends them as image blocks and `OpenAICompatible` as `image_url` parts with a data URL, and the stores
  save them. They count 1,600 tokens each in `agent.context_tokens(state)`.
- **Model switching.** Any Agent continues any State: `agent.copy(model="gpt-...").run(state)`. Each request records
  `ModelRequestEntry` with the model it was sent to, so the history shows which model answered which turn.
- **`store.save(state)` and `await store.asave(state)`**, to save right away what the Agent's save points would save
  later, such as the person's next message in a chat loop. See the
  [Save and resume](https://tgoddessana.github.io/alpineagents/guides/resume/) guide.
- **`StateInfo`** (what `store.list()` returns) and **`AgentInfo`** (name, model as `provider/name`, system prompt
  hash, tool names, MCP server names), exported from `alpineagents`.
- **New history entry kinds**, all exported: `ModelRequestEntry` (`"model_request"`, recorded when `think` sends a
  request), `RunStartEntry` (`"run_start"`, holds the `AgentInfo` of the Agent that started the run),
  `StopEntry` (`"stop"`, holds why the run is set to stop: `finish`, a permission, `until` or `limit`) and
  `ExtraDataEntry` (`"extra_data"`, the changed keys and the removed ones). `ModelReplyEntry` replaces `ReplyEntry`.
  Every public value of a State is a fold of these entries.
- **`ToolResultEntry.outcome`**, a `ToolOutcomeKind`, on every tool result: `DONE`, `ERROR`, `INPUT_ERROR`,
  `ABORTED`, `INTERRUPTED`, `DENIED` or `CANCELLED`. A denied or cancelled call is a `tool_result` entry now.
- `ExchangeEntry.usage`, the tokens an `ask` used (`None` for a question to a human), and `ContextChange.usage`, the
  tokens of a compaction the model wrote. `state.usage` is the sum of replies, asks and compactions, and a loaded
  State has the same total as the State that was saved. `ContextChange` also has `kept`, `cleared`, `messages` and
  `restored_to` for the kinds above.
- **`agent.context_tokens(state)` and `agent.context_used(state)`**: the estimate of what the next request holds,
  and the fraction of the model's context window it fills, with this Agent's system prompt and tool definitions.
  `compact_if_full` uses them.

### Changed

- **History is the only source of truth.** A State holds one frozen `StateSnapshot`, and every change builds one
  history entry and folds it into a new snapshot. There is no second path for "recording" and for "replaying": a
  State rebuilt from its history (`State(history=...)`, `store.load`, `state.fork()`) equals the one that made it.
  `state.history` is a tuple that only grows.
- **`store.load` replays every saved entry** through the same fold, instead of starting from a snapshot and
  replaying the entries after it. A State that was saved in the middle of a turn is closed with entries that get
  saved with the next save: a request that was waiting on the model gets an `ErrorEntry` ("process stopped while
  waiting for the model", which takes the request back), and each tool call without a saved result gets a
  `ToolResultEntry` with outcome `ABORTED` saying it may or may not have run. The tool is not run again. The
  `UNSAVED_STEPS` notice and the case "a tail that cannot be replayed" no longer exist.
- **A failed `think` is taken back by its `ErrorEntry`.** The request is on record (`ModelRequestEntry`), and an error
  before the reply returns `messages` and `turn` to what they were. An exception after the reply was recorded, such
  as one raised by `Reporter.on_think_end`, no longer undoes the reply. The `"rollback"` context change and its
  Reporter notification are gone.
- **`run` records `RunStartEntry` at its start**, which clears `state.stopped`. A loop's `until` or `limit` stop and
  `finish()` are `StopEntry` entries. `finish()` called twice: the last answer given wins, and `finish()` with no
  answer keeps an earlier one.
- **Values in history are frozen**, so an entry never changes after it is recorded: `entry.content` of a
  tool call's arguments, an `ask` answer, a `finish` answer, and `extra_data` values.
- The Anthropic and OpenAI adapters send plain copies of `ToolCall.args` and `RawBlock.data` to their SDKs.
- Context changes reach the Reporter for `compact`, `clear_tool_results` and `restore` (not for the import of
  `State(messages=...)`), on the Reporter of the Agent that last ran or thought on the State. The Terminal prints
  `context restored` for a restore.
- `Message.user("", image)` holds the image only: an empty text block is not sent.

### Fixed

- `FileStore` no longer writes a duplicate entry when two `FileStore` objects for the same folder save the same
  State.

### Migrating from 0.4

| 0.4 | 0.5 |
| --- | --- |
| `State("task", id="x")` | `State(messages=[Message.user("task")], id="x")` |
| `state.task` | the first message in `state.messages`, or `StateInfo.first_message` |
| `state.context` | `state.messages` |
| `state.data["k"] = v` | `with state.edit_extra_data() as data: data["k"] = v` |
| `state.data` | `state.extra_data` |
| `with state.lock: ...` | `state.edit_extra_data()` (atomic), or `state.snapshot()` for a consistent read |
| `state.add_user_message(text)` | `state.add_message(Message.user(text))` |
| `state.add_notice(text)` | `state.add_message(Message.notice(text))` |
| `state.start_from(summary)` | `state.compact(summary)` |
| `state.wants_tools()` | `state.pending_calls` (a tuple: truthy when there are calls) |
| `state.is_finished()` | `state.finished` |
| `@loop(until=State.is_answered, ...)` | `@loop(until=waiting_for_user, ...)` with your own function |
| `state.context_tokens`, `state.context_used` | `agent.context_tokens(state)`, `agent.context_used(state)` |
| `agent.save(state)`, `agent.asave(state)` | `store.save(state)`, `await store.asave(state)` |
| `SavedState`, `SavedState.task` | `StateInfo`, `StateInfo.first_message` |
| `SavedState.from_snapshot(...)` | `StateInfo.from_info(state_id, info)` |
| `Store.write(id, entries, snapshot)`, `Record(entries, snapshot)` | `Store.write(id, entries, info)`, `Record(entries, info)` |
| `ReplyEntry`, `entry.kind == "reply"` | `ModelReplyEntry`, `entry.kind == "model_reply"` |
| `NotRunEntry`, kinds `"denied"`, `"cancelled"` | `ToolResultEntry` with `outcome` `DENIED`, `CANCELLED` |
| `entry.is_error` on a tool result | unchanged to read; it is `entry.outcome != ToolOutcomeKind.DONE` |
| `ContextChange` kind `"start_from"` | `"compact"` |
| `ContextChange` kind `"rollback"` | none: a failed `think` records an `ErrorEntry` |
| a second Agent thinking on a State raised `ValueError` | allowed: `agent.copy(model=...).run(state)` |
| `state.stopped == StoppedByFinish()` | `isinstance(state.stopped, StoppedByFinish)`, or `state.finished` |
| `state.finish(Review(...))`, then `state.answer` is a `Review` | `state.answer` is a read-only dict: `Review(**state.answer)`. Non-JSON answers raise `TypeError` |
| a custom `Model` read `message.text` and never saw an image | user messages can hold `Image` blocks: convert them or raise |
| States saved by 0.4 | not loadable by 0.5 (format 3, no converter) |

## [0.4.0] - 2026-09-30

### Added

- **Permissions.** `Agent(permissions=[...])` decides, before any tool of a turn runs, whether each call may run. A
  refused call does not run, and the model gets the reason as its error result. See the
  [Ask before a tool runs](https://tgoddessana.github.io/alpineagents/guides/approval/) guide.
  - The names live in `alpineagents.permissions`. A permission subclasses `DenyPermission` (may only refuse),
    `AllowPermission` (may only allow) or `DecidePermission` (may do either), and implements `check(state, call,
    tool)`, `async acheck(...)`, or both. It returns `Allowed()`, `Denied(reason)`, or `None` to let the next one
    decide.
  - Every `DenyPermission` runs first, wherever it is in the list. Then the others in list order, and the first
    verdict decides.
  - With `permissions=`, some permission must allow a call. A call nobody allows is refused with a
    `PermissionWarning`, so `permissions=[]` refuses every call. Leave `permissions=` out, or pass `None`, to run every
    call without checks: `agent.copy(permissions=None)`.
  - `Denied(reason, stop=True)` also cancels the other calls of the turn (none of them runs) and stops the run before
    its next turn. The State is not finished: add the person's next message and run again. See
    [When the person says no](https://tgoddessana.github.io/alpineagents/guides/approval/#when-the-person-says-no).
  - Raising `ToolError(message)` in a permission refuses the call with that message. Any other exception stops the
    run, as one from a tool does. A failure never counts as allowed.
  - `agent.permissions`: the permissions, as a tuple. A run fails before the first model call when a permission could
    not work: `NoHumanError` for a `DecideByHuman()` when the Agent has no human, and, in `run`, `TypeError` for a
    permission that only implements `acheck`.
- Built-in permissions in `alpineagents.permissions`:
  - `DenyByName(["delete_*"], reason=None)` and `AllowByName(["read_file", "github__get_*"])` match tool names with
    globs. MCP tools are named `{server}__{tool}`, so `"github__*"` matches a whole server.
  - `AllowByReadOnly(trust_mcp=False)` allows a call the tool says is read-only. It does not believe MCP servers'
    annotations unless you pass `trust_mcp=True`.
  - `DecideByHuman(human=None)` asks the person `Run write_file(path="README.md", ...)?`. `yes` allows the call; `no`
    refuses it with `stop=True`. The question and the answer are recorded in `state.history`, like `agent.ask_human`.
    Override `question(call, tool)` to ask something else.
  - `AllowByDefault()` allows every call. Put it last to allow what nobody refused.
- `PermissionWarning`, for a call that no permission allowed.

- **Hints for one call.** `@tool(hints_for=fn)` tells what one call does: `fn(args)` returns `Hints(...)`, or `None`
  when unsure, which keeps the tool's own hints. A shell tool whose `ls` changes nothing can say so while the tool
  stays not read-only. See
  [Hints for one call](https://tgoddessana.github.io/alpineagents/concepts/tools/#hints-for-one-call).
  - `Hints(read_only=True, open_world=False)` takes the four hints with the rules of `@tool`: a hint left out assumes
    the worst. Subclass it to carry more facts about a call.
  - `tool.hints_for(args)` returns the hints of one call. A `Tool` subclass can override it. `AllowByReadOnly` reads
    it.
  - `tool.copy(hints_for=...)` adds or replaces the function, and `hints_for=None` removes it.

- **Why a run stopped, as values.** `state.stopped` is `StoppedByUntil(name)`, `StoppedByLimit(turns)`,
  `StoppedByFinish()`, `StoppedByPermission(call, permission)`, or `None`. `str()` of each gives a short text such as
  `stopped by is_answered` or `stopped at limit 30`. `alpineagents.types.Stopped` is their union. See the
  [Stop conditions](https://tgoddessana.github.io/alpineagents/guides/stop-conditions/) guide.
  - `state.finish()` and a permission's `stop=True` set `state.stopped` right away. A loop written without `@loop`
    stops on them by checking `state.stopped is None` before each turn.
- `ToolOutcomeKind`, the kinds of `ToolOutcome.kind` as a `StrEnum`, with the new `CANCELLED`: a call that did not
  run because a permission stopped its turn. Its history entry has the new kind `"cancelled"`, and the Terminal shows
  `cancelled {name}: ...`.
- `ToolOutcome.decided_by`: the `repr()` of the permission that decided a `DENIED` or `CANCELLED` call, for a Reporter
  that logs who refused what.

- **Options with a model string.** `resolve_model(model, **options)` passes keyword options to the Model it picks:
  `resolve_model("anthropic/claude-sonnet-5", max_tokens=16_000)` is `Anthropic("claude-sonnet-5",
  max_tokens=16_000)`. See
  [Options with a model string](https://tgoddessana.github.io/alpineagents/guides/models/#options-with-a-model-string).
  - With `base_url=`, the model is always `OpenAICompatible` and the whole string is its name, not split. Routers
    such as OpenRouter name models `"anthropic/claude-sonnet-5"`. For a proxy that speaks the Anthropic API, create
    `Anthropic(..., base_url=...)` directly.
  - `"ollama/<name>"` still removes the prefix, and an explicit `base_url=` replaces the local Ollama URL.
  - An option the picked Model does not take raises `TypeError` that lists the ones it takes. A Model object with
    options raises `TypeError`.
- The [Models that stall or drop streams](https://tgoddessana.github.io/alpineagents/guides/unreliable-models/)
  guide: retry a stream that dropped mid-reply by wrapping a Model, hand back a reply with no tool call with a notice,
  and stop a model that repeats itself with an `until` function.

### Changed

- **History entries have a class per kind.** `state.history` holds `MessageEntry` (`user`, `notice`), `ReplyEntry`,
  `ToolResultEntry`, `NotRunEntry` (`denied`, `cancelled`), `ExchangeEntry` (`ask`, `human`), `ContextChangeEntry`,
  `ModelEventEntry` and `ErrorEntry`. `HistoryEntry` is now their union. The `kind` values and `entry.content` do not
  change, and checking `kind` now tells a type checker what `content` is: after `if entry.kind == "reply":`,
  `entry.content` is a `Reply`. `match entry: case ToolResultEntry(call=call): ...` works too.
  - Fields are only on the classes where they mean something. `call` is on `ToolResultEntry`, `NotRunEntry` (never
    `None` there) and `ErrorEntry`; `late` only on `ToolResultEntry`; `is_error` on `ToolResultEntry` and
    `NotRunEntry`; `error` on those two and `ErrorEntry`. Reading `entry.call` without checking the kind first raises
    `AttributeError` on other entries: filter with `entry.kind in ("tool_result", "denied", "cancelled")`.
  - `HistoryEntry(...)` cannot be called any more. Make the class you need, with keywords:
    `MessageEntry(kind="user", content="Hi", turn=0)`.
  - `HistoryEntry.substate` is removed. It was reserved for subagents and always `None`.
  - Stores save entries in the same format, so saved States load as before.

- `Anthropic` raises `ProviderError` when the connection drops or a read times out while a reply is streaming. It
  raised the HTTP library's own error before, which `except ProviderError` did not catch. The original error is in
  `__cause__`. It is not retried: the SDK retries only a request whose reply has not started.
- `ToolOutcome.kind` is a `ToolOutcomeKind`. It is a `StrEnum`, so `outcome.kind == "denied"` keeps working, and
  `ToolOutcome("denied")` still takes a string.
- `print(state)` and the Terminal's last line say `stopped by ...`: `done: stopped by is_answered (2 turns)` instead
  of `done: is_answered (2 turns)`, and `done: stopped at limit 30 (30 turns)` instead of `done: limit(30) (30 turns)`.
  The Terminal's line for a reached limit (`done: reached limit(30), ...`) does not change.
- An `until` function can be named `finish` or `limit`. These names were reserved, because `stopped_by` was a string
  and could not tell them apart from the built-in reasons.
- The [Ask before a tool runs](https://tgoddessana.github.io/alpineagents/guides/approval/) guide uses permissions
  instead of a loop that calls `state.deny`.
- Stores save in format version 3. Earlier versions of alpineagents cannot load States saved from now on; this
  version still loads States they saved, and turns their `stopped_by` into the new values.
- `store.load` refuses a snapshot that misses keys with `ValueError` ("is damaged: its snapshot has no ...") instead
  of failing with a `KeyError` halfway through the load.

### Removed

- `state.deny(call, reason)`. Refuse calls with permissions:

  ```python
  # before
  @loop(until=State.is_answered, limit=30)
  def careful(agent, state):
      agent.think(state)
      for call in state.pending_calls:
          found = agent.tool_map.get(call.name)
          if found is not None and not found.read_only:
              if agent.ask_human(state, f"Run {call.name}?", returns=Literal["yes", "no"]) == "no":
                  state.deny(call, "The user declined this call")
      if state.wants_tools():
          agent.use_tools(state)

  agent = Agent(model=..., tools=[...], loop=careful)

  # now
  from alpineagents.permissions import AllowByReadOnly, DecideByHuman

  agent = Agent(model=..., tools=[...], permissions=[AllowByReadOnly(), DecideByHuman()])
  ```

  A rule of your own is a `DenyPermission`, `AllowPermission` or `DecidePermission` subclass whose `check` returns
  `Denied(reason)` where the loop called `state.deny`.

- `state.stopped_by` and `state.stopped_limit`. Use `state.stopped`:

  ```python
  # before
  if state.stopped_by == "limit":
      print(f"gave up after {state.stopped_limit} turns")
  elif state.stopped_by == "spent_too_much":
      ...

  # now
  from alpineagents import StoppedByLimit, StoppedByUntil

  if isinstance(state.stopped, StoppedByLimit):
      print(f"gave up after {state.stopped.turns} turns")
  elif state.stopped == StoppedByUntil("spent_too_much"):
      ...
  ```

  `"finish"` is `StoppedByFinish()`. A loop written without `@loop` checks `state.stopped is None` before each turn,
  so it stops on `finish()` and on a permission's `stop=True`.

- `SavedState.stopped_by`, in `store.list()`. Use `SavedState.stopped`, which holds the same values as
  `state.stopped`:

  ```python
  # before
  unfinished = [s.id for s in store.list() if s.stopped_by == "limit"]

  # now
  unfinished = [s.id for s in store.list() if isinstance(s.stopped, StoppedByLimit)]
  ```

- `Model._lookup_price`, an extension point that always returned `None`. `usage.cost` comes only from the Model's
  `price`. A subclass that overrode `_lookup_price` passes the price instead:

  ```python
  # before
  class MyModel(OpenAICompatible):
      def _lookup_price(self):
          return Price(input=3.0, output=15.0)

  # now
  model = OpenAICompatible("my-model", price=Price(input=3.0, output=15.0))
  ```

- The planned `alpineagents[prices]` extra (cost from genai-prices) is dropped, and the README no longer mentions it.
  Pass `price=Price(...)` to the Model, as before.

### Fixed

- Type checkers see `FileStore(...)`, `DenyByName(...)`, `Terminal()` and your own `Store`, `Permission` and `Human`
  subclasses as their own class. They saw the base class before (`Store`, `Permission`, `Human`), so methods of the
  subclass were unknown to them.

## [0.3.0] - 2026-09-28

### Added

- **Error results from tools.** A tool can report a failure the model should see and handle, such as a missing file
  or an HTTP 404, without stopping the run. Any other exception from a tool still stops the run. See the
  [Let the model handle tool failures](https://tgoddessana.github.io/alpineagents/guides/tool-failures/) guide.
  - `ToolError("message")`: raise it in a tool. The model gets the message as the call's error result.
  - `@tool(exception_handler=...)`: a function that takes one exception and returns the message for the model. The
    type hint of its parameter names the exceptions it takes (`httpx.HTTPError`,
    `FileNotFoundError | PermissionError`), so one handler serves every tool that raises them.
  - `Tool.copy(...)`: a tool with some `@tool` options changed, for example `fetch_url.copy(exception_handler=None)`
    for an Agent where every failure should stop the run.
  - The `tool_result` history entry of such a call has `is_error=True`, and its `error` holds the `ToolError`
    (with the handled exception in `__cause__`).

- **Tool hints.** `@tool(read_only=..., destructive=..., idempotent=..., open_world=...)` says what a tool does, with
  the meaning of MCP tool annotations. The model does not see them; your code reads them, for example to ask before
  calls that are not read-only. A hint left out assumes the worst. MCP tools get them from the server's annotations.
- `agent.tool_map`: every tool the model can call, by name (`@tool` functions, `@tool` methods of objects, and MCP
  tools while connected). Look up a call's tool with `agent.tool_map.get(call.name)`.
- `MCPTool`, the type of an MCP server's tool in `agent.tool_map`.
- `think(tools=...)` accepts the values of `agent.tool_map`, including MCP tools, so
  `[t for t in agent.tool_map.values() if t.read_only]` works.

- **Tools that are not functions.** Subclass `Tool`, call `super().__init__(name=..., description=...,
  input_schema=..., ...)` and implement `run(args, state)` (or `async def run`), for tools whose name and schema come
  as data, such as rows in a database or an OpenAPI spec. `run` returns the result as a `@tool` function does, and
  signals with `ToolInputError` (wrong arguments) and `ToolError` (a failure the model should handle).
- `ToolInputError` is public: raise it from a tool for arguments it cannot use. The model gets `(input error: ...)`.

- **Images from tools.** A tool can `return Image(png_bytes)`, or a list of text and images such as
  `["Screenshot of the page", Image.from_path("shot.png")]`, and the model sees the image. PNG, JPEG, GIF and WebP.
  See [Images](https://tgoddessana.github.io/alpineagents/concepts/tools/#images).
  - `Anthropic` sends images in the tool result. `OpenAICompatible` sends them in a user message right after the
    tool messages, between `<tool_result>` tags, since Chat Completions tool messages take only text.
  - MCP servers' image content reaches the model as images instead of a note that images are not supported.
  - Stores save images (base64 in the JSON), and `clear_tool_results` clears them.
- `result_text(result)` in `alpineagents.types`: a tool result as one string, for display.

### Changed

- `Tool` is now the base class of every tool. `@tool` makes a `FunctionTool` (a `Tool` subclass), and an MCP server's
  tool is an `MCPTool` (also a `Tool` subclass). Code that created a tool with `Tool(fn)` uses `FunctionTool(fn)`;
  `isinstance(t, Tool)` is true for every tool.
- The [Ask before a tool runs](https://tgoddessana.github.io/alpineagents/guides/approval/) guide asks before calls
  whose tool is not `read_only`, instead of before tools in a list of names.
- The quick start and the other examples raise `ToolError` for a missing file instead of returning a string, which
  the model would have taken as an ordinary result. The [Testing](https://tgoddessana.github.io/alpineagents/guides/testing/)
  guide shows how to test a tool that fails.
- `print(state)` shows error results as `tool_result fetch_url (error): HTTP 404: ...` instead of their size.
- `ToolResultBlock.content`, the `content` of a `tool_result` history entry and the `result` of `Reporter.on_tool_end`
  are a tuple of `TextBlock` and `Image` when the tool returned an image; they stay a `str` otherwise.
- Stores save in format version 2. Earlier versions of alpineagents cannot load States saved from now on; this
  version still loads States they saved.

### Fixed

- `OpenAICompatible` now starts the text of an error result with `Error: `. Its tool message has no error flag, so the
  model could not tell a denied, interrupted or failed call from an ordinary result.

## [0.2.0] - 2026-09-27

### Added

- **Save and resume.** `Agent(store=...)` saves each State as the run goes, and `store.load(id)` rebuilds it in
  another process, for example after a crash, a deploy, or when a person comes back to a conversation. See the
  [Save and resume](https://tgoddessana.github.io/alpineagents/guides/resume/) guide.
  - `Store`, the role for keeping States outside the process. Implement `write` and `read` (and optionally `list`,
    `delete`), or their async versions.
  - `FileStore(path)`: one folder per State, with a history log and a snapshot. Files are readable only by their
    owner and flushed to disk on each write.
  - `store.load(id)`, `store.list()` (returns `SavedState`), `store.delete(id)`, and their async versions.
  - `agent.save(state)` / `await agent.asave(state)` save changes made outside the Agent's steps, such as the
    person's next message in a chat loop.
  - Each tool result is saved as soon as it is recorded, so a long or expensive tool's result is kept even if the
    process stops while other tools of the same turn still run.
  - A tool call that was running when the process stopped is closed, on load, with a result telling the model it
    may or may not have run. The tool is not run again.
  - If saving fails while the run is already failing, the original exception is raised with a note about the
    failed save, so `except` clauses keep working.
- `State(task, id=...)` and `state.id`: the name a store saves the State under. Random when left out.
- `ResumeWarning`: the first `run` of a loaded State warns when its Agent differs from the one that saved it (name,
  model, system prompt, tools, MCP servers). `.changes` holds the differences.
- `Reply.model`: the model that actually answered, as the provider reported it (it can differ from the requested
  name with aliases, routers and fallbacks).
- `HistoryEntry.at`: when each history entry was recorded, in UTC. `State.created_at` and `State.updated_at`.
- `HistoryEntry.is_error`: whether a `tool_result` or `denied` entry is an error result for the model.

### Changed

- The approval guide example keeps approved tool names in a list instead of a set, so it also works with a store.

### Removed

- The `state.save(path)` and `State.load(path)` placeholders, which raised `NotImplementedError`. Use a `Store`.

## [0.1.0] - 2026-09-26

First release: `Agent`, `State`, loops as plain functions (`@loop`, `default_loop`), `@tool`, MCP servers,
Anthropic and OpenAI-compatible model adapters, `Reporter`/`Human`/`Terminal`, async API, and `FakeModel`/`FakeHuman`
for tests.
