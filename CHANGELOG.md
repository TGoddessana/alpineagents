# Changelog

All notable changes to alpineagents are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/)
(while the version is 0.x, a minor release may change the API).

## [Unreleased]

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
