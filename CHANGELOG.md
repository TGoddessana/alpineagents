# Changelog

All notable changes to alpineagents are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/)
(while the version is 0.x, a minor release may change the API).

## [Unreleased]

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
