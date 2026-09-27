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

### Changed

- `print(state)` shows error results as `tool_result fetch_url (error): HTTP 404: ...` instead of their size.

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
