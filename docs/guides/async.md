# Async

Use the async versions in async code, such as a web server. They have the same behavior and an `a` prefix.

| Sync | Async |
| --- | --- |
| `agent.run` | `await agent.arun` |
| `agent.think` | `await agent.athink` |
| `agent.use_tools` | `await agent.ause_tools` |
| `agent.compact` | `await agent.acompact` |
| `agent.ask` | `await agent.aask` |
| `agent.ask_human` | `await agent.aask_human` |
| `compact_if_full` | `await acompact_if_full` |
| `CompactIfFull(...)(agent, state)` | `await CompactIfFull(...).acall(agent, state)` |
| `default_loop` | `adefault_loop` |

State methods have no async versions. They do not wait on anything.

## Example

```python
--8<-- "docs_src/async_agent.py"
```

1. `@loop` on an `async def` body makes an async loop. `until` functions stay plain functions.
2. The body awaits the async actions.
3. `agent.arun` runs the loop. An Agent without `loop=` uses `adefault_loop`.

## Rules

- `arun` needs an async loop or the default loop. `run` needs a sync loop. The wrong pair raises `TypeError`.
- `async def` tools run as tasks on your event loop. Other tools run on worker threads. Calls to `parallel=True`
  tools still run at the same time.
- A sync action such as `agent.think` inside an async loop raises `TypeError`, because it would block the event loop.
  The message names the async action to use.
- `compact_if_full` in an async loop raises only when it first compacts, which can be many turns in. Use
  `acompact_if_full` from the start.
- Cancelling the task follows the rules of Ctrl+C. Examples are a client disconnect or `asyncio.timeout`.
- On cancel, the model request is rolled back. Pending calls are closed with `(interrupted by user)`. `async def`
  tools are cancelled.
- `Anthropic` and `OpenAICompatible` use the providers' async clients, so cancelling also closes the HTTP request.

## Related

- [Errors and interruptions](../concepts/errors.md#ctrlc-and-cancellation)
- [Progress and questions](progress.md#write-a-human): a Human with `async def aask`
