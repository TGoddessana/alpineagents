# Errors and interruptions

## Mistakes in your code

A wrong use of the library raises `TypeError` or `ValueError` as early as possible: when you create the Agent, apply
`@tool` or `@loop`, or call a method. The message says what is wrong, how to fix it, and gives an example:

```text
TypeError: @loop needs both until and limit
Fix: set both
Example:
    from alpineagents import waiting_for_user

    @loop(until=waiting_for_user, limit=50)
```

Each such error is listed under `Raises` in the [API reference](../api/agent.md) of the method that raises it.

## Errors from outside

Every exception below inherits from `AlpineAgentsError`.

| Exception | Raised when |
| --- | --- |
| `RateLimitError` | The provider returned 429 after the SDK's retries |
| `AuthError` | The API key is missing or rejected |
| `ContextTooLongError` | The request is larger than the model's context window. Add `compact_if_full` to the loop |
| `ProviderError` | Any other provider failure. The three above are subclasses of it |
| `OutputError` | `ask(returns=...)` did not get a valid answer after its retries, or `compact` got an empty summary |
| `NoHumanError` | `ask_human` was called on an Agent with `human=None` |
| `MCPConnectionError` | An MCP server could not connect, or the connection was lost |

For `ProviderError` and `MCPConnectionError`, the original exception is in `__cause__`.

## Nothing is swallowed

Exceptions from tools, from the loop body and from `until` functions propagate as they are. alpineagents does not
catch them or turn them into results, with the exceptions you choose: to let the model handle a failure, raise
`ToolError` (or `ToolInputError` for wrong arguments) in the tool, or name the exceptions in the tool's
`exception_handler`. See [Tool calls](tools.md#when-a-call-goes-wrong).

## The State after an exception

| The exception happens in | The State afterwards |
| --- | --- |
| `think` | `state.messages` is as it was before `think`, and `state.turn` does not change. History keeps the `ModelRequestEntry` and adds an `ErrorEntry` |
| `use_tools` | Results of the calls that finished are recorded. The other calls stay pending |
| Anything, and leaves `run` | The error is recorded in `state.history` as an `ErrorEntry`. Pending calls are closed with a `ToolResultEntry` such as `(aborted: TimeoutError)`, with outcome `aborted` |

After `run` raises, `state.stopped` is `None`, and you can call `run` with the same State again:

```python
import time

from alpineagents import Agent, Message, RateLimitError, State

agent = Agent(model="claude-sonnet-5")
state = State(messages=[Message.user("Find the bug in main.py")])
try:
    agent.run(state)
except RateLimitError:
    time.sleep(60)
    agent.run(state)
```

With a [store](../guides/production.md), the State is saved after an exception too. If that save also fails, the
original exception is raised with a note about the failed save, so `except RateLimitError:` still catches it.

If the loop body catches an exception from `use_tools`, the calls that did not finish stay pending, and the next
`think` raises `ValueError`. Call `use_tools` again first (calls that already have a result do not run again), or let
the exception leave the run, which closes them.

## Ctrl+C and cancellation

- Ctrl+C during `run` closes pending calls with the result `(interrupted by user)`, outcome `interrupted`.
  `run(state)` continues from there.
- A sync tool that is still running keeps running in its thread. If it finishes later, the model gets its result as a
  notice at the next `think`.
- In async code, cancelling the task follows the same rules. See [Use async](../guides/async.md).
- A process that is killed (SIGKILL, or SIGTERM without a handler) stops at once, without closing calls. With a
  store, `store.load(id)` closes the calls that were running with a result telling the model they may or may not
  have run (outcome `aborted`), and a request that never got its reply with an `ErrorEntry`. See
  [Survive crashes and restarts](../guides/production.md#if-the-process-stops).
