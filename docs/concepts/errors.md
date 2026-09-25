# Errors and interruptions

## Mistakes in your code

A wrong use of the library raises `TypeError` or `ValueError` as early as possible: when you create the Agent, apply
`@tool` or `@loop`, or call a method. The message says what is wrong, how to fix it, and gives an example:

```text
TypeError: @loop needs both until and limit
Fix: set both
Example:
    @loop(until=State.is_answered, limit=50)
```

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
catch them or turn them into results. To let the model handle a failure, return it from the tool as a string. See
[Tools](tools.md#when-a-call-goes-wrong).

## The State after an exception

| The exception happens in | The State afterwards |
| --- | --- |
| `think` | The context is as it was before `think`. `state.turn` does not change |
| `use_tools` | Results of the calls that finished are recorded. The other calls stay pending |
| Anything, and leaves `run` | The error is recorded in `state.history`. Pending calls are closed with a result such as `(aborted: TimeoutError)` |

After `run` raises, `state.stopped_by` is `None`, and you can call `run` with the same State again. `agent` is the
Agent from the [quick start](../index.md#quick-start):

```python
import time

from alpineagents import RateLimitError, State

state = State("Find the bug in main.py")
try:
    agent.run(state)
except RateLimitError:
    time.sleep(60)
    agent.run(state)
```

## Catch an exception inside the loop body

If the loop body catches an exception from `use_tools`, the calls that did not finish stay pending, and the next
`think` raises `ValueError`. Close them with `state.deny` first:

```python
try:
    agent.use_tools(state)
except TimeoutError:
    for call in state.pending_calls:
        state.deny(call, "The tool timed out")
```

## Ctrl+C and cancellation

- Ctrl+C during `run` closes pending calls with the result `(interrupted by user)`. `run(state)` continues from there.
- A sync tool that is still running keeps running in its thread. If it finishes later, the model gets its result as a
  notice at the next `think`.
- In async code, cancelling the task follows the same rules. See [Async](../guides/async.md).
