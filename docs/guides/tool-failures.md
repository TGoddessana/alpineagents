# Let the model handle tool failures

Some failures are part of the job: a page that does not exist, a server that is down, a request that timed out. The
model should hear about them and try something else. Other failures are bugs in your code, and the model cannot fix
those. A tool says which is which, and anything it does not name stops the run.

## Branch on the failure

This tool reads a JSON API with [httpx](https://www.python-httpx.org/). Its `exception_handler` turns the HTTP
failures the model can work around into messages, one per kind:

```python
import httpx

from alpineagents import Agent, tool

client = httpx.Client(base_url="https://api.example.com")


def http_errors(error: httpx.HTTPError) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        status, url = error.response.status_code, error.request.url
        if 400 <= status < 500:
            return f"HTTP {status} for {url}. Check the path or id and try another request"
        if status >= 500:
            return f"HTTP {status} for {url}. The server failed. Find the information another way"
    if isinstance(error, httpx.TimeoutException):
        return f"Timed out: {error.request.url}. Try again later"
    raise error


@tool(exception_handler=http_errors)
def get_json(path: str) -> dict:
    """GET a JSON endpoint of the API

    Args:
        path: A path such as /users/42
    """
    return client.get(path).raise_for_status().json()


agent = Agent(model="claude-sonnet-5", tools=[get_json])
```

| What happens in `get_json` | Taken by `http_errors`? | Result |
| --- | --- | --- |
| 404 | Yes | The model gets `HTTP 404 for ... Check the path or id ...`. The run continues |
| 503 | Yes | The model gets `HTTP 503 for ... The server failed ...`. The run continues |
| Timeout | Yes | The model gets `Timed out: ...`. The run continues |
| 302 | Yes, but the handler raises it again | The run stops with `httpx.HTTPStatusError` |
| A 200 response that is not JSON | No, `JSONDecodeError` is not an `httpx.HTTPError` | The run stops with `JSONDecodeError` |

The type hint `httpx.HTTPError` decides which exceptions reach the handler. Inside it, `raise error` sends one back
out, for a failure the model can do nothing about.

## What you see

When the handler takes a failure, the terminal shows `error` and the run goes on:

```text
[turn 1] thinking
  tool get_json(path="/users/999")
  error get_json: HTTP 404 for https://api.example.com/users/999. Check the path or id and try another request
[turn 2] thinking
  tool get_json(path="/orders")
  tool get_json(path="/slow")
  error get_json: HTTP 503 for https://api.example.com/orders. The server failed. Find the information another way
  error get_json: Timed out: https://api.example.com/slow. Try again later
[turn 3] thinking
  tool get_json(path="/users/42")
  done 25B
[turn 4] thinking
User 42 is Ana. The orders could not be fetched because the server failed.
done: stopped by is_answered (4 turns)
```

When an exception gets through, the terminal shows `aborted` and `run` raises it:

```text
[turn 1] thinking
  tool get_json(path="/legacy")
  aborted get_json: (aborted: JSONDecodeError)
done: error JSONDecodeError: Expecting value: line 1 column 1 (char 0) (1 turn)
```

## When an exception gets through

`run` raises the exception as it is, so `except` catches it by its own type. It has a note naming the call, and the
State keeps what happened:

```python
from alpineagents import Message, State

state = State(messages=[Message.user("Get the settings from the legacy API")])
try:
    agent.run(state)
except Exception as error:
    print(error.__notes__)  # ['exception raised in tool get_json(path="/legacy")']
    print(state)
```

```text
[turn 0] context_change import: 1 messages
[turn 0] run_start: anthropic/claude-sonnet-5
[turn 1] model_request: anthropic/claude-sonnet-5
[turn 1] model_reply: get_json(path="/legacy")
[turn 1] error get_json: JSONDecodeError: Expecting value: line 1 column 1 (char 0)
[turn 1] tool_result get_json (error): (aborted: JSONDecodeError)
```

The call is closed with `(aborted: JSONDecodeError)`, so you can call `agent.run(state)` again. The model then sees
that result and continues from there.

## Decide where a failure belongs

When an `aborted` line shows up, decide whether it is a bug or a failure the model should know about:

| The failure | Where to handle it |
| --- | --- |
| Only this tool knows it, like a response that is not JSON | Catch it in the tool and raise `ToolError` |
| Exceptions of a library, raised by several tools | The type hint of an `exception_handler` |
| A bug in your code | Nowhere. Let it stop the run, and fix the code |

For the JSON case, the tool knows what went wrong and can say so:

```python
from alpineagents import ToolError


@tool(exception_handler=http_errors)
def get_json(path: str) -> dict:
    """GET a JSON endpoint of the API

    Args:
        path: A path such as /users/42
    """
    response = client.get(path).raise_for_status()
    try:
        return response.json()
    except ValueError as error:
        content_type = response.headers.get("content-type")
        raise ToolError(f"{path} did not return JSON (content-type: {content_type})") from error
```

```text
  tool get_json(path="/legacy")
  error get_json: /legacy did not return JSON (content-type: text/html)
```

Adding `ValueError` to the handler's type hint would work too, but a `ValueError` from a bug in the tool would then
reach the model instead of you. Keep the type hint to the exceptions you expect.

## Related

- [Tools: when a call goes wrong](../concepts/tools.md#when-a-call-goes-wrong)
- [Errors and interruptions](../concepts/errors.md)
- [Progress and questions](progress.md): `outcome.kind == "error"` in your own Reporter
