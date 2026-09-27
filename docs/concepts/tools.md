# Tools

A tool is a Python function the model can call.

## Define a tool

```python
from pathlib import Path

from alpineagents import tool


@tool
def read_file(path: str, max_lines: int = 200) -> str:
    """Read a text file

    Args:
        path: Path relative to the repository root
        max_lines: How many lines to return
    """
    file = Path(path)
    if not file.exists():
        return f"No such file: {path}"
    return "\n".join(file.read_text().splitlines()[:max_lines])
```

The model sees the tool built from the function:

| From the function | The model sees |
| --- | --- |
| Function name | Tool name: `read_file` |
| First paragraph of the docstring | Tool description |
| Type hints | Input schema (JSON Schema) |
| `Args:` section of the docstring | Description of each parameter |
| Default values | Optional parameters |

`read_file.spec` shows exactly what the model sees.

- Every parameter needs a type hint. Supported: `str`, `int`, `float`, `bool`, `Literal[...]`, `Enum`, `list[T]`,
  `dict[str, T]`, `T | None`, dataclass, `TypedDict`, Pydantic model. On Python 3.11, import `TypedDict` from
  `typing_extensions`.
- Options: `@tool(name="...", description="...", parallel=False, exception_handler=...)`. See
  [When a call goes wrong](#when-a-call-goes-wrong). The hints `read_only=`, `destructive=`, `idempotent=` and
  `open_world=` are in [Describe what a tool does](#describe-what-a-tool-does).
- A mistake in the function raises `TypeError` when `@tool` runs, before any model request.

## Return values

| The tool returns | The model gets |
| --- | --- |
| `str` | The string |
| `None` | `(done)` |
| Anything else | JSON |

A value that cannot become JSON raises `TypeError`, which stops the run like any exception from a tool.

## When a call goes wrong

| Situation | What happens | The run |
| --- | --- | --- |
| The model sends invalid arguments or an unknown tool name | The model gets `(input error: ...)` as the result | Continues |
| The tool raises `ToolError("...")` | The model gets the message as an error result | Continues |
| The tool raises an exception its `exception_handler` takes | The model gets the handler's message as an error result | Continues |
| The tool raises any other exception | The exception propagates out of `use_tools` and `run` | Stops |

An exception stops the run because it is usually a bug, and the model cannot fix your code. For a failure the model
can do something about, such as a missing file or an HTTP 404, raise `ToolError`:

```python
from pathlib import Path

from alpineagents import ToolError, tool


@tool
def read_file(path: str) -> str:
    """Read a file"""
    if not Path(path).is_file():
        raise ToolError(f"No such file: {path}. Use list_files to see what exists")
    return Path(path).read_text()
```

When the failures come as a library's exceptions, name them in an `exception_handler` instead of catching them in
every tool. The type hint of its one parameter says which exceptions it takes, and it returns the message for the
model:

```python
import httpx


def http_errors(error: httpx.HTTPError) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP {error.response.status_code}: {error.request.url}"
    return f"Request failed: {error}"


@tool(exception_handler=http_errors)
def fetch_url(url: str) -> str:
    """Fetch a web page"""
    return httpx.get(url).raise_for_status().text


@tool(exception_handler=http_errors)
def post_json(url: str, body: dict[str, str]) -> str:
    """Send JSON to a URL"""
    return httpx.post(url, json=body).raise_for_status().text
```

- Use `FileNotFoundError | PermissionError` for several exception types, and `Exception` to take every exception.
- To let one exception stop the run anyway, raise it in the handler.
- A mistake in the handler (no type hint, not an exception class, more than one required parameter) raises
  `TypeError` when `@tool` runs.
- `fetch_url.copy(exception_handler=None)` gives a tool without the handler, for an Agent where every failure should
  stop the run. The Agent itself does not know about handlers.

[Let the model handle tool failures](../guides/tool-failures.md) branches on the status code, shows what happens to
an exception the handler does not take, and how to decide where a failure belongs.

An error result shows up as `error fetch_url: ...` in the terminal, as `outcome.kind == "error"` in a
[Reporter](../guides/progress.md), and as a `tool_result` entry with `is_error=True` in `state.history`. That entry's
`error` holds the `ToolError`, and its `__cause__` holds the exception the handler took.

## Tools that use the State

A parameter typed `State` is hidden from the model and receives the current State:

```python
from alpineagents import State, tool


@tool
def submit(summary: str, state: State) -> None:
    """Submit the finished work"""
    state.finish(summary)
```

See [Tools that use the State](../guides/tool-state.md).

## Tools on an object

`@tool` works on methods. Pass the object to give the Agent every `@tool` method, or pass one method:

```python
--8<-- "docs_src/tool_object.py"
```

Use an object when tools share settings, such as a root folder or a client.

## Describe what a tool does

Four hints say what a tool does. The model does not see them. They are for your code, such as a loop that asks the
person before a call that changes something:

```python
@tool(read_only=True, open_world=False)
def read_file(path: str) -> str:
    """Read a file in the project"""
    return Path(path).read_text()


@tool(open_world=False)
def write_file(path: str, content: str) -> None:
    """Create a file, or replace its content"""
    Path(path).write_text(content)
```

| Hint | The tool | Left out |
| --- | --- | --- |
| `read_only` | Changes nothing | `False` |
| `destructive` | May delete or overwrite something | `True`, or `False` for a read-only tool |
| `idempotent` | Has no further effect when called again with the same arguments | `False`, or `True` for a read-only tool |
| `open_world` | Reaches outside its own domain: the web, a shell, another system | `True` |

A hint left out assumes the worst, so a forgotten hint makes a rule stricter, never looser. `read_only=True` with
`destructive=True` or `idempotent=False` raises `ValueError`. The hints have the meaning of MCP tool annotations.

### Find the tool of a call

`agent.tool_map` has every tool the model can call, by the name it calls it: `@tool` functions, the `@tool` methods of
objects, and the tools of MCP servers. Look up a call's tool there to read its hints:

```python
for call in state.pending_calls:
    found = agent.tool_map.get(call.name)  # None for a name the model made up
    if found is not None and not found.read_only:
        ...  # ask the person, see "Ask before a tool runs"
```

Pick tools by their hints for `think(tools=...)`:

```python
readers = [t for t in agent.tool_map.values() if t.read_only]
agent.think(state, tools=readers)
```

- `agent.tool_map` is read-only. To change the tools, use `agent.copy(tools=...)`.
- MCP servers' tools are in it only while the servers are connected: during a run, which covers the loop, blocks and
  Reporters, or inside `with agent:`. Looking up one of their names before that raises a `KeyError` that says so.
- An MCP tool (`MCPTool`) gets its hints from the server's annotations (`readOnlyHint` and so on), with the same
  defaults. The server says them about itself, so do not trust them more than the server. `found.server` says which
  server the tool is from.
- A Reporter gets the State and the call, not the Agent. To read hints there, give it the Agent after creating both:
  `reporter.agent = agent`.

See [Ask before a tool runs](../guides/approval.md).

## Several calls in one reply

A reply can ask for several tool calls. `use_tools` runs them like this:

1. Tools with `parallel=True` (the default) run at the same time, on worker threads.
2. Then tools with `parallel=False` run one at a time, in the order the model asked.
3. The results go into the context in the order the model asked.

Use `parallel=False` for tools that must not overlap, such as two writes to the same file.

## MCP servers

MCP servers go in `tools=` too. See [MCP servers](../guides/mcp.md).
