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
- Options: `@tool(name="...", description="...", parallel=False)`.
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
| The tool returns an error message as a string | The model gets the string | Continues |
| The tool raises an exception | The exception propagates out of `use_tools` and `run` | Stops |

Return a string for failures the model can fix, such as a missing file or a failing command. Raise for failures that
should stop the run.

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

## Several calls in one reply

A reply can ask for several tool calls. `use_tools` runs them like this:

1. Tools with `parallel=True` (the default) run at the same time, on worker threads.
2. Then tools with `parallel=False` run one at a time, in the order the model asked.
3. The results go into the context in the order the model asked.

Use `parallel=False` for tools that must not overlap, such as two writes to the same file.

## MCP servers

MCP servers go in `tools=` too. See [MCP servers](../guides/mcp.md).
