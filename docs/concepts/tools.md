# Tool calls

This page is the rules for what happens between the model asking for a tool and getting its result. To write your
first tool, see [Give the agent tools](../learn/tools.md).

## What the model sees

A tool is something the model can call. `@tool` makes one from a function:

```python
from pathlib import Path

from alpineagents import ToolError, tool


@tool
def read_file(path: str, max_lines: int = 200) -> str:
    """Read a text file

    Args:
        path: Path relative to the repository root
        max_lines: How many lines to return
    """
    file = Path(path)
    if not file.exists():
        raise ToolError(f"No such file: {path}")
    return "\n".join(file.read_text().splitlines()[:max_lines])


print(read_file.spec)
```

| From the function | The model sees |
| --- | --- |
| Function name | Tool name: `read_file` |
| First paragraph of the docstring | Tool description |
| Type hints | Input schema (JSON Schema) |
| `Args:` section of the docstring | Description of each parameter |
| Default values | Optional parameters |

- Every parameter needs a type hint. Supported: `str`, `int`, `float`, `bool`, `Literal[...]`, `Enum`, `list[T]`,
  `dict[str, T]`, `T | None`, dataclass, `TypedDict`, Pydantic model. On Python 3.11, import `TypedDict` from
  `typing_extensions`.
- Options: `@tool(name="...", description="...", parallel=False, exception_handler=...)`, and the hints `read_only=`,
  `destructive=`, `idempotent=` and `open_world=`.
- The hints say what a tool does, for your code. The model does not see them. They are explained, with `hints_for`, in
  [Write permission rules](../guides/permissions.md).
- A mistake in the function raises `TypeError` when `@tool` runs, before any model request.
- Other kinds of tools are in [Tools on objects and from data](../guides/more-tools.md): `@tool` methods of an
  object, `Tool` subclasses, and MCP servers ([Use MCP servers](../guides/mcp.md)).

## Return values

| The tool returns | The model gets |
| --- | --- |
| `str` | The string |
| `None` | `(done)` |
| `Image` | The image |
| A list with an `Image` in it | Its text and images, in order |
| Anything else | JSON |

A value that cannot become JSON raises `TypeError`, which stops the run like any exception from a tool.

The result the model gets is a tuple of blocks, such as `(TextBlock("..."), Image(image/png, 34.2KB))`. It is the
`content` of the call's `ToolResultBlock` and `ToolResultEntry`, and the `result` a Reporter's `on_tool_end` gets.
`result_text(result)` from `alpineagents.types` turns it into one line of text, with each image as
`(image/png, 34.2KB)`. Images are explained in [Tools on objects and from data](../guides/more-tools.md).

## When a call goes wrong

| Situation | What happens | The run |
| --- | --- | --- |
| The model sends invalid arguments or an unknown tool name | The model gets `(input error: ...)` as the result | Continues |
| The tool raises `ToolInputError("...")` | The model gets `(input error: ...)` as the result | Continues |
| The tool raises `ToolError("...")` | The model gets the message as an error result | Continues |
| The tool raises an exception its `exception_handler` takes | The model gets the handler's message as an error result | Continues |
| The tool raises any other exception | The exception propagates out of `use_tools` and `run` | Stops |
| A permission refuses the call | The call does not run, and the model gets the reason as an error result | Continues, unless the permission stopped it |

An exception stops the run because it is usually a bug, and the model cannot fix your code. For a failure the model
can do something about, such as a missing file or an HTTP 404, raise `ToolError`, or name the exceptions in an
`exception_handler`. [Let the model handle tool failures](../guides/tool-failures.md) shows both.

- The type hint of the handler's one parameter says which exceptions it takes. `Exception` takes every exception.
  To let one exception stop the run anyway, raise it in the handler.
- `tool.copy(exception_handler=None)` gives a tool without the handler. The Agent itself does not know about handlers.

An error result shows up in three places:

- in the terminal as `error read_file: ...`,
- as `outcome.kind == "error"` in a [Reporter](../guides/progress.md),
- as a `ToolResultEntry` with `outcome == "error"` in `state.history` (`is_error` is true). The entry's `error` holds
  the `ToolError`, and its `__cause__` holds the exception the handler took. A store does not save `error`; the
  entry's `content` keeps the text.

## Several calls in one reply

A reply can ask for several tool calls. `use_tools` runs them like this:

1. Tools with `parallel=True` (the default) run at the same time, on worker threads.
2. Then tools with `parallel=False` run one at a time, in the order the model asked.
3. The results go into `state.messages` in the order the model asked, together, once the last call has one.

Use `parallel=False` for tools that must not overlap, such as two writes to the same file.

## The State parameter

A parameter typed `State` is hidden from the model and receives the current State:

```python
from alpineagents import State, tool


@tool
def submit(summary: str, state: State) -> None:
    """Submit the finished work"""
    state.finish(summary)
```

A tool can read `state.messages`, `state.extra_data` and the rest, and change the State through its methods:
`state.finish(...)`, `state.add_message(...)` and `state.edit_extra_data()`. See
[Let tools use the State](../guides/tool-state.md).
