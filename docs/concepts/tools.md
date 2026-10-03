# Tools

A tool is something the model can call. Most tools are Python functions with `@tool`. A tool that is not a function,
such as one built from a JSON Schema, is an object of a [`Tool` subclass](#tools-that-are-not-functions).

## Define a tool

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
| `Image` | The image |
| A list with an `Image` in it | Its text and images, in order |
| Anything else | JSON |

A value that cannot become JSON raises `TypeError`, which stops the run like any exception from a tool.

### Images

Return an `Image` for a tool whose result the model should look at, such as a screenshot or a chart. Put it in a list
to send text with it:

```python
from alpineagents import Image, tool


@tool
def screenshot(url: str) -> list:
    """Take a screenshot of a web page"""
    png = browser.screenshot(url)  # bytes
    return [f"Screenshot of {url}", Image(png)]
```

- `Image(data)` takes the image file's bytes. `Image.from_path("chart.png")` reads a file, and
  `Image.from_base64(text)` decodes base64 text. PNG, JPEG, GIF and WebP are supported; the type is read from the
  bytes (or pass `media_type="image/png"`).
- In the list, a `str` is text, and anything else that is not an `Image` is sent as JSON text.
- The result is the tuple of blocks the model gets: `(TextBlock("Screenshot of ..."), Image(image/png, 34.2KB))`. It
  is the `content` of the call's `ToolResultBlock` and `ToolResultEntry` history entry, and the `result` a Reporter's
  `on_tool_end` gets. `result_text(result)` from `alpineagents.types` turns it into one line of text, with each image
  as `(image/png, 34.2KB)`.
- Anthropic takes images in the tool result. Chat Completions APIs (`OpenAICompatible`) take only text there, so the
  adapter sends the images in a user message right after, marked as the call's result. The model must accept images.
- The context size estimate (`agent.context_tokens(state)`) counts each image as 1,600 tokens. Images are large:
  `state.clear_tool_results()` clears them with the rest of old results, and a compaction summary keeps only what the model wrote about them. History
  and the store keep them.

## When a call goes wrong

| Situation | What happens | The run |
| --- | --- | --- |
| The model sends invalid arguments or an unknown tool name | The model gets `(input error: ...)` as the result | Continues |
| The tool raises `ToolInputError("...")` | The model gets `(input error: ...)` as the result | Continues |
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
[Reporter](../guides/progress.md), and as a `ToolResultEntry` with `outcome == "error"` (so `is_error` is true) in
`state.history`. That entry's `error` holds the `ToolError`, and its `__cause__` holds the exception the handler took.
A store does not save `error`; the entry's `content` keeps the text.

## Tools that use the State

A parameter typed `State` is hidden from the model and receives the current State:

```python
from alpineagents import State, tool


@tool
def submit(summary: str, state: State) -> None:
    """Submit the finished work"""
    state.finish(summary)
```

A tool can read `state.messages`, `state.extra_data` and the rest, and change the State through its methods:
`state.finish(...)`, `state.add_message(...)`, and `state.edit_extra_data()` to keep values of your own.

```python
@tool
def remember(note: str, state: State) -> None:
    """Remember something for later"""
    with state.edit_extra_data() as data:
        data.setdefault("notes", []).append(note)
```

See [Tools that use the State](../guides/tool-state.md).

## Tools on an object

`@tool` works on methods. Pass the object to give the Agent every `@tool` method, or pass one method:

```python
--8<-- "docs_src/tool_object.py"
```

Use an object when tools share settings, such as a root folder or a client.

## Tools that are not functions

`@tool` needs a Python function with type hints. For a tool whose name, description and input schema come as data,
such as tools stored in a database or built from an OpenAPI spec, subclass `Tool`:

```python
import httpx

from alpineagents import Agent, State, Tool, ToolError, ToolInputError


class Webhook(Tool):
    def __init__(self, name: str, description: str, schema: dict, url: str):
        super().__init__(name=name, description=description, input_schema=schema, open_world=True)
        self.url = url

    def run(self, args: dict, state: State) -> dict:
        if "title" not in args:
            raise ToolInputError("title is required")
        response = httpx.post(self.url, json=args)
        if response.status_code >= 400:
            raise ToolError(f"HTTP {response.status_code}")
        return response.json()


agent = Agent(model="claude-sonnet-5", tools=[Webhook(**row) for row in rows])
```

- `super().__init__` takes `name`, `description`, `input_schema` (an object schema, left out for no arguments),
  `parallel` and the [hints](#describe-what-a-tool-does).
- `run(args, state)` gets the model's arguments as a dict, not checked against `input_schema`. It is your own copy:
  changing it does not change the recorded call, whose `args` are read-only. Check what you rely
  on and raise `ToolInputError` when it is wrong.
- What `run` returns (an `Image` included) and raises works as for `@tool`: see [Return values](#return-values) and
  [When a call goes wrong](#when-a-call-goes-wrong). Arguments that are not valid JSON never reach `run`.
- `run` can be `async def`.
- A subclass that does not call `super().__init__` or does not define `run`, or passing the class instead of an
  object, raises `TypeError` when you create the Agent.

`@tool` makes a `FunctionTool`, and an MCP server's tool is an `MCPTool`. Both are `Tool` subclasses, so every value of
`agent.tool_map` is a `Tool`.

## Describe what a tool does

Four hints say what a tool does. The model does not see them. They are for your code, such as a
[permission](../guides/approval.md) that runs read-only calls without asking the person:

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

### Hints for one call

The hints of `@tool` hold for every call, so they are the worst case. A shell tool that can run `rm` is not read-only,
even though most of its calls only look. `hints_for=` gives the facts for one call:

```python
--8<-- "docs_src/tool_hints.py"
```

```python
bash.hints_for({"command": "ls"}).read_only  # True
bash.hints_for({"command": "rm -rf build"}).read_only  # False: the tool's own hints
bash.read_only  # False: the worst case stays
```

- `Hints(...)` takes the same four hints with the same rules: one left out assumes the worst, and `read_only=True`
  sets the other two. Every hint is a `bool` after that.
- The function gets the model's arguments before they are checked against the type hints. A value can be missing or
  of any type, so read with `args.get(...)` and `isinstance`, and return `None` when unsure: the tool's own hints then
  apply. Anything other than `Hints` or `None` raises `TypeError`.
- Hints are facts about the call, not decisions. Say what the call does; the code that reads the hints decides what
  to allow. `AllowByReadOnly()` reads `hints_for`, so with it `bash("ls")` runs without asking. See
  [Ask before a tool runs: decide by what a call does](../guides/approval.md#decide-by-what-a-call-does).
- `tool.copy(hints_for=...)` adds or replaces the function, and `hints_for=None` removes it.
- A `Tool` subclass overrides `hints_for(self, args)`, and returns `super().hints_for(args)` when unsure. An MCP
  tool's `hints_for` returns its annotations for every call.
- To tell more about a call, subclass `Hints` with fields that have defaults, such as
  `@dataclass(frozen=True) class FileHints(Hints): paths: tuple[str, ...] = ()`, and return that.

### Find the tool of a call

`agent.tool_map` has every tool the model can call, by the name it calls it: `@tool` functions, the `@tool` methods of
objects, and the tools of MCP servers. Look up a call's tool there to read its hints:

```python
for call in state.pending_calls:
    found = agent.tool_map.get(call.name)  # None for a name the model made up
    if found is not None and not found.read_only:
        ...  # log it, for example. To ask the person first, see "Ask before a tool runs"
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
  defaults. The server says them about itself, so do not trust them more than the server: `AllowByReadOnly()`
  ignores them unless `trust_mcp=True` (see [MCP tools](../guides/approval.md#mcp-tools)). `found.server` says which
  server the tool is from.
- A Reporter gets the State and the call, not the Agent. To read hints there, give it the Agent after creating both:
  `reporter.agent = agent`.

See [Ask before a tool runs](../guides/approval.md).

## Several calls in one reply

A reply can ask for several tool calls. `use_tools` runs them like this:

1. Tools with `parallel=True` (the default) run at the same time, on worker threads.
2. Then tools with `parallel=False` run one at a time, in the order the model asked.
3. The results go into `state.messages` in the order the model asked, together, once the last call has one.

Use `parallel=False` for tools that must not overlap, such as two writes to the same file.

## MCP servers

MCP servers go in `tools=` too. See [MCP servers](../guides/mcp.md).
