# Tools on objects and from data

[Give the agent tools](../learn/tools.md) showed `@tool` on a function. This page covers the other ways to make a tool.

## Tools on an object

`@tool` works on methods. Pass the object to give the Agent every `@tool` method, or pass one method:

```python
--8<-- "docs_src/tool_object.py"
```

1. `workspace` in `tools=` gives the model `read_file` and `list_files`.
2. `workspace.read_file` gives it only that one.
3. The model does not see `self`.

Use an object when tools share settings, such as a root folder or a client.

## Tools from data

`@tool` needs a Python function with type hints. For a tool whose name, description and input schema come as data,
such as rows in a database or an OpenAPI spec, subclass `Tool`:

```python
--8<-- "docs_src/tool_subclass.py"
```

1. `super().__init__` takes `name`, `description`, `input_schema` (an object schema, left out for no arguments),
   `parallel` and the [hints](permissions.md#say-what-a-tool-does).
2. `run(args, state)` gets the model's arguments as a dict. They are not checked against `input_schema`, so check what
   you rely on and raise `ToolInputError` when it is wrong.
3. What `run` returns and raises works as for `@tool`: see [Tool calls](../concepts/tools.md). `run` can be
   `async def`.

`@tool` makes a `FunctionTool` and an MCP server's tool is an `MCPTool`. Both are `Tool` subclasses.

## Images from a tool

Return an `Image` for a tool whose result the model should look at, such as a screenshot or a chart. Put it in a list
to send text with it:

```python
--8<-- "docs_src/tool_image.py"
```

- `Image(data)` takes the file's bytes. `Image.from_path("chart.png")` reads a file, and `Image.from_base64(text)`
  decodes base64 text. PNG, JPEG, GIF and WebP work. The type is read from the bytes, or pass
  `media_type="image/png"`.
- In the list, a `str` is text, and anything else that is not an `Image` is sent as JSON text.
- The model must read images. `Anthropic` takes them in the tool result. `OpenAICompatible` sends them in a user
  message right after it.
- An image counts as 1,600 tokens, so `state.clear_tool_results()` clears old ones with the text results
  ([Keep the context small](context-size.md)). History and a store keep them.
- A Reporter gets the result as a tuple of blocks. `result_text(result)` from `alpineagents.types` makes it one line.
- To send an image from the person, see [Keep the conversation](../learn/conversation.md#send-an-image).

## Tools that must not overlap

A reply can ask for several calls. Tools with `parallel=True` (the default) run at the same time. Then tools with
`parallel=False` run one at a time, in the order the model asked. Use `parallel=False` for two writes to the same
file:

```python
from alpineagents import tool


@tool(parallel=False)
def append_line(path: str, line: str) -> None:
    """Add a line to the end of a file"""
    with open(path, "a") as file:
        file.write(line + "\n")
```

## Test a tool alone

A `@tool` function runs as a plain function: `read_file("missing.py")` raises its `ToolError`. A `Tool` subclass has
`run(args, state)`: `Webhook(**row).run({"title": "x"}, State())`. To test a tool inside a run, see
[Test your agent](../learn/testing.md).

## Related

- [Tool calls](../concepts/tools.md)
- [Let tools use the State](tool-state.md)
- [Let the model handle tool failures](tool-failures.md)
