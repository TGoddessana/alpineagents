# Give the agent tools

A tool is a Python function the model can call. This page shows what the model sees of one, and what happens when
the call fails.

## Define a tool

```python
--8<-- "docs_src/learn_tools.py"
```

1. `@tool` makes a tool from the function. A tool is still callable as a plain function in your own code.
2. `Args:` in the docstring describes each parameter for the model.
3. `max_lines` has a default value, so the model may leave it out.
4. `tools=[read_file]` gives the tool to the Agent.

## What the model sees

`print(read_file.spec)` shows exactly what goes to the model:

```text
ToolSpec(name='read_file', description='Read a text file', input_schema={'additionalProperties': False, 'properties': {'path': {'description': 'Path of the file', 'type': 'string'}, 'max_lines': {'default': 200, 'description': 'How many lines to return', 'type': 'integer'}}, 'required': ['path'], 'type': 'object'})
```

| From the function | The model sees |
| --- | --- |
| The function name | The tool name: `read_file` |
| The first paragraph of the docstring | The tool description |
| The type hints | The input schema, as JSON Schema |
| The `Args:` section | The description of each parameter |
| A default value | An optional parameter |

The model only has this text to decide when and how to call the tool. Write the name, the description and the
`Args:` for the model as a reader. Every parameter needs a type hint.

## What a tool returns

Return a `str` for text. Return `None` when there is nothing to say, and the model gets `(done)`. Return a list,
a dict or a number, and the model gets it as JSON. A tool can also return an image. The full table is in
[Tool calls](../concepts/tools.md#return-values).

## When a call fails

| What happens in the tool | The model gets | The run |
| --- | --- | --- |
| It raises `ToolError("...")` | The message, as an error result | Continues |
| The model asked for a tool that does not exist, or sent arguments that do not fit | An `(input error: ...)` result | Continues |
| It raises any other exception | Nothing | Stops, and `agent.run` raises it |

Use `ToolError` for what the model can do something about, such as a missing file. It reads the message and tries
something else, for example a different path. Any other exception is usually a bug in your code, and the model
cannot fix that, so the run stops and you see the traceback.

A good message says what to do next: `No such file: notes.md. Use list_files to see what exists`.

## Next

[Write your own loop](loop.md). To go deeper on tools:

- [Let the model handle tool failures](../guides/tool-failures.md)
- [Let tools use the State](../guides/tool-state.md)
- [Tools on objects and from data](../guides/more-tools.md)
- [Tool calls](../concepts/tools.md)
