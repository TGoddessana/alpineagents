# Your first agent

This page installs alpineagents and runs an agent that reads a file. You need Python 3.11 or later.

## Install

```bash
pip install alpineagents     # or: uv add alpineagents
```

Anthropic models read `ANTHROPIC_API_KEY`. OpenAI-compatible servers read `OPENAI_API_KEY`. To try the library first
without a key, see [Run without an API key](#run-without-an-api-key).

## Run it

```python
--8<-- "docs_src/quickstart.py"
```

1. `@tool` turns the function into a tool. The type hints and the docstring tell the model how to call it.
2. `ToolError` tells the model the call failed, and the run continues. Any other exception stops the run.
3. `Agent` holds the settings: the model and the tools.
4. `agent.run` repeats turns until the model answers, then returns the answer.

## What you see

Progress is printed to the terminal while the agent works:

```text
[turn 1] thinking
  tool read_file(path="README.md")
  done 6.7KB
[turn 2] thinking
README.md describes a Python agent framework.
done: stopped by is_answered (2 turns)
```

1. In turn 1 the model asked for `read_file`. The Agent ran it, and the file was 6.7KB.
2. In turn 2 the model had the file and answered, and the answer was printed.
3. A turn is one model reply and the tool calls it asked for. The last line says why the run ended.
   `is_answered` means the model answered without asking for a tool. [Write your own loop](loop.md#the-name-is_answered) explains the name.

`agent.run` returns the answer, and `print` shows it a second time here. To hide the progress lines, pass
`reporter=None` to `Agent`.

## Run without an API key

`FakeModel` returns prepared replies in order. It needs no key and no network. This is the same agent with a fake
model:

```python
--8<-- "docs_src/no_api_key.py"
```

The first reply asks for `read_file`. The Agent runs the tool, and the second reply is the answer. Run it from the
folder that has `README.md`. You will use `FakeModel` again in [Test your agent](testing.md).

## Next

[Give the agent tools](tools.md): what the model sees of a tool, and what happens when it fails.
