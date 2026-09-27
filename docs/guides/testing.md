# Testing

Test an agent without an API key or network. `FakeModel` replaces the model and returns prepared replies in order.

## A test

```python
--8<-- "docs_src/test_example.py"
```

1. `FakeModel` gets two replies: a call to `read_file`, then the answer.
2. `agent.copy(model=fake, reporter=None)` is a new Agent with the same settings, except the fake model and no
   terminal output.
3. The assertions check the answer, why the run stopped, and that every reply was used.

## Replies FakeModel accepts

Each model request takes the next item: every `think`, every `ask` attempt and every `compact` (also from
`compact_if_full`). When the items run out, the request raises `RuntimeError`.

| Item | The reply |
| --- | --- |
| `"text"` | A text reply without tool calls |
| `tool_call("name", arg=value)` | A reply with one tool call |
| A list of text and tool calls | A reply with those parts in order, for example `["Let me check.", tool_call(...)]` |
| An exception instance, such as `RateLimitError("slow down")` | The request raises it |
| A `Reply` object | Used as is |
| A function that takes the `Request` | Its return value, read by the rules above |

The replies FakeModel builds have `model` set to the fake's name, `"fake"` by default.

`fake.requests` holds every request the model received. Use it to check what the model saw:

```python
assert [t.name for t in fake.requests[0].tools] == ["read_file"]
```

## A tool that fails

A failure the model should handle, raised with `ToolError`, becomes an error result, and the run continues:

```python
--8<-- "docs_src/test_tool_failure.py"
```

- The `tool_result` entry in `state.history` has the message the model got, and `is_error=True`.
- `state.stopped_by == "is_answered"` shows the run went on after the failure. An exception that is not a
  `ToolError` would have stopped the run and been raised by `run`.
- To test only the tool, call it without an Agent. A `@tool` function runs as a plain function
  (`read_file("missing.py")` raises the `ToolError`), and a [`Tool` subclass](../concepts/tools.md#tools-that-are-not-functions)
  has `run(args, state)`: `Webhook(...).run({"title": "x"}, State("test"))`.

## Questions to the person

`FakeHuman` answers `ask_human` with prepared answers in order. `agent` is the Agent from
[Ask before a tool runs](approval.md):

```python
from alpineagents.testing import FakeHuman, FakeModel, tool_call

fake = FakeModel([tool_call("write_file", path="a.md", content="x"), "Skipped"])
human = FakeHuman(["no"])
agent.copy(model=fake, human=human, reporter=None).run("Write a.md")
assert human.remaining == 0
```

An answer that does not fit the question's `returns` type is skipped, and the next one is used.

## What to assert

| To check | Assert on |
| --- | --- |
| The result | `state.answer` |
| Why the run stopped | `state.stopped_by` |
| How many turns | `state.turn` |
| What happened, in order | `state.history`, for example `[h.kind for h in state.history]`. The kinds are listed in the [Data types API](../api/types.md) |
| What the model saw | `fake.requests` |

## Test saving and resuming

Use a `FileStore` in pytest's `tmp_path`, then load the State as another process would:

```python
store = FileStore(tmp_path)
agent.copy(model=FakeModel(["Done"]), reporter=None, store=store).run(State("Fix it", id="t1"))

state = store.load("t1")
assert state.answer == "Done"
```

## Related

- [Testing API](../api/testing.md)
