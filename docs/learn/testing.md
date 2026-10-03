# Test your agent

A test needs no API key and no network. `FakeModel` replaces the model and returns prepared replies in order.

## A test

```python
--8<-- "docs_src/test_example.py"
```

1. `FakeModel` gets two replies: a call to `read_file`, then the answer.
2. `agent.copy(model=fake, reporter=None)` is a new Agent with the same settings, a fake model and no terminal
   output. The Agent in your code stays as it is.
3. The assertions check the answer, why the run stopped (`is_answered`, see
   [Write your own loop](loop.md#the-name-is_answered)), and that every reply was used.

## Replies FakeModel accepts

Every model request takes the next item. When the items run out, the request raises `RuntimeError`.

| Item | The reply |
| --- | --- |
| `"text"` | A text reply without tool calls |
| `tool_call("name", arg=value)` | A reply with one tool call |
| A list of text and tool calls | A reply with those parts in order, for example `["Let me check.", tool_call(...)]` |
| An exception instance, such as `RateLimitError("slow down")` | The request raises it |
| A `Reply` object | Used as is |
| A function that takes the `Request` | Its return value, read by the rules above |

`fake.requests` holds every request the model received. Use it to check what the model saw. In the test above,
after `run`:

```python
assert [t.name for t in fake.requests[0].tools] == ["read_file"]
```

## What to assert

| To check | Assert on |
| --- | --- |
| The result | `state.answer` |
| Why the run stopped | `state.stopped`, for example `StoppedByUntil("is_answered")` |
| How many turns | `state.turn` |
| What happened, in order | `state.history`, for example `[h.kind for h in state.history]` |
| What the model saw | `fake.requests` |

## A tool that fails

A `ToolError` becomes an error result, and the run goes on. The test shows both:

```python
--8<-- "docs_src/test_tool_failure.py"
```

The `tool_result` entry in `state.history` has the message the model got and `is_error=True`. The stop name shows
that the run went on after the failure. To test only the tool, call it as a plain function.

## The person's answers

`FakeHuman` answers the questions of `DecideByHuman` with prepared answers, in order:

```python
--8<-- "docs_src/test_approval_flow.py"
```

`human.remaining == 0` checks that every answer was asked for. An answer that does not fit the question is skipped,
and the next one is used.

## Save and resume

Use a `FileStore` in pytest's `tmp_path`, then load the State as another process would:

```python
--8<-- "docs_src/test_resume.py"
```

## Next

You have the whole path: tools, loops, the State, permissions, stores and tests. Pick a task in the Guides, for
example [Use other models](../guides/models.md) or [Handle unreliable models](../guides/unreliable-models.md). How it
all fits together is in [How a run works](../concepts/overview.md).
