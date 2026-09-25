# Structured output

Get a typed result instead of text. There are two ways.

| Way | When to use |
| --- | --- |
| `agent.ask(state, prompt, returns=Type)` | During or after a run, to extract a result without changing the run |
| A tool that calls `state.finish(value)` | When the model should decide when it is done, and hand over the result itself |

## ask

```python
--8<-- "docs_src/structured_output.py"
```

1. `agent.run(state)` does the work.
2. `agent.ask(...)` asks the model one more question about the same context.
3. `returns=Review` makes the answer a `Review` object.

About `ask`:

- `returns` accepts `str`, a dataclass or a Pydantic model.
- The question and answer are recorded in `state.history`, but not added to the context. The run continues as if the
  question was never asked.
- The model cannot call tools while answering.
- If the answer does not fit `returns`, `ask` asks again up to `retries` times (default 2), then raises `OutputError`.
- After `state.finish()`, `ask` raises `ValueError`. So do not combine it with a submit tool on the same State.

## A submit tool

The model fills the tool's typed parameters, and the tool ends the run with them. See
[Stop conditions: stop from a tool](stop-conditions.md#stop-from-a-tool).

## Related

- [Agent: ask compared with think](../concepts/agent.md#ask-compared-with-think)
