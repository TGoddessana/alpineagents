# Ask before a tool runs

The person approves risky tool calls before they run. Refused calls do not run, and the model is told why.

## Yes or no

```python
--8<-- "docs_src/approval.py"
```

1. `agent.think(state)` gets the reply. Tool calls wait in `state.pending_calls`.
2. For each risky call, `agent.ask_human` asks the person. `returns=Literal["yes", "no"]` accepts only those answers.
3. `state.deny(call, reason)` refuses the call. The model gets `reason` as that call's result.
4. `agent.use_tools(state)` runs the calls that are still pending. Denied calls are no longer pending.

The default `human` is the terminal, which shows `yes/no` after the question and asks again on any other answer.

## Add "always"

Let the person approve a tool once for the rest of the run. Replace the loop with:

```python
--8<-- "docs_src/approval_always.py"
```

- `state.data["allowed"]` keeps the approved tool names across turns.
- The model never sees `state.data`.

## Tell the model how to do it differently

A refusal with a reason from the person helps the model choose another way. Inside the loop body above:

```python
how = agent.ask_human(state, "What should it do instead?")
state.deny(call, f"The user declined this call. They said: {how}")
```

## Related

- [Progress and questions](progress.md): answer questions from somewhere other than the terminal
- [Testing](testing.md): test the approval flow with `FakeHuman`
