# Check the work before finishing

When the model answers, your code checks the work. If the check fails, the model keeps working.

```python
--8<-- "docs_src/verify.py"
```

1. When the reply asks for tools, the turn runs them and ends with `return`.
2. When the reply is an answer, `failing_tests()` runs the tests.
3. If they fail, `state.add_notice(...)` adds the output to the context.
4. The notice comes after the answer, so `State.is_answered` is false and the loop continues.
5. If the tests pass, nothing is added. `State.is_answered` is true and the loop stops before the next turn.

`limit=40` still caps the run if the model cannot fix the tests.

## Why a notice

`add_notice` adds a message from your code, marked with `[notice] ` so the model can tell it apart from the person.
Use `add_user_message` for messages from the person.

## Related

- [State: add a message](../concepts/state.md#add-a-message)
- [Stop conditions](stop-conditions.md)
