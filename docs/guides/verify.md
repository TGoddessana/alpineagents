# Check the work before finishing

When the model answers, your code checks the work. If the check fails, the model keeps working.

```python
--8<-- "docs_src/verify.py"
```

1. When the reply asks for tools, the turn runs them and ends with `return`.
2. When the reply is an answer, `failing_tests()` runs the tests.
3. If they fail, `state.add_message(Message.notice(...))` adds the output to the context.
4. The notice comes after the answer, so `waiting_for_user` is false and the loop continues.
5. If the tests pass, nothing is added. `waiting_for_user` is true and the loop stops before the next turn.

`limit=40` still caps the run if the model cannot fix the tests. `waiting_for_user` is explained in
[Write your own loop](../learn/loop.md).

## Why a notice

`Message.notice(text)` makes a message from your code, marked with `[notice] ` so the model can tell it apart from the
person. Use `Message.user(text)` for messages from the person. Both go in with `state.add_message(...)`
([Keep the conversation](../learn/conversation.md#messages-from-your-code)).

## Related

- [Plan before acting](plan-first.md): the same notice technique
- [Stop a run](stop-conditions.md)
