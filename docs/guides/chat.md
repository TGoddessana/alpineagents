# Chat

The person and the agent talk in turns. The model sees the whole conversation.

```python
--8<-- "docs_src/chat.py"
```

1. One State holds the whole conversation.
2. `agent.run(state)` runs until the model answers, and returns the answer.
3. `state.add_user_message(...)` adds the person's next message. `State.is_answered` becomes false.
4. The next `agent.run(state)` continues the same conversation.

Without step 3, `run` would stop at once, because `State.is_answered` is still true.

## Long conversations

The default loop compacts the context when it is more than 60% full. For your own loop, see
[Context size](context-size.md).

## Related

- [State: run the same State again](../concepts/state.md#run-the-same-state-again)
