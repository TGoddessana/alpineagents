# Chat

The person and the agent talk in turns. The model sees the whole conversation.

```python
--8<-- "docs_src/chat.py"
```

1. One State holds the whole conversation.
2. `agent.run(state)` runs until the model answers, and returns the answer.
3. `state.add_message(Message.user(...))` adds the person's next message. The last message is now the person's, so the
   default loop does not stop at once.
4. The next `agent.run(state)` continues the same conversation.

Without step 3, `run` would stop at once, because the last message is still the model's answer.

## Send an image

A user message can carry images after its text. Create the `Image` from a file or from bytes, and pass it to
`Message.user`:

```python
--8<-- "docs_src/user_image.py"
```

- The image goes to the model with the message, in the order you gave. `Message.user("", image)` sends an image
  without text.
- `Anthropic` and `OpenAICompatible` both send it. Use a model that reads images: nothing checks that for you, and a
  server that cannot read images answers with its own error.
- Each image counts as 1,600 tokens when `agent.context_used(state)` estimates the context.
- A [store](resume.md) saves it with the State, as base64, like an image in a tool result.
- An image a tool returns is a different thing: see [Tools: images](../concepts/tools.md#images).

## Long conversations

The default loop compacts the context when it is more than 60% full. For your own loop, see
[Context size](context-size.md).

## Continue later

To continue a conversation in another process, give the Agent a store. See [Save and resume](resume.md).

## Related

- [State: run the same State again](../concepts/state.md#run-the-same-state-again)
