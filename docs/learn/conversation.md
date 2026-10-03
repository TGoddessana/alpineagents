# Keep the conversation

`agent.run("text")` creates a State, runs it and throws it away. To look at the run afterwards, or to talk to the
agent again, create the State yourself.

## Create and keep a State

```python
--8<-- "docs_src/learn_conversation.py"
```

1. `State(messages=[Message.user(...)])` is one conversation. Its first message is the task.
2. `agent.run(state)` runs until the model answers and returns the answer. The State stays yours.
3. After a run, `state.answer` is the answer, `state.stopped` says why the run ended, `state.turn` counts the model's
   replies, and `state.usage` has the tokens and the requests.
4. `state.add_message(Message.user(...))` adds the next message. The last message is now the person's, so the
   next `agent.run(state)` continues the same conversation instead of stopping at once.

`print(state)` shows what happened, in order, one line for each entry of `state.history`. An entry is one thing that
happened: a request to the model, a reply, a stop. `context_change import` records the messages the State started with.

```text
[turn 0] context_change import: 1 messages
[turn 0] run_start: anthropic/claude-sonnet-5
[turn 1] model_request: anthropic/claude-sonnet-5
[turn 1] model_reply: Flask, Django and FastAPI.
[turn 1] stop: stopped by is_answered
done: stopped by is_answered (1 turn)
```

History is all a State is. `messages`, `turn` and `answer` are computed from it, and every change is a new entry.
Nothing is hidden. See [State and history](../concepts/state.md).

## A chat

A chat is the same two lines in a loop. This is the whole program:

```python
--8<-- "docs_src/chat.py"
```

The model sees the whole conversation each time. Without the `add_message` line, `run` would stop at once, because the
last message would still be the model's own answer. Stop the program with Ctrl+C.

## Messages from your code

`Message.user` is a message from the person. `Message.notice` is a message from your code, such as a test result.
It continues the example above:

```python
state.add_message(Message.notice("The tests failed"))
```

The model sees it with the prefix `[notice] `, so it can tell your code from the person. A notice added after the
model's answer makes the next run continue, like a user message.

## Send an image

A user message can carry images after its text:

```python
--8<-- "docs_src/user_image.py"
```

1. `Image.from_path` reads an image file. `Image(bytes)` takes bytes you already have.
2. `Message.user("", image)` sends an image without text.
3. Use a model that reads images. Nothing checks that for you, and a server that cannot read images answers with
   its own error.

## Next

[Ask before a tool runs](approval.md). To go deeper:

- [Save and resume](save-resume.md): keep a conversation when the process ends
- [Keep the context small](../guides/context-size.md): what to do when a conversation gets long
- [State and history](../concepts/state.md): every value and every history entry
