# Handle unreliable models

Some models, often small or local ones, are less steady than others. A stream breaks halfway through a reply. The
model stops without calling a tool, in the middle of the task. It calls the same tool with the same arguments again
and again. Each recipe below handles one of these in your own code.

| The model | Recipe |
| --- | --- |
| Its stream breaks after text started arriving | [Retry a stream that dropped mid-reply](#retry-a-stream-that-dropped-mid-reply) |
| Replies with no tool call before the task is done | [Hand back a reply with no tool call](#hand-back-a-reply-with-no-tool-call) |
| Repeats the same tool calls | [Stop when the model repeats itself](#stop-when-the-model-repeats-itself) |

## Retry a stream that dropped mid-reply

A Model that wraps another Model can retry the reply:

```python
--8<-- "docs_src/retry_stream.py"
```

1. `RetryDroppedStream` passes every request to the wrapped Model, and copies its name, prices and settings.
2. `_Watch` wraps `on_text`, so it knows whether any text arrived before the failure.
3. A `ProviderError` after the stream started is retried, up to `attempts` in total. The request is sent again
   as it was, so the model starts the reply over. The partial text is not kept.
4. `AuthError` and `ContextTooLongError` are raised as they are. Sending the same request again cannot fix them.
5. Before each retry, `on_event(ModelEvent("retry", ...))` tells the Reporter, and the Agent records it in
   history as `model_event`.
6. `arespond` does the same for `athink` and `arun`.

The Reporter has already shown the partial text. A Reporter that keeps the text, for a UI that redraws it, throws
it away on the event:

```python
--8<-- "docs_src/retry_reporter.py"
```

`Terminal` cannot take back printed text. It prints the event on its own line, `model: stream dropped (...),
starting the reply over`, and the new reply below it.

The tokens of a dropped attempt are not in `state.usage`, because no `Reply` came back for it.

### Why the framework does not retry

The provider SDK retries a failed request (`retries=2` by default), but only before the stream starts: a
connection error, a 429, a 500. Once text is streaming, the SDK raises, and the Model raises `ProviderError`.
Retrying then has a cost: the text shown so far is thrown away, the tokens are paid again, and the new reply can
differ from the first. Whether that is worth it, and what the person sees, is up to your app.

## Hand back a reply with no tool call

Some models stop to say what they will do next ("I will now read the file") instead of doing it. That reply has no
tool call, so `waiting_for_user` is true and the loop stops ([Write your own loop](../learn/loop.md)). This loop hands it back:

```python
--8<-- "docs_src/nudge.py"
```

1. When the reply calls tools, the counter in `state.extra_data` goes back to 0.
2. When it calls none, `state.add_message(Message.notice(NUDGE))` asks the model to go on, up to `NUDGES` times in a row.
3. The notice comes after the reply, so `waiting_for_user` is false and the loop runs another turn.
4. After `NUDGES` notices, nothing is added. `waiting_for_user` is true and the loop stops.

The notice is in history as kind `notice`, and the model sees it as a user message that starts with `[notice] `.

A finished task costs up to `NUDGES` more replies, because the model gives its answer again after each notice.
Keep `NUDGES` low.

## Stop when the model repeats itself

An `until` function can look at the last replies and stop the loop when the model is stuck:

```python
--8<-- "docs_src/repeating.py"
```

1. `repeating` takes the tool calls of the last `REPEATS` `model_reply` entries of `state.history`.
2. It compares names and arguments, not call ids, which are new in every reply.
3. When all of them are the same, the loop stops before the next turn with
   `state.stopped == StoppedByUntil("repeating")`.

To give the model a chance first, add a notice such as `"You made the same calls 3 times. Try something else."`
from the loop, as in the recipe above, and stop only if it repeats again.

## Related

- [Stop a run](stop-conditions.md)
- [Check the work before finishing](verify.md): another loop that adds a notice
- [Use other models: write a Model](models.md#write-a-model)
- [Show progress and ask the person](progress.md): Reporters
