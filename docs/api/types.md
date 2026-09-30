# Data types

The data objects that the other classes take and return.

::: alpineagents.Message

::: alpineagents.ToolCall

::: alpineagents.ToolSpec

::: alpineagents.Request

::: alpineagents.Reply

::: alpineagents.Usage

::: alpineagents.Price

## History entries

The items of `state.history`. `alpineagents.HistoryEntry` is their union: check `kind` (or use `isinstance` or
`match`) to know which one an entry is, and so what its `content` holds.

| Class | `kind` | `content` |
| --- | --- | --- |
| `MessageEntry` | `user`, `notice` | `str`. The first entry is the task |
| `ReplyEntry` | `reply` | `Reply` |
| `ToolResultEntry` | `tool_result` | The result sent to the model: `str`, or a tuple of `TextBlock` and `Image` |
| `NotRunEntry` | `denied`, `cancelled` | `str`, what the model was told |
| `ExchangeEntry` | `ask`, `human` | An object with `question` and `answer` |
| `ContextChangeEntry` | `context_change` | `ContextChange` |
| `ModelEventEntry` | `model_event` | `ModelEvent` |
| `ErrorEntry` | `error` | `str` such as `"TimeoutError: ..."` |

Every entry has `kind`, `content`, `turn` and `at`. The other fields are only on the classes where they mean
something: `call` on `ToolResultEntry`, `NotRunEntry` and `ErrorEntry`, for example.

::: alpineagents.MessageEntry

::: alpineagents.ReplyEntry

::: alpineagents.ToolResultEntry

::: alpineagents.NotRunEntry

::: alpineagents.ExchangeEntry

::: alpineagents.ContextChangeEntry

::: alpineagents.ModelEventEntry

::: alpineagents.ErrorEntry

## Other records

::: alpineagents.ContextChange

::: alpineagents.ModelEvent

::: alpineagents.ToolOutcome

::: alpineagents.ToolOutcomeKind

## Why a loop stopped

The values of `state.stopped`. `alpineagents.types.Stopped` is their union.

::: alpineagents.StoppedByUntil

::: alpineagents.StoppedByLimit

::: alpineagents.StoppedByFinish

::: alpineagents.StoppedByPermission

## Content blocks

The parts of a `Message`. `ToolCall` above is also one.

::: alpineagents.types.TextBlock

::: alpineagents.types.ToolResultBlock

::: alpineagents.Image
    options:
      filters: ["!^_"]

::: alpineagents.types.result_text

::: alpineagents.types.RawBlock
