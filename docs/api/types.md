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
`match`) to know which one an entry is, and so what its `content` holds. There are 11 classes. `Model...` entries are
about the model, `ToolResultEntry` about a tool call.

| Class | `kind` | `content` |
| --- | --- | --- |
| `MessageEntry` | `user`, `notice` | `str`, or a tuple of `TextBlock` and `Image` when the message has images |
| `ModelRequestEntry` | `model_request` | `str`, the model name that was asked (`provider/name`) |
| `ModelReplyEntry` | `model_reply` | `Reply` |
| `ModelEventEntry` | `model_event` | `ModelEvent` |
| `ToolResultEntry` | `tool_result` | The result sent to the model: `str`, or a tuple of `TextBlock` and `Image`. `outcome` says how the call ended, including `denied` and `cancelled` |
| `ExchangeEntry` | `ask`, `human` | An object with `question` and `answer`. An `ask` entry also has `usage` |
| `ContextChangeEntry` | `context_change` | `ContextChange`: `compact`, `clear_tool_results`, `import` or `restore` |
| `RunStartEntry` | `run_start` | `AgentInfo` |
| `StopEntry` | `stop` | `Stopped`, or `None` for a stop that was cleared |
| `ExtraDataEntry` | `extra_data` | The keys that changed with their new values. `removed` lists the deleted keys |
| `ErrorEntry` | `error` | `str` such as `"TimeoutError: ..."` |

Every entry has `kind`, `content`, `turn` and `at`. The other fields are only on the classes where they mean
something: `call` on `ToolResultEntry` and `ErrorEntry`, for example.

Values in history are read-only. A `ToolCall`'s `args`, a `ModelEvent`'s `data`, an `ExtraDataEntry`'s `content` and
an `Exchange`'s `answer` are frozen dicts and lists: they work with indexing, `json.dumps` and `==`, and a change
raises `TypeError`. `dict(value)` makes an editable copy.

::: alpineagents.MessageEntry

::: alpineagents.ModelRequestEntry

::: alpineagents.ModelReplyEntry

::: alpineagents.ModelEventEntry

::: alpineagents.ToolResultEntry

::: alpineagents.ExchangeEntry

::: alpineagents.ContextChangeEntry

::: alpineagents.RunStartEntry

::: alpineagents.StopEntry

::: alpineagents.ExtraDataEntry

::: alpineagents.ErrorEntry

## Other records

::: alpineagents.ContextChange

::: alpineagents.types.Exchange

::: alpineagents.AgentInfo

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

::: alpineagents.types.format_call

::: alpineagents.types.RawBlock
