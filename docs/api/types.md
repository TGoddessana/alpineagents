# Data types

The data objects that the other classes take and return.

::: alpineagents.Message

::: alpineagents.ToolCall

::: alpineagents.ToolSpec

::: alpineagents.Request

::: alpineagents.Reply

::: alpineagents.Usage

::: alpineagents.Price

::: alpineagents.HistoryEntry
    options:
      filters: ["!^_", "!^(substate)$"]

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
