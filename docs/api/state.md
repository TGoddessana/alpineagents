# State

Concepts: [State](../concepts/state.md).

::: alpineagents.State
    options:
      filters: ["!^_", "!^(parent|root|depth)$"]

## StateSnapshot

What `state.snapshot()` returns: the frozen value of a State at one moment. Pass it to `state.restore(snapshot)`.

::: alpineagents.StateSnapshot
    options:
      filters: ["!^_"]
      show_signature: false
