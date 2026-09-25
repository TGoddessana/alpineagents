# Plan before acting

The model writes a plan in the first turn, without tools. From the second turn it carries out the plan.

```python
--8<-- "docs_src/plan_first.py"
```

1. In the first turn, `agent.think(state, tools=[])` shows the model no tools, so the reply is text: the plan.
2. A reply without tool calls makes `State.is_answered` true. `state.add_notice(...)` adds a message after it, so the
   loop continues.
3. From the second turn, `agent.think(state)` shows every tool.

## Other tool sets per turn

`think(state, tools=[...])` accepts any subset of the Agent's tools. For example, only reading tools for the first
five turns. Inside the loop body above:

```python
tools = [read_file, list_files] if state.turn < 5 else None
agent.think(state, tools=tools)
```

`tools=None` shows all of the Agent's tools. A tool the Agent does not have raises `ValueError`.

## Related

- [Agent: think and use_tools](../concepts/agent.md#think-and-use_tools)
- [Check the work before finishing](verify.md): the same notice technique
