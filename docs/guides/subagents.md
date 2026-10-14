# Use subagents

A subagent is an Agent that another Agent's model calls like a tool. It runs its own loop with its own tools, then
hands back its answer. Use one to keep a side task, such as reading many files, out of the main conversation, or to
give a task different tools, a different system prompt or a cheaper model.

The rule, in one sentence: **a run started inside a tool call is a subagent of the run that made the call.** The rest
of this page uses it. The precise rules are in [Subagents](../concepts/subagents.md).

## Give an Agent to `tools=`

Put an Agent in `tools=` next to the functions:

```python
--8<-- "docs_src/subagent.py"
```

1. A subagent needs a `name` and a `description`: they are the name and the description of its tool. Without them,
   creating the Agent that lists it raises `ValueError`.
2. The model of `lead` sees one tool, `researcher`, that takes a `prompt`. Calling it is the same as running
   `researcher.run(prompt)`. The researcher sees that prompt alone, not the conversation of `lead`.
3. The tool result is what `researcher.run` returned: its answer.

- The prompt is all the subagent knows, so the model has to write the background into it. Say that in the
  `description`, or in the `system` prompt of the lead.
- Several subagent calls in one reply run at the same time, each with its own State.
- The tool does not say it is read-only, so `AllowByReadOnly()` does not allow a call to it. Allow it by name, as in
  [Write permission rules](permissions.md#permissions-in-subagents).
- The terminal shows the subagent's lines indented under the call that started it:

```text
[turn 1] thinking
  tool researcher(prompt="Read pyproject.toml and list the supported Python versions")
  researcher started
  [researcher turn 1] thinking
    tool read_file(path="pyproject.toml")
    done 26B
  [researcher turn 2] thinking
  researcher done: stopped by waiting_for_user (2 turns)
  done 21B
[turn 2] thinking
It supports Python 3.11 and later.
done: stopped by waiting_for_user (2 turns)
```

## What the model is told when a subagent does not finish

A subagent that ends in the usual way hands back its answer. When it does not, the model of the lead is told how, and
the lead goes on:

| The subagent | The lead's model gets | The call's outcome |
| --- | --- | --- |
| Hands back an answer, or calls `finish(answer)` | The answer | `done` |
| Reaches its loop's `limit` | The answer so far, and `(researcher stopped after 30 turns before finishing)` | `done` |
| Is stopped by a permission that denied a call with `stop=True` | `(researcher was stopped by the user: reason)` | `done` |
| Raises an exception | `(researcher failed: TimeoutError: took too long)` | `error` |

Ctrl+C and task cancellation are not turned into a result: they stop the lead too. Code that must tell the cases apart
reads the outcome of the call and the subagent's State ([below](#look-inside-after-the-run)).

## Change what the model passes

`prompt` is not always the right input. To give the model your own parameters, write an ordinary `@tool` and run the
Agent inside it. Nothing else is needed: a run started inside a tool call is linked to that call.

```python
--8<-- "docs_src/subagent_tool.py"
```

1. `reviewer` is not in `tools=`. The tool starts it.
2. The model passes a list of files and a focus. The tool builds each reviewer's prompt.
3. `asyncio.gather` runs the reviews at the same time. The reviews are subagents of the same call, so
   `entry.child_ids` lists both.
4. The tool returns a list, which the model receives as JSON.

For sync code, call `reviewer.run(...)` in a `@tool` function instead.

- **Exceptions follow the rules of tools.** The summary table above is only for an Agent in `tools=`. If `reviewer.run`
  raises inside your tool, that is an exception from a tool: it stops the run unless you catch it and raise
  `ToolError`, or name it in `exception_handler=`. See [Let the model handle tool failures](tool-failures.md).
- **Threads you make yourself lose the link.** The link travels with the call's context. Tasks of `asyncio.gather` and
  the threads alpineagents starts keep it. A thread you start, such as a `ThreadPoolExecutor`, does not, so a run in
  it is a run of its own: not a subagent, with no parent and without the permissions, the person and the reporter of
  the lead. Use `asyncio.gather` with `arun`, or run the agents one after another.
- A run started anywhere else is not a subagent either: in your own loop, a Reporter or a permission.

## Show the tree

Every Reporter event of a subagent comes with the **subagent's State**, and `state.parent_id` is the id of the parent
State. `None` means the top-level run. A Reporter that wants a tree keeps one number per State:

```python
--8<-- "docs_src/subagent_reporter.py"
```

1. `on_run_start` and `on_run_end` come for each run, subagents included.
2. The parent's run started before its subagent's, so the parent's depth is known.
3. `researcher` is the Agent from the first example. It has no `reporter=`, so it reports to the reporter of the lead.
   Give it `reporter=None` to keep it silent, or a Reporter of its own.

```text
researcher({'prompt': 'Read pyproject.toml and list the supported Python versions'})
  read_file({'path': 'pyproject.toml'})
  done: stopped by waiting_for_user
done: stopped by waiting_for_user
```

Events can come from several threads when subagents run at the same time, so make the Reporter thread-safe.

## Look inside after the run

The lead's State records which tool call started which subagents, and the subagent's State is a State like any other:

```python
--8<-- "docs_src/subagent_inspect.py"
```

1. `entry.child_ids` holds the ids of the subagents the call started, in the order they started. A call that started
   none has an empty tuple.
2. `state.child(child_id)` is the subagent's State: its `answer`, `stopped`, `history` and `usage`. For a grandchild,
   chain: `state.child(a).child(b)`. An id that is not a direct child raises `KeyError`.
3. `child.parent_id` and `child.parent_call_id` point back to the lead's State and to the `ToolCall.id` of the call.
4. `state.usage` of the lead includes its subagents: tokens, requests and cost.

While the process that ran the subagents is alive, `state.child(id)` returns the very State the subagent ran with. A
State from `store.load` opens the subagents from the store instead, which needs the subagents to be saved in it: they
are, in the store of the run that started them, unless their Agent has a store of its own. The next section has the
details. Use `await state.achild(id)` with a store that works only asynchronously.

### Save and list

An Agent with `store=` saves its subagents too, each under its own id. After a restart, `state.child` opens them from
the store, and `store.list()` lists them beside the runs. Keep the ones without a parent to list only the runs you
started:

```python
--8<-- "docs_src/subagent_store.py"
```

`store.load(child_id)` also gives a subagent's State alone. It is a normal State: `agent.run(child)` continues it.

### The cost

`state.usage.cost` is the cost of the whole tree, because each request is priced with its own model's price. It is
`None` if any model in the tree has no price. A model you pay for with a subscription, or run locally, costs nothing
extra. Say so with `Price(input=0, output=0)`, so the total stays known
([Use other models](models.md#models-with-no-price)).

## Test with FakeModel

Give each Agent its own `FakeModel`. The lead's tool call is `tool_call("researcher", prompt=...)`:

```python
--8<-- "docs_src/test_subagent.py"
```

1. `researcher.copy(model=...)` is the researcher with a fake model. `lead.copy(tools=[...])` lists it in place of the
   original.
2. The lead's first reply calls the researcher. The researcher's fake model answers, and the lead's second reply is the
   final answer.
3. `state.child(...)` is the researcher's State. It shows what the researcher saw.
4. `remaining == 0` checks that every prepared reply was used.

## Related

- [Subagents](../concepts/subagents.md): the rules for links, permissions, the end of a subagent and usage
- [Write permission rules](permissions.md#permissions-in-subagents): what `permissions=` does to a subagent's calls
- [Show progress and ask the person](progress.md): Reporter and Human
- [Use async](async.md)
