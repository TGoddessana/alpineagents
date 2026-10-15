# Subagents

This page is the rules. To use subagents, see [Use subagents](../guides/subagents.md).

## The rule

**A run started inside a tool call is a subagent of the run that made the call.**

That is the whole idea. There is no `as_tool()` and no other API to learn. An Agent in `tools=` does it for you, and
a tool you write does it when it runs an Agent:

```python
from alpineagents import Agent, tool

researcher = Agent(model="claude-sonnet-5", name="researcher", description="Reads files and reports what they say")

# 1. The framework writes the tool: the model calls researcher(prompt=...)
agent = Agent(model="claude-sonnet-5", tools=[researcher])


# 2. You write the tool: the run inside it is a subagent too
@tool
def review(path: str, focus: str) -> str:
    """Review a file"""
    return researcher.run(f"Review {path}. Focus on {focus}.")
```

"Inside a tool call" is exact:

- It includes everything the tool does while it runs: `agent.run` and `agent.arun`, runs from tasks of
  `asyncio.gather`, from threads that alpineagents started for the tool, from a sync tool in an async run. The tool's
  context is passed to all of them.
- It includes a run started by a `Human` that the tool reaches through `agent.ask_human`.
- It does not include a thread you made yourself (`threading.Thread`, `ThreadPoolExecutor`), which does not get the
  context. Such a run has no parent.
- It does not include your own loop, a `Reporter` method or a permission's `check`: they are not tool calls.
- A run that a subagent's own loop starts is not a subagent of the tool call around it, because the loop is not in a
  tool call.

The Agent in `tools=` is shown to the model as a tool with the Agent's `name` and `description` and one parameter,
`prompt`. Calling it runs `agent.run(prompt)` on a new State that has that prompt as its only message. The parent's
conversation is not passed on. The tool is `parallel`, so several calls in one reply run at the same time, each with
its own State. Its [hints](tools.md) are not given, so a permission that reads hints, such as `AllowByReadOnly`, treats
it as one that may change things.

## The record

A subagent is a State of its own: its own id, its own history, its own answer. The two States point to each other by
id, never by object, so the links survive `store.load`.

| Where | What | Holds |
| --- | --- | --- |
| Subagent | `state.parent_id` | The id of the State whose tool call started it. `None` for a run no tool call started |
| Subagent | `state.parent_call_id` | The `ToolCall.id` of that call |
| Subagent | `RunStartEntry.parent_id`, `parent_call_id` | The same, as recorded by each run. `None` when the run has no parent |
| Parent | `ToolResultEntry.child_ids` | The ids of the subagents the call started, in the order they started. `()` for none |
| Parent | `ToolResultEntry.usage` | What those subagents (and theirs) used. See [Usage](#usage) |
| Parent | `state.child(child_id)` | The subagent's State |

- `parent_id` and `parent_call_id` are the first link in the history, so they are the same during the run and after
  `store.load`. A State that is run again, from another call, keeps its first parent. The new `RunStartEntry` records
  where it was started from this time.
- `child_ids` is filled when a subagent **starts**, so the entry holds it however the call ends: with a result, an
  error, an abort or Ctrl+C. A call a permission refused started no subagent.
- A result that arrives after its call was closed (a `late` result) repeats the call's `child_ids`.
- A grandchild belongs to the subagent: `state.child(a).child(b)`. `state.child` finds direct children only.

### Open a subagent

`state.child(child_id)` and `await state.achild(child_id)` (for a store that works only asynchronously) return the
subagent's State:

| Situation | Result |
| --- | --- |
| The subagent ran in this process | The very State it ran with. `state.child(a) is state.child(a)` |
| The State was loaded with `store.load`, or is a `fork()` of one | Loaded from the store. Not cached: each call loads a new State |
| `child_id` is not a direct child of this State | `KeyError` |
| It is a child, but the State has no store and no memory of it | `ValueError` that says to run with `Agent(store=...)` |
| The subagent has a store of its own, and the State is loaded from another | `LookupError` from the store: open it with that store's `load(child_id)` |

`fork()` and `restore()` keep what is needed to open the children. `State(history=...)` has neither a store nor a
memory of the children, so `child` raises `ValueError` there for an id in its history.

### Save

A subagent is saved under its own id, at the same points as any run, while it runs. It goes to the Agent's `store=`
if it has one, otherwise to the store of the run that started it. There is no way to say "save this one nowhere" when
the run above has a store.

`StateInfo.parent_id` and the `parent_id` key of the info dict hold the same value as `state.parent_id`, so a listing
can tell subagents from the runs you started: `[i for i in store.list() if i.parent_id is None]`. A State saved in
format 4 (before subagents) lists with `parent_id=None`. The format version is 5, and 4 is still read. See
[Survive crashes and restarts](../guides/production.md#rules).

## What a subagent takes from the run above

| Setting | A subagent... |
| --- | --- |
| `model`, `system`, `tools`, `loop`, `name`, `description` | Always uses its Agent's own |
| `reporter` | Uses its own if the Agent was given one (`None` too, which is silent). Otherwise the reporter of the run above |
| `human` | Likewise: its own if given (`None` means nobody can answer), otherwise the human of the run above |
| `store` | Uses its own if given, otherwise the store of the run above |
| `permissions` | Is checked by the permissions of the run. See [Permissions](#permissions) |

- A Reporter has no new methods for subagents. It gets its usual events with the **subagent's State**, and tells them
  apart with `state.parent_id`. `on_run_start` and `on_run_end` come once for every run.
- The default `Terminal` indents a subagent's lines one level deeper than the call that started it, and starts them
  with the Agent's name (`[researcher turn 1] thinking`, `researcher done: ...`). It does not show a subagent's
  streamed text.
- A question a subagent asks with `agent.ask_human` is recorded in the subagent's history, and goes to the person the
  subagent's `human` points to.
- `agent.reporter`, `agent.human` and `agent.store` always show the Agent's own settings, not what it takes from the
  run above. A different Agent that you call on a State of a running run (`other.think(state)`) uses its own settings.

## Permissions

**`permissions=` applies to every tool call in the run, including the calls inside its subagents.** A call of a
subagent is checked by:

1. The `DenyPermission`s of every run above it, from the top. A subagent cannot widen them.
2. Then, if the subagent's Agent has `permissions=None`, the whole list of the run above it. If it has a list of its
   own, that list.

The order inside the checks is the usual one: every `DenyPermission` first, then the rest in list order, and a call
nobody allows is refused. See [How permissions decide](../guides/permissions.md#how-permissions-decide).

- A subagent with its own list is not checked by the allow and decide permissions of the run above. A parent's
  `[AllowByReadOnly()]` means "refuse the rest" for the parent's own calls, and does not mean that for such a subagent.
  Limits that must hold everywhere go in `DenyBy...` permissions.
- `permissions=[]` on a subagent refuses all its calls that no deny rule above refused, with a `PermissionWarning`.
- A permission asks the `human` of the Agent that **wrote** it. A `DecideByHuman()` the lead wrote asks the lead's
  person about a subagent's call. A subagent's own `DecideByHuman()` asks the subagent's `human`, which is the lead's
  unless the subagent has one.
- The checks before a run starts (`NoHumanError` for a `DecideByHuman()` with nobody to ask, async-only permissions in
  a sync run) cover the inherited list too.
- A refusal is recorded in the subagent's history. `Denied(..., stop=True)` stops only the subagent.

## How a subagent ends

What happens in a subagent ends in the subagent's State. The parent gets a result and goes on. Only Ctrl+C and
cancellation, which belong to the whole process, stop the parent too.

For an Agent in `tools=`, the result is `run`'s return value, converted as any tool result is (see
[Return values](tools.md#return-values)). The tool adds a note for an ending that is not a plain finish:

| The subagent | The parent's model gets | Outcome of the call |
| --- | --- | --- |
| Hands the turn back, or `state.finish(answer)`, or a custom loop returns | The answer | `done` |
| Reaches its loop's `limit` | The answer, then a line `(researcher stopped after 30 turns before finishing)` | `done` |
| A permission denies a call with `stop=True` | `(researcher was stopped by the user: reason)` | `done` |
| An `Exception` leaves its run after it started | `(researcher failed: TimeoutError: took too long)` | `error` |
| An `Exception` from the checks before the run starts | Nothing: it stops the parent's run | `aborted` |
| `KeyboardInterrupt`, cancellation (`BaseException`) | `(interrupted by user)`, and the parent's run stops too | `interrupted` |

- The permission stop is `done`, because the call did run: `denied` means a call that did not run.
- The exception becomes an error result like a `ToolError`: the entry's `error` holds it and its `__cause__` the
  original exception, and the subagent's own history has the details.
- The checks before the run starts (`NoHumanError` for a `DecideByHuman()` with nobody to ask, `TypeError` for an
  async-only permission, store or human in a sync run, an invalid State) are a mistake in the setup, not something that
  happened to the subagent. Their exception is not turned into a result: it propagates as the exception of any tool
  does, and the parent's run stops with it. The subagent never started, so `child_ids` is `()`.
- `state.finish(answer)` in the subagent ends only the subagent. The parent goes on.
- To tell these apart in code, read the call's `outcome` and `state.child(child_id).stopped`, which is
  `StoppedByLimit`, `StoppedByPermission` and so on.

A `@tool` that runs an Agent itself gets none of these conversions. What it returns is converted as usual, and an
exception follows the [rules of tools](tools.md#when-a-call-goes-wrong): a `ToolError` is an error result, anything
else stops the run. The links, permissions, human, reporter and usage work the same.

### Known limit: Ctrl+C in a sync run

In a sync run, a subagent runs in a worker thread, and a thread cannot be stopped from outside. Ctrl+C stops the
parent's run and its call ends `interrupted`, but a subagent that is still running keeps going until it ends. Its
result and usage are then recorded once, as a `late` result of the call (see [Usage](#usage)). Use `arun`: cancelling
the task of the parent cancels its subagents with it.

## Usage

`state.usage` of the parent includes the usage of all its descendants: tokens, requests and cost.

- A call's subagents' usage is carried by the `ToolResultEntry` that closes the call (`entry.usage`), and added to
  `state.usage` when the entry is recorded. So while a subagent runs, the parent does not show its usage yet; it
  appears when the call closes. A subagent that is still running then is counted up to that moment.
- `entry.usage` is "not counted yet": a subagent counted when the call closed is not counted again. A `late` result
  holds only what the subagents used after the call closed.
- `restore` does not take usage back, as for any usage. Tokens that were spent stay counted.
- `cost` is computed per request with the price of the Model that made it, so it is right for a tree of different
  models. It is `None` when any model has no price: give a model you do not pay per token for `Price(input=0, output=0)`
  ([Use other models](../guides/models.md#models-with-no-price)). Token counts add up over all models.
- There is no per-model breakdown. Count `ModelReplyEntry`s, whose `reply.model` names the model, in the parent's
  history and in each `state.child(id)`.
- To stop the lead when the tree has spent too much, check `state.usage` in your own loop between turns. A subagent
  that is still running is not in it until its call closes.
