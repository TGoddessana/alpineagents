# Save and resume

Give the Agent a store, and it saves each State as the run goes. Another process can load the State later and
continue it, for example after a crash or a deploy, or when a person comes back to a conversation.

```python
--8<-- "docs_src/resume.py"
```

1. `FileStore(".agent-runs")` keeps each State in a folder of its own under `.agent-runs/`.
2. `Agent(store=store)` saves every State this Agent runs. Nothing else in the loop changes.
3. `store.list()` returns the saved States, most recently updated first. `store.load(id)` rebuilds one.
4. `store.save(state)` saves the person's message right away, so it is not lost if the process stops before the
   next `run`.

`python chat.py --resume` continues the last conversation where it stopped.

## Name a State

Each State has an `id`, random unless you give one. Give your own when you want to find the State again by name:

```python
state = State(id="review-pr-42", messages=[Message.user("Review pull request 42")])
```

An id has letters, digits, `_` and `-`, at most 128 characters. It names one conversation. A new State with an id the
store already has is refused before the first model request (`ValueError`): continue the saved one with
`store.load(id)`, or pick another id.

## When the Agent saves

| When | What is saved |
| --- | --- |
| `run` starts and ends | Everything not saved yet |
| Before and after each `think` | Everything not saved yet |
| After each tool result | The result, right away, so a long or expensive tool's result is kept even if the process stops while other tools of the same turn still run |
| After `use_tools` and `compact` | Everything not saved yet |
| `store.save(state)` | Everything not saved yet. Use it for changes outside the Agent's steps. `await store.asave(state)` in async code |

A save writes only what the store does not have yet, and nothing when nothing changed.

## If the process stops

Ctrl+C, an exception or `sys.exit` end the run the usual way: calls without a result are closed, and the State is
saved. A process that is killed does not get that far. History is the only thing a store keeps, and `store.load`
replays all of it, so the loaded State is exactly what was saved. If it was saved in the middle of a turn, `load`
closes that turn with entries of its own, and the next save writes them:

- A tool call that was running when the process stopped gets this result, with `outcome == "aborted"`:
  `(unknown: the run stopped before this result was saved, so the tool may or may not have run. Check whether it
  took effect before relying on it or running it again)`. The tool is not run again. The model decides, for
  example by checking whether the email was sent.
- A model request that was running gets an `error` entry, `(process stopped while waiting for the model)`, and is
  taken back, as if `think` had failed. The next `run` asks again.
- A saved message that was waiting for the reply or the results goes into the messages, in order.

Nothing is lost: `state.history` has every entry that was saved, and a State loaded twice from the same save is the
same State.

### Stop cleanly on SIGTERM

`docker stop`, Kubernetes and most process managers send SIGTERM before they kill a process. Python ends at once on
SIGTERM, without closing calls or saving. Turn it into `SystemExit`, and the run ends the way it does on an
exception:

```python
--8<-- "docs_src/sigterm.py"
```

Calls still running are closed with `(aborted: SystemExit)` and the State is saved. For `arun`, cancel the task
instead with `loop.add_signal_handler(signal.SIGTERM, task.cancel)`.

## What changes after loading

A loaded State equals the one that was saved: `store.load(id).snapshot() == state.snapshot()` after a save. Values
are already in their JSON form when they are recorded (see [State: history](../concepts/state.md#history)), so a
store has nothing to convert. One thing is not saved:

| Value | After loading |
| --- | --- |
| `error` of history entries | `None`. The entry's text still holds the exception type and message. `==` does not compare it |
| The Agent | Not saved. The Agent in your code runs the State, and the first `run` warns with `ResumeWarning` if it differs from the one that saved it. See [State](../concepts/state.md#a-state-saved-by-another-agent). Switching models in your own code never warns: see [Models](models.md#switch-models-in-one-conversation) |

## If saving fails

- If saving fails during a run that is going well, the run raises the store's exception.
- If the run is already failing, it raises the original exception, with the failed save as a note:
  `saving the state also failed: OSError: [Errno 28] No space left on device`. `except RateLimitError:` still catches
  it. Ctrl+C or cancellation while saving is raised instead of the original.
- A failed save after one tool result does not stop the other tools. The save at the end of `use_tools` writes
  what is missing, or raises.

Nothing that failed to save is lost from the State: the next save writes it.

## Rules

- A State is saved in one store. Running it with an Agent whose store is a different one raises `ValueError`. Two
  `FileStore` objects for the same folder count as the same store.
- One process at a time runs a given State. Two processes running the same id at once are not supported.
- `FileStore` files are readable only by their owner, because history holds tool results and messages.
- Images in tool results are saved in the files as base64, so a run with many screenshots makes large files.
- Saved States have a format version. `store.load` raises `ValueError` for a State saved in another format: one saved
  by alpineagents 0.4 (format 3) does not load in 0.5, and one saved by a newer alpineagents says to upgrade. There
  is no converter, so finish or export 0.4 conversations before you upgrade.

## Go back, or try something else

`state.snapshot()` returns a `StateSnapshot`: the State's value at that moment, frozen, so it never changes. Take one
before something you may want to undo, and `state.restore(snapshot)` goes back:

```python
--8<-- "docs_src/snapshot.py"
```

- `messages`, `turn`, `extra_data`, `finished`, `answer` and `stopped` go back to what they were. `usage` does not:
  the tokens were spent. `history` does not shrink either: it keeps everything and gets one `context_change` entry
  with `kind == "restore"`, so a store saves the going back like any other step.
- `restore` takes a snapshot taken from the same State. It raises `ValueError` for another State's snapshot, while
  tool calls are pending, and while the model is being waited on. Take snapshots between turns.
- A snapshot lives in memory. To keep one, keep what a store keeps, its history:
  `State(history=snapshot.history)` rebuilds the same value, in this process or another.

`state.fork()` makes a new State with the same history and a new id, bound to no store and to no running Agent. Use it
to try a different next step and keep the original:

```python
--8<-- "docs_src/fork.py"
```

- A fork is independent: running or changing it leaves `state` as it was.
- With `Agent(store=...)`, the fork is saved under its own id the first time the Agent runs it.
- `fork().snapshot() == state.snapshot()`: equal histories give equal snapshots.
- Fork between turns. `fork()` raises `ValueError` while the model is being waited on: fork before `think`, or after
  it returns (for example in `Reporter.on_think_end`).

## List and delete

```python
for saved in store.list():  # a StateInfo for each
    print(saved.id, saved.first_message, saved.updated_at, saved.stopped)

store.delete("review-pr-42")
```

## Your own store

Subclass `Store` and implement `write` and `read` (and `list`, `delete` if you want them), or their async versions
for a store reached asynchronously. `load` is built on `read`.

```python
from alpineagents import Store
from alpineagents.store import Record


class PostgresStore(Store):
    async def awrite(self, state_id, entries, info, *, create=False):
        ...  # insert entries whose seq is new, then replace the info, in one transaction

    async def aread(self, state_id):
        ...  # return Record(entries, info), or None
```

- `entries` are history entries as dicts, each with `seq`, its position in history. Skip entries whose `seq` you
  already have: after a failed write, the Agent sends them again.
- `info` is a small JSON dict with what `store.list()` shows (`first_message`, `created_at`, `updated_at`, `turn`,
  `stopped`, `finished`) and the format version under `"v"`. It is never `None`. Keep it as it is, and give it back in
  `Record.info`. `StateInfo.from_info(state_id, info)` turns it into a `StateInfo` for your `list`.
- Nothing else is saved: `load` rebuilds the State by replaying `entries`.
- `create=True` is the first write of a new State. Refuse an id you already have with `ValueError`, and write all of
  it or nothing.

An Agent with an async-only store runs with `arun`. A sync store also works with `arun`: its methods run on a worker
thread.

## Related

- [State](../concepts/state.md)
- [Errors and interruptions](../concepts/errors.md)
- [Store API](../api/store.md)
