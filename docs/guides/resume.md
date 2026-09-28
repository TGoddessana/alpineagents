# Save and resume

Give the Agent a store, and it saves each State as the run goes. Another process can load the State later and
continue it, for example after a crash or a deploy, or when a person comes back to a conversation.

```python
--8<-- "docs_src/resume.py"
```

1. `FileStore(".agent-runs")` keeps each State in a folder of its own under `.agent-runs/`.
2. `Agent(store=store)` saves every State this Agent runs. Nothing else in the loop changes.
3. `store.list()` returns the saved States, most recently updated first. `store.load(id)` rebuilds one.
4. `agent.save(state)` saves the person's message right away, so it is not lost if the process stops before the
   next `run`.

`python chat.py --resume` continues the last conversation where it stopped.

## Name a State

Each State has an `id`, random unless you give one. Give your own when you want to find the State again by name:

```python
state = State("Review pull request 42", id="review-pr-42")
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
| `agent.save(state)` | Everything not saved yet. Use it for changes outside the Agent's steps |

A save writes only what the store does not have yet, and nothing when nothing changed.

## If the process stops

Ctrl+C, an exception or `sys.exit` end the run the usual way: calls without a result are closed, and the State is
saved. A process that is killed does not get that far. The loaded State then continues from the last steps that
were saved:

- A tool call that was running when the process stopped gets this result:
  `(unknown: the run stopped before this result was saved, so the tool may or may not have run. Check whether it
  took effect before relying on it or running it again)`. The tool is not run again. The model decides, for
  example by checking whether the email was sent.
- A model request that was running is dropped, as if `think` had failed. The next `run` asks again.
- If a store failed halfway, the saved steps can end after a step that cannot be replayed, such as a compaction.
  The State then continues from the last complete save with a notice that the steps after it are missing. The
  notice names the tool calls that may or may not have run. `history` still has every saved entry.

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

A store keeps JSON, so a loaded State differs from the one that was saved in a few ways:

| Value | After loading |
| --- | --- |
| `state.data` | Must hold only JSON values: dicts with string keys, lists, strings, numbers, booleans, `None`. Anything else raises `TypeError` when saving, before the next model request. Tuples come back as lists |
| `finish(answer)` and `ask` answers | Pydantic models and dataclasses come back as dicts. Other values must be JSON values |
| `HistoryEntry.error` | `None`. The entry's text still holds the exception type and message |
| The Agent | Not saved. The Agent in your code runs the State, and the first `run` warns with `ResumeWarning` if it differs from the one that saved it. See [State](../concepts/state.md#a-state-saved-by-another-agent) |

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
- A State saved by a newer alpineagents may not load in an older one: `store.load` raises `ValueError` and says to
  upgrade. States saved by older versions load.

## List and delete

```python
for saved in store.list():
    print(saved.id, saved.task, saved.updated_at, saved.stopped_by)

store.delete("review-pr-42")
```

## Your own store

Subclass `Store` and implement `write` and `read` (and `list`, `delete` if you want them), or their async versions
for a store reached asynchronously. `load` is built on `read`.

```python
from alpineagents import Store
from alpineagents.store import Record


class PostgresStore(Store):
    async def awrite(self, state_id, entries, snapshot, *, create=False):
        ...  # insert entries whose seq is new, then replace the snapshot, in one transaction

    async def aread(self, state_id):
        ...  # return Record(entries, snapshot), or None
```

- `entries` are history entries as dicts, each with `seq`, its position in history. Skip entries whose `seq` you
  already have: after a failed write, the Agent sends them again.
- `snapshot` is a dict with the rest of the State, or `None` to keep the current one.
- `create=True` is the first write of a new State. Refuse an id you already have with `ValueError`, and write all of
  it or nothing.

An Agent with an async-only store runs with `arun`. A sync store also works with `arun`: its methods run on a worker
thread.

## Related

- [State](../concepts/state.md)
- [Errors and interruptions](../concepts/errors.md)
- [Store API](../api/store.md)
