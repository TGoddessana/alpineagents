# Survive crashes and restarts

[Save and resume](../learn/save-resume.md) gave the Agent a store. This page is about what the store holds when
things go wrong: when the Agent saves, what a loaded State looks like after a crash, and what happens if saving
fails.

## When the Agent saves

| When | What is saved |
| --- | --- |
| `run` starts and ends | Everything not saved yet |
| Before and after each `think` | Everything not saved yet |
| After each tool result | The result, right away, so a long or expensive tool's result is kept even if the process stops while other tools of the same turn still run |
| After `use_tools` and `compact` | Everything not saved yet |
| `store.save(state)` | Everything not saved yet. Use it for changes outside the Agent's steps. `await store.asave(state)` in async code |

A save writes only what the store does not have yet, and nothing when nothing changed. What the permission checks
recorded (refused calls, the person's answers) is saved before any tool of the turn runs.

## If the process stops

Ctrl+C, an exception or `sys.exit` end the run the usual way: calls without a result are closed, and the State is
saved. A process that is killed does not get that far. `store.load` replays the saved history, so the loaded State is
exactly what was saved. If it was saved in the middle of a turn, `load` closes that turn with entries of its own, and
the next save writes them:

- A tool call that was running gets a result with `outcome == "aborted"`: `(unknown: the run stopped before this
  result was saved, so the tool may or may not have run. ...)`. The tool is not run again. The model decides, for
  example by checking whether the email was sent.
- A model request that was running gets an `error` entry, `(process stopped while waiting for the model)`, and is
  taken back, as if `think` had failed. The next `run` asks again.
- A saved message that was waiting for the reply or the results goes into the messages, in order.

Nothing is lost: `state.history` has every entry that was saved, and a State loaded twice from the same save is the
same State. One thing is not saved: the `error` of history entries is `None` after loading, and the entry's text
still holds the exception type and message.

### Stop cleanly on SIGTERM

`docker stop`, Kubernetes and most process managers send SIGTERM before they kill a process. Python ends at once on
SIGTERM, without closing calls or saving. Turn it into `SystemExit`, and the run ends the way it does on an
exception:

```python
--8<-- "docs_src/sigterm.py"
```

1. `stop` raises `SystemExit` with the usual exit code for SIGTERM.
2. Calls still running are closed with `(aborted: SystemExit)`, and the State is saved before the process ends.
3. For `arun`, cancel the task instead: `loop.add_signal_handler(signal.SIGTERM, task.cancel)`.

## If saving fails

- During a run that is going well, the run raises the store's exception.
- If the run is already failing, it raises the original exception, with the failed save as a note:
  `saving the state also failed: OSError: [Errno 28] No space left on device`. `except RateLimitError:` still catches
  it. Ctrl+C or cancellation while saving is raised instead of the original.
- A failed save after one tool result does not stop the other tools. The save at the end of `use_tools` writes what is
  missing, or raises.

Nothing that failed to save is lost from the State: the next save writes it.

## Rules

- A State is saved in one store. Running it with an Agent whose store is a different one raises `ValueError`. Two
  `FileStore` objects for the same folder count as the same store.
- One process at a time runs a given State. Two processes running the same id at once are not supported.
- A new State with an id the store already has is refused before the first model request. Continue the saved one
  with `store.load(id)`, or pick another id.
- `FileStore` files are readable only by their owner, because history holds tool results and messages. Images in tool
  results are saved as base64, so a run with many screenshots makes large files.
- Saved States have a format version. `store.load` raises `ValueError` for a State saved in another format. There is
  no converter.

## List and delete

```python
for saved in store.list():  # a StateInfo for each, most recently updated first
    print(saved.id, saved.first_message, saved.updated_at, saved.stopped)

store.delete("review-pr-42")
```

## Write your own store

Subclass `Store` and implement `write` and `read`. Add `list` and `delete` if you want those calls. A store reached
asynchronously implements `awrite`, `aread` and so on instead. `load` is built on `read`.

```python
--8<-- "docs_src/own_store.py"
```

1. `entries` are history entries as dicts, each with `seq`, its position in history. Skip entries whose `seq` you
   already have: after a failed write, the Agent sends them again.
2. `info` is a small JSON dict with what `store.list()` shows and the format version under `"v"`. Keep it as it is,
   and give it back in `Record.info`. `StateInfo.from_info(state_id, info)` turns it into a `StateInfo` for `list`.
3. `create=True` is the first write of a new State. Refuse an id you already have with `ValueError`, and write all of
   it or nothing. Without `create`, an id you do not have raises `LookupError`.
4. Nothing else is saved: `load` rebuilds the State by replaying `entries`.

An Agent with an async-only store runs with `arun`. A sync store also works with `arun`: its methods run on a worker
thread.

## Related

- [Errors and interruptions](../concepts/errors.md)
- [Go back or try another path](undo-and-fork.md)
- [Store API](../api/store.md)
