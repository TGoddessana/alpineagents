# Save and resume

Give the Agent a store, and it saves each State while the run goes. Another process can load the State later and
continue it, for example when a person comes back to a chat.

## A chat that survives

```python
--8<-- "docs_src/resume.py"
```

1. `FileStore(".agent-runs")` keeps each State in a folder of its own under `.agent-runs/`.
2. `Agent(store=store)` saves every State this Agent runs. Nothing else in your code changes.
3. `store.save(state)` saves the person's message right away, so it is not lost if the process stops before the next
   `run`.
4. `store.list()` returns the saved States, most recently updated first. `store.load(id)` rebuilds one.

Run `python chat.py`, say something, stop it with Ctrl+C, and run `python chat.py --resume`. The conversation goes on
where it stopped.

## Name a State

Each State has an `id`, random unless you give one. Give your own to find the State again by name:

```python
from alpineagents import Message, State

state = State(id="review-pr-42", messages=[Message.user("Review pull request 42")])
```

An id has letters, digits, `_` and `-`, at most 128 characters. A new State with an id the store already has is
refused: continue the saved one with `store.load(id)`, or pick another id.

## Look at what is saved

`store.list()` has one `StateInfo` for each saved State:

```python
from alpineagents import FileStore

store = FileStore(".agent-runs")
for saved in store.list():
    print(saved.id, saved.first_message, saved.updated_at, saved.stopped)
```

A store saves the history of the State, and nothing else. `store.load` rebuilds the State by replaying it, so a loaded
State is equal to the one that was saved: `store.load(id).snapshot() == state.snapshot()`.

The Agent is not saved. Your code makes the Agent that runs a loaded State. If it differs from the one that saved
it, the first `run` warns with `ResumeWarning`, and the run goes on.

## Next

[Test your agent](testing.md). To go deeper:

- [Survive crashes and restarts](../guides/production.md): when the Agent saves, a killed process, SIGTERM, a failed
  save, and your own store
- [Go back or try another path](../guides/undo-and-fork.md): snapshots, `restore` and `fork`
