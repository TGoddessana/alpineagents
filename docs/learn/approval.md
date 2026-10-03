# Ask before a tool runs

Some tool calls change something. A permission decides, before a call runs, whether it may run. Here the person
approves every call that writes, and reads run freely.

## Yes or no

```python
--8<-- "docs_src/approval_stop.py"
```

1. `@tool(read_only=True)` says `read_file` changes nothing. `open_world=False` says a tool stays inside its own
   domain, with no web and no shell. The model does not see these hints. Your permissions do.
2. `permissions=[...]` is a list of rules. Before any tool of a turn runs, they decide about every call the model
   asked for.
3. `AllowByReadOnly()` allows a call when the tool says it is read-only. Its calls run without asking.
4. `DecideByHuman()` asks the person about every other call. A call nobody decides about is refused.
5. The `while` loop is for the answer `no`. It is explained below.

The terminal shows the question, and the person types `yes` or `no`:

```text
[turn 1] thinking
Run write_file(path="README.md", content="# Hi")? (yes/no)
  denied write_file: The user declined this call. Wait for their next message.
done: stopped by permission DecideByHuman() (1 turn)
What should it do instead? Use README.txt instead
[turn 2] thinking
Run write_file(path="README.txt", content="Hi")? (yes/no)
  tool write_file(path="README.txt", content="Hi")
  done 6B
[turn 3] thinking
Wrote README.txt
done: stopped by waiting_for_user (3 turns)
```

| The person says | What happens |
| --- | --- |
| `yes` | The tool runs, and the run goes on |
| `no` | The tool does not run. The model is told the person declined, and the run stops |

## Continue after no

`no` stops the run. `state.stopped` is `StoppedByPermission(call, "DecideByHuman()")`, and `run` returns normally,
without an exception. The State is not finished. The loop in the example asks the person what to do instead, adds
the answer as a message and runs again. The model then sees that its call was declined, and the new message.

The other calls the model asked for in the same turn are cancelled. They do not run, and nobody is asked about
them. The model gets `(not run: the user stopped this turn)` for each, and `state.history` records them as
`tool_result` entries with outcome `cancelled`.

## Next

[Save and resume](save-resume.md). To go deeper:

- [Write permission rules](../guides/permissions.md): deny by name, decide by what a call does, MCP tools, your own
  permission, and why permissions are not a security boundary
- [Show progress and ask the person](../guides/progress.md): ask somewhere other than the terminal
