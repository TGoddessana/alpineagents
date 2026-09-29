# Ask before a tool runs

Permissions decide, before a tool call runs, whether it may run. The person approves the calls that change
something, rules in your code refuse or allow the rest, and a refused call does not run: the model is told why.

## Yes or no

```python
--8<-- "docs_src/approval.py"
```

- `permissions=` takes a list of permissions. Before any tool of a turn runs, they decide about every call the model
  asked for.
- `AllowByReadOnly()` allows a call when the tool says it is read-only. `read_file` says `read_only=True`, so its calls
  run without asking.
- `DecideByHuman()` asks the person about every other call: `Run write_file(path="README.md", content="...")?`. The
  answer `yes` runs it. The question and the answer are recorded in `state.history`, like `agent.ask_human`.
- The answer `no` refuses the call and stops the run, so the person can say what to do instead (see
  [When the person says no](#when-the-person-says-no)). The other calls of the same turn do not run either.

The rule reads what the tool says about itself, not a list of names, so a new tool follows it without changing the
Agent. See [Tools: describe what a tool does](../concepts/tools.md#describe-what-a-tool-does).

The default `human` is the terminal, which shows `yes/no` after the question and asks again on any other answer.
`DecideByHuman(human=...)` asks someone else, for example an approvals channel. A `DecideByHuman()` without its own
`human` on an Agent with `human=None` raises `NoHumanError` when the run starts, before the first model call.

## How permissions decide

For each call of the turn, in the order the model asked for them:

1. Every `DenyPermission` in the list, in list order, wherever it is in the list. The first one that refuses the call
   decides. No allow rule can let a refused call through.
2. Otherwise the `AllowPermission`s and `DecidePermission`s, in list order. The first one with an opinion decides.
3. Nobody decided: the call is refused.

So `[AllowByDefault(), DenyByName(["delete_*"])]` allows every call except those to `delete_*` tools.

Someone must allow a call. A call no permission decides about is refused with `No permission allowed this call.`, and
Python shows a `PermissionWarning` that names the tool. To allow every call no permission refused, put
`AllowByDefault()` at the end of the list. This holds for any list, so `permissions=[]` refuses every call.

Leaving `permissions=` out (or `None`) turns the checks off: every call runs. To turn them off on a copy of an Agent,
use `agent.copy(permissions=None)`.

Calls to unknown tools and calls whose arguments are not valid JSON are
[input errors](../concepts/tools.md#when-a-call-goes-wrong) before permissions are asked. Every other call is checked
before any tool of the turn runs, and then the allowed calls run together, as in
[Several calls in one reply](../concepts/tools.md#several-calls-in-one-reply).

A refused call:

- does not run, and the model gets the reason as the call's error result;
- is recorded in `state.history` as a `denied` entry with the reason and the call, and leaves `state.pending_calls`;
- gets `on_tool_end` on the Reporter (without `on_tool_start`), with `outcome.kind == "denied"` and
  `outcome.decided_by` set to the `repr()` of the permission that refused it, such as `"DecideByHuman()"`
  (`None` when nobody decided).

With a [store](resume.md), what the checks recorded (refused calls, the person's answers) is saved before any tool of
the turn runs.

## Built-in permissions

They live in `alpineagents.permissions`:

| Permission | Kind | Decides |
| --- | --- | --- |
| `DenyByName(["delete_*"], reason=None)` | Deny | Refuses calls to matching tools, with `reason` or `"{name} is not allowed."` |
| `AllowByName(["read_file", "github__get_*"])` | Allow | Allows calls to matching tools |
| `AllowByReadOnly(trust_mcp=False)` | Allow | Allows a call the tool says is read-only: `tool.hints_for(call.args).read_only` |
| `DecideByHuman(human=None)` | Decide | Asks the person: `yes` allows, `no` refuses and stops |
| `AllowByDefault()` | Allow | Allows every call. Put it last |

- The names are `fnmatch` patterns on the name the model calls: `*` matches anything, so `"delete_*"` matches
  `delete_file` and `delete_branch`.
- Put `DecideByHuman()` after the permissions that decide without asking, so the person is asked only about the calls
  they left open.

## When the person says no

`no` stops the run with `state.stopped == StoppedByPermission(call, "DecideByHuman()")`. The model was told
`The user declined this call. Wait for their next message.` Add the person's next message and run again:

```python
--8<-- "docs_src/approval_stop.py"
```

- The run ends normally, without an exception, and the State is not finished, so `run` continues it.
- The other calls of the same turn are cancelled: they do not run and nobody is asked about them. The model gets
  `(not run: the user stopped this turn)` for each, `state.history` records them as `cancelled`, and the Reporter gets
  `outcome.kind == "cancelled"`.
- Any permission can stop a run this way by returning `Denied(reason, stop=True)`.

`state.stopped` is set as soon as the person says `no`, before the tools of the turn would run. A `@loop` stops by
itself before its next turn. A loop written without `@loop` checks it before each turn:

```python
def my_loop(agent: Agent, state: State):
    while state.stopped is None and not state.is_answered():
        agent.think(state)
        if state.wants_tools():
            agent.use_tools(state)
```

## Decide by what a call does

The hints of `@tool` hold for every call, so a shell tool is never read-only: it can run `rm`. With `hints_for=`,
the tool says what one call does, and a permission reads that:

```python
--8<-- "docs_src/approval_hints.py"
```

- `bash_hints` says `ls` and `cat` change nothing, and `mkdir` and `touch` delete and overwrite nothing. For
  anything else, including commands joined with `;` or `|`, it returns `None`, and the tool's own hints (the worst
  case) apply.
- `AllowNotDestructive` allows the calls whose hints say `destructive=False`. `DecideByHuman()` asks about the rest,
  such as `rm -rf build`.
- `AllowByReadOnly()` reads `hints_for` the same way, so with it `ls` runs without asking and `mkdir` asks.
- Keep hints about facts ("this call deletes nothing") and decisions in permissions ("run what deletes nothing").

See [Tools: hints for one call](../concepts/tools.md#hints-for-one-call).

## MCP tools

An [MCP server](mcp.md)'s tools are named `{server}__{tool}`, where `server` is the `name=` you gave `MCP(...)`. The
names are yours, so `DenyByName` and `AllowByName` work on them: `DenyByName(["github__delete_*"])`,
`AllowByName(["github__get_*"])`.

The hints of an MCP tool come from the server, which describes its own tools. `AllowByReadOnly()` does not believe
them: it has no opinion about an MCP tool, and the next permission decides. `AllowByReadOnly(trust_mcp=True)` believes
the server's `readOnlyHint`. Do this only for a server you trust as much as your own code. A permission of your own
that reads hints checks `isinstance(tool, MCPTool)` the same way, as `AllowNotDestructive` above does.

## Write your own permission

Subclass one of three kinds, and implement `check(self, state, call, tool)`:

| Kind | `check` returns |
| --- | --- |
| `DenyPermission` | `Denied(reason)` or `None`. Asked first, wherever it is in the list |
| `AllowPermission` | `Allowed()` or `None` |
| `DecidePermission` | `Allowed()`, `Denied(reason)` or `None` |

`None` means no opinion: the next permission decides. This one refuses paths outside a folder:

```python
--8<-- "docs_src/approval_deny.py"
```

- `call.args` are the model's arguments before they are checked against the tool's parameters: any value may be
  missing or have the wrong type. Read them with `.get(...)` and `isinstance`.
- `tool` is the `Tool` the call would run. `call.name` is the name the model called.
- `repr()` of the permission is what `outcome.decided_by` and `StoppedByPermission.permission` show. Give a
  permission with settings a `__repr__` that shows them; without one, it is the class name, `DenyOutsideFolder()`.
- Creating a permission raises `TypeError` when its class subclasses `Permission` directly, or implements neither
  `check` nor `acheck`. Returning a verdict its kind may not give, such as `Allowed()` from a `DenyPermission`, raises
  `TypeError` during the run.

### Add "always"

Let the person approve a tool once for the rest of the task:

```python
--8<-- "docs_src/approval_always.py"
```

- `Denied(reason, stop=True)` also stops the run, like `DecideByHuman`'s `no`.
- `state.root.data["always"]` keeps the approved tool names across turns and runs, and a [store](resume.md) saves
  it. The model never sees `state.data`.
- `self.human.ask(...)` asks directly, so the question is not recorded in `state.history` the way `DecideByHuman`
  records it.

### When a check fails

- Raise `ToolError(message)` when the check itself fails in a way the model should hear about, for example the
  approval service did not answer. The call does not run, the model gets `message` as its error result, and the run
  continues. The outcome is `"denied"`, with the `ToolError` as `outcome.error`.
- Any other exception propagates out of `use_tools` and `run`, like one raised by a tool. The call does not run.
- A failure never counts as allowed.

### Async permissions

A permission that awaits something, such as an approval service over HTTP, implements `async def acheck(self, state,
call, tool)` instead of `check`, with the same verdicts. `arun` and `ause_tools` await it. A sync `run` cannot, so it
raises `TypeError` when the run starts, before the first model call.

A permission with only `check` works with `arun` too: `acheck` runs `check` on a worker thread, so it does not block
the event loop. `DecideByHuman` implements both, and asks the Human with `aask` in `arun`. A `DecideByHuman`
subclass that overrides only `acheck` counts as async-only: a sync `run` raises `TypeError` rather than ask the Human.

## Not a security boundary

Permissions keep an honest model from doing what you did not want. They do not stop a determined one:

- `bash_hints` above says `cat` is read-only, and it is, but `cat ~/.ssh/id_rsa` still reads a secret.
- A command can do more than its first word says, through an alias, a script it runs or a program's own options.
- A tool can do things its hints do not say.

Treat permissions as a way to ask the person at the right moments, not as a way to contain the agent. When a tool
runs commands or code the model wrote, run the agent in a sandbox: a container or a virtual machine with only the
files, network access and credentials the task needs.

## Related

- [Progress and questions](progress.md): answer questions from somewhere other than the terminal
- [Stop conditions](stop-conditions.md): every way a run stops
- [Testing](testing.md): test the approval flow with `FakeHuman`
- [API reference: Permissions](../api/permissions.md)
