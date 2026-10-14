# Write permission rules

[Ask before a tool runs](../learn/approval.md) showed `AllowByReadOnly()` and `DecideByHuman()`. This page is for
the rest: how permissions decide, the other built-in rules, and rules of your own.

## How permissions decide

For each call of the turn, in the order the model asked for them:

1. Every `DenyPermission` in the list, in list order, wherever it is in the list. The first one that refuses the call
   decides. No allow rule can let a refused call through.
2. Otherwise the `AllowPermission`s and `DecidePermission`s, in list order. The first one with an opinion decides.
3. Nobody decided: the call is refused.

So `[AllowByDefault(), DenyByName(["delete_*"])]` allows every call except those to `delete_*` tools. A call nobody
decides about is refused with `No permission allowed this call.`, and Python shows a `PermissionWarning` that names
the tool. `permissions=[]` refuses every call.

`permissions=None` (or leaving it out) turns the checks off: every call runs. `agent.copy(permissions=None)` does that
for a copy.

A refused call does not run. The model gets the reason as its error result. `state.history` has a `tool_result` entry
with `outcome == "denied"`, and the Reporter gets `on_tool_end` with `outcome.kind == "denied"` and
`outcome.decided_by` set to the `repr()` of the permission that refused it (`None` when nobody decided). Calls to
unknown tools and calls with invalid JSON arguments are [input errors](../concepts/tools.md#when-a-call-goes-wrong)
before permissions are asked.

## Built-in permissions

They live in `alpineagents.permissions`:

| Permission | Kind | Decides |
| --- | --- | --- |
| `DenyByName(["delete_*"], reason=None)` | Deny | Refuses calls to matching tools, with `reason` or `"{name} is not allowed."` |
| `AllowByName(["read_file", "github__get_*"])` | Allow | Allows calls to matching tools |
| `AllowByReadOnly(trust_mcp=False)` | Allow | Allows a call the tool says is read-only: `tool.hints_for(call.args).read_only` |
| `DecideByHuman(human=None)` | Decide | Asks the person: `yes` allows, `no` refuses and stops |
| `AllowByDefault()` | Allow | Allows every call. Put it last |

- Names are `fnmatch` patterns on the name the model calls: `"delete_*"` matches `delete_file` and `delete_branch`.
- Put `DecideByHuman()` after the permissions that decide without asking, so the person is asked only about the calls
  they left open.
- `DecideByHuman(human=...)` asks someone else, for example an approvals channel. A `DecideByHuman()` with no
  `human` of its own, on an Agent with `human=None`, raises `NoHumanError` when the run starts.

## Say what a tool does

Four hints say what a tool does. The model does not see them. Permissions read them:

```python
from pathlib import Path

from alpineagents import tool


@tool(read_only=True, open_world=False)
def read_file(path: str) -> str:
    """Read a file in the project"""
    return Path(path).read_text()
```

| Hint | The tool | Left out |
| --- | --- | --- |
| `read_only` | Changes nothing | `False` |
| `destructive` | May delete or overwrite something | `True`, or `False` for a read-only tool |
| `idempotent` | Has no further effect when called again with the same arguments | `False`, or `True` for a read-only tool |
| `open_world` | Reaches outside its own domain: the web, a shell, another system | `True` |

A hint left out assumes the worst, so a forgotten hint makes a rule stricter, never looser. The hints have the meaning
of MCP tool annotations.

### Decide by what one call does

The hints of `@tool` hold for every call, so a shell tool is never read-only: it can run `rm`. With `hints_for=`,
the tool says what one call does, and a permission reads that:

```python
--8<-- "docs_src/approval_hints.py"
```

1. `bash_hints` says `ls` and `cat` change nothing, and `mkdir` and `touch` delete and overwrite nothing. For
   anything else, including commands joined with `;` or `|`, it returns `None`, and the tool's own hints (the worst
   case) apply.
2. `AllowNotDestructive` allows the calls whose hints say `destructive=False`. `DecideByHuman()` asks about the rest,
   such as `rm -rf build`.
3. `AllowByReadOnly()` reads `hints_for` the same way, so with it `ls` runs without asking and `mkdir` asks.

- `Hints(...)` takes the same four hints with the same rules: one left out assumes the worst, and `read_only=True`
  sets the other two.
- The function gets the model's arguments before they are checked: a value can be missing or of any type. Read with
  `args.get(...)` and `isinstance`, and return `None` when unsure.
- Keep hints about facts ("this call deletes nothing") and decisions in permissions ("run what deletes nothing").
- `tool.copy(hints_for=...)` adds or replaces the function. A `Tool` subclass overrides `hints_for(self, args)`.
- To read a call's hints in your own code, look the tool up: `agent.tool_map.get(call.name)`.

## MCP tools

An [MCP server](mcp.md)'s tools are named `{server}__{tool}`, where `server` is the `name=` you gave `MCP(...)`. The
names are yours, so `DenyByName(["github__delete_*"])` and `AllowByName(["github__get_*"])` work.

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
  permission with settings a `__repr__` that shows them.
- A verdict the kind may not give, such as `Allowed()` from a `DenyPermission`, raises `TypeError` during the run.

### Add "always"

Let the person approve a tool once for the rest of the task:

```python
--8<-- "docs_src/approval_always.py"
```

- `Denied(reason, stop=True)` also stops the run, like `DecideByHuman`'s `no`.
- `state.extra_data["always"]` keeps the approved tool names across turns and runs, and a store saves it.
  `edit_extra_data()` records the change as one history entry. The model never sees `extra_data`.
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
raises `TypeError` when the run starts, before the first model call. A permission with only `check` works with
`arun` too: it runs on a worker thread. `DecideByHuman` implements both. A `DecideByHuman` subclass that overrides
only `acheck` counts as async-only: a sync `run` raises `TypeError` rather than ask the Human. One that overrides only
`check` decides the same way in `arun`.

## Permissions in subagents

`permissions=` applies to **every tool call in the run, including the calls inside its [subagents](subagents.md)**.
You write the list once, on the Agent you run, and a subagent cannot get around it. The code below has two subagents:

```python
--8<-- "docs_src/subagent_permissions.py"
```

1. `editor` has no `permissions=`, so the whole list of `lead` checks its calls. `delete_*` is refused, and
   `write_file` goes to `DecideByHuman()`, which asks the person of `lead`.
2. `drafter` has a list of its own. It decides for `drafter`'s calls instead of the allow and decide rules of `lead`:
   `write_file` is allowed without asking.
3. `delete_*` is still refused for both. A subagent cannot widen a `DenyPermission` of the runs above it.
4. `AllowByName(["editor", "drafter"])` lets `lead` call the subagents. A subagent's tool does not say it is
   read-only, so `AllowByReadOnly()` does not allow it.

The rule for the calls of one subagent:

| The subagent's `permissions=` | Which permissions check its calls |
| --- | --- |
| `None` (left out) | The whole list of the run above it |
| A list | The `DenyPermission`s of every run above it, from the top, then its own list |

So:

- **Put what must always hold in `DenyBy...` permissions**, such as `DenyByName(["delete_*"])`. They reach every
  subagent and every subagent of a subagent.
- **A list of allow rules does not bind a subagent with a list of its own.** `[AllowByName(["read_file"])]` on the
  lead means "refuse the rest" for the lead's own calls. A subagent with its own list is checked by that list, so a
  call that list allows is not refused by the lead's "refuse the rest". To refuse it everywhere, use a
  `DenyPermission`.
- A subagent with `permissions=[]` has every call refused (with a `PermissionWarning`), after the `DenyPermission`s
  above it.
- **A permission asks the person of the Agent that wrote it.** `DecideByHuman()` in the lead's list asks the lead's
  `human` about a call the editor makes. A `DecideByHuman()` in the subagent's own list asks the subagent's `human`,
  which is the lead's when the subagent has none of its own
  ([Subagents](../concepts/subagents.md#what-a-subagent-takes-from-the-run-above)).
- `outcome.decided_by` in the Reporter names the permission that decided, wherever it was written.
- The checks before the run starts apply to the inherited list too: a `DecideByHuman()` with no person to ask, or an
  async-only permission under a sync `run`, raises before the first request.

## Not a security boundary

Permissions keep an honest model from doing what you did not want. They do not stop a determined one:

- `bash_hints` above says `cat` is read-only, and it is, but `cat ~/.ssh/id_rsa` still reads a secret.
- A command can do more than its first word says, through an alias, a script it runs or a program's own options.
- A tool can do things its hints do not say.

Treat permissions as a way to ask the person at the right moments, not as a way to contain the agent. When a tool
runs commands or code the model wrote, run the agent in a sandbox: a container or a virtual machine with only the
files, network access and credentials the task needs.

## Related

- [Use subagents](subagents.md)
- [Show progress and ask the person](progress.md): answer questions from somewhere other than the terminal
- [Stop a run](stop-conditions.md): every way a run stops
- [Test your agent](../learn/testing.md): test the approval flow with `FakeHuman`
- [API reference: Permissions](../api/permissions.md)
