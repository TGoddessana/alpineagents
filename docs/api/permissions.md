# Permissions

Guide: [Ask before a tool runs](../guides/approval.md).

Import these from `alpineagents.permissions`. `Agent(permissions=[...])` takes a list of them. Before any tool of a
turn runs, `use_tools` asks them about every call: first every `DenyPermission` in list order, then the
`AllowPermission`s and `DecidePermission`s in list order. The first verdict decides. A call nobody decides about is
denied, with a [`PermissionWarning`](errors.md).

## Verdicts

::: alpineagents.permissions.Allowed

::: alpineagents.permissions.Denied

## Kinds of permission

::: alpineagents.permissions.Permission

::: alpineagents.permissions.DenyPermission

::: alpineagents.permissions.AllowPermission

::: alpineagents.permissions.DecidePermission

## Built-in permissions

::: alpineagents.permissions.DenyByName

::: alpineagents.permissions.AllowByName

::: alpineagents.permissions.AllowByReadOnly

::: alpineagents.permissions.DecideByHuman

::: alpineagents.permissions.AllowByDefault
