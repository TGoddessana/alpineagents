# MCP servers

Give an Agent the tools of an MCP server. Install the extra first:

```bash
pip install "alpineagents[mcp]"
```

## Add a server

Put the server in `tools=` like any other tool:

```python
--8<-- "docs_src/mcp_servers.py"
```

| Argument | Meaning |
| --- | --- |
| `command` (first argument) | The command that starts a local server. `env=` sets extra environment variables. The server does not get your full environment |
| `url=` | The URL of a remote server. `headers=` adds HTTP headers |
| `name=` | Required. The model sees each tool as `{name}__{tool}`, for example `github__create_issue` |

Pass exactly one of `command`, `url=` and `server=`. `server=` takes an in-process server, for tests.

## Pick tools

| Pass | The Agent gets |
| --- | --- |
| `github` | Every tool of the server |
| `linear.list_issues` | One tool |
| `linear["list-issues"]` | One tool, for a name that is not a Python identifier |

## Connections

- `agent.run` connects the servers at the start and disconnects them at the end.
- `with agent:` (or `async with agent:`) keeps them connected across several runs.
- Agents and concurrent runs that use the same `MCP` object share one connection.
- A tool name that collides with another tool, or a picked tool the server does not have, raises `ValueError` when
  the server connects, before the first `think`.

## Errors

| Situation | What happens |
| --- | --- |
| The server returns an error for a call | The model gets it as the tool result. The run continues |
| The connection fails or is lost | `MCPConnectionError` is raised |

## Related

- [Tools](../concepts/tools.md)
- [MCP API](../api/tools.md)
