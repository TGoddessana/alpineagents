# Tools and MCP

Learn: [Give the agent tools](../learn/tools.md). Concepts: [Tool calls](../concepts/tools.md). Guides: [Tools on objects and from data](../guides/more-tools.md), [Use MCP servers](../guides/mcp.md).

::: alpineagents.tool.tool

::: alpineagents.Tool
    options:
      filters: ["!^_"]

::: alpineagents.FunctionTool
    options:
      filters: ["!^_", "!^(prepare|invoke|fn|needs_self|bound_to)$"]

::: alpineagents.Hints

::: alpineagents.MCP

::: alpineagents.MCPTool
    options:
      filters: ["!^_"]
