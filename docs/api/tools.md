# Tools and MCP

Concepts: [Tools](../concepts/tools.md). Guide: [MCP servers](../guides/mcp.md).

::: alpineagents.tool.tool

::: alpineagents.Tool
    options:
      filters: ["!^_"]

::: alpineagents.FunctionTool
    options:
      filters: ["!^_", "!^(prepare|invoke|fn|needs_self|bound_to)$"]

::: alpineagents.MCP

::: alpineagents.MCPTool
    options:
      filters: ["!^_"]
