import os

from alpineagents import MCP, Agent

github = MCP(
    "npx -y @modelcontextprotocol/server-github",
    name="github",
    env={"GITHUB_TOKEN": os.environ["GITHUB_TOKEN"]},
)
linear = MCP(
    url="https://mcp.linear.app/mcp",
    name="linear",
    headers={"Authorization": f"Bearer {os.environ['LINEAR_API_KEY']}"},
)

agent = Agent(model="claude-sonnet-5", tools=[github, linear.list_issues])
