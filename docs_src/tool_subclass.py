import httpx

from alpineagents import Agent, State, Tool, ToolError, ToolInputError

rows = [  # for example from a database or an OpenAPI spec
    {
        "name": "create_ticket",
        "description": "Create a support ticket",
        "schema": {
            "type": "object",
            "properties": {"title": {"type": "string", "description": "A short title"}},
            "required": ["title"],
        },
        "url": "https://example.com/hooks/tickets",
    },
]


class Webhook(Tool):
    def __init__(self, name: str, description: str, schema: dict, url: str):
        super().__init__(name=name, description=description, input_schema=schema, open_world=True)
        self.url = url

    def run(self, args: dict, state: State) -> dict:
        if "title" not in args:
            raise ToolInputError("title is required")
        response = httpx.post(self.url, json=args)
        if response.status_code >= 400:
            raise ToolError(f"HTTP {response.status_code}")
        return response.json()


agent = Agent(model="claude-sonnet-5", tools=[Webhook(**row) for row in rows])
print(agent.run("Open a ticket: the login page is down"))
