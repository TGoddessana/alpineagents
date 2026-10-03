from alpineagents import Agent, Image, tool


@tool
def show_chart(path: str) -> list:
    """Look at a chart image"""
    return [f"The chart in {path}", Image.from_path(path)]


agent = Agent(model="claude-sonnet-5", tools=[show_chart])
print(agent.run("What does sales.png show?"))
