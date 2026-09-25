from alpineagents import Agent, State, tool


@tool
def submit(summary: str, files_changed: list[str], state: State) -> None:
    """Submit the finished work. Call it once, at the end."""
    state.finish({"summary": summary, "files_changed": files_changed})


agent = Agent(
    model="claude-sonnet-5",
    system="When the work is done, call submit.",
    tools=[submit],
)
print(agent.run("Rename the helper functions in utils.py"))
