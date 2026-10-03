from alpineagents import Agent, State, loop, tool, waiting_for_user

REPEATS = 3


@tool
def look(path: str) -> str:
    """Look at a path"""
    return "nothing here"


def repeating(state: State) -> bool:
    replies = [entry.content for entry in state.history if entry.kind == "model_reply"][-REPEATS:]
    if len(replies) < REPEATS:
        return False
    calls = [[(call.name, call.args) for call in reply.tool_calls] for reply in replies]
    return bool(calls[0]) and all(same == calls[0] for same in calls)


@loop(until=[waiting_for_user, repeating], limit=40)
def watched(agent: Agent, state: State):
    agent.think(state)
    if state.pending_calls:
        agent.use_tools(state)


agent = Agent(model="ollama/qwen3:8b", tools=[look], loop=watched)
print(agent.run("Find the config file"))
