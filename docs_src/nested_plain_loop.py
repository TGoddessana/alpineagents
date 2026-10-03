from alpineagents import Agent, State, StoppedByFinish, StoppedByPermission, loop, tool, waiting_for_user


@tool
def look(topic: str) -> str:
    """Look something up"""
    return f"Notes on {topic}"


@loop(until=waiting_for_user, limit=5)
def research(agent: Agent, state: State):
    agent.think(state)
    if state.pending_calls:
        agent.use_tools(state)


def three_rounds(agent: Agent, state: State):
    for _ in range(3):
        # Not `state.stopped is None`: research may have left StoppedByLimit(5) there.
        if isinstance(state.stopped, (StoppedByFinish, StoppedByPermission)) or waiting_for_user(state):
            return
        research(agent, state)


agent = Agent(model="claude-sonnet-5", tools=[look], loop=three_rounds)
print(agent.run("Research how the cache works"))
