from alpineagents import Agent, State, loop

REPEATS = 3


def repeating(state: State) -> bool:
    replies = [entry.content for entry in state.history if entry.kind == "reply"][-REPEATS:]
    if len(replies) < REPEATS:
        return False
    calls = [[(call.name, call.args) for call in reply.tool_calls] for reply in replies]
    return bool(calls[0]) and all(same == calls[0] for same in calls)


@loop(until=[State.is_answered, repeating], limit=40)
def watched(agent: Agent, state: State):
    agent.think(state)
    if state.wants_tools():
        agent.use_tools(state)
