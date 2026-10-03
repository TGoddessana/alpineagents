from alpineagents import Agent, State, loop

REPEATS = 3


def waiting_for_user(state: State) -> bool:
    if state.pending_calls or not state.messages:
        return False
    last = state.messages[-1]
    return last.role == "assistant" and not last.tool_calls


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
