from alpineagents import Agent, State, loop


def waiting_for_user(state: State) -> bool:
    if state.pending_calls or not state.messages:
        return False
    last = state.messages[-1]
    return last.role == "assistant" and not last.tool_calls


def spent_too_much(state: State) -> bool:
    return state.usage.output_tokens > 20_000


@loop(until=[waiting_for_user, spent_too_much], limit=30)
def careful(agent: Agent, state: State):
    agent.think(state)
    if state.pending_calls:
        agent.use_tools(state)
