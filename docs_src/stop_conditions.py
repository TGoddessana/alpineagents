from alpineagents import Agent, State, loop


def spent_too_much(state: State) -> bool:
    return state.usage.output_tokens > 20_000


@loop(until=[State.is_answered, spent_too_much], limit=30)
def careful(agent: Agent, state: State):
    agent.think(state)
    if state.wants_tools():
        agent.use_tools(state)
