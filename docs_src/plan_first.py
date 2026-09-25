from alpineagents import Agent, State, loop


@loop(until=State.is_answered, limit=30)
def plan_then_act(agent: Agent, state: State):
    if state.turn == 0:
        agent.think(state, tools=[])
        state.add_notice("Now carry out the plan.")
        return
    agent.think(state)
    if state.wants_tools():
        agent.use_tools(state)
