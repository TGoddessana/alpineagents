from alpineagents import Agent, State, loop

NUDGES = 2
NUDGE = (
    "Your last reply called no tool. If the task is not done, continue it with a tool call. "
    "If it is done, give your final answer again."
)


@loop(until=State.is_answered, limit=40)
def nudging(agent: Agent, state: State):
    agent.think(state)
    if state.wants_tools():
        state.data["nudges"] = 0
        agent.use_tools(state)
        return
    nudges = state.data.get("nudges", 0)
    if nudges < NUDGES:
        state.data["nudges"] = nudges + 1
        state.add_notice(NUDGE)
