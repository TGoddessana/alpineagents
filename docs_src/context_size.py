from alpineagents import Agent, CompactIfFull, State, loop

compact = CompactIfFull(at=0.5, instructions="Keep file paths and failing test names")


@loop(until=State.is_answered, limit=100)
def long_task(agent: Agent, state: State):
    compact(agent, state)
    agent.think(state)
    if state.wants_tools():
        agent.use_tools(state)
        state.clear_tool_results(keep_last=10)
