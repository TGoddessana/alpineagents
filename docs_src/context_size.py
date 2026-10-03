from alpineagents import Agent, CompactIfFull, State, loop

compact = CompactIfFull(at=0.5, instructions="Keep file paths and failing test names")


def waiting_for_user(state: State) -> bool:
    if state.pending_calls or not state.messages:
        return False
    last = state.messages[-1]
    return last.role == "assistant" and not last.tool_calls


@loop(until=waiting_for_user, limit=100)
def long_task(agent: Agent, state: State):
    compact(agent, state)
    agent.think(state)
    if state.pending_calls:
        agent.use_tools(state)
        state.clear_tool_results(keep_last=10)
