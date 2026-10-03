from alpineagents import Agent, State, StoppedByFinish, StoppedByPermission, loop


def waiting_for_user(state: State) -> bool:
    if state.pending_calls or not state.messages:
        return False
    last = state.messages[-1]
    return last.role == "assistant" and not last.tool_calls


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
