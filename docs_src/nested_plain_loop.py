from alpineagents import Agent, State, StoppedByFinish, StoppedByPermission, loop


@loop(until=State.is_answered, limit=5)
def research(agent: Agent, state: State):
    agent.think(state)
    if state.wants_tools():
        agent.use_tools(state)


def three_rounds(agent: Agent, state: State):
    for _ in range(3):
        # Not `state.stopped is None`: research may have left StoppedByLimit(5) there.
        if isinstance(state.stopped, (StoppedByFinish, StoppedByPermission)) or state.is_answered():
            return
        research(agent, state)
