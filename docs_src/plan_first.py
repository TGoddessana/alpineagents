from alpineagents import Agent, Message, State, loop


def waiting_for_user(state: State) -> bool:
    if state.pending_calls or not state.messages:
        return False
    last = state.messages[-1]
    return last.role == "assistant" and not last.tool_calls


@loop(until=waiting_for_user, limit=30)
def plan_then_act(agent: Agent, state: State):
    if state.turn == 0:
        agent.think(state, tools=[])
        state.add_message(Message.notice("Now carry out the plan."))
        return
    agent.think(state)
    if state.pending_calls:
        agent.use_tools(state)
