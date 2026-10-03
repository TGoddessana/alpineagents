from alpineagents import Agent, Message, State, loop

NUDGES = 2
NUDGE = (
    "Your last reply called no tool. If the task is not done, continue it with a tool call. "
    "If it is done, give your final answer again."
)


def waiting_for_user(state: State) -> bool:
    if state.pending_calls or not state.messages:
        return False
    last = state.messages[-1]
    return last.role == "assistant" and not last.tool_calls


@loop(until=waiting_for_user, limit=40)
def nudging(agent: Agent, state: State):
    agent.think(state)
    if state.pending_calls:
        with state.edit_extra_data() as data:
            data["nudges"] = 0
        agent.use_tools(state)
        return
    nudges = state.extra_data.get("nudges", 0)
    if nudges < NUDGES:
        with state.edit_extra_data() as data:
            data["nudges"] = nudges + 1
        state.add_message(Message.notice(NUDGE))
