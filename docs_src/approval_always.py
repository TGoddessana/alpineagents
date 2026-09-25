@loop(until=State.is_answered, limit=30)
def careful(agent: Agent, state: State):
    agent.think(state)
    allowed = state.data.setdefault("allowed", set())
    for call in state.pending_calls:
        if call.name in RISKY and call.name not in allowed:
            question = f"Run {call.name}({call.args})?"
            choices = Literal["yes", "no", "always"]
            answer = agent.ask_human(state, question, returns=choices)
            if answer == "always":
                allowed.add(call.name)
            elif answer == "no":
                state.deny(call, "The user declined this call")
    if state.wants_tools():
        agent.use_tools(state)
