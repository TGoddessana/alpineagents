from alpineagents import Message, State

state = State(messages=[Message.user("Which Python versions does this project support? Ask the researcher")])
lead.run(state)

for entry in state.history:
    if entry.kind == "tool_result" and entry.child_ids:
        for child_id in entry.child_ids:
            child = state.child(child_id)
            print(entry.call.name, entry.outcome, child.parent_id == state.id, child.stopped, child.turn)

print(state.usage.requests)  # the lead's requests and the researcher's
