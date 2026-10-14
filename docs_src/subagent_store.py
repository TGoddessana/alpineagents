from alpineagents import FileStore, Message, State

store = FileStore(".agent-runs")
lead = Agent(model="claude-sonnet-5", tools=[researcher], store=store)
lead.run(State(id="review-42", messages=[Message.user("Which Python versions are supported? Ask the researcher")]))

# After a restart: the subagents come from the store
state = store.load("review-42")
for entry in state.history:
    if entry.kind == "tool_result":
        for child_id in entry.child_ids:
            print(state.child(child_id).answer)  # loaded from the store

# Keep the States without a parent to list only the runs you started
top_level = [info for info in store.list() if info.parent_id is None]
print([info.id for info in top_level])
