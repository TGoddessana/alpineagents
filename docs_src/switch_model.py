from alpineagents import Agent, Message, State

fast = Agent(model="claude-haiku-5", system="You are a helpful assistant.")
strong = fast.copy(model="claude-opus-5")

state = State(messages=[Message.user("Outline a plan to move the users table to a new schema")])
print(fast.run(state))

state.add_message(Message.user("Now write the data backfill. It must not lock the table"))
print(strong.run(state))
