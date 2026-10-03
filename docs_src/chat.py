from alpineagents import Agent, Message, State

agent = Agent(model="claude-sonnet-5", system="You are a helpful assistant.")

state = State(messages=[Message.user(input("> "))])
while True:
    print(agent.run(state))
    state.add_message(Message.user(input("> ")))
