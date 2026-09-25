from alpineagents import Agent, State

agent = Agent(model="claude-sonnet-5", system="You are a helpful assistant.")

state = State(input("> "))
while True:
    print(agent.run(state))
    state.add_user_message(input("> "))
