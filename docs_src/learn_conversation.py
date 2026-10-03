from alpineagents import Agent, Message, State

agent = Agent(model="claude-sonnet-5", system="You are a helpful assistant.")

state = State(messages=[Message.user("Name three Python web frameworks")])
print(agent.run(state))

print(state.stopped)  # why the run ended
print(state.turn)  # how many times the model answered
print(state.usage)  # tokens and requests
print(state)  # one line per history entry

state.add_message(Message.user("Which one is the smallest?"))
print(agent.run(state))
