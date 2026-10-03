from alpineagents import Message, State, StoppedByPermission

state = State(messages=[Message.user("Write a short README for this folder")])
agent.run(state)
while isinstance(state.stopped, StoppedByPermission):
    state.add_message(Message.user(input("What should it do instead? ")))
    agent.run(state)
print(state.answer)
