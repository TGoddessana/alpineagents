from alpineagents import State, StoppedByPermission

state = State("Write a short README for this folder")
agent.run(state)
while isinstance(state.stopped, StoppedByPermission):
    state.add_user_message(input("What should it do instead? "))
    agent.run(state)
print(state.answer)
