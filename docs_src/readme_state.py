from alpineagents import Message, State

state = State(messages=[Message.user("Add a test for the add() function in calc.py")])
agent.run(state)

print(state.answer, state.stopped, state.usage.cost)
state.add_message(Message.user("Now run the tests"))  # the person's next message
agent.run(state)  # continues the same conversation
