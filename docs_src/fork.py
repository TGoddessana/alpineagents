from alpineagents import Agent, Message, State

agent = Agent(model="claude-sonnet-5")

state = State(messages=[Message.user("Fix calc.py")])
agent.run(state)

attempt = state.fork()  # the same content, a new id, nothing shared
attempt.add_message(Message.user("Rewrite it as a class"))
agent.run(attempt)

print(state.answer)  # unchanged by the attempt
print(attempt.answer)
