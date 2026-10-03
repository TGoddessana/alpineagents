from alpineagents import Agent, Image, Message, State

agent = Agent(model="claude-sonnet-5")

state = State(messages=[Message.user("What is wrong in this screenshot?", Image.from_path("screen.png"))])
print(agent.run(state))

state.add_message(Message.user("And in this one?", Image.from_path("screen-2.png")))
print(agent.run(state))
