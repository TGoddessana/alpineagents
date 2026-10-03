from alpineagents import Agent, Message, State

agent = Agent(model="claude-sonnet-5")

state = State(messages=[Message.user("Fix calc.py")])
agent.run(state)

before = state.snapshot()  # a frozen value: it never changes
state.add_message(Message.user("Rewrite it as a class"))
agent.run(state)

if "class" not in state.answer:
    state.restore(before)  # the State is back where the snapshot was taken
    print(len(state.history) > len(before.history))  # True: history keeps everything, and says it went back
