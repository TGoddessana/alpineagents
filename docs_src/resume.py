import sys

from alpineagents import Agent, FileStore, Message, State

store = FileStore(".agent-runs")
agent = Agent(model="claude-sonnet-5", system="You are a helpful assistant.", store=store)

if "--resume" in sys.argv:
    state = store.load(store.list()[0].id)  # the most recently updated conversation
else:
    state = State(messages=[Message.user(input("> "))])

while True:
    print(agent.run(state))
    state.add_message(Message.user(input("> ")))
    store.save(state)
