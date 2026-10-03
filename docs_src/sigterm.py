import signal

from alpineagents import Agent, FileStore, Message, State


def stop(signum, frame):
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, stop)

agent = Agent(model="claude-sonnet-5", store=FileStore(".agent-runs"))
state = State(id="nightly-report", messages=[Message.user("Write the nightly report")])
print(agent.run(state))
