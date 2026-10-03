import subprocess

from alpineagents import Agent, Message, State, loop


def failing_tests() -> str | None:
    """The pytest output if any test fails, otherwise None."""
    result = subprocess.run(["pytest", "-q"], capture_output=True, text=True)
    return None if result.returncode == 0 else result.stdout[-3000:]


def waiting_for_user(state: State) -> bool:
    if state.pending_calls or not state.messages:
        return False
    last = state.messages[-1]
    return last.role == "assistant" and not last.tool_calls


@loop(until=waiting_for_user, limit=40)
def fix_until_green(agent: Agent, state: State):
    agent.think(state)
    if state.pending_calls:
        agent.use_tools(state)
        return
    failures = failing_tests()
    if failures:
        state.add_message(Message.notice(f"The tests still fail. Fix them, then answer.\n{failures}"))
