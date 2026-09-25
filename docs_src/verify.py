import subprocess

from alpineagents import Agent, State, loop


def failing_tests() -> str | None:
    """The pytest output if any test fails, otherwise None."""
    result = subprocess.run(["pytest", "-q"], capture_output=True, text=True)
    return None if result.returncode == 0 else result.stdout[-3000:]


@loop(until=State.is_answered, limit=40)
def fix_until_green(agent: Agent, state: State):
    agent.think(state)
    if state.wants_tools():
        agent.use_tools(state)
        return
    failures = failing_tests()
    if failures:
        state.add_notice(f"The tests still fail. Fix them, then answer.\n{failures}")
