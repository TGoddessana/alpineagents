import asyncio

from alpineagents import Agent, State, acompact_if_full, loop, tool


@tool
async def run_command(command: str) -> str:
    """Run a shell command and return its output"""
    process = await asyncio.create_subprocess_shell(
        command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    output, _ = await process.communicate()
    return output.decode()[-5000:]


@loop(until=State.is_answered, limit=30)
async def working(agent: Agent, state: State):
    await acompact_if_full(agent, state)
    await agent.athink(state)
    if state.wants_tools():
        await agent.ause_tools(state)


agent = Agent(model="claude-sonnet-5", tools=[run_command], loop=working)


async def main():
    print(await agent.arun("Which Python version is installed?"))


asyncio.run(main())
