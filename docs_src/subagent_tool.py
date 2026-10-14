import asyncio
from pathlib import Path

from alpineagents import Agent, tool

reviewer = Agent(
    model="claude-sonnet-5",
    system="Review the file you are given. Report bugs only, with line numbers.",
)


@tool
async def review_files(paths: list[str], focus: str) -> list[dict]:
    """Review several files at the same time

    Args:
        paths: The files to review
        focus: What to look for, for example "error handling"
    """

    async def one(path: str) -> dict:
        code = Path(path).read_text()
        review = await reviewer.arun(f"Review {path}. Focus on {focus}.\n\n{code}")
        return {"path": path, "review": review}

    return await asyncio.gather(*(one(path) for path in paths))


lead = Agent(model="claude-sonnet-5", tools=[review_files])


async def main():
    print(await lead.arun("Review calc.py and parse.py for error handling"))


asyncio.run(main())
