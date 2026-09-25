from typing import Literal, get_args, get_origin

from alpineagents import Human


class Unattended(Human):
    """For runs nobody watches. Says no to every yes/no question."""

    def ask(self, state, prompt, returns=str):
        if returns is bool:
            return False
        if get_origin(returns) is Literal and "no" in get_args(returns):
            return "no"
        raise RuntimeError(f"Nobody can answer this: {prompt}")
