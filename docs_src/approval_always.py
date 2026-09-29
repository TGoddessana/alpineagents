from typing import Literal

from alpineagents import Terminal
from alpineagents.permissions import AllowByReadOnly, Allowed, DecidePermission, Denied
from alpineagents.types import format_call


class AskWithAlways(DecidePermission):
    """Asks before each call. "always" allows the tool for the rest of the task."""

    def __init__(self, human):
        self.human = human

    def check(self, state, call, tool):
        always = state.root.data.setdefault("always", [])
        if call.name in always:
            return Allowed()
        choices = Literal["yes", "no", "always"]
        answer = self.human.ask(state, f"Run {format_call(call)}?", choices)
        if answer == "no":
            return Denied("The user declined this call. Wait for their next message.", stop=True)
        if answer == "always":
            always.append(call.name)
        return Allowed()


agent = agent.copy(permissions=[AllowByReadOnly(), AskWithAlways(Terminal())])
