"""The Human role: asks a person and gets an answer.

The one required member is ``ask(state, prompt, returns)``, or ``async aask(...)`` for a person reached
asynchronously (e.g. over a web socket). ``returns`` is ``str``, ``bool`` or ``Literal[...]``.
The implementation asks again when an answer does not fit the format. It asks one question at a time and queues
concurrent calls. A Human does not change State, does not call the model, and does not interpret answers.

Every implementation (Terminal, FakeHuman) shares one answer conversion rule: :func:`parse_answer`.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin

from ._async import run_in_thread
from .errors import fix_message

if TYPE_CHECKING:
    from .state import State

__all__ = ["Human", "check_returns", "parse_answer", "describe_choices"]

_YES = {"y", "yes", "true", "1"}
_NO = {"n", "no", "false", "0"}


class Human(ABC):
    """A conversation partner that asks a person. A subclass implements ``ask``, ``aask``, or both."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Human:
        # Checked here, not at class creation, so abstract intermediate classes still work.
        if cls.ask is Human.ask and cls.aask is Human.aask:
            raise TypeError(
                fix_message(
                    f"{cls.__name__} implements neither ask nor aask",
                    "Implement ask(state, prompt, returns), or async aask(...) for a person reached asynchronously",
                    "class WebHuman(Human):\n"
                    "    async def aask(self, state, prompt, returns=str):\n"
                    "        return parse_answer(await ask_on_the_page(prompt), returns)",
                )
            )
        return super().__new__(cls)

    def ask(self, state: State, prompt: str, returns: Any = str) -> Any:
        """Ask a person and return the answer in the ``returns`` format. Ask again when the answer does not fit.

        The default is for a Human that implements only ``aask``: ``TypeError`` pointing to ``aask_human``.
        """
        raise TypeError(
            fix_message(
                f"{type(self).__name__} answers only asynchronously (it implements aask, not ask)",
                "Ask it from an async loop with await agent.aask_human(...)",
                'ok = await agent.aask_human(state, "Run it?", returns=bool)',
            )
        )

    async def aask(self, state: State, prompt: str, returns: Any = str) -> Any:
        """The async version of ``ask``, used by ``agent.aask_human``. The default runs ``ask`` on a worker
        thread (a blocking ``input()`` does not block the event loop)."""
        return await run_in_thread(self.ask, state, prompt, returns)


def check_returns(returns: Any) -> None:
    """Check that ``returns`` for ``ask_human`` is a supported format. ``TypeError`` if not."""
    if returns is str or returns is bool:
        return
    if get_origin(returns) is Literal and get_args(returns):
        return
    raise TypeError(
        fix_message(
            f"ask_human does not support returns={returns!r}",
            "set returns to str, bool or Literal[...]",
            'agent.ask_human(state, "Run it?", returns=Literal["yes", "no", "always"])',
        )
    )


def describe_choices(returns: Any) -> str | None:
    """Choice hint to put after a question. ``"yes/no"`` for ``bool``, ``"a/b/c"`` for Literal, ``None`` for ``str``."""
    if returns is bool:
        return "yes/no"
    if get_origin(returns) is Literal:
        return "/".join(str(option) for option in get_args(returns))
    return None


def parse_answer(answer: Any, returns: Any) -> Any:
    """Convert a person's answer to the ``returns`` format. ``ValueError`` if it does not fit (the message is the
    hint shown when asking again).

    - ``str``: the string with surrounding whitespace stripped. Asks again when empty (an empty answer cannot go
      into ``state.add_user_message`` either; this way pressing just Enter in a chat loop does not stop the run)
    - ``bool``: ``y/yes/true/1`` → True, ``n/no/false/0`` → False (case-insensitive)
    - ``Literal[...]``: the choice value, when the answer equals a choice (comparing as strings also works)
    """
    check_returns(returns)
    if returns is str:
        text = str(answer).strip()
        if not text:
            raise ValueError("Empty answers are not accepted. Please type something")
        return text
    if returns is bool:
        if isinstance(answer, bool):
            return answer
        text = str(answer).strip().lower()
        if text in _YES:
            return True
        if text in _NO:
            return False
        raise ValueError("Answer 'yes' or 'no'")
    options = get_args(returns)
    for option in options:
        if answer == option:
            return option
    text = str(answer).strip()
    for option in options:
        if str(option) == text:
            return option
    raise ValueError(f"Answer with one of: {', '.join(str(o) for o in options)}")
