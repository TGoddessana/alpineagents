"""Error types for alpineagents.

Principle 5 (fail honestly): wrap into common types only at the provider boundary and keep the original in
``__cause__``. Misuse does not get a new type; it raises ``TypeError``/``ValueError`` with how to fix it
(build the message with :func:`fix_message`).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AlpineAgentsError",
    "ProviderError",
    "RateLimitError",
    "ContextTooLongError",
    "AuthError",
    "OutputError",
    "NoHumanError",
    "ToolInputError",
    "ResumeWarning",
    "fix_message",
]


class AlpineAgentsError(Exception):
    """Common parent of the errors alpineagents raises itself."""


class ProviderError(AlpineAgentsError):
    """The model API failed even after SDK retries. The original SDK exception is in ``__cause__``."""


class RateLimitError(ProviderError):
    """The provider's rate limit (429)."""


class ContextTooLongError(ProviderError):
    """The context is larger than the model's context window. Add a compaction block such as ``compact_if_full``
    to the loop."""


class AuthError(ProviderError):
    """Authentication failed (missing API key, wrong key, no permission)."""


class OutputError(AlpineAgentsError):
    """``agent.ask(returns=...)`` did not get a well-formed answer, even after asking again.

    The last validation error is in ``__cause__``.
    """


class NoHumanError(AlpineAgentsError):
    """``ask_human`` was called on an Agent with ``human=None``."""


class MCPConnectionError(AlpineAgentsError):
    """Could not connect to an MCP server, or the connection was lost. The original exception is in ``__cause__``.

    Errors the server reports for a tool call (``isError``) go to the model as the result instead.
    """


class ResumeWarning(UserWarning):
    """The Agent resuming a saved State differs from the Agent that saved it.

    A State holds the conversation, while the Agent's settings live in code. If the tools, model or system prompt
    changed between saving and resuming, the run still works but may behave differently: the model gets an error
    result when it calls a removed tool, and provider-specific blocks are not sent to a model of another provider.
    This may be what you want after a deploy, so it is a warning and not an error. Each loaded State is checked
    once, at its first ``run`` or ``arun``. Python's default warnings filter shows the same message from the same
    line only once per process; a server that resumes many States should use ``catch_warnings`` or the
    ``"always"`` filter to see each one.

    To refuse such a resume, turn it into an exception with
    ``warnings.filterwarnings("error", category=ResumeWarning)``. To show a notice in your UI, catch it with
    ``warnings.catch_warnings(record=True)`` and read ``.changes``.
    """

    def __init__(self, message: str, changes: dict[str, tuple[Any, Any]]) -> None:
        super().__init__(message)
        self.changes = changes
        """What changed: ``{key: (saved_value, current_value)}`` for each differing key of the Agent summary
        (``name``, ``model``, ``system_sha256``, ``tools``, ``mcp_servers``)."""

    def __reduce__(self) -> Any:
        # args holds only the message, so the default would rebuild without changes (pickle, copy).
        return (type(self), (str(self), self.changes))


class ToolInputError(Exception):
    """Internal. The tool arguments the model gave do not match the schema.

    Never raised to the caller. The Agent catches it and records ``(input error: {message})`` as the tool result.
    """


def fix_message(problem: str, fix: str, example: str | None = None) -> str:
    """Give every mistake-proofing error message the same format.

    >>> fix_message("until got a call result", "pass the function, no parens", "@loop(until=State.is_answered)")
    'until got a call result\\nFix: pass the function, no parens\\nExample:\\n    @loop(until=State.is_answered)'
    """
    message = f"{problem}\nFix: {fix}"
    if example:
        indented = "\n".join("    " + line for line in example.strip("\n").splitlines())
        message += f"\nExample:\n{indented}"
    return message
