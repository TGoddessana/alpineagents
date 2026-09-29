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
    "ToolError",
    "ToolInputError",
    "ResumeWarning",
    "PermissionWarning",
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


class PermissionWarning(UserWarning):
    """No permission in ``Agent(permissions=[...])`` allowed or denied a tool call, so it was denied.

    A call no permission decides about is never run. If that is what you want, turn the warning off with
    ``warnings.filterwarnings("ignore", category=PermissionWarning)``. To allow every call no permission denied,
    put ``AllowByDefault()`` at the end of the list.
    """


class ToolError(Exception):
    """Raised by a tool for a failure the model should see and handle, such as a missing file or an HTTP 404.

    The model gets the message as the call's error result, and the run continues. Any other exception from a tool
    stops the run, because it is usually a bug. To turn a library's exceptions into error results without
    catching them in every tool, use ``@tool(exception_handler=...)``.

    Does not inherit from ``AlpineAgentsError``: tools raise it, not alpineagents.

    Example:
        ```python
        @tool
        def fetch_url(url: str) -> str:
            \"\"\"Fetch a web page\"\"\"
            response = httpx.get(url)
            if response.status_code >= 400:
                raise ToolError(f"HTTP {response.status_code}: {url}")
            return response.text
        ```
    """

    def __init__(self, message: str) -> None:
        """
        Args:
            message: What the model is told. Not empty.

        Raises:
            TypeError: ``message`` is not a string.
            ValueError: ``message`` is empty or whitespace only.
        """
        example = 'raise ToolError(f"HTTP {status}: {url}")'
        if not isinstance(message, str):
            raise TypeError(
                fix_message(
                    f"ToolError takes a string, the message the model is told (got: {message!r})",
                    "Pass the message as a string",
                    example,
                )
            )
        if not message.strip():
            raise ValueError(
                fix_message(
                    "ToolError got an empty message. Providers reject empty tool results",
                    "Say what went wrong, so the model can do something about it",
                    example,
                )
            )
        super().__init__(message)


class ToolInputError(Exception):
    """Raised by a tool when the arguments the model gave are wrong, such as a missing or invalid value.

    The model gets ``(input error: {message})`` as the call's result, and the run continues. ``@tool`` raises it for
    arguments that do not fit the type hints; a ``Tool`` subclass raises it from ``run`` for arguments it cannot use.

    Example:
        ```python
        def run(self, args: dict, state: State) -> str:
            if "title" not in args:
                raise ToolInputError("title is required")
            ...
        ```
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
