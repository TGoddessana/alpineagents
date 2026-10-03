"""Permissions: decide, before a tool call runs, whether it may run.

``Agent(permissions=[...])`` takes a list of permissions. ``use_tools`` asks them about every call of the turn
(after unknown tools and broken arguments became input errors) and before any tool runs:

1. Every ``DenyPermission`` in list order, wherever it sits in the list. The first ``Denied`` decides.
2. Otherwise every ``AllowPermission`` and ``DecidePermission`` in list order. The first verdict that is not
   ``None`` decides.
3. Nobody decided: the call is denied, with a ``PermissionWarning``.

A denied call does not run and the model gets the reason as its error result. ``Denied(..., stop=True)`` also
cancels the other calls of the turn and sets ``state.stopped`` to ``StoppedByPermission`` right away, so the loop
stops before its next turn. The runner (``_runner.run_calls``) applies the verdicts; this module only asks.

Layer: next to ``loop.py`` and ``blocks.py``. It never imports ``agent.py`` (the running Agent is ``state._agent``).
"""

from __future__ import annotations

import fnmatch
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Self

from ._async import run_in_thread
from .errors import ToolError, fix_message
from .human import Human
from .mcp_tools import MCPTool
from .types import format_call

if TYPE_CHECKING:
    from .state import State
    from .tool import Tool
    from .types import ToolCall

__all__ = [
    "Permission",
    "DenyPermission",
    "AllowPermission",
    "DecidePermission",
    "Allowed",
    "Denied",
    "DenyByName",
    "AllowByName",
    "AllowByReadOnly",
    "DecideByHuman",
    "AllowByDefault",
]


# ---------------------------------------------------------------- verdicts


@dataclass(frozen=True)
class Allowed:
    """The verdict that lets a call run. Returned by an ``AllowPermission`` or a ``DecidePermission``."""


@dataclass(frozen=True)
class Denied:
    """The verdict that refuses a call. Returned by a ``DenyPermission`` or a ``DecidePermission``.

    The call does not run, and the model gets ``reason`` as its error result.

    Raises:
        TypeError: ``reason`` is not a string, or ``stop`` is not a bool.
        ValueError: ``reason`` is empty or whitespace only.

    Example:
        ```python
        return Denied("Deleting files is not allowed")
        return Denied("The user declined this call. Wait for their next message.", stop=True)
        ```
    """

    reason: str
    """What the model is told."""
    stop: bool = False
    """Also stop the turn: the other calls of the turn are cancelled (not run, not asked about), and
    ``state.stopped`` becomes ``StoppedByPermission(call, permission)`` right away, so the loop stops before its next
    turn. The run ends normally; ``state.add_message(Message.user(...))`` and ``run`` again continue it. A ``@loop`` stops by
    itself; a loop written without ``@loop`` checks ``state.stopped is None`` before each turn."""

    def __post_init__(self) -> None:
        example = 'Denied("Deleting files is not allowed")'
        if not isinstance(self.reason, str):
            raise TypeError(
                fix_message(
                    f"Denied takes a reason string, what the model is told (got: {self.reason!r})",
                    "Pass the reason as a string",
                    example,
                )
            )
        if not self.reason.strip():
            raise ValueError(
                fix_message(
                    "Denied got an empty reason. Providers reject empty tool results",
                    "Say why the call was refused, so the model can do something else",
                    example,
                )
            )
        if not isinstance(self.stop, bool):
            raise TypeError(
                fix_message(
                    f"Denied: stop= takes True or False (got: {self.stop!r})",
                    "Pass stop=True to also stop the turn, or leave it out",
                    'Denied("The user declined this call", stop=True)',
                )
            )


# ---------------------------------------------------------------- permission kinds


class Permission:
    """Decides whether a tool call may run. Subclass one of ``DenyPermission``, ``AllowPermission`` or
    ``DecidePermission`` (not this class), and implement ``check``, or ``async acheck``, or both.

    ``check(state, call, tool)`` returns a verdict (``Allowed()`` or ``Denied(...)``) or ``None`` for no opinion,
    which lets the next permission decide. What a permission may return depends on its kind.

    - ``call.args`` are the model's arguments, parsed from JSON but not yet checked against the tool's input schema.
    - ``tool`` is always a ``Tool`` (calls to unknown tools, or with broken JSON arguments, are input errors before
      permissions are asked).
    - Raising ``ToolError(message)`` refuses the call: the model gets ``message`` as its error result, and the run
      continues. Any other exception propagates out of ``use_tools``, like one raised by a tool. A failure never
      counts as allowed.

    ``repr()`` of a permission names it in ``ToolOutcome.decided_by`` and ``StoppedByPermission.permission``, so
    give a permission with settings a ``__repr__`` that shows them.

    Raises:
        TypeError: When creating an instance of a class that subclasses ``Permission`` directly or more than one
            kind, or that implements neither ``check`` nor ``acheck``.

    Example:
        ```python
        class DenyOutsideProject(DenyPermission):
            def check(self, state, call, tool):
                path = call.args.get("path")
                if isinstance(path, str) and not path.startswith("project/"):
                    return Denied(f"{path} is outside the project folder")
                return None
        ```
    """

    #: The verdict types this kind may return (besides ``None``). Set by each kind.
    _verdicts: tuple[type, ...] = ()

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        # Checked here, not at class creation, so abstract intermediate classes still work (like Human.__new__).
        kinds = [kind.__name__ for kind in _KINDS if issubclass(cls, kind)]
        if len(kinds) != 1:
            problem = (
                f"{cls.__name__} subclasses Permission directly"
                if not kinds
                else f"{cls.__name__} subclasses more than one kind of permission ({', '.join(kinds)})"
            )
            raise TypeError(
                fix_message(
                    problem,
                    "Subclass exactly one of DenyPermission (may only deny), AllowPermission (may only allow) or "
                    "DecidePermission (may do either)",
                    "class DenyDeletes(DenyPermission):\n"
                    "    def check(self, state, call, tool):\n"
                    '        return Denied("Deleting is not allowed") if call.name == "delete_file" else None',
                )
            )
        if cls.check is Permission.check and cls.acheck is Permission.acheck:
            raise TypeError(
                fix_message(
                    f"{cls.__name__} implements neither check nor acheck",
                    "Implement check(state, call, tool), or async acheck(...) for a permission that awaits "
                    "something",
                    f"class {cls.__name__}({kinds[0]}):\n"
                    "    def check(self, state, call, tool):\n"
                    "        return None  # no opinion: the next permission decides",
                )
            )
        return super().__new__(cls)

    def check(self, state: State, call: ToolCall, tool: Tool) -> Allowed | Denied | None:
        """Decides about one call: a verdict, or ``None`` to let the next permission decide.

        Args:
            state: The State of the run.
            call: The tool call. ``call.args`` are not yet checked against the tool's input schema.
            tool: The tool the call would run.

        Raises:
            TypeError: The permission implements only ``acheck``. Run the Agent with ``arun``.
        """
        raise TypeError(
            fix_message(
                f"{self!r} checks calls only asynchronously (it implements acheck, not check)",
                "Run the Agent with await agent.arun(...), or implement check as well",
                'answer = await agent.arun("...")',
            )
        )

    async def acheck(self, state: State, call: ToolCall, tool: Tool) -> Allowed | Denied | None:
        """The async version of ``check``, used by ``arun`` and ``ause_tools``. By default it runs ``check`` on a
        worker thread, so a blocking check does not block the event loop."""
        return await run_in_thread(self.check, state, call, tool)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class DenyPermission(Permission):
    """A permission that may only deny: ``check`` returns ``Denied(...)`` or ``None``.

    Deny permissions are asked first, wherever they are in the list, so no allow rule can let a denied call through.
    """

    _verdicts = (Denied,)


class AllowPermission(Permission):
    """A permission that may only allow: ``check`` returns ``Allowed()`` or ``None``."""

    _verdicts = (Allowed,)


class DecidePermission(Permission):
    """A permission that may allow or deny: ``check`` returns ``Allowed()``, ``Denied(...)`` or ``None``.

    Asked in list order with the allow permissions, after every deny permission.
    """

    _verdicts = (Allowed, Denied)


_KINDS = (DenyPermission, AllowPermission, DecidePermission)


# ---------------------------------------------------------------- built-in permissions


def _patterns(value: Any, owner: str) -> tuple[str, ...]:
    """``patterns=`` of ``DenyByName``/``AllowByName`` as a tuple of strings."""
    example = f'{owner}(["delete_*", "github__merge_pull_request"])'
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise TypeError(
            fix_message(
                f"{owner} takes a list of name patterns (got: {value!r})",
                "Wrap it in square brackets, even if there is only one",
                example,
            )
        )
    patterns = tuple(value)
    wrong = [p for p in patterns if not isinstance(p, str)]
    if wrong:
        raise TypeError(
            fix_message(
                f"{owner} takes name patterns as strings (got: {wrong[0]!r})",
                "Give tool names, or glob patterns such as \"github__*\"",
                example,
            )
        )
    return patterns


class DenyByName(DenyPermission):
    """Denies calls whose tool name matches one of ``patterns``.

    A pattern is an ``fnmatch`` glob on the name the model calls (``call.name``), so an MCP tool is
    ``{server}__{tool}``: ``"github__*"`` denies every tool of the ``github`` server.

    Example:
        ```python
        Agent(model=..., tools=[...], permissions=[DenyByName(["delete_*"]), AllowByDefault()])
        ```
    """

    def __init__(self, patterns: Iterable[str], reason: str | None = None) -> None:
        """
        Args:
            patterns: Tool names or glob patterns.
            reason: What the model is told. Defaults to ``"{call.name} is not allowed."``.

        Raises:
            TypeError: ``patterns`` is not a list of strings, or ``reason`` is not a string.
            ValueError: ``reason`` is empty or whitespace only.
        """
        self.patterns = _patterns(patterns, "DenyByName")
        """The tool names or ``fnmatch`` globs, as a tuple."""
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            problem = (
                f"DenyByName: reason= takes a string, what the model is told (got: {reason!r})"
                if not isinstance(reason, str)
                else "DenyByName got an empty reason=. Providers reject empty tool results"
            )
            error = TypeError if not isinstance(reason, str) else ValueError
            raise error(
                fix_message(
                    problem,
                    "Say why the call is refused, so the model can do something else, or leave reason= out",
                    'DenyByName(["delete_*"], reason="Deleting files is not allowed")',
                )
            )
        self.reason = reason
        """What the model is told, or ``None`` for ``"{call.name} is not allowed."``."""

    def check(self, state: State, call: ToolCall, tool: Tool) -> Denied | None:
        """Refuses a call whose name matches one of ``patterns``, with ``reason``. Has no opinion otherwise."""
        if any(fnmatch.fnmatchcase(call.name, pattern) for pattern in self.patterns):
            return Denied(self.reason if self.reason is not None else f"{call.name} is not allowed.")
        return None

    def __repr__(self) -> str:
        extra = f", reason={self.reason!r}" if self.reason is not None else ""
        return f"{type(self).__name__}({list(self.patterns)!r}{extra})"


class AllowByName(AllowPermission):
    """Allows calls whose tool name matches one of ``patterns`` (``fnmatch`` globs on ``call.name``, so an MCP tool
    is ``{server}__{tool}``).

    Example:
        ```python
        permissions=[AllowByName(["read_file", "github__get_*"]), DecideByHuman()]
        ```
    """

    def __init__(self, patterns: Iterable[str]) -> None:
        """
        Args:
            patterns: Tool names or glob patterns.

        Raises:
            TypeError: ``patterns`` is not a list of strings.
        """
        self.patterns = _patterns(patterns, "AllowByName")
        """The tool names or ``fnmatch`` globs, as a tuple."""

    def check(self, state: State, call: ToolCall, tool: Tool) -> Allowed | None:
        """Allows a call whose name matches one of ``patterns``. Has no opinion otherwise."""
        if any(fnmatch.fnmatchcase(call.name, pattern) for pattern in self.patterns):
            return Allowed()
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self.patterns)!r})"


class AllowByReadOnly(AllowPermission):
    """Allows a call when the tool says this call is read-only: ``tool.hints_for(call.args).read_only``.

    An MCP server's tools describe themselves, so they are not allowed this way unless ``trust_mcp=True``.
    """

    def __init__(self, trust_mcp: bool = False) -> None:
        """
        Args:
            trust_mcp: Also believe MCP servers' ``readOnlyHint``.

        Raises:
            TypeError: ``trust_mcp`` is not a bool.
        """
        if not isinstance(trust_mcp, bool):
            raise TypeError(
                fix_message(
                    f"AllowByReadOnly: trust_mcp= takes True or False (got: {trust_mcp!r})",
                    "Pass trust_mcp=True to believe MCP servers' read-only hints, or leave it out",
                    "AllowByReadOnly(trust_mcp=True)",
                )
            )
        self.trust_mcp = trust_mcp
        """Whether MCP servers' ``readOnlyHint`` is believed."""

    def check(self, state: State, call: ToolCall, tool: Tool) -> Allowed | None:
        """Allows a call when ``tool.hints_for(call.args).read_only`` is true. Has no opinion otherwise, and
        always for an ``MCPTool`` unless ``trust_mcp`` is true."""
        if isinstance(tool, MCPTool) and not self.trust_mcp:
            return None
        return Allowed() if tool.hints_for(call.args).read_only else None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(trust_mcp=True)" if self.trust_mcp else f"{type(self).__name__}()"


class AllowByDefault(AllowPermission):
    """Allows every call. Put it last in the list to allow everything no permission before it denied."""

    def check(self, state: State, call: ToolCall, tool: Tool) -> Allowed:
        """Allows the call."""
        return Allowed()


#: The reason ``DecideByHuman`` gives when the person answers no.
_DECLINED = "The user declined this call. Wait for their next message."
_CHOICES = Literal["yes", "no"]


class DecideByHuman(DecidePermission):
    """Asks the person whether each call may run. ``yes`` allows it; ``no`` denies it and stops the turn
    (``Denied(..., stop=True)``), so the person can say what to do instead.

    The question goes to the running Agent's ``human`` (the terminal by default) and is recorded in
    ``state.history`` like ``agent.ask_human``. Put it after the permissions that decide without asking, so the
    person is asked only about the calls they left open.

    Example:
        ```python
        permissions=[DenyByName(["delete_*"]), AllowByReadOnly(), DecideByHuman()]
        ```
    """

    def __init__(self, human: Human | None = None) -> None:
        """
        Args:
            human: Who to ask instead of the Agent's ``human``, for example a chat app for approvals.

        Raises:
            TypeError: ``human`` is not a Human object.
        """
        if human is not None and (
            isinstance(human, type)
            or not (callable(getattr(human, "ask", None)) or callable(getattr(human, "aask", None)))
        ):
            raise TypeError(
                fix_message(
                    f"DecideByHuman: human={human!r} is not a Human object",
                    "Pass an object of an alpineagents.Human subclass, or leave it out to ask the Agent's human",
                    'DecideByHuman(human=FakeHuman(["yes"]))',
                )
            )
        self.human = human
        """The Human asked instead of the Agent's ``human``, or ``None`` for the Agent's."""

    def question(self, call: ToolCall, tool: Tool) -> str:
        """The question to ask about ``call``. Override it to ask differently.

        Returns:
            By default ``Run {call}?``, e.g. ``Run write_file(path="a.md", content="x")?``.
        """
        return f"Run {format_call(call)}?"

    def check(self, state: State, call: ToolCall, tool: Tool) -> Allowed | Denied | None:
        """Asks the person ``question(call, tool)`` and allows the call on ``yes``. Any other answer denies it
        with ``stop=True``. Raises ``ValueError`` if the State has no running Agent."""
        if type(self).acheck is not DecideByHuman.acheck and type(self).check is DecideByHuman.check:
            # A subclass that changed acheck (and not check) is async-only: check must not ask the human instead.
            return super().check(state, call, tool)
        agent = _running_agent(state, self)
        answer = agent._ask_human(state, self.question(call, tool), _CHOICES, "ask_human", self.human)
        return self._verdict(answer)

    async def acheck(self, state: State, call: ToolCall, tool: Tool) -> Allowed | Denied | None:
        """The async version of ``check``: asks the Human with ``aask`` (or ``ask`` on a worker thread)."""
        if type(self).check is not DecideByHuman.check:
            # A subclass that changed check (and not acheck) decides the same way in arun.
            return await super().acheck(state, call, tool)
        agent = _running_agent(state, self)
        answer = await agent._aask_human(state, self.question(call, tool), _CHOICES, "aask_human", self.human)
        return self._verdict(answer)

    @staticmethod
    def _verdict(answer: Any) -> Allowed | Denied:
        return Allowed() if answer == "yes" else Denied(_DECLINED, stop=True)

    def __repr__(self) -> str:
        name = type(self).__name__
        if self.human is None:
            return f"{name}()"
        # The Human's own repr only when its class wrote one: object's default has a memory address, which changes
        # from run to run (decided_by and StoppedByPermission.permission are saved and compared).
        human = self.human
        shown = repr(human) if type(human).__repr__ is not object.__repr__ else f"{type(human).__name__}()"
        return f"{name}(human={shown})"


def _running_agent(state: State, permission: Permission) -> Any:
    """The Agent running ``state`` (the one linked to it when its run, ``think``, ``use_tools`` or ``compact``
    started)."""
    agent = state._agent
    if agent is None:
        raise ValueError(
            fix_message(
                f"{permission!r} asks through the Agent running the State, and this State has no Agent yet",
                "Let the Agent call it: give it in Agent(permissions=[...]) and run the Agent",
                "agent = Agent(model=..., tools=[...], permissions=[DecideByHuman()])",
            )
        )
    return agent


# ---------------------------------------------------------------- asking the list (used by _runner)


class _Decision(NamedTuple):
    """What the permissions decided about one call."""

    #: ``Allowed()``, ``Denied(...)``, or ``None`` when nobody decided or a permission raised ``ToolError``.
    verdict: Allowed | Denied | None
    #: The permission that decided (or raised ``ToolError``). ``None`` when nobody decided.
    permission: Permission | None
    #: The ``ToolError`` a permission raised.
    error: ToolError | None = None


def _steps(permissions: tuple[Permission, ...]) -> Generator[Permission, Any, _Decision]:
    """The order of asking, without the asking: yields each permission, receives its verdict, returns the decision.
    Deny permissions first, then the rest in list order. ``TypeError`` for a verdict the permission may not give."""
    for permission in permissions:
        if isinstance(permission, DenyPermission):
            verdict = yield permission
            _check_verdict(permission, verdict)
            if verdict is not None:
                return _Decision(verdict, permission)
    for permission in permissions:
        if not isinstance(permission, DenyPermission):
            verdict = yield permission
            _check_verdict(permission, verdict)
            if verdict is not None:
                return _Decision(verdict, permission)
    return _Decision(None, None)


def _check_verdict(permission: Permission, verdict: Any) -> None:
    allowed = type(permission)._verdicts
    if verdict is None or isinstance(verdict, allowed):
        return
    kind = next(k.__name__ for k in _KINDS if isinstance(permission, k))
    example = (
        'return Allowed() if call.name == "read_file" else None'
        if kind == "AllowPermission"
        else 'return Denied("Deleting files is not allowed") if call.name == "delete_file" else None'
    )
    may = " or ".join([*(f"{v.__name__}(...)" if v is Denied else f"{v.__name__}()" for v in allowed), "None"])
    raise TypeError(
        fix_message(
            f"{permission!r} returned {verdict!r}, and {'an' if kind[0] in 'AEIOU' else 'a'} {kind} may return only "
            f"{may}",
            "Return one of those"
            + (", or subclass DecidePermission to both allow and deny" if kind != "DecidePermission" else ""),
            example,
        )
    )


def _note(error: BaseException, permission: Permission, call: ToolCall) -> None:
    error.add_note(f"exception raised in permission {permission!r} checking {format_call(call)}")


def _check_call(permissions: tuple[Permission, ...], state: State, call: ToolCall, tool: Tool) -> _Decision:
    """Asks ``permissions`` about ``call`` with ``check``. A ``ToolError`` decides as a denial; any other exception
    propagates (with a note naming the permission)."""
    steps = _steps(permissions)
    try:
        permission = next(steps)
    except StopIteration as done:  # an empty list: nobody decides
        return done.value
    while True:
        try:
            verdict = permission.check(state, call, tool)
        except ToolError as e:
            return _Decision(None, permission, e)
        except Exception as e:
            _note(e, permission, call)
            raise
        try:
            permission = steps.send(verdict)
        except StopIteration as done:
            return done.value


async def _acheck_call(permissions: tuple[Permission, ...], state: State, call: ToolCall, tool: Tool) -> _Decision:
    """The async version of ``_check_call``, with ``acheck``."""
    steps = _steps(permissions)
    try:
        permission = next(steps)
    except StopIteration as done:  # an empty list: nobody decides
        return done.value
    while True:
        try:
            verdict = await permission.acheck(state, call, tool)
        except ToolError as e:
            return _Decision(None, permission, e)
        except Exception as e:
            _note(e, permission, call)
            raise
        try:
            permission = steps.send(verdict)
        except StopIteration as done:
            return done.value


def _only_async(permission: Permission) -> bool:
    """Whether ``permission`` implements only ``acheck`` (so a sync ``run`` cannot use it). For a ``DecideByHuman``
    subclass: it changed ``acheck`` and not ``check``."""
    cls = type(permission)
    if isinstance(permission, DecideByHuman) and cls.check is DecideByHuman.check:
        return cls.acheck is not DecideByHuman.acheck
    return cls.check is Permission.check


def _asks_only_async(permission: Permission, agent_human: Any) -> Any:
    """The Human a ``DecideByHuman`` asks (its own, else ``agent_human``) when that Human answers only
    asynchronously (``aask`` without ``ask``), so a sync ``run`` cannot ask it. ``None`` otherwise, and for a
    subclass that changed ``check`` (it may not ask at all)."""
    if not isinstance(permission, DecideByHuman) or type(permission).check is not DecideByHuman.check:
        return None
    human = permission.human if permission.human is not None else agent_human
    if human is None:
        return None
    if isinstance(human, Human):
        return human if type(human).ask is Human.ask else None
    return None if callable(getattr(human, "ask", None)) else human


def _needs_agent_human(permission: Permission) -> bool:
    """Whether ``permission`` asks the Agent's ``human`` (a ``DecideByHuman`` without its own)."""
    return isinstance(permission, DecideByHuman) and permission.human is None
