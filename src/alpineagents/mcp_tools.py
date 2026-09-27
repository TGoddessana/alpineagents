"""MCP servers as tools (ARCHITECTURE.md "MCP").

``MCP("npx -y @modelcontextprotocol/server-github", name="github")`` or ``MCP(url="https://...", name="linear")``
goes into ``Agent(tools=[...])``. Its tools are shown to the model as ``github__create_issue``. ``gh.search_code``
(or ``gh["search-code"]``) picks one tool.

Connections live on one background event loop thread shared by every server: the MCP SDK is async and must enter
and leave a connection in the same task, and this way the sync and async APIs use a connection the same way. A
server object counts its users (Agents holding it) and connects on the first, disconnects after the last.

The ``mcp`` package (``pip install "alpineagents[mcp]"``) is imported only when a server connects.
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import threading
from collections.abc import Mapping
from concurrent.futures import Future, wait
from contextlib import AsyncExitStack
from typing import Any

from .errors import MCPConnectionError, ToolError, ToolInputError, fix_message
from .tool import DONE, Tool, hint_values

__all__ = ["MCP", "MCPTool", "MCPToolRef", "SEPARATOR"]

#: Between the server name and the tool name: ``github__create_issue``.
SEPARATOR = "__"
#: Longest tool name the providers accept.
_MAX_NAME = 64
_SERVER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*(?:_[A-Za-z0-9-]+)*$")
_NOT_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]")

_INSTALL_HINT = 'pip install "alpineagents[mcp]"'


# ---------------------------------------------------------------- background loop


class _Loop:
    """One daemon thread running an event loop for every MCP connection."""

    _lock = threading.Lock()
    _loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    def get(cls) -> asyncio.AbstractEventLoop:
        with cls._lock:
            if cls._loop is None:
                loop = asyncio.new_event_loop()
                threading.Thread(target=loop.run_forever, name="alpineagents-mcp", daemon=True).start()
                cls._loop = loop
            return cls._loop

    @classmethod
    def submit(cls, coro: Any) -> Future[Any]:
        return asyncio.run_coroutine_threadsafe(coro, cls.get())


def wait_sync(future: Future[Any]) -> Any:
    """Waits for ``future`` on a sync caller's thread in short steps, so Ctrl+C gets through."""
    while not wait([future], timeout=0.1).done:
        pass
    return future.result()


# ---------------------------------------------------------------- the server object


class MCP:
    """An MCP server, put in ``Agent(tools=[...])`` like any other tool. Needs ``pip install "alpineagents[mcp]"``.

    The model sees the server's tools as ``{name}__{tool}``, e.g. ``github__create_issue``. To give the Agent
    only one tool, pass ``gh.search_code``, or ``gh["search-code"]`` for a name that is not a Python identifier.

    ``agent.run`` connects the servers and disconnects them when it ends. ``with agent:`` (or
    ``async with agent:``) keeps them connected across runs. Agents and concurrent runs that share one ``MCP``
    object share one connection.

    Example:
        ```python
        github = MCP("npx -y @modelcontextprotocol/server-github", name="github", env={"GITHUB_TOKEN": token})
        linear = MCP(url="https://mcp.linear.app/mcp", name="linear", headers={"Authorization": f"Bearer {key}"})
        agent = Agent(model="claude-sonnet-5", tools=[github, linear.list_issues])
        ```
    """

    def __init__(
        self,
        command: str | None = None,
        *,
        name: str | None = None,
        url: str | None = None,
        server: Any = None,
        env: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Takes exactly one of ``command``, ``url`` or ``server``. Nothing connects until the Agent uses it.

        Args:
            command: The command that starts a stdio server.
            name: Required. Letters, digits, ``-`` and single ``_``. Prefixes the tool names.
            url: The URL of a Streamable HTTP server.
            server: Anything ``mcp.Client`` accepts, such as an in-process server in tests.
            env: Extra environment variables for ``command``, added to the MCP SDK's safe default environment.
            headers: HTTP headers for ``url``, e.g. ``{"Authorization": "Bearer ..."}``.
            cwd: The working directory for ``command``.
            timeout: Seconds to wait for connecting and for each call. ``None`` waits without a limit.

        Raises:
            TypeError: No server or more than one is given, ``env`` without ``command``, or ``headers`` without
                ``url``.
            ValueError: ``name`` is missing or not allowed, or ``command`` is empty.
        """
        given = [label for label, value in (("command", command), ("url", url), ("server", server)) if value]
        if len(given) != 1:
            raise TypeError(
                fix_message(
                    "MCP() takes exactly one server: a command, url= or server="
                    + (f" (got: {', '.join(given)})" if given else ""),
                    "Pass the command that starts the server, or the URL of a running one",
                    'MCP("npx -y @modelcontextprotocol/server-github", name="github")\n'
                    'MCP(url="https://mcp.linear.app/mcp", name="linear")',
                )
            )
        if name is None:
            raise ValueError(
                fix_message(
                    "MCP() needs name=: the model sees this server's tools as {name}__{tool}",
                    "Give the server a short name",
                    'MCP("npx -y @modelcontextprotocol/server-github", name="github")',
                )
            )
        if not isinstance(name, str) or not _SERVER_NAME.match(name):
            raise ValueError(
                fix_message(
                    f"MCP name {name!r} is not allowed",
                    "Use letters, digits, '-' and single '_' (no '__': it separates the server from the tool)",
                    'MCP(..., name="github")',
                )
            )
        if env is not None and command is None:
            raise TypeError(fix_message("env= is for a server started by a command", "Remove env="))
        if headers is not None and url is None:
            raise TypeError(fix_message("headers= is for a server reached by url=", "Remove headers="))

        self.name = name
        """The server name that prefixes its tool names."""
        self.command = command
        """The command that starts a stdio server, or ``None``."""
        self.url = url
        """The URL of a Streamable HTTP server, or ``None``."""
        self.server = server
        """The object given as ``server=``, or ``None``."""
        self.env = dict(env) if env is not None else None
        """Extra environment variables for ``command``, or ``None``."""
        self.headers = dict(headers) if headers is not None else None
        """HTTP headers for ``url``, or ``None``."""
        self.cwd = cwd
        """The working directory for ``command``, or ``None``."""
        self.timeout = timeout
        """Seconds to wait for connecting and for each call, or ``None`` for no limit."""
        if command is not None:
            argv = shlex.split(command)
            if not argv:
                raise ValueError(fix_message("MCP() got an empty command", "Pass the command that starts the server"))
            self._argv = argv

        self._lock = threading.Lock()
        self._users = 0
        #: The current connection (``None`` when nobody uses the server). A new one after each disconnect.
        self._conn: _Connection | None = None

    def __getattr__(self, tool_name: str) -> MCPToolRef:
        """``gh.search_code``: one tool of this server, for ``tools=[...]`` or ``think(tools=[...])``."""
        if tool_name.startswith("_"):
            raise AttributeError(tool_name)
        return MCPToolRef(self, tool_name)

    def __getitem__(self, tool_name: str) -> MCPToolRef:
        """``gh["search-code"]``: the same, for a tool name that is not a Python identifier."""
        return MCPToolRef(self, tool_name)

    def __repr__(self) -> str:
        target = self.command if self.command is not None else self.url if self.url is not None else self.server
        return f"MCP({target!r}, name={self.name!r})"

    # ------------------------------------------------------------ connection (any thread)

    def _acquire(self) -> Future[tuple[Any, ...]]:
        """One more user. The first connects. Returns the future of the tool list."""
        with self._lock:
            self._users += 1
            if self._conn is None:
                conn = _Connection()
                conn.serving = _Loop.submit(self._serve(conn))
                self._conn = conn
            return self._conn.ready

    def _release(self) -> Future[None] | None:
        """One user less. After the last, disconnects; returns the future of the disconnect."""
        with self._lock:
            self._users -= 1
            conn = self._conn
            if self._users > 0 or conn is None:
                return None
            self._conn = None
        _Loop.get().call_soon_threadsafe(conn.close)
        return conn.serving

    async def _serve(self, conn: _Connection) -> None:
        """The connection's task on the background loop: connects, lists the tools, waits for ``_release``."""
        ready = conn.ready
        conn.task = asyncio.current_task()
        try:
            try:
                import mcp
            except ImportError as e:
                raise MCPConnectionError(
                    fix_message(
                        f"MCP server {self.name!r} needs the mcp package, which is not installed",
                        f"Install it: {_INSTALL_HINT}",
                    )
                ) from e
            async with AsyncExitStack() as stack:
                target = await self._target(stack)
                client = mcp.Client(target, read_timeout_seconds=self.timeout)
                async with asyncio.timeout(self.timeout):
                    await stack.enter_async_context(client)
                    tools = await _list_tools(client)
                conn.client = client
                ready.set_result(tools)
                await conn.stop.wait()
        except BaseException as e:
            if not ready.done():
                error = e if isinstance(e, MCPConnectionError) else MCPConnectionError(
                    f"Could not connect to MCP server {self.name!r} ({self!r}): {_describe(e)}"
                )
                if error is not e:
                    error.__cause__ = e
                ready.set_exception(error)
            else:
                conn.lost = e
            if isinstance(e, asyncio.CancelledError):
                raise
        finally:
            conn.client = None

    async def _target(self, stack: AsyncExitStack) -> Any:
        """What ``mcp.Client`` connects to."""
        if self.server is not None:
            return self.server
        if self.url is not None:
            if self.headers is None:
                return self.url
            from mcp.client.streamable_http import streamable_http_client
            from mcp.shared._httpx_utils import create_mcp_http_client

            http = await stack.enter_async_context(create_mcp_http_client(headers=self.headers))
            return streamable_http_client(self.url, http_client=http)
        from mcp import StdioServerParameters

        return StdioServerParameters(command=self._argv[0], args=self._argv[1:], env=self.env, cwd=self.cwd)

    async def _call(self, remote_name: str, args: dict[str, Any]) -> Any:
        """Calls a tool on the background loop. A server error response becomes an error result; a broken
        connection raises ``MCPConnectionError``."""
        conn = self._conn
        client = conn.client if conn is not None else None
        if client is None:
            raise self._lost_error(conn.lost if conn is not None else None)
        try:
            return await client.call_tool(remote_name, args)
        except Exception as e:
            if not _is_transport_error(e):
                # The server answered, but with an error or a result the SDK rejects: the model sees it.
                return _ErrorResult(f"(error from MCP server {self.name!r}: {_describe(e)})")
            lost = e
        if conn is not None and conn.lost is None:
            conn.lost = lost
            conn.client = None
        raise MCPConnectionError(f"Lost the connection to MCP server {self.name!r}: {_describe(lost)}") from lost

    def _lost_error(self, lost: BaseException | None) -> MCPConnectionError:
        error = MCPConnectionError(
            fix_message(
                f"MCP server {self.name!r} is not connected"
                + (f" (connection lost: {_describe(lost)})" if lost is not None else ""),
                "A lost connection is opened again by the next run (or the next with agent:) after this one "
                "ends. Tools of an MCP server work inside agent.run(), or inside with agent: / async with agent:",
            )
        )
        error.__cause__ = lost
        return error


class _Connection:
    """One connection's state. ``ready`` (any thread) holds the tool list; the rest belongs to the background
    loop."""

    def __init__(self) -> None:
        self.ready: Future[tuple[Any, ...]] = Future()
        self.serving: Future[None] | None = None
        self.task: asyncio.Task[Any] | None = None
        self.stop = asyncio.Event()
        self.client: Any = None
        self.lost: BaseException | None = None

    def close(self) -> None:
        """On the background loop: a connected server stops waiting; one still connecting is cancelled."""
        if self.ready.done():
            self.stop.set()
        elif self.task is not None:
            self.task.cancel()


class MCPToolRef:
    """``gh.search_code``: one tool of a server, looked up when the server connects."""

    def __init__(self, server: MCP, tool_name: str) -> None:
        self.server = server
        self.tool_name = tool_name

    @property
    def name(self) -> str:
        """The name the model sees: ``github__search_code``."""
        return tool_name(self.server, self.tool_name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MCPToolRef) and other.server is self.server and other.tool_name == self.tool_name

    def __hash__(self) -> int:
        return hash((id(self.server), self.tool_name))

    def __repr__(self) -> str:
        return f"{self.server.name}.{self.tool_name}"


class MCPTool(Tool):
    """A tool of a connected MCP server, as found in ``agent.tool_map`` under ``{server}__{tool}``.

    It has the same ``name``, ``description``, ``input_schema`` and hints as any ``Tool``. The hints come from the
    server's tool annotations (``readOnlyHint`` and so on), with the same defaults as ``@tool`` for the ones it
    leaves out. They are what the server says about itself, not a guarantee: do not trust them more than the server.
    """

    server: MCP
    """The MCP server the tool belongs to."""
    remote_name: str
    """The tool's name on the server (``name`` has the ``{server}__`` prefix)."""
    read_only: bool
    """From ``readOnlyHint``. ``False`` if the server does not say."""
    destructive: bool
    """From ``destructiveHint``. ``True`` if the server does not say, and ``False`` for a read-only tool."""
    idempotent: bool
    """From ``idempotentHint``. ``False`` if the server does not say, and ``True`` for a read-only tool."""
    open_world: bool
    """From ``openWorldHint``. ``True`` if the server does not say."""

    def __init__(self, server: MCP, remote: Any) -> None:
        self.server = server
        self.remote_name = remote.name
        schema = dict(remote.input_schema or {})
        schema.setdefault("type", "object")
        # A server's hints are read leniently: a value that is not a bool counts as not given.
        annotations = getattr(remote, "annotations", None)

        def hint(field: str) -> bool | None:
            value = getattr(annotations, field, None)
            return value if isinstance(value, bool) else None

        read_only, destructive, idempotent, open_world = hint_values(
            hint("read_only_hint"), hint("destructive_hint"), hint("idempotent_hint"), hint("open_world_hint")
        )
        super().__init__(
            name=tool_name(server, remote.name),
            description=remote.description or "",
            input_schema=schema,
            read_only=read_only,
            destructive=destructive,
            idempotent=idempotent,
            open_world=open_world,
        )

    async def run(self, args: dict[str, Any], state: Any) -> str:
        """Calls the tool on the server. Missing required arguments are an input error; the server checks the
        rest. An error the server reports (``isError``, or an error response) becomes an error result; a lost
        connection raises ``MCPConnectionError``.

        Text blocks are joined by newlines, other blocks become short placeholders, ``structured_content`` is sent
        as JSON when there are no blocks, and ``(done)`` when there is nothing."""
        missing = [key for key in self.input_schema.get("required", ()) if key not in args]
        if missing:
            raise ToolInputError(f"missing required arguments: {', '.join(missing)}")
        # Runs on the background loop; awaited from the caller's loop, so a cancel reaches it.
        result = await asyncio.wrap_future(_Loop.submit(self.server._call(self.remote_name, args)))
        if isinstance(result, _ErrorResult):
            raise ToolError(result.text)
        parts = [_content_text(block) for block in (result.content or ())]
        text = "\n".join(part for part in parts if part)
        if not text and result.structured_content is not None:
            text = json.dumps(result.structured_content, ensure_ascii=False)
        if getattr(result, "is_error", False):
            raise ToolError(text or f"(MCP server {self.server.name!r} reported an error without a message)")
        return text or DONE


class _ErrorResult:
    """A server error response (JSON-RPC error) turned into an error result for the model."""

    def __init__(self, text: str) -> None:
        self.text = text


# ---------------------------------------------------------------- helpers


def tool_name(server: MCP, remote_name: str) -> str:
    """``{server}__{tool}`` with characters providers reject replaced by ``_``."""
    return f"{server.name}{SEPARATOR}{_NOT_NAME_CHARS.sub('_', remote_name)}"


async def _list_tools(client: Any) -> tuple[Any, ...]:
    tools: list[Any] = []
    cursor = None
    while True:
        page = await client.list_tools(cursor=cursor)
        tools.extend(page.tools)
        cursor = page.next_cursor
        if not cursor:
            return tuple(tools)


def _content_text(block: Any) -> str:
    kind = getattr(block, "type", None)
    if kind == "text":
        return block.text
    if kind == "resource":
        resource = block.resource
        text = getattr(resource, "text", None)
        return text if text is not None else f"(resource {resource.uri}: {getattr(resource, 'mime_type', None)})"
    if kind == "resource_link":
        return f"(resource link: {block.uri})"
    if kind in ("image", "audio"):
        return f"({kind} {getattr(block, 'mime_type', '')} not shown: {kind} results are not supported yet)"
    return f"({kind} content)"


def _is_transport_error(error: BaseException) -> bool:
    """Whether a call failed because the connection broke (not because of what the server answered)."""
    import anyio
    import httpx2
    from mcp.shared.exceptions import MCPError
    from mcp_types import CONNECTION_CLOSED

    if isinstance(error, MCPError):
        return getattr(error.error, "code", None) == CONNECTION_CLOSED
    return isinstance(
        error,
        (OSError, EOFError, anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream,
         httpx2.TransportError),
    )


def _describe(error: BaseException | None) -> str:
    """A one-line description; an exception group shows its first leaf (anyio wraps transport errors)."""
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    if error is None:
        return ""
    message = str(error)
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def check_tool_names(server: MCP, tools: tuple[Any, ...], only: frozenset[str] | None) -> list[MCPTool]:
    """The server's tools this Agent uses, checked: every name in ``only`` exists, and each name the model sees
    is unique and at most 64 characters."""
    by_remote = {remote.name: remote for remote in tools}
    if only is not None:
        missing = sorted(only - set(by_remote))
        if missing:
            raise ValueError(
                fix_message(
                    f"MCP server {server.name!r} has no tool {', '.join(map(repr, missing))} "
                    f"(its tools: {', '.join(by_remote) or '(none)'})",
                    "Use one of the server's tool names",
                    f"tools=[{server.name}[{next(iter(by_remote), 'tool_name')!r}]]",
                )
            )
    # Names are checked before the MCPTools are made (Tool.__init__ would reject a long name less clearly).
    seen: dict[str, str] = {}
    chosen = []
    for remote_name, remote in by_remote.items():
        if only is not None and remote_name not in only:
            continue
        name = tool_name(server, remote_name)
        if len(name) > _MAX_NAME:
            raise ValueError(
                fix_message(
                    f"MCP tool name {name!r} is longer than {_MAX_NAME} characters",
                    "Give the server a shorter name=",
                )
            )
        if name in seen:
            raise ValueError(
                fix_message(
                    f"MCP server {server.name!r} has tools {seen[name]!r} and {remote_name!r}, "
                    f"which both become {name!r}",
                    f"Pick one of them with tools=[{server.name}[...]]",
                )
            )
        seen[name] = remote_name
        chosen.append(MCPTool(server, remote))
    return chosen
