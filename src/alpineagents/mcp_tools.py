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

from .errors import MCPConnectionError, ToolInputError, fix_message
from .tool import DONE
from .types import INVALID_ARGS_KEY, TRUNCATED_ARGS_MESSAGE, ToolSpec

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
    """An MCP server: ``MCP(command, name=...)`` (stdio), ``MCP(url=..., name=...)`` (Streamable HTTP), or
    ``MCP(server=..., name=...)`` (anything ``mcp.Client`` accepts, such as an in-process ``MCPServer``).

    - ``name`` is required: tools show up as ``{name}__{tool}``. Letters, digits, ``-`` and single ``_``.
    - ``env``: extra environment variables for a command (added to the MCP SDK's safe default environment).
    - ``headers``: HTTP headers for a URL (e.g. ``{"Authorization": "Bearer ..."}``).
    - ``timeout``: seconds to wait for connecting and for each call (``None``: no limit).

    Mistakes are reported here: no server or more than one, a missing or invalid ``name``, ``env`` without a
    command, ``headers`` without a URL.
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
        self.command = command
        self.url = url
        self.server = server
        self.env = dict(env) if env is not None else None
        self.headers = dict(headers) if headers is not None else None
        self.cwd = cwd
        self.timeout = timeout
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


class MCPTool:
    """A connected server's tool in an Agent's tool map. Same interface as ``Tool`` for the runner: ``spec``,
    ``parallel``, ``is_async``, ``prepare``, ``invoke``, ``format_result``, ``is_error_result``."""

    parallel = True
    is_async = True

    def __init__(self, server: MCP, remote: Any) -> None:
        self.server = server
        self.remote_name: str = remote.name
        self.name = tool_name(server, remote.name)
        self.description: str = remote.description or ""
        schema = dict(remote.input_schema or {})
        schema.setdefault("type", "object")
        self.input_schema: dict[str, Any] = schema

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    def prepare(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """Argument JSON that could not be read, and missing required arguments, are input errors. The server
        checks the rest (its errors go to the model)."""
        if INVALID_ARGS_KEY in args:
            raw = args[INVALID_ARGS_KEY]
            if raw == TRUNCATED_ARGS_MESSAGE:
                raise ToolInputError(raw)
            raise ToolInputError(f"arguments are not valid JSON: {raw}")
        missing = [key for key in self.input_schema.get("required", ()) if key not in args]
        if missing:
            raise ToolInputError(f"missing required arguments: {', '.join(missing)}")
        return dict(args)

    def invoke(self, kwargs: Mapping[str, Any], state: Any) -> Any:
        return self._ainvoke(dict(kwargs))

    async def _ainvoke(self, args: dict[str, Any]) -> Any:
        """Runs the call on the background loop and awaits it from the caller's loop (a cancel reaches it)."""
        return await asyncio.wrap_future(_Loop.submit(self.server._call(self.remote_name, args)))

    @staticmethod
    def format_result(result: Any) -> str:
        """Text blocks joined by newlines; other blocks as short placeholders; ``structured_content`` as JSON
        when there are no blocks; ``(done)`` when there is nothing."""
        if isinstance(result, _ErrorResult):
            return result.text
        parts = [_content_text(block) for block in (result.content or ())]
        text = "\n".join(part for part in parts if part)
        if not text and result.structured_content is not None:
            text = json.dumps(result.structured_content, ensure_ascii=False)
        return text or DONE

    @staticmethod
    def is_error_result(result: Any) -> bool:
        return isinstance(result, _ErrorResult) or bool(getattr(result, "is_error", False))

    def __repr__(self) -> str:
        return f"MCPTool({self.name!r})"


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
    chosen = [MCPTool(server, remote) for name, remote in by_remote.items() if only is None or name in only]
    seen: dict[str, str] = {}
    for mcp_tool in chosen:
        if len(mcp_tool.name) > _MAX_NAME:
            raise ValueError(
                fix_message(
                    f"MCP tool name {mcp_tool.name!r} is longer than {_MAX_NAME} characters",
                    "Give the server a shorter name=",
                )
            )
        if mcp_tool.name in seen:
            raise ValueError(
                fix_message(
                    f"MCP server {server.name!r} has tools {seen[mcp_tool.name]!r} and {mcp_tool.remote_name!r}, "
                    f"which both become {mcp_tool.name!r}",
                    f"Pick one of them with tools=[{server.name}[...]]",
                )
            )
        seen[mcp_tool.name] = mcp_tool.remote_name
    return chosen
