"""A small MCP server for the tests: ``python mcp_demo_server.py`` (stdio) or
``python mcp_demo_server.py http PORT`` (Streamable HTTP on 127.0.0.1)."""

import logging
import os
import sys
import time

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

logging.disable(logging.CRITICAL)

server = MCPServer("demo")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@server.tool()
def greeting() -> str:
    """The GREETING environment variable"""
    return os.environ.get("GREETING", "(unset)")


@server.tool()
def fail(message: str) -> str:
    """Always fails"""
    raise ToolError(message)


@server.tool()
def crash() -> str:
    """Ends the server process"""
    os._exit(1)


if __name__ == "__main__":
    time.sleep(float(os.environ.get("START_DELAY", "0")))  # a slow server, to cancel while connecting
    if len(sys.argv) > 1 and sys.argv[1] == "http":
        server.run("streamable-http", host="127.0.0.1", port=int(sys.argv[2]))
    else:
        server.run()
