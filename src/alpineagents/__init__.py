"""alpineagents: an agent looks at the state, thinks, and uses tools when needed, until there is an answer."""

from .agent import Agent
from .blocks import CompactIfFull, acompact_if_full, compact_if_full
from .errors import (
    AlpineAgentsError,
    AuthError,
    ContextTooLongError,
    MCPConnectionError,
    NoHumanError,
    OutputError,
    ProviderError,
    RateLimitError,
)
from .human import Human
from .loop import Loop, adefault_loop, default_loop, loop
from .mcp_tools import MCP
from .models import Anthropic, Model, OpenAICompatible
from .reporter import Reporter
from .state import State
from .terminal import Terminal
from .tool import Tool, tool
from .types import (
    ContextChange,
    HistoryEntry,
    Message,
    ModelEvent,
    Price,
    Reply,
    Request,
    ToolCall,
    ToolOutcome,
    ToolSpec,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    # Objects
    "Agent",
    "State",
    "loop",
    "Loop",
    "default_loop",
    "adefault_loop",
    "tool",
    "Tool",
    "compact_if_full",
    "acompact_if_full",
    "MCP",
    "CompactIfFull",
    # Roles and implementations
    "Model",
    "Anthropic",
    "OpenAICompatible",
    "Reporter",
    "Human",
    "Terminal",
    # Data
    "Price",
    "Usage",
    "ToolCall",
    "Reply",
    "Request",
    "Message",
    "ToolSpec",
    "HistoryEntry",
    "ContextChange",
    "ModelEvent",
    "ToolOutcome",
    # Errors
    "AlpineAgentsError",
    "ProviderError",
    "RateLimitError",
    "ContextTooLongError",
    "AuthError",
    "OutputError",
    "NoHumanError",
    "MCPConnectionError",
]
