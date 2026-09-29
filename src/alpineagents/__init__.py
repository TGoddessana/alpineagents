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
    PermissionWarning,
    ProviderError,
    RateLimitError,
    ResumeWarning,
    ToolError,
    ToolInputError,
)
from .human import Human
from .loop import Loop, adefault_loop, default_loop, loop
from .mcp_tools import MCP, MCPTool
from .models import Anthropic, Model, OpenAICompatible
from .reporter import Reporter
from .state import State
from .store import FileStore, SavedState, Store
from .terminal import Terminal
from .tool import FunctionTool, Hints, Tool, tool
from .types import (
    ContextChange,
    HistoryEntry,
    Image,
    Message,
    ModelEvent,
    Price,
    Reply,
    Request,
    StoppedByFinish,
    StoppedByLimit,
    StoppedByPermission,
    StoppedByUntil,
    ToolCall,
    ToolOutcome,
    ToolOutcomeKind,
    ToolSpec,
    Usage,
)

__version__ = "0.3.0"

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
    "FunctionTool",
    "Hints",
    "compact_if_full",
    "acompact_if_full",
    "MCP",
    "MCPTool",
    "CompactIfFull",
    # Roles and implementations
    "Model",
    "Anthropic",
    "OpenAICompatible",
    "Reporter",
    "Human",
    "Store",
    "FileStore",
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
    "ToolOutcomeKind",
    "StoppedByUntil",
    "StoppedByLimit",
    "StoppedByFinish",
    "StoppedByPermission",
    "SavedState",
    "Image",
    # Errors
    "AlpineAgentsError",
    "ProviderError",
    "RateLimitError",
    "ContextTooLongError",
    "AuthError",
    "OutputError",
    "NoHumanError",
    "MCPConnectionError",
    "ResumeWarning",
    "PermissionWarning",
    "ToolError",
    "ToolInputError",
]
