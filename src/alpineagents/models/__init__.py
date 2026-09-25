"""The Model role and its adapters."""

from .anthropic import Anthropic
from .base import Model
from .openai_compatible import OpenAICompatible
from .resolve import resolve_model

__all__ = ["Model", "Anthropic", "OpenAICompatible", "resolve_model"]
