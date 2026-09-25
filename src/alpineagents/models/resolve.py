"""Turns a model string into an adapter (ARCHITECTURE.md "Model string rules")."""

from __future__ import annotations

from typing import Any

from ..errors import fix_message
from .anthropic import Anthropic
from .base import Model
from .openai_compatible import OpenAICompatible

__all__ = ["resolve_model", "OLLAMA_BASE_URL"]

OLLAMA_BASE_URL = "http://localhost:11434/v1"

_KNOWN_PROVIDERS = ("anthropic", "openai", "ollama")


def resolve_model(model: Any) -> Model:
    """Turns the value passed to ``Agent(model=...)`` into a Model object.

    - ``Model`` instance → as is
    - ``"provider/model"`` (split on the first ``/``; ``:`` is an Ollama tag, not a separator):
        - ``anthropic/x`` → ``Anthropic("x")``
        - ``openai/x`` → ``OpenAICompatible("x")`` (base_url=None, i.e. the default OpenAI URL.
          An extension point: switches to a dedicated ``OpenAI`` adapter once one exists)
        - ``ollama/x`` → ``OpenAICompatible("x", base_url=OLLAMA_BASE_URL)``
        - Any other prefix → ``ValueError``: unknown provider, the known list (anthropic, openai, ollama),
          and the fix ``OpenAICompatible("model", base_url="...")``
        - ``ValueError`` if the model part is empty
    - Names without a prefix: starting with ``claude-`` → Anthropic, starting with ``gpt-`` → same as ``openai/``.
      Otherwise ``ValueError``: the provider is ambiguous; shows the candidates ``"anthropic/{s}"``,
      ``"openai/{s}"``, ``"ollama/{s}"``.
    - Empty string → ``ValueError``. Neither a string nor a Model → ``TypeError``.

    No network or API key is needed at construction (adapter rule).
    """
    if isinstance(model, Model):
        return model

    if not isinstance(model, str):
        raise TypeError(
            fix_message(
                f"model got a value that is neither a Model instance nor a string ({type(model).__name__})",
                "Pass a Model instance or a 'provider/model' string",
                "Agent(model='anthropic/claude-sonnet-5')",
            )
        )

    if not model:
        raise ValueError(
            fix_message(
                "The model string is empty",
                "Fill it in as 'provider/model' (e.g. 'anthropic/claude-sonnet-5')",
                "Agent(model='anthropic/claude-sonnet-5')",
            )
        )

    if "/" in model:
        provider, _, name = model.partition("/")
        if not name:
            raise ValueError(
                fix_message(
                    f"The model string {model!r} has no model name",
                    "Write it as 'provider/model'",
                    "Agent(model='anthropic/claude-sonnet-5')",
                )
            )
        if provider == "anthropic":
            return Anthropic(name)
        if provider == "openai":
            return OpenAICompatible(name)
        if provider == "ollama":
            return OpenAICompatible(name, base_url=OLLAMA_BASE_URL)
        raise ValueError(
            fix_message(
                f"Unknown provider {provider!r}",
                f"Known providers are {', '.join(_KNOWN_PROVIDERS)}. For any other provider, "
                "connect directly with OpenAICompatible(model, base_url='...')",
                f"OpenAICompatible({name!r}, base_url='https://my-server/v1')",
            )
        )

    if model.startswith("claude-"):
        return Anthropic(model)
    if model.startswith("gpt-"):
        return resolve_model(f"openai/{model}")

    raise ValueError(
        fix_message(
            f"The provider is ambiguous from the model name {model!r} alone",
            "Add the provider as a prefix",
            f"anthropic/{model}\nopenai/{model}\nollama/{model}",
        )
    )
