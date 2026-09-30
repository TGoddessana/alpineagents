"""Turns a model string into an adapter (ARCHITECTURE.md "Model string rules")."""

from __future__ import annotations

import inspect
from typing import Any

from ..errors import fix_message
from .anthropic import Anthropic
from .base import Model
from .openai_compatible import OpenAICompatible

__all__ = ["resolve_model", "OLLAMA_BASE_URL"]

OLLAMA_BASE_URL = "http://localhost:11434/v1"

_KNOWN_PROVIDERS = ("anthropic", "openai", "ollama")


def resolve_model(model: Any, **options: Any) -> Model:
    """Turns the value passed to ``Agent(model=...)`` into a Model object.

    Keyword ``options`` go to the adapter it picks: ``resolve_model("anthropic/claude-sonnet-5",
    max_tokens=16_000)`` is ``Anthropic("claude-sonnet-5", max_tokens=16_000)``.

    - ``Model`` instance → as is. With options → ``TypeError`` (set them when creating the Model)
    - ``base_url=`` given (not ``None``) → ``OpenAICompatible(model, **options)`` with the whole string as the
      model name, not split: routers such as OpenRouter use names like ``"anthropic/claude-sonnet-5"``.
      ``ollama/x`` is the one exception (below). For an Anthropic-format proxy, create
      ``Anthropic(..., base_url=...)`` directly.
    - ``"provider/model"`` (split on the first ``/``; ``:`` is an Ollama tag, not a separator):
        - ``anthropic/x`` → ``Anthropic("x", **options)``
        - ``openai/x`` → ``OpenAICompatible("x", **options)`` (base_url=None, i.e. the default OpenAI URL.
          An extension point: switches to a dedicated ``OpenAI`` adapter once one exists)
        - ``ollama/x`` → ``OpenAICompatible("x", base_url=OLLAMA_BASE_URL, **options)``; an explicit
          ``base_url`` wins (an Ollama server on another machine)
        - Any other prefix → ``ValueError``: unknown provider, the known list (anthropic, openai, ollama),
          and the fix ``OpenAICompatible("model", base_url="...")``
        - ``ValueError`` if the model part is empty
    - Names without a prefix: starting with ``claude-`` → Anthropic, starting with ``gpt-`` → same as ``openai/``.
      Otherwise ``ValueError``: the provider is ambiguous; shows the candidates ``"anthropic/{s}"``,
      ``"openai/{s}"``, ``"ollama/{s}"``.
    - Empty string → ``ValueError``. Neither a string nor a Model → ``TypeError``.
    - An option the chosen adapter does not accept → ``TypeError`` naming the adapter and the options it accepts.
      Only the names are checked here; the adapter checks the values.

    No network or API key is needed at construction (adapter rule).
    """
    if isinstance(model, Model):
        if options:
            raise TypeError(
                fix_message(
                    f"resolve_model got a Model instance ({type(model).__name__}) together with options "
                    f"({', '.join(options)})",
                    "Options are only for model strings. Set them when you create the Model",
                    "Anthropic('claude-sonnet-5', max_tokens=16_000)",
                )
            )
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

    # base_url=None means "not given": drop it so the rules below (and the Ollama default URL) apply.
    if "base_url" in options and options["base_url"] is None:
        del options["base_url"]

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
        if provider == "ollama":
            return _build(OpenAICompatible, model, name, {"base_url": OLLAMA_BASE_URL, **options})
        if "base_url" in options:
            return _build(OpenAICompatible, model, model, options)
        if provider == "anthropic":
            return _build(Anthropic, model, name, options)
        if provider == "openai":
            return _build(OpenAICompatible, model, name, options)
        raise ValueError(
            fix_message(
                f"Unknown provider {provider!r}",
                f"Known providers are {', '.join(_KNOWN_PROVIDERS)}. For any other provider, "
                "connect directly with OpenAICompatible(model, base_url='...')",
                f"OpenAICompatible({name!r}, base_url='https://my-server/v1')",
            )
        )

    if "base_url" in options or model.startswith("gpt-"):
        return _build(OpenAICompatible, model, model, options)
    if model.startswith("claude-"):
        return _build(Anthropic, model, model, options)

    raise ValueError(
        fix_message(
            f"The provider is ambiguous from the model name {model!r} alone",
            "Add the provider as a prefix",
            f"anthropic/{model}\nopenai/{model}\nollama/{model}",
        )
    )


def _build(
    adapter: type[Anthropic] | type[OpenAICompatible], model: str, name: str, options: dict[str, Any]
) -> Model:
    """``adapter(name, **options)``, after checking that ``adapter`` accepts every option name.

    ``model`` is the string as given, for the error message. The check runs before the call, so a ``TypeError``
    raised inside the adapter is never reported as an unknown option.
    """
    parameters = inspect.signature(adapter).parameters.values()
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        keyword_kinds = (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        accepted = [p.name for p in parameters if p.kind in keyword_kinds and p.name != "name"]
        unknown = [key for key in options if key not in accepted]
        if unknown:
            raise TypeError(
                fix_message(
                    f"The model string {model!r} picked {adapter.__name__}, which does not accept "
                    f"{', '.join(f'{key}=' for key in unknown)}",
                    f"Pass only options {adapter.__name__} accepts: {', '.join(accepted)}",
                    f"resolve_model({model!r}, max_tokens=16_000)",
                )
            )
    return adapter(name, **options)
