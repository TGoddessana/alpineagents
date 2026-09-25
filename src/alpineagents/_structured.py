"""Format instructions and validation for ``agent.ask(returns=...)``. Owned by Agent.

``ask``'s ``returns`` takes ``str``, a dataclass or a Pydantic model (and any other type Pydantic's
``TypeAdapter`` can handle). If the reply does not match the format, the validation error is shown to the model
and it is asked again.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .errors import fix_message

__all__ = ["check_returns", "question_text", "parse_reply", "retry_text"]

_FENCE = re.compile(r"```[ \t]*[A-Za-z0-9_-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)
_ADAPTERS: dict[Any, TypeAdapter[Any]] = {}


def _adapter(returns: Any) -> TypeAdapter[Any]:
    """Build a ``TypeAdapter``. Hashable types are built only once."""
    try:
        cached = _ADAPTERS.get(returns)
    except TypeError:  # unhashable type
        return TypeAdapter(returns)
    if cached is None:
        cached = TypeAdapter(returns)
        _ADAPTERS[returns] = cached
    return cached


def _schema_text(returns: Any) -> str:
    return json.dumps(_adapter(returns).json_schema(), ensure_ascii=False)


def check_returns(returns: Any) -> None:
    """``TypeError`` if no JSON schema can be built from ``returns`` (``TypeAdapter(returns).json_schema()``
    fails, ``None``, etc.). Fix: pass str, a dataclass or a Pydantic model."""
    if returns is str:
        return
    problem = f"ask(returns={returns!r}) is not a supported type"
    fix = "pass the type itself to returns, such as str, a dataclass or a Pydantic model"
    example = (
        "@dataclass\n"
        "class Plan:\n"
        "    steps: list[str]\n"
        "\n"
        'plan = agent.ask(state, "Break the task into steps", returns=Plan)'
    )
    if returns is None or isinstance(returns, (str, bytes, int, float, bool)):
        raise TypeError(fix_message(problem, fix, example))
    try:
        _schema_text(returns)
    except Exception as e:  # a type Pydantic cannot handle
        raise TypeError(fix_message(problem, fix, example)) from e


def question_text(prompt: str, returns: Any) -> str:
    """The question sent to the model. For ``str``, ``prompt`` as is.

    Otherwise ``f"{prompt}\\n\\nReply with a single JSON value matching the JSON schema below. Do not write
    anything else.\\n{schema}"`` (schema is ``json.dumps(TypeAdapter(returns).json_schema(), ensure_ascii=False)``).
    """
    if returns is str:
        return prompt
    return (
        f"{prompt}\n\nReply with a single JSON value matching the JSON schema below. Do not write anything else.\n"
        f"{_schema_text(returns)}"
    )


def _strip_fence(text: str) -> str:
    match = _FENCE.search(text)
    return match.group(1).strip() if match else text


def _balanced_end(text: str, start: int) -> int | None:
    """Index of the closing bracket matching ``text[start]`` (``{`` or ``[``). ``None`` if there is none."""
    stack: list[str] = []
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or stack.pop() != char:
                return None
            if not stack:
                return index
    return None


def _extract_json(text: str) -> str | None:
    """Find one JSON object or array amid surrounding chatter."""
    for start, char in enumerate(text):
        if char not in "{[":
            continue
        end = _balanced_end(text, start)
        if end is None:
            continue
        candidate = text[start : end + 1]
        try:
            json.loads(candidate)
        except ValueError:
            continue
        return candidate
    return None


def _describe(error: ValidationError) -> str:
    """Shorten a Pydantic error so the model can fix it."""
    parts = []
    for item in error.errors(include_url=False):
        loc = ".".join(str(part) for part in item.get("loc", ()))
        message = item.get("msg", "")
        parts.append(f"{loc}: {message}" if loc else message)
    return "; ".join(parts) or str(error)


def parse_reply(text: str, returns: Any) -> Any:
    """Convert the model's reply text to ``returns``. On mismatch, ``ValueError`` (the message is the validation
    error shown to the model).

    - ``str``: ``text.strip()`` (``ValueError`` if the reply is empty)
    - otherwise: if the whole reply is JSON, use it as is. If not, strip the code fence (```json ... ```), and if
      there is surrounding chatter, take the first ``{``/``[`` up to its matching end as the JSON (from the raw
      text if it is not found inside the fence). Validate with ``TypeAdapter(returns).validate_json`` (an object
      for a dataclass, an instance for a Pydantic model). ``pydantic.ValidationError``/JSON errors are re-raised
      as ``ValueError``.
    """
    stripped = (text or "").strip()
    if returns is str:
        if not stripped:
            raise ValueError("the reply is empty")
        return stripped
    if not stripped:
        raise ValueError("the reply is empty. Reply with a single JSON value")
    candidate = stripped
    try:
        json.loads(candidate)  # already JSON: use as is (so ``` fences inside string values are not stripped)
    except ValueError:
        candidate = _strip_fence(stripped)
        try:
            json.loads(candidate)
        except ValueError:
            # if JSON in a fence has ``` inside a string value, the fence regex cuts it short: also search the raw text
            extracted = _extract_json(candidate) or _extract_json(stripped)
            if extracted is None:
                raise ValueError("no JSON found in the reply. Reply with a single JSON value only") from None
            candidate = extracted
    try:
        return _adapter(returns).validate_json(candidate)
    except ValidationError as e:
        raise ValueError(_describe(e)) from e


def retry_text(error: ValueError) -> str:
    """The user message added when asking again:
    ``f"The reply does not match the format: {error}\\nReply again in the required format."``"""
    return f"The reply does not match the format: {error}\nReply again in the required format."
