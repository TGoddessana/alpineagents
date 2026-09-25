"""Tests for ``models.resolve.resolve_model`` (ARCHITECTURE.md "Model string rules"). No network."""

import pytest

from alpineagents.models.anthropic import Anthropic
from alpineagents.models.openai_compatible import OpenAICompatible
from alpineagents.models.resolve import OLLAMA_BASE_URL, resolve_model


def test_model_instance_passthrough():
    model = Anthropic("claude-sonnet-5")
    assert resolve_model(model) is model


def test_anthropic_prefix():
    model = resolve_model("anthropic/claude-sonnet-5")
    assert isinstance(model, Anthropic)
    assert model.name == "claude-sonnet-5"


def test_bare_claude_name():
    model = resolve_model("claude-sonnet-5")
    assert isinstance(model, Anthropic)
    assert model.name == "claude-sonnet-5"


def test_openai_prefix_uses_default_url():
    model = resolve_model("openai/gpt-5")
    assert isinstance(model, OpenAICompatible)
    assert model.name == "gpt-5"
    assert model.base_url is None


def test_bare_gpt_name_same_as_openai_prefix():
    model = resolve_model("gpt-5")
    assert isinstance(model, OpenAICompatible)
    assert model.name == "gpt-5"
    assert model.base_url is None


def test_ollama_prefix_sets_base_url_and_keeps_colon_tag():
    model = resolve_model("ollama/llama3:8b")
    assert isinstance(model, OpenAICompatible)
    assert model.name == "llama3:8b"
    assert model.base_url == OLLAMA_BASE_URL


def test_unknown_prefix_raises_value_error_listing_known_providers():
    with pytest.raises(ValueError, match="Unknown provider"):
        resolve_model("mystery/x")


def test_ambiguous_bare_name_raises_value_error_with_candidates():
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_model("mystery-model")


def test_missing_model_part_raises_value_error():
    with pytest.raises(ValueError):
        resolve_model("anthropic/")


def test_empty_string_raises_value_error():
    with pytest.raises(ValueError):
        resolve_model("")


def test_non_str_non_model_raises_type_error():
    with pytest.raises(TypeError):
        resolve_model(123)


def test_no_network_or_credentials_at_construction(monkeypatch):
    """resolve_model itself does not import an SDK or require credentials."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    resolve_model("anthropic/claude-sonnet-5")
    resolve_model("openai/gpt-5")
    resolve_model("ollama/llama3")
