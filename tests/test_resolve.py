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


# ==================================================================
# resolve_model(model, **options)
# ==================================================================


def test_options_are_passed_to_anthropic():
    model = resolve_model("anthropic/claude-sonnet-5", max_tokens=16_000, temperature=0.2)
    assert isinstance(model, Anthropic)
    assert model.name == "claude-sonnet-5"
    assert model.max_tokens == 16_000
    assert model.temperature == 0.2


def test_options_are_passed_for_bare_claude_and_gpt_names():
    claude = resolve_model("claude-sonnet-5", retries=5)
    assert isinstance(claude, Anthropic) and claude.retries == 5
    gpt = resolve_model("gpt-5", context_window=400_000)
    assert isinstance(gpt, OpenAICompatible)
    assert gpt.name == "gpt-5" and gpt.base_url is None and gpt.context_window == 400_000


def test_options_are_passed_to_openai():
    model = resolve_model("openai/gpt-5", api_key="sk-test", timeout=30.0)
    assert isinstance(model, OpenAICompatible)
    assert model.name == "gpt-5"
    assert model.api_key == "sk-test"
    assert model.timeout == 30.0
    assert model.base_url is None


def test_ollama_options_keep_the_default_url():
    model = resolve_model("ollama/qwen3:8b", context_window=32_000)
    assert isinstance(model, OpenAICompatible)
    assert model.name == "qwen3:8b"
    assert model.base_url == OLLAMA_BASE_URL
    assert model.context_window == 32_000


def test_ollama_explicit_base_url_wins():
    model = resolve_model("ollama/qwen3:8b", base_url="http://gpu-box:11434/v1")
    assert model.name == "qwen3:8b"
    assert model.base_url == "http://gpu-box:11434/v1"


def test_base_url_uses_openai_compatible_with_the_whole_string():
    """Routers such as OpenRouter name models "anthropic/claude-sonnet-5", so the string is not split."""
    model = resolve_model("anthropic/claude-sonnet-5", base_url="https://openrouter.ai/api/v1", api_key="k")
    assert isinstance(model, OpenAICompatible)
    assert model.name == "anthropic/claude-sonnet-5"
    assert model.base_url == "https://openrouter.ai/api/v1"
    assert model.api_key == "k"


@pytest.mark.parametrize("string", ["openai/gpt-5", "moonshotai/kimi-k2", "claude-sonnet-5", "qwen3:8b"])
def test_base_url_skips_the_provider_rules(string):
    """With base_url, an unknown prefix or a bare name is not an error: the server knows its own names."""
    model = resolve_model(string, base_url="https://my-server/v1")
    assert isinstance(model, OpenAICompatible)
    assert model.name == string
    assert model.base_url == "https://my-server/v1"


def test_base_url_none_means_not_given():
    anthropic = resolve_model("anthropic/claude-sonnet-5", base_url=None)
    assert isinstance(anthropic, Anthropic)
    assert anthropic.name == "claude-sonnet-5"
    ollama = resolve_model("ollama/llama3", base_url=None)
    assert ollama.base_url == OLLAMA_BASE_URL
    with pytest.raises(ValueError, match="Unknown provider"):
        resolve_model("mystery/x", base_url=None)


def test_base_url_does_not_hide_string_errors():
    with pytest.raises(ValueError, match="empty"):
        resolve_model("", base_url="https://my-server/v1")
    with pytest.raises(ValueError, match="no model name"):
        resolve_model("anthropic/", base_url="https://my-server/v1")


def test_unknown_option_raises_type_error_naming_the_adapter_and_accepted_options():
    with pytest.raises(TypeError) as info:
        resolve_model("openai/gpt-5", thinking=True)
    message = str(info.value)
    assert "OpenAICompatible" in message
    assert "thinking=" in message
    assert "Fix:" in message
    for accepted in ["base_url", "api_key", "max_tokens", "context_window", "supports"]:
        assert accepted in message


def test_unknown_option_for_anthropic_lists_anthropic_options():
    with pytest.raises(TypeError) as info:
        resolve_model("claude-sonnet-5", max_token=100)
    message = str(info.value)
    assert "Anthropic" in message and "max_token=" in message
    assert "thinking" in message and "cache" in message


def test_name_is_not_an_option():
    with pytest.raises(TypeError, match="does not accept name="):
        resolve_model("anthropic/claude-sonnet-5", name="other")


def test_adapter_value_errors_are_not_reported_as_unknown_options():
    """Only option names are checked here; the adapter still checks the values."""
    with pytest.raises(ValueError, match="does not support"):
        resolve_model("anthropic/claude-sonnet-5", supports=[], thinking=True)


def test_model_instance_with_options_raises_type_error():
    with pytest.raises(TypeError, match="Model instance") as info:
        resolve_model(Anthropic("claude-sonnet-5"), max_tokens=100)
    assert "Fix:" in str(info.value)
