"""Black-box tests for Model (talking to providers).

Only tests derived directly from the contract sentences (ARCHITECTURE.md "Model adapter contract"), not from
reading the implementation. No network: the SDK client is always replaced by faking ``_client``, or blocked by
monkeypatching the SDK constructor itself.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx2
import pytest
from anthropic import (
    AuthenticationError as SDKAnthropicAuthError,
    BadRequestError as SDKAnthropicBadRequest,
    PermissionDeniedError as SDKAnthropicPermissionDenied,
    RateLimitError as SDKAnthropicRateLimit,
    RequestTooLargeError as SDKAnthropicRequestTooLarge,
)
from openai import (
    AuthenticationError as SDKOpenAIAuthError,
    BadRequestError as SDKOpenAIBadRequest,
    RateLimitError as SDKOpenAIRateLimit,
)

from alpineagents.errors import AuthError, ContextTooLongError, ProviderError, RateLimitError
from alpineagents.models.anthropic import Anthropic
from alpineagents.models.base import Model
from alpineagents.models.openai_compatible import OpenAICompatible
from alpineagents.models.resolve import resolve_model
from alpineagents.types import (
    Message,
    Price,
    RawBlock,
    Reply,
    Request,
    TextBlock,
    ToolCall,
    Usage,
)

# ==================================================================
# Helpers: fake response bodies (httpx2 response shell)
# ==================================================================


def _fake_response(status_code: int, error_type: str, message: str, extra_body: dict | None = None):
    req = httpx2.Request("POST", "https://example.invalid/v1")
    body = {"error": {"type": error_type, "message": message, **(extra_body or {})}}
    return httpx2.Response(status_code, request=req, json=body)


# ==================================================================
# Helpers: fake SDK client for Anthropic
# ==================================================================


class _AnthropicStreamCM:
    """Stand-in for the context manager returned by ``client.messages.stream(...)``."""

    def __init__(self, chunks=(), final=None, error=None):
        self._chunks = list(chunks)
        self._final = final
        self._error = error

    def __enter__(self):
        if self._error is not None:
            raise self._error
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        return iter(self._chunks)

    def get_final_message(self):
        return self._final


def _install_anthropic_client(monkeypatch, model, cm):
    calls: list[dict] = []

    def fake_stream(**kwargs):
        calls.append(kwargs)
        return cm

    fake_client = SimpleNamespace(messages=SimpleNamespace(stream=fake_stream))
    monkeypatch.setattr(model, "_client", lambda: fake_client)
    return calls


def _anthropic_block(type_, **fields):
    ns = SimpleNamespace(type=type_, **fields)
    ns.model_dump = lambda mode=None, exclude_none=None: {"type": type_, **fields}
    return ns


def _anthropic_usage(**kw):
    defaults = dict(input_tokens=0, output_tokens=0, cache_read_input_tokens=None, cache_creation_input_tokens=None)
    defaults.update(kw)
    return SimpleNamespace(**defaults)


# ==================================================================
# Helpers: fake SDK client for OpenAI-compatible
# ==================================================================


def _install_openai_client(monkeypatch, model, chunks=(), error=None):
    calls: list[dict] = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        if error is not None:
            raise error
        return iter(chunks)

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setattr(model, "_client", lambda: fake_client)
    return calls


def _oai_chunk(content=None, tool_calls=None, finish_reason=None, usage=None, reasoning_content=None):
    delta_kwargs = {"content": content, "tool_calls": tool_calls}
    if reasoning_content is not None:
        delta_kwargs["reasoning_content"] = reasoning_content
    delta = SimpleNamespace(**delta_kwargs)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def _oai_tool_delta(index, id=None, name=None, arguments=None):
    function = None
    if name is not None or arguments is not None:
        function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=id, function=function)


def _oai_usage(prompt_tokens, completion_tokens, cached_tokens=None):
    details = None if cached_tokens is None else SimpleNamespace(cached_tokens=cached_tokens)
    return SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, prompt_tokens_details=details)


# ==================================================================
# Helpers: minimal adapter with only the two required members (to check the Model defaults)
# ==================================================================


class _RecordingModel(Model):
    """"A new provider works once it implements the two required members."""

    def __init__(self, window: int = 100_000):
        self.name = "recording"
        self._window = window
        self.received_requests: list[Request] = []

    @property
    def context_window(self) -> int:
        return self._window

    def respond(self, request: Request, on_text=None, on_event=None) -> Reply:
        self.received_requests.append(request)
        if on_text is not None:
            on_text("answer")
        return Reply(message=Message("assistant", (TextBlock("answer"),)), usage=Usage(requests=1))


class _PricedModel(Model):
    """Minimal adapter for checking the ``_cost``/``_lookup_price`` priority."""

    def __init__(self, price=None, lookup_price=None):
        self.name = "priced"
        self.price = price
        self._lookup_result = lookup_price

    @property
    def context_window(self) -> int:
        return 100_000

    def respond(self, request: Request, on_text=None, on_event=None) -> Reply:  # pragma: no cover - unused
        raise NotImplementedError

    def _lookup_price(self):
        return self._lookup_result


# ==================================================================
# Model: required/optional members
# ==================================================================


def test_respond_is_a_required_member():
    """"respond(request, on_text) -> reply | required." Without it the adapter cannot be created."""

    class MissingRespond(Model):
        @property
        def context_window(self) -> int:
            return 1000

    with pytest.raises(TypeError):
        MissingRespond()


def test_context_window_is_a_required_member():
    """"context_window | required. The context window size." Without it the adapter cannot be created."""

    class MissingContextWindow(Model):
        def respond(self, request, on_text=None, on_event=None):
            raise NotImplementedError

    with pytest.raises(TypeError):
        MissingContextWindow()


def test_new_adapter_needs_only_the_two_required_members():
    """"A new provider works once it implements the two required members."""
    model = _RecordingModel()
    assert model.context_window == 100_000
    reply = model.respond(Request(system=None, messages=()))
    assert reply.text == "answer"


def test_count_tokens_has_a_default_implementation():
    """"count_tokens(request) -> int | optional. The default adds an estimate of the newly added part to the last
    reply's usage."""
    model = _RecordingModel()
    anchored = Message("assistant", (TextBlock("previous answer"),), tokens=1000)
    request = Request(system=None, messages=(anchored, Message.user("follow-up question")))
    # Only the new message estimate is added to the last anchor (1000), so it must be above 1000 and below a
    # full recount.
    assert model.count_tokens(request) > 1000


def test_mark_cache_has_a_default_implementation_that_returns_request_unchanged():
    """"mark_cache(request) -> Request | optional. Decides where prompt cache markers go." The default returns it
    unchanged."""
    model = _RecordingModel()
    request = Request(system="sys", messages=(Message.user("hello"),))
    assert model.mark_cache(request) == request


def test_compact_has_a_default_implementation_calling_respond_with_tool_choice_none():
    """"compact(request, instructions) -> summary | optional. The default is a summary request that works with every
    model."""
    model = _RecordingModel()
    request = Request(system=None, messages=(Message.user("work so far"),))
    reply = model.compact(request, instructions="keep the changed files")
    sent = model.received_requests[-1]
    assert sent.tool_choice == "none"
    assert sent.messages[-1].role == "user"
    assert "keep the changed files" in sent.messages[-1].text
    assert reply.text == "answer"


def test_supports_defaults_to_empty_when_adapter_declares_nothing():
    """"supports | the list of supported features." A minimal adapter that declares nothing supports no features."""
    model = _RecordingModel()
    assert model.supports == frozenset()


# ==================================================================
# Model: model string rules
# ==================================================================


def test_model_string_format_is_provider_slash_model():
    """"Model string: the format is "provider/model" ("anthropic/claude-sonnet-5", ...)."""
    model = resolve_model("anthropic/claude-sonnet-5")
    assert isinstance(model, Anthropic)
    assert model.name == "claude-sonnet-5"


def test_colon_is_not_a_separator_because_it_collides_with_ollama_tags():
    """"":` is not used as a separator because it collides with Ollama tags."" (the provider is up to the first `/`)"""
    model = resolve_model("ollama/llama3:8b")
    assert isinstance(model, OpenAICompatible)
    assert model.name == "llama3:8b"


def test_unambiguous_bare_name_may_omit_provider_prefix():
    """"Models whose provider is clear from the name alone (`claude-*`, `gpt-*`) may omit the prefix."""
    assert isinstance(resolve_model("claude-sonnet-5"), Anthropic)
    assert isinstance(resolve_model("gpt-5"), OpenAICompatible)


def test_ambiguous_bare_name_raises_error_showing_candidates():
    """"If ambiguous, raise an error that shows the candidates."""
    with pytest.raises(ValueError) as exc_info:
        resolve_model("mystery-model")
    message = str(exc_info.value)
    assert "anthropic/mystery-model" in message
    assert "openai/mystery-model" in message
    assert "ollama/mystery-model" in message


# ==================================================================
# Model: the Anthropic / OpenAICompatible settings objects
# ==================================================================


@pytest.mark.parametrize("adapter_cls", [Anthropic, OpenAICompatible])
def test_common_settings_share_the_same_names_across_adapter_classes(adapter_cls):
    """"Shared settings (max_tokens, temperature, timeout, retries, price) have the same names in every class"""
    price = Price(input=1.0, output=2.0)
    model = adapter_cls("some-model", max_tokens=111, temperature=0.3, timeout=9.5, retries=4, price=price)
    assert model.max_tokens == 111
    assert model.temperature == 0.3
    assert model.timeout == 9.5
    assert model.retries == 4
    assert model.price == price


def test_provider_specific_settings_exist_only_on_their_own_class():
    """"Provider-specific settings (thinking=, cache=) exist only on their own class."""
    Anthropic("claude-sonnet-5", thinking=True)  # Anthropic has it
    with pytest.raises(TypeError):
        OpenAICompatible("gpt-5", thinking=True)  # OpenAICompatible does not
    with pytest.raises(TypeError):
        OpenAICompatible("gpt-5", cache=True)


@pytest.mark.parametrize("adapter_cls, model_name", [(Anthropic, "claude-sonnet-5"), (OpenAICompatible, "gpt-5")])
def test_adapter_construction_needs_no_network_or_credentials(monkeypatch, adapter_cls, model_name):
    """"No network or credentials are used at construction. The SDK client is built once, on the first respond."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    model = adapter_cls(model_name)  # must be created without an exception
    assert model.name == model_name


# ==================================================================
# Model: price and usage.cost rules
# ==================================================================


def test_explicit_price_takes_priority_over_the_lookup_table():
    """""price=Price(input=3.0, output=15.0)" (dollars per million tokens) takes priority over that."""
    explicit = Price(input=1.0, output=2.0)
    looked_up = Price(input=100.0, output=200.0)
    model = _PricedModel(price=explicit, lookup_price=looked_up)
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000, requests=1)
    assert model._cost(usage) == explicit.cost(usage)


def test_lookup_table_used_only_when_no_explicit_price_is_given():
    """"With genai-prices installed via pip install alpineagents[prices], cost is computed from its table" (table
    only)"""
    looked_up = Price(input=5.0, output=10.0)
    model = _PricedModel(price=None, lookup_price=looked_up)
    usage = Usage(input_tokens=1_000_000, output_tokens=0, requests=1)
    assert model._cost(usage) == 5.0


def test_cost_is_none_not_zero_when_price_is_unknown():
    """"With neither, or for an unknown model, it is `None`. This keeps "unknown" apart from 0."""
    model = _PricedModel(price=None, lookup_price=None)
    usage = Usage(input_tokens=100, output_tokens=100, requests=1)
    cost = model._cost(usage)
    assert cost is None
    assert cost != 0


# ==================================================================
# Model: unsupported feature errors
# ==================================================================


def test_setting_unsupported_feature_raises_value_error_not_silently_ignored():
    """"Adapters declare the features they support. Setting an unsupported feature raises an error instead of being
    silently ignored."""
    with pytest.raises(ValueError, match="does not support"):
        Anthropic("claude-sonnet-5", supports=frozenset(), thinking=True)


def test_unsupported_feature_error_message_lists_what_the_adapter_supports():
    """Mistake-proofing errors: 'setting a feature the adapter does not support | message gist: the features this
    adapter supports'"""
    with pytest.raises(ValueError, match="Features this adapter supports") as exc_info:
        Anthropic("claude-sonnet-5", supports=frozenset({"vision"}), cache=True)
    assert "vision" in str(exc_info.value)


# ==================================================================
# Model: retries use the SDK's
# ==================================================================


def test_anthropic_retries_setting_is_passed_to_the_sdk_client(monkeypatch):
    """"Retries use the SDK's retries (`retries=`)."""
    import anthropic as sdk

    captured: dict = {}
    monkeypatch.setattr(sdk, "Anthropic", lambda **kw: captured.update(kw) or SimpleNamespace())
    model = Anthropic("claude-sonnet-5", retries=7)
    model._client()
    assert captured["max_retries"] == 7


def test_openai_compatible_retries_setting_is_passed_to_the_sdk_client(monkeypatch):
    """"Retries use the SDK's retries (`retries=`)."""
    import openai as sdk

    captured: dict = {}
    monkeypatch.setattr(sdk, "OpenAI", lambda **kw: captured.update(kw) or SimpleNamespace())
    model = OpenAICompatible("gpt-5", api_key="any-key", retries=9)
    model._client()
    assert captured["max_retries"] == 9


# ==================================================================
# Model: request/reply conversion - Anthropic (fake SDK client)
# ==================================================================


def test_anthropic_respond_streams_text_via_on_text_callback(monkeypatch):
    """"respond(request, on_text) -> reply | required. ... Passes text to `on_text` as it arrives."""
    model = Anthropic("claude-sonnet-5")
    final = SimpleNamespace(
        content=[_anthropic_block("text", text="Hello there")],
        usage=_anthropic_usage(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    _install_anthropic_client(monkeypatch, model, _AnthropicStreamCM(chunks=["Hel", "lo there"], final=final))

    received: list[str] = []
    model.respond(Request(system=None, messages=(Message.user("hello"),)), on_text=received.append)

    assert received == ["Hel", "lo there"]


def test_anthropic_respond_converts_tool_use_block_to_tool_call(monkeypatch):
    """When the model requests a tool call, that call arrives as is in the reply blocks."""
    model = Anthropic("claude-sonnet-5")
    final = SimpleNamespace(
        content=[_anthropic_block("tool_use", id="call1", name="search", input={"q": "weather"})],
        usage=_anthropic_usage(input_tokens=1, output_tokens=1),
        stop_reason="tool_use",
    )
    _install_anthropic_client(monkeypatch, model, _AnthropicStreamCM(final=final))

    reply = model.respond(Request(system=None, messages=(Message.user("What is the weather?"),)))

    assert reply.tool_calls == (ToolCall("search", {"q": "weather"}, "call1"),)


def test_anthropic_respond_preserves_thinking_block_data_verbatim(monkeypatch):
    """"Received messages are not modified. Provider-specific data such as thinking blocks is kept as the original
    and sent back unchanged."""
    model = Anthropic("claude-sonnet-5", thinking=True)
    thinking_payload = {"type": "thinking", "thinking": "read the file first", "signature": "sig-abc"}
    final = SimpleNamespace(
        content=[_anthropic_block("thinking", thinking="read the file first", signature="sig-abc")],
        usage=_anthropic_usage(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    _install_anthropic_client(monkeypatch, model, _AnthropicStreamCM(final=final))

    reply = model.respond(Request(system=None, messages=(Message.user("Think it over"),)))

    assert reply.message.content == (RawBlock("anthropic", thinking_payload),)


def test_anthropic_respond_maps_usage_fields_from_sdk_usage(monkeypatch):
    """"Adapters map Usage to one meaning: `input_tokens` is input that did not go through the cache, ..."
    (ARCHITECTURE.md "Requests and replies")"""
    model = Anthropic("claude-sonnet-5")
    final = SimpleNamespace(
        content=[_anthropic_block("text", text="ok")],
        usage=_anthropic_usage(input_tokens=100, output_tokens=20, cache_read_input_tokens=30, cache_creation_input_tokens=5),
        stop_reason="end_turn",
    )
    _install_anthropic_client(monkeypatch, model, _AnthropicStreamCM(final=final))

    reply = model.respond(Request(system=None, messages=(Message.user("hi"),)))

    assert reply.usage.input_tokens == 100
    assert reply.usage.output_tokens == 20
    assert reply.usage.cache_read_tokens == 30
    assert reply.usage.cache_write_tokens == 5
    assert reply.usage.requests == 1


def test_anthropic_respond_usage_missing_cache_fields_default_to_zero(monkeypatch):
    """"cost is `Model._cost(usage)` (`None` if the price is unknown)."" Before that, checks the premise that cache
    fields are normalized to 0 when the cache is not used."""
    model = Anthropic("claude-sonnet-5")
    final = SimpleNamespace(
        content=[_anthropic_block("text", text="ok")],
        usage=_anthropic_usage(input_tokens=5, output_tokens=5, cache_read_input_tokens=None, cache_creation_input_tokens=None),
        stop_reason="end_turn",
    )
    _install_anthropic_client(monkeypatch, model, _AnthropicStreamCM(final=final))

    reply = model.respond(Request(system=None, messages=(Message.user("hi"),)))

    assert reply.usage.cache_read_tokens == 0
    assert reply.usage.cache_write_tokens == 0


def test_anthropic_respond_cost_is_none_when_no_price_configured(monkeypatch):
    """"With neither, or for an unknown model, it is `None`."" (checked on the usage.cost that respond really
    returns)"""
    model = Anthropic("claude-sonnet-5")
    final = SimpleNamespace(
        content=[_anthropic_block("text", text="ok")], usage=_anthropic_usage(input_tokens=5, output_tokens=5), stop_reason="end_turn"
    )
    _install_anthropic_client(monkeypatch, model, _AnthropicStreamCM(final=final))
    reply = model.respond(Request(system=None, messages=(Message.user("hi"),)))
    assert reply.usage.cost is None


def test_anthropic_respond_cost_computed_from_given_price(monkeypatch):
    """"price=Price(input=3.0, output=15.0)" (dollars per million tokens) takes priority over that."" → reflected
    in the reply's usage.cost."""
    price = Price(input=3.0, output=15.0)
    model = Anthropic("claude-sonnet-5", price=price)
    final = SimpleNamespace(
        content=[_anthropic_block("text", text="ok")],
        usage=_anthropic_usage(input_tokens=1_000_000, output_tokens=1_000_000),
        stop_reason="end_turn",
    )
    _install_anthropic_client(monkeypatch, model, _AnthropicStreamCM(final=final))
    reply = model.respond(Request(system=None, messages=(Message.user("hi"),)))
    assert reply.usage.cost == pytest.approx(3.0 + 15.0)


def test_anthropic_sends_own_provider_raw_block_back_verbatim(monkeypatch):
    """"Provider-specific data such as thinking blocks is kept as the original and sent back unchanged.""
    (same provider)"""
    model = Anthropic("claude-sonnet-5")
    raw_payload = {"type": "thinking", "thinking": "that earlier thought", "signature": "sig-xyz"}
    history = Message("assistant", (RawBlock("anthropic", raw_payload),))
    final = SimpleNamespace(
        content=[_anthropic_block("text", text="Continuing")], usage=_anthropic_usage(input_tokens=1, output_tokens=1), stop_reason="end_turn"
    )
    calls = _install_anthropic_client(monkeypatch, model, _AnthropicStreamCM(final=final))

    model.respond(Request(system=None, messages=(history, Message.user("Go on"))))

    sent_messages = calls[0]["messages"]
    assistant_content = sent_messages[0]["content"]
    assert raw_payload in assistant_content


def test_anthropic_drops_raw_block_from_a_different_provider(monkeypatch):
    """"Only an adapter with the same `provider` (`Model.provider`) sends it back unchanged; other adapters drop it.""
    (ARCHITECTURE.md "Message representation"; the counterpart of the keep-the-original rule)"""
    model = Anthropic("claude-sonnet-5")
    history = Message("assistant", (RawBlock("openai_compatible", {"reasoning_content": "other provider"}),))
    final = SimpleNamespace(
        content=[_anthropic_block("text", text="Continuing")], usage=_anthropic_usage(input_tokens=1, output_tokens=1), stop_reason="end_turn"
    )
    calls = _install_anthropic_client(monkeypatch, model, _AnthropicStreamCM(final=final))

    model.respond(Request(system=None, messages=(history, Message.user("Go on"))))

    sent_messages = calls[0]["messages"]
    assistant_content = sent_messages[0]["content"]
    assert assistant_content == []


# ==================================================================
# Model: request/reply conversion - OpenAICompatible (fake SDK client)
# ==================================================================


def test_openai_compatible_respond_streams_text_via_on_text_callback(monkeypatch):
    """"respond(request, on_text) -> reply | required. ... Passes text to `on_text` as it arrives."" (the
    OpenAI-compatible adapter has the same contract)"""
    model = OpenAICompatible("gpt-5", api_key="x")
    chunks = [_oai_chunk(content="Hel"), _oai_chunk(content="lo"), _oai_chunk(finish_reason="stop")]
    _install_openai_client(monkeypatch, model, chunks=chunks)

    received: list[str] = []
    model.respond(Request(system=None, messages=(Message.user("hello"),)), on_text=received.append)

    assert received == ["Hel", "lo"]


def test_openai_compatible_respond_accumulates_streamed_tool_call_by_index(monkeypatch):
    """Even when the provider streams a tool call in pieces, it is merged into one call in the reply."""
    model = OpenAICompatible("gpt-5", api_key="x")
    chunks = [
        _oai_chunk(tool_calls=[_oai_tool_delta(0, id="call_1", name="search", arguments="")]),
        _oai_chunk(tool_calls=[_oai_tool_delta(0, arguments='{"q": ')]),
        _oai_chunk(tool_calls=[_oai_tool_delta(0, arguments='"weather"}')]),
        _oai_chunk(finish_reason="tool_calls"),
    ]
    _install_openai_client(monkeypatch, model, chunks=chunks)

    reply = model.respond(Request(system=None, messages=(Message.user("What is the weather?"),)))

    assert reply.tool_calls == (ToolCall("search", {"q": "weather"}, "call_1"),)


def test_openai_compatible_fallback_call_ids_are_unique_across_turns(monkeypatch):
    """A server that gives no ids: the ids made up instead must not repeat as ``call_0`` every turn."""
    model = OpenAICompatible("llama3", base_url="http://localhost:11434/v1", api_key="x")
    ids = []
    for _ in range(2):
        chunks = [
            _oai_chunk(tool_calls=[_oai_tool_delta(0, name="search", arguments='{"q": "a"}')]),
            _oai_chunk(tool_calls=[_oai_tool_delta(1, name="search", arguments='{"q": "b"}')]),
            _oai_chunk(finish_reason="tool_calls"),
        ]
        _install_openai_client(monkeypatch, model, chunks=chunks)
        reply = model.respond(Request(system=None, messages=(Message.user("Find it"),)))
        ids.extend(call.id for call in reply.tool_calls)
    assert len(ids) == 4
    assert len(set(ids)) == 4
    assert all(call_id.startswith("call_") for call_id in ids)


def test_openai_compatible_respond_maps_usage_and_subtracts_cached_from_input(monkeypatch):
    """"Adapters map Usage to one meaning: input_tokens is input that did not go through the cache, total input =
    input + cache_read + cache_write."""
    model = OpenAICompatible("gpt-5", api_key="x")
    usage = _oai_usage(prompt_tokens=100, completion_tokens=20, cached_tokens=30)
    chunks = [_oai_chunk(content="ok"), _oai_chunk(finish_reason="stop", usage=usage)]
    _install_openai_client(monkeypatch, model, chunks=chunks)

    reply = model.respond(Request(system=None, messages=(Message.user("hi"),)))

    assert reply.usage.cache_read_tokens == 30
    assert reply.usage.input_tokens == 100 - 30
    assert reply.usage.output_tokens == 20


def test_openai_compatible_respond_cost_is_none_when_no_price_configured(monkeypatch):
    """"With neither, or for an unknown model, it is `None`."" (checked on the OpenAI-compatible adapter's respond
    result)"""
    model = OpenAICompatible("gpt-5", api_key="x")
    usage = _oai_usage(prompt_tokens=10, completion_tokens=10)
    chunks = [_oai_chunk(content="ok"), _oai_chunk(finish_reason="stop", usage=usage)]
    _install_openai_client(monkeypatch, model, chunks=chunks)

    reply = model.respond(Request(system=None, messages=(Message.user("hi"),)))
    assert reply.usage.cost is None


def test_openai_compatible_own_provider_raw_block_merged_back_verbatim(monkeypatch):
    """"Provider-specific data such as thinking blocks is kept as the original and sent back unchanged.""
    (same provider)"""
    model = OpenAICompatible("gpt-5", api_key="x")
    raw_data = {"reasoning_content": "this is what I thought before"}
    history = Message("assistant", (RawBlock("openai_compatible", raw_data),))
    chunks = [_oai_chunk(content="Continuing"), _oai_chunk(finish_reason="stop")]
    calls = _install_openai_client(monkeypatch, model, chunks=chunks)

    model.respond(Request(system=None, messages=(history, Message.user("Go on"))))

    sent_messages = calls[0]["messages"]
    assistant_message = next(m for m in sent_messages if m["role"] == "assistant")
    assert assistant_message["reasoning_content"] == raw_data["reasoning_content"]


def test_openai_compatible_drops_raw_block_from_a_different_provider(monkeypatch):
    """"Only an adapter with the same `provider` sends it back unchanged; other adapters drop it.""
    (ARCHITECTURE.md "Message representation" — RawBlock is kept as the original only when the provider matches)"""
    model = OpenAICompatible("gpt-5", api_key="x")
    history = Message("assistant", (RawBlock("anthropic", {"type": "thinking", "thinking": "another provider"}),))
    chunks = [_oai_chunk(content="Continuing"), _oai_chunk(finish_reason="stop")]
    calls = _install_openai_client(monkeypatch, model, chunks=chunks)

    model.respond(Request(system=None, messages=(history, Message.user("Go on"))))

    sent_messages = calls[0]["messages"]
    assistant_message = next(m for m in sent_messages if m["role"] == "assistant")
    assert "thinking" not in assistant_message
    assert "type" not in assistant_message


# ==================================================================
# Model: wrapping in ProviderError, and __cause__
# ==================================================================


def test_anthropic_rate_limit_wrapped_as_rate_limit_error_with_cause(monkeypatch):
    """"Model (API): if it still fails after the SDK's retries, alpineagents.ProviderError and its subtypes
    (RateLimitError, ...). The original SDK exception is in __cause__."""
    model = Anthropic("claude-sonnet-5")
    sdk_exc = SDKAnthropicRateLimit("slow down", response=_fake_response(429, "rate_limit_error", "slow down"), body=None)
    _install_anthropic_client(monkeypatch, model, _AnthropicStreamCM(error=sdk_exc))

    with pytest.raises(RateLimitError) as exc_info:
        model.respond(Request(system=None, messages=()))
    assert exc_info.value.__cause__ is sdk_exc


@pytest.mark.parametrize(
    "sdk_exc_factory",
    [
        lambda: SDKAnthropicAuthError("bad key", response=_fake_response(401, "authentication_error", "bad key"), body=None),
        lambda: SDKAnthropicPermissionDenied("no perm", response=_fake_response(403, "permission_error", "no perm"), body=None),
    ],
)
def test_anthropic_auth_failures_wrapped_as_auth_error(monkeypatch, sdk_exc_factory):
    """"...ContextTooLongError, AuthError). The original SDK exception is in __cause__."""
    model = Anthropic("claude-sonnet-5")
    sdk_exc = sdk_exc_factory()
    _install_anthropic_client(monkeypatch, model, _AnthropicStreamCM(error=sdk_exc))

    with pytest.raises(AuthError) as exc_info:
        model.respond(Request(system=None, messages=()))
    assert exc_info.value.__cause__ is sdk_exc


def test_anthropic_request_too_large_wrapped_as_context_too_long_error(monkeypatch):
    """"...ContextTooLongError, AuthError). The original SDK exception is in __cause__."""
    model = Anthropic("claude-sonnet-5")
    sdk_exc = SDKAnthropicRequestTooLarge("too big", response=_fake_response(413, "request_too_large", "too big"), body=None)
    _install_anthropic_client(monkeypatch, model, _AnthropicStreamCM(error=sdk_exc))

    with pytest.raises(ContextTooLongError) as exc_info:
        model.respond(Request(system=None, messages=()))
    assert exc_info.value.__cause__ is sdk_exc


def test_anthropic_other_sdk_errors_wrapped_as_generic_provider_error(monkeypatch):
    """"If it still fails after the SDK's retries, alpineagents.ProviderError and its subtypes" (anything else gets
    the base type)"""
    model = Anthropic("claude-sonnet-5")
    body = {"error": {"type": "invalid_request_error", "message": "missing field"}}
    sdk_exc = SDKAnthropicBadRequest("bad", response=_fake_response(400, "invalid_request_error", "missing field"), body=body)
    _install_anthropic_client(monkeypatch, model, _AnthropicStreamCM(error=sdk_exc))

    with pytest.raises(ProviderError) as exc_info:
        model.respond(Request(system=None, messages=()))
    assert not isinstance(exc_info.value, ContextTooLongError)
    assert exc_info.value.__cause__ is sdk_exc


def test_openai_compatible_rate_limit_wrapped_as_rate_limit_error_with_cause(monkeypatch):
    """"Model (API): if it still fails after the SDK's retries, alpineagents.ProviderError and its subtypes
    (RateLimitError, ...). The original SDK exception is in __cause__." (the OpenAI-compatible adapter follows the
    same contract)"""
    model = OpenAICompatible("gpt-5", api_key="x")
    sdk_exc = SDKOpenAIRateLimit("slow down", response=_fake_response(429, "rate_limit_error", "slow down"), body=None)
    _install_openai_client(monkeypatch, model, error=sdk_exc)

    with pytest.raises(RateLimitError) as exc_info:
        model.respond(Request(system=None, messages=()))
    assert exc_info.value.__cause__ is sdk_exc


def test_openai_compatible_auth_failure_wrapped_as_auth_error(monkeypatch):
    """"...ContextTooLongError, AuthError). The original SDK exception is in __cause__."""
    model = OpenAICompatible("gpt-5", api_key="x")
    sdk_exc = SDKOpenAIAuthError("bad key", response=_fake_response(401, "authentication_error", "bad key"), body=None)
    _install_openai_client(monkeypatch, model, error=sdk_exc)

    with pytest.raises(AuthError) as exc_info:
        model.respond(Request(system=None, messages=()))
    assert exc_info.value.__cause__ is sdk_exc


def test_openai_compatible_context_length_exceeded_wrapped_as_context_too_long_error(monkeypatch):
    """"...ContextTooLongError, AuthError). The original SDK exception is in __cause__." (when the server reports it
    via code)"""
    model = OpenAICompatible("gpt-5", api_key="x")
    body = {"type": "invalid_request_error", "code": "context_length_exceeded", "message": "too long"}
    sdk_exc = SDKOpenAIBadRequest(
        "too long", response=_fake_response(400, "invalid_request_error", "too long", {"code": "context_length_exceeded"}), body=body
    )
    _install_openai_client(monkeypatch, model, error=sdk_exc)

    with pytest.raises(ContextTooLongError) as exc_info:
        model.respond(Request(system=None, messages=()))
    assert exc_info.value.__cause__ is sdk_exc


def test_openai_compatible_other_bad_request_wrapped_as_generic_provider_error(monkeypatch):
    """"If it still fails after the SDK's retries, alpineagents.ProviderError and its subtypes" (anything else gets
    the base type)"""
    model = OpenAICompatible("gpt-5", api_key="x")
    body = {"type": "invalid_request_error", "message": "missing field"}
    sdk_exc = SDKOpenAIBadRequest("bad", response=_fake_response(400, "invalid_request_error", "missing field"), body=body)
    _install_openai_client(monkeypatch, model, error=sdk_exc)

    with pytest.raises(ProviderError) as exc_info:
        model.respond(Request(system=None, messages=()))
    assert not isinstance(exc_info.value, ContextTooLongError)
    assert exc_info.value.__cause__ is sdk_exc


def test_provider_error_hierarchy_lets_generic_except_survive_provider_swap():
    """"Why only provider errors are wrapped: user code with `except anthropic.RateLimitError` breaks the moment the
    provider changes. So errors are wrapped in common types only at the provider boundary."""
    assert issubclass(RateLimitError, ProviderError)
    assert issubclass(ContextTooLongError, ProviderError)
    assert issubclass(AuthError, ProviderError)
