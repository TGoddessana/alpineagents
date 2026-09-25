"""``models.anthropic.Anthropic`` tests. No network: the SDK client is replaced with a fake by monkeypatching
``_client``."""

from types import SimpleNamespace

import httpx2
import pytest
from anthropic import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    RateLimitError as SDKRateLimitError,
    RequestTooLargeError,
)

from alpineagents.errors import AuthError, ContextTooLongError, ProviderError, RateLimitError
from alpineagents.models.anthropic import Anthropic, _looks_like_context_overflow
from alpineagents.types import (
    Message,
    Price,
    RawBlock,
    Request,
    TextBlock,
    ToolCall,
    ToolResultBlock,
    ToolSpec,
)

# ---------------------------------------------------------------- fake stream/client


class FakeStreamCM:
    """Stand-in for the context manager returned by ``client.messages.stream(...)``."""

    def __init__(self, text_chunks=(), final_message=None, error=None):
        self.text_chunks = list(text_chunks)
        self.final_message = final_message
        self.error = error

    def __enter__(self):
        if self.error is not None:
            raise self.error
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    @property
    def text_stream(self):
        return iter(self.text_chunks)

    def get_final_message(self):
        return self.final_message


class FakeMessages:
    def __init__(self, cm):
        self._cm = cm
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return self._cm


class FakeClient:
    def __init__(self, cm):
        self.messages = FakeMessages(cm)


def fake_block(type_, **fields):
    ns = SimpleNamespace(type=type_, **fields)
    ns.model_dump = lambda mode=None, exclude_none=None: {"type": type_, **fields}
    return ns


def fake_usage(**kw):
    defaults = dict(input_tokens=0, output_tokens=0, cache_read_input_tokens=None, cache_creation_input_tokens=None)
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def install_fake_client(monkeypatch, model, cm):
    fake = FakeClient(cm)
    monkeypatch.setattr(model, "_client", lambda: fake)
    return fake


def make_response(status_code, error_type, message):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"error": {"type": error_type, "message": message}}
    return httpx2.Response(status_code, request=req, json=body)


# ---------------------------------------------------------------- __init__


def test_strips_anthropic_prefix():
    assert Anthropic("anthropic/claude-sonnet-5").name == "claude-sonnet-5"


def test_no_client_built_at_construction():
    model = Anthropic("claude-sonnet-5")
    assert model._sdk_client is None


def test_default_supports_and_cache():
    model = Anthropic("claude-sonnet-5")
    assert model.supports == frozenset({"thinking", "cache", "vision"})
    assert model.cache is True
    assert model.context_window == 200_000


@pytest.mark.parametrize(
    "thinking, expected",
    [
        (True, {"type": "adaptive"}),
        (4096, {"type": "enabled", "budget_tokens": 4096}),
        (False, None),
    ],
)
def test_thinking_setting(thinking, expected):
    model = Anthropic("claude-sonnet-5", thinking=thinking)
    assert model._thinking == expected


def test_thinking_with_temperature_raises_value_error():
    with pytest.raises(ValueError, match="temperature"):
        Anthropic("claude-sonnet-5", thinking=True, temperature=0.5)


def test_thinking_unsupported_raises_value_error():
    with pytest.raises(ValueError):
        Anthropic("claude-sonnet-5", supports=frozenset(), thinking=True)


def test_cache_default_off_when_unsupported():
    model = Anthropic("claude-sonnet-5", supports=frozenset())
    assert model.cache is False


def test_cache_true_unsupported_raises_value_error():
    with pytest.raises(ValueError):
        Anthropic("claude-sonnet-5", supports=frozenset(), cache=True)


def test_cache_false_turns_off_even_if_supported():
    model = Anthropic("claude-sonnet-5", cache=False)
    assert model.cache is False


# ---------------------------------------------------------------- message/tool conversion


def test_to_anthropic_messages_merges_same_role():
    model = Anthropic("claude-sonnet-5")
    msgs = (
        Message("user", (TextBlock("hello"),)),
        Message("user", (TextBlock("a follow-up message"),)),
    )
    merged = model._to_anthropic_messages(msgs)
    assert len(merged) == 1
    assert len(merged[0]["content"]) == 2


def test_to_anthropic_content_reorders_tool_result_before_text():
    model = Anthropic("claude-sonnet-5")
    user_msg = Message(
        "user",
        (TextBlock("text first"), ToolResultBlock(call_id="c1", content="(done)", name="t")),
    )
    content = model._to_anthropic_content(user_msg)
    assert [c["type"] for c in content] == ["tool_result", "text"]


def test_to_anthropic_content_drops_empty_text():
    model = Anthropic("claude-sonnet-5")
    msg = Message("assistant", (TextBlock(""), ToolCall("t", {"x": 1}, "id1")))
    content = model._to_anthropic_content(msg)
    assert content == [{"type": "tool_use", "id": "id1", "name": "t", "input": {"x": 1}}]


def test_to_anthropic_content_keeps_own_provider_raw_block_verbatim():
    model = Anthropic("claude-sonnet-5")
    msg = Message("assistant", (RawBlock("anthropic", {"type": "thinking", "thinking": "hm"}),))
    content = model._to_anthropic_content(msg)
    assert content == [{"type": "thinking", "thinking": "hm"}]


def test_to_anthropic_content_drops_other_provider_raw_block():
    model = Anthropic("claude-sonnet-5")
    msg = Message("assistant", (RawBlock("openai_compatible", {"foo": "bar"}),))
    assert model._to_anthropic_content(msg) == []


def test_tool_defs_pass_through_declared_tools():
    model = Anthropic("claude-sonnet-5")
    request = Request(system=None, messages=(), tools=(ToolSpec("t1", "description", {"type": "object"}),))
    tools, choice = model._tool_defs_and_choice(request)
    assert tools == [{"name": "t1", "description": "description", "input_schema": {"type": "object"}}]
    assert choice is None


def test_tool_defs_placeholder_when_empty_but_referenced():
    model = Anthropic("claude-sonnet-5")
    request = Request(
        system=None,
        messages=(Message("assistant", (ToolCall("search", {"q": "x"}, "call1"),)),),
        tools=(),
    )
    tools, choice = model._tool_defs_and_choice(request)
    assert tools == [{"name": "search", "description": "", "input_schema": {"type": "object"}}]
    assert choice == {"type": "none"}


def test_build_request_kwargs_respects_tool_choice_none():
    model = Anthropic("claude-sonnet-5")
    request = Request(
        system=None, messages=(), tools=(ToolSpec("t1", "d", {"type": "object"}),), tool_choice="none"
    )
    kwargs = model._build_request_kwargs(request)
    assert kwargs["tool_choice"] == {"type": "none"}


def test_build_request_kwargs_system_and_no_tools_key():
    model = Anthropic("claude-sonnet-5")
    request = Request(system="system", messages=(Message("user", (TextBlock("hello"),)),))
    kwargs = model._build_request_kwargs(request)
    assert kwargs["system"] == "system"
    assert "tools" not in kwargs


def test_build_request_kwargs_cache_control_on_by_default():
    model = Anthropic("claude-sonnet-5")
    request = Request(system=None, messages=(Message("user", (TextBlock("hello"),)),))
    kwargs = model._build_request_kwargs(request)
    assert kwargs["cache_control"] == {"type": "ephemeral"}


def test_build_request_kwargs_no_cache_control_when_cache_off():
    model = Anthropic("claude-sonnet-5", cache=False)
    request = Request(system=None, messages=(Message("user", (TextBlock("hello"),)),))
    kwargs = model._build_request_kwargs(request)
    assert "cache_control" not in kwargs


def test_build_request_kwargs_thinking_passed():
    model = Anthropic("claude-sonnet-5", thinking=True)
    request = Request(system=None, messages=(Message("user", (TextBlock("hello"),)),))
    kwargs = model._build_request_kwargs(request)
    assert kwargs["thinking"] == {"type": "adaptive"}


# ---------------------------------------------------------------- guessing context overflow


@pytest.mark.parametrize(
    "message, expected",
    [
        ("Error: prompt is too long: 250000 tokens > 200000 maximum", True),
        ("invalid api key", False),
    ],
)
def test_looks_like_context_overflow(message, expected):
    assert _looks_like_context_overflow(message) is expected


# ---------------------------------------------------------------- respond()


def test_respond_streams_text_and_converts_blocks(monkeypatch):
    model = Anthropic("claude-sonnet-5")
    final_message = SimpleNamespace(
        content=[
            fake_block("text", text="Hello there"),
            fake_block("tool_use", id="call1", name="search", input={"q": "weather"}),
            fake_block("thinking", thinking="hmm..."),
        ],
        usage=fake_usage(
            input_tokens=100, output_tokens=20, cache_read_input_tokens=30, cache_creation_input_tokens=5
        ),
        stop_reason="tool_use",
    )
    install_fake_client(monkeypatch, model, FakeStreamCM(text_chunks=["Hel", "lo there"], final_message=final_message))

    received = []
    reply = model.respond(
        Request(system=None, messages=(Message("user", (TextBlock("hello"),)),)), on_text=received.append
    )

    assert received == ["Hel", "lo there"]
    assert reply.message.content[0] == TextBlock("Hello there")
    assert reply.message.content[1] == ToolCall("search", {"q": "weather"}, "call1")
    assert isinstance(reply.message.content[2], RawBlock)
    assert reply.message.content[2].provider == "anthropic"
    assert reply.message.content[2].data == {"type": "thinking", "thinking": "hmm..."}

    assert reply.usage.input_tokens == 100
    assert reply.usage.output_tokens == 20
    assert reply.usage.cache_read_tokens == 30
    assert reply.usage.cache_write_tokens == 5
    assert reply.usage.requests == 1
    assert reply.usage.cost is None  # no price info
    assert reply.context_tokens == 100 + 30 + 5 + 20
    assert reply.stop_reason == "tool_use"


def test_respond_without_on_text_callback(monkeypatch):
    model = Anthropic("claude-sonnet-5")
    final_message = SimpleNamespace(
        content=[fake_block("text", text="ok")], usage=fake_usage(input_tokens=1, output_tokens=1), stop_reason="end_turn"
    )
    install_fake_client(monkeypatch, model, FakeStreamCM(text_chunks=["ok"], final_message=final_message))
    reply = model.respond(Request(system=None, messages=(Message("user", (TextBlock("hi"),)),)))
    assert reply.text == "ok"


def test_respond_normalizes_missing_cache_usage_to_zero(monkeypatch):
    model = Anthropic("claude-sonnet-5")
    final_message = SimpleNamespace(
        content=[fake_block("text", text="ok")],
        usage=fake_usage(input_tokens=5, output_tokens=5, cache_read_input_tokens=None, cache_creation_input_tokens=None),
        stop_reason="end_turn",
    )
    install_fake_client(monkeypatch, model, FakeStreamCM(text_chunks=[], final_message=final_message))
    reply = model.respond(Request(system=None, messages=(Message("user", (TextBlock("hi"),)),)))
    assert reply.usage.cache_read_tokens == 0
    assert reply.usage.cache_write_tokens == 0


def test_respond_computes_cost_when_price_given(monkeypatch):
    model = Anthropic("claude-sonnet-5", price=Price(input=3.0, output=15.0))
    final_message = SimpleNamespace(
        content=[fake_block("text", text="ok")], usage=fake_usage(input_tokens=5, output_tokens=5), stop_reason="end_turn"
    )
    install_fake_client(monkeypatch, model, FakeStreamCM(text_chunks=[], final_message=final_message))
    reply = model.respond(Request(system=None, messages=(Message("user", (TextBlock("hi"),)),)))
    assert reply.usage.cost is not None
    assert reply.usage.cost > 0


def test_respond_on_text_exception_propagates_unchanged(monkeypatch):
    model = Anthropic("claude-sonnet-5")
    final_message = SimpleNamespace(
        content=[fake_block("text", text="x")], usage=fake_usage(input_tokens=1, output_tokens=1), stop_reason="end_turn"
    )
    install_fake_client(monkeypatch, model, FakeStreamCM(text_chunks=["x"], final_message=final_message))

    class Boom(Exception):
        pass

    def boom(_chunk):
        raise Boom("callback exploded")

    with pytest.raises(Boom):
        model.respond(Request(system=None, messages=()), on_text=boom)


# ---------------------------------------------------------------- error wrapping


def test_rate_limit_error_wrapped(monkeypatch):
    model = Anthropic("claude-sonnet-5")
    sdk_exc = SDKRateLimitError("slow down", response=make_response(429, "rate_limit_error", "slow down"), body=None)
    install_fake_client(monkeypatch, model, FakeStreamCM(error=sdk_exc))
    with pytest.raises(RateLimitError) as exc_info:
        model.respond(Request(system=None, messages=()))
    assert exc_info.value.__cause__ is sdk_exc


def test_authentication_error_wrapped_as_auth_error(monkeypatch):
    model = Anthropic("claude-sonnet-5")
    sdk_exc = AuthenticationError("bad key", response=make_response(401, "authentication_error", "bad key"), body=None)
    install_fake_client(monkeypatch, model, FakeStreamCM(error=sdk_exc))
    with pytest.raises(AuthError) as exc_info:
        model.respond(Request(system=None, messages=()))
    assert exc_info.value.__cause__ is sdk_exc


def test_permission_denied_error_wrapped_as_auth_error(monkeypatch):
    model = Anthropic("claude-sonnet-5")
    sdk_exc = PermissionDeniedError("no perm", response=make_response(403, "permission_error", "no perm"), body=None)
    install_fake_client(monkeypatch, model, FakeStreamCM(error=sdk_exc))
    with pytest.raises(AuthError):
        model.respond(Request(system=None, messages=()))


def test_request_too_large_wrapped_as_context_too_long(monkeypatch):
    model = Anthropic("claude-sonnet-5")
    sdk_exc = RequestTooLargeError("too big", response=make_response(413, "request_too_large", "too big"), body=None)
    install_fake_client(monkeypatch, model, FakeStreamCM(error=sdk_exc))
    with pytest.raises(ContextTooLongError):
        model.respond(Request(system=None, messages=()))


def test_bad_request_with_overflow_text_wrapped_as_context_too_long(monkeypatch):
    model = Anthropic("claude-sonnet-5")
    body = {"error": {"type": "invalid_request_error", "message": "prompt is too long: too many tokens"}}
    sdk_exc = BadRequestError(
        f"Error code: 400 - {body}",
        response=make_response(400, "invalid_request_error", "prompt is too long: too many tokens"),
        body=body,
    )
    install_fake_client(monkeypatch, model, FakeStreamCM(error=sdk_exc))
    with pytest.raises(ContextTooLongError):
        model.respond(Request(system=None, messages=()))


def test_plain_bad_request_wrapped_as_provider_error(monkeypatch):
    model = Anthropic("claude-sonnet-5")
    body = {"error": {"type": "invalid_request_error", "message": "missing required field"}}
    sdk_exc = BadRequestError(
        f"Error code: 400 - {body}",
        response=make_response(400, "invalid_request_error", "missing required field"),
        body=body,
    )
    install_fake_client(monkeypatch, model, FakeStreamCM(error=sdk_exc))
    with pytest.raises(ProviderError) as exc_info:
        model.respond(Request(system=None, messages=()))
    assert not isinstance(exc_info.value, ContextTooLongError)


def test_connection_error_wrapped_as_provider_error(monkeypatch):
    model = Anthropic("claude-sonnet-5")
    sdk_exc = APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    install_fake_client(monkeypatch, model, FakeStreamCM(error=sdk_exc))
    with pytest.raises(ProviderError):
        model.respond(Request(system=None, messages=()))


# ---------------------------------------------------------------- lazy _client() construction


def test_client_is_built_lazily_and_reused():
    import anthropic as sdk

    model = Anthropic("claude-sonnet-5")
    assert model._sdk_client is None
    client = model._client()
    assert isinstance(client, sdk.Anthropic)
    assert model._client() is client
