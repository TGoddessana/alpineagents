"""Reproduction tests for Model adapter defects confirmed in review (models/anthropic.py,
models/openai_compatible.py).

They follow the fake SDK client pattern of tests/test_model.py. The two credential tests rely on the exceptions
the installed SDK really raises when credentials are missing (TypeError / openai.OpenAIError). Those are raised
synchronously before any network access (at the header/constructor step), so they reproduce quickly and safely
offline without monkeypatching.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from alpineagents import Agent, State, tool
from alpineagents.errors import AlpineAgentsError, AuthError, ProviderError
from alpineagents.models.anthropic import Anthropic
from alpineagents.models.openai_compatible import OpenAICompatible
from alpineagents.testing import FakeModel
from alpineagents.types import INVALID_ARGS_KEY, TRUNCATED_ARGS_MESSAGE, Message, Request, TextBlock, ToolCall

# ==================================================================
# Helpers: fake SDK clients (same shape as tests/test_model.py)
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


def _anthropic_usage(**kw):
    defaults = dict(input_tokens=0, output_tokens=0, cache_read_input_tokens=None, cache_creation_input_tokens=None)
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _anthropic_block(type_, **fields):
    """A fake SDK block for ``message.content`` (same shape as ``tests/test_model.py``)."""
    ns = SimpleNamespace(type=type_, **fields)
    ns.model_dump = lambda mode=None, exclude_none=None: {"type": type_, **fields}
    return ns


def _install_openai_client(monkeypatch, model, chunks=(), error=None):
    calls: list[dict] = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        if error is not None:
            raise error
        return _FakeOpenAIStream(chunks)

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setattr(model, "_client", lambda: fake_client)
    return calls


class _FakeOpenAIStream:
    """Stand-in that supports ``close()`` like the real ``openai._streaming.Stream``."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    def __iter__(self):
        return iter(self._chunks)

    def close(self):
        self.closed = True


def _oai_chunk(content=None, tool_calls=None, finish_reason=None, usage=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def _oai_tool_delta(index, id=None, name=None, arguments=None):
    """One ``delta.tool_calls`` item (same shape as ``tests/test_model.py``)."""
    function = None
    if name is not None or arguments is not None:
        function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=id, function=function)


def _oai_usage(prompt_tokens, completion_tokens, cached_tokens=None, cache_write_tokens=None):
    details = None
    if cached_tokens is not None or cache_write_tokens is not None:
        details = SimpleNamespace(cached_tokens=cached_tokens, cache_write_tokens=cache_write_tokens)
    return SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, prompt_tokens_details=details)


def _req(messages=(Message.user("hi"),)) -> Request:
    return Request(system=None, messages=messages)


# ==================================================================
# 1. temperature= is not a top-level kwarg of anthropic 1.8 Messages.stream()
# ==================================================================


def test_anthropic_temperature_is_sent_via_extra_body_not_a_top_level_kwarg(monkeypatch):
    """anthropic 1.8 Messages.stream()/MessageCreateParams has no top-level ``temperature``, so passing it as a
    keyword raises ``TypeError``. It must go through ``extra_body``."""
    model = Anthropic("claude-sonnet-5", temperature=0.2, api_key="sk-test-dummy")
    final = SimpleNamespace(content=[], usage=_anthropic_usage(), stop_reason="end_turn")
    calls = _install_anthropic_client(monkeypatch, model, _AnthropicStreamCM(final=final))

    reply = model.respond(_req())

    assert len(calls) == 1
    sent = calls[0]
    assert "temperature" not in sent, "sending temperature as a top-level kwarg makes the SDK raise TypeError"
    assert sent.get("extra_body", {}).get("temperature") == 0.2
    assert reply.stop_reason == "end_turn"


def test_anthropic_unexpected_sdk_typeerror_is_wrapped_as_provider_error(monkeypatch):
    """If an SDK signature mismatch still raises ``TypeError`` (and it is not a credentials problem), it must not
    leak unwrapped; it is wrapped as ``AlpineAgentsError`` (``ProviderError``)."""
    model = Anthropic("claude-sonnet-5", api_key="sk-test-dummy")
    _install_anthropic_client(
        monkeypatch, model, _AnthropicStreamCM(error=TypeError("stream() got an unexpected keyword argument 'bogus'"))
    )

    with pytest.raises(AlpineAgentsError) as exc_info:
        model.respond(_req())
    assert isinstance(exc_info.value, ProviderError)
    assert not isinstance(exc_info.value, AuthError)


# ==================================================================
# 2. Missing credentials -> AuthError (not a raw TypeError / openai.OpenAIError)
# ==================================================================


def test_anthropic_missing_api_key_raises_autherror_not_raw_typeerror(monkeypatch):
    """With no API key at all, the anthropic SDK raises a bare ``TypeError`` at request time (header validation
    runs lazily). The adapter's docstring: "no API key → AuthError"."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    model = Anthropic("claude-sonnet-5")

    with pytest.raises(AuthError):
        model.respond(_req())


def test_openai_compatible_missing_api_key_raises_autherror_not_raw_sdk_error(monkeypatch):
    """With no API key at all, constructing ``openai.OpenAI(...)`` itself raises ``openai.OpenAIError`` (it is not
    caught if ``_client()`` is outside respond()'s try)."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    model = OpenAICompatible("gpt-5")

    with pytest.raises(AuthError):
        model.respond(_req())


# ==================================================================
# 3. An empty or whitespace-only assistant message becomes invalid content when sent again
# ==================================================================


def test_anthropic_empty_assistant_message_gets_non_empty_placeholder_content():
    """The ``Message("assistant", ())`` stored by state._record_reply (when Claude ended with content=[]) must be
    sent with valid (non-empty) content even when a user message follows -- Anthropic rejects empty content on
    any message that is not the last one."""
    model = Anthropic("claude-sonnet-5")

    window = (
        Message("user", (TextBlock("do the thing"),)),
        Message("assistant", ()),  # what _record_reply stores for content=[]
        Message("user", (TextBlock("continue"),)),  # state.add_user_message
    )

    sent = model._to_anthropic_messages(window)

    assistant_messages = [m for m in sent if m["role"] == "assistant"]
    assert len(assistant_messages) == 1
    content = assistant_messages[0]["content"]
    assert content != [], "Anthropic rejects empty content ([]) on a message that is not the last one"
    assert all(block["type"] != "text" or block["text"].strip() for block in content)


def test_anthropic_whitespace_only_text_block_is_not_sent_as_whitespace_only_content():
    """A whitespace-only text block (``"\\n\\n"``) is truthy, so ``if block.text:`` does not filter it out --
    Anthropic rejects whitespace-only text blocks, so it must be replaced with a placeholder."""
    model = Anthropic("claude-sonnet-5")

    window = (
        Message("user", (TextBlock("do the thing"),)),
        Message("assistant", (TextBlock("\n\n"),)),
        Message("user", (TextBlock("continue"),)),
    )

    sent = model._to_anthropic_messages(window)
    assistant_messages = [m for m in sent if m["role"] == "assistant"]
    content = assistant_messages[0]["content"]

    assert content != [{"type": "text", "text": "\n\n"}]
    assert all(block["type"] != "text" or block["text"].strip() for block in content)


def test_openai_compatible_empty_assistant_message_gets_content_when_no_tool_calls():
    """OpenAI rejects an assistant message with neither content nor tool_calls (just ``{"content": None}``)."""
    model = OpenAICompatible("gpt-5", api_key="x")

    message = Message("assistant", ())  # empty reply, no tool calls
    sent = model._assistant_message(message)

    assert sent["content"] is not None
    assert "tool_calls" not in sent


# ==================================================================
# 4. The thinking budget is checked at construction (catch mistakes early)
# ==================================================================


def test_thinking_budget_at_or_above_max_tokens_raises_at_construction():
    with pytest.raises(ValueError):
        Anthropic("claude-sonnet-5", thinking=16000)  # default max_tokens=8192


def test_thinking_budget_below_sdk_minimum_raises_at_construction():
    with pytest.raises(ValueError):
        Anthropic("claude-sonnet-5", thinking=500)  # the SDK minimum is 1024


def test_thinking_zero_is_not_treated_as_off_and_still_raises():
    """``0 is False`` is not true, so 0 is not treated as "off" -- it must be rejected as below the minimum."""
    with pytest.raises(ValueError):
        Anthropic("claude-sonnet-5", thinking=0)


def test_thinking_budget_within_bounds_constructs_fine():
    model = Anthropic("claude-sonnet-5", thinking=4096, max_tokens=8192)
    assert model._thinking == {"type": "enabled", "budget_tokens": 4096}


# ==================================================================
# 5. OPENAI_API_KEY from the environment must not be ignored when base_url is given
# ==================================================================


def test_openai_compatible_env_api_key_used_when_base_url_given(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    model = OpenAICompatible("x", base_url="https://openrouter.ai/api/v1")
    client = model._client()
    assert client.api_key == "sk-real"


def test_openai_compatible_falls_back_to_not_needed_only_without_any_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    model = OpenAICompatible("x", base_url="http://localhost:11434/v1")
    client = model._client()
    assert client.api_key == "not-needed"


# ==================================================================
# 6. o-series (reasoning) models get max_completion_tokens instead of max_tokens
# ==================================================================


def test_reasoning_model_uses_max_completion_tokens_not_max_tokens(monkeypatch):
    model = OpenAICompatible("o3", api_key="x", max_tokens=4000)
    calls = _install_openai_client(monkeypatch, model, chunks=[_oai_chunk(content="ok", finish_reason="stop")])

    model.respond(_req())

    sent = calls[0]
    assert "max_tokens" not in sent, f"sent max_tokens, which o-series models reject: {sent!r}"
    assert sent["max_completion_tokens"] == 4000


def test_non_reasoning_model_still_uses_max_tokens(monkeypatch):
    model = OpenAICompatible("gpt-5", api_key="x", max_tokens=4000)
    calls = _install_openai_client(monkeypatch, model, chunks=[_oai_chunk(content="ok", finish_reason="stop")])

    model.respond(_req())

    sent = calls[0]
    assert sent["max_tokens"] == 4000
    assert "max_completion_tokens" not in sent


# ==================================================================
# 7. cache_write_tokens must be mapped too (cache writes must not be counted as uncached input)
# ==================================================================


def test_openai_compatible_maps_cache_write_tokens(monkeypatch):
    """types.Usage: cache_write_tokens means "input tokens newly written to the cache", and adapters map SDK values
    to that meaning. When the openai SDK's PromptTokensDetails.cache_write_tokens carries that value, it must be
    used (and taken out of input_tokens)."""
    model = OpenAICompatible("gpt-5", api_key="x")
    usage = _oai_usage(prompt_tokens=10_000, completion_tokens=20, cached_tokens=0, cache_write_tokens=8_000)
    chunks = [_oai_chunk(content="ok"), _oai_chunk(finish_reason="stop", usage=usage)]
    _install_openai_client(monkeypatch, model, chunks=chunks)

    reply = model.respond(_req())

    assert reply.usage.cache_write_tokens == 8_000
    assert reply.usage.input_tokens == 10_000 - 8_000


def test_openai_compatible_cache_write_tokens_defaults_to_zero_when_absent(monkeypatch):
    model = OpenAICompatible("gpt-5", api_key="x")
    usage = _oai_usage(prompt_tokens=100, completion_tokens=20, cached_tokens=10)
    chunks = [_oai_chunk(content="ok"), _oai_chunk(finish_reason="stop", usage=usage)]
    _install_openai_client(monkeypatch, model, chunks=chunks)

    reply = model.respond(_req())

    assert reply.usage.cache_write_tokens == 0
    assert reply.usage.input_tokens == 90


# ==================================================================
# 8. When a consumer raises (reporter.on_text etc.), the stream/response must be closed
# ==================================================================


def test_openai_compatible_closes_stream_when_on_text_callback_raises(monkeypatch):
    """Wrapping in ``with client.chat.completions.create(...) as stream:`` closes the stream even when an
    exception from ``on_text`` escapes (like ``with client.messages.stream(...)`` in the Anthropic adapter)."""
    model = OpenAICompatible("gpt-5", api_key="x")
    chunks = [_oai_chunk(content="hello")]
    calls: list[dict] = []
    fake_stream = _FakeOpenAIStream(chunks)

    def fake_create(**kwargs):
        calls.append(kwargs)
        return fake_stream

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setattr(model, "_client", lambda: fake_client)

    def raising_on_text(chunk):
        raise RuntimeError("reporter blew up mid-stream")

    with pytest.raises(RuntimeError):
        model.respond(_req(), on_text=raising_on_text)

    assert fake_stream.closed, "the stream was not closed when respond() exited on an exception from on_text"


# ==================================================================
# 9. Tool calls must not run when the reply was cut off at the output token limit
# ==================================================================


def _agent_runs_tool_and_reports_first_result(reply, tool_fn):
    """Runs the Agent for one turn with a single fake Reply and returns the tool result string (whether the tool
    was really called is checked separately with the ``called`` list)."""
    fake = FakeModel([reply, "done"])
    agent = Agent(model=fake, tools=[tool_fn], reporter=None, human=None)
    state = State("Task")
    agent.think(state)
    agent.use_tools(state)
    return state.context[-1].content[0].content


def test_anthropic_max_tokens_truncated_tool_call_is_marked_invalid(monkeypatch):
    """With ``stop_reason == "max_tokens"``, the last tool_use block must not run even if its arguments happen to
    parse as a valid dict (e.g. ``command`` was cut off entirely)."""
    model = Anthropic("claude-sonnet-5")
    final = SimpleNamespace(
        content=[_anthropic_block("tool_use", id="call1", name="bash", input={"cwd": "/"})],
        usage=_anthropic_usage(input_tokens=1, output_tokens=1),
        stop_reason="max_tokens",
    )
    _install_anthropic_client(monkeypatch, model, _AnthropicStreamCM(final=final))

    reply = model.respond(_req())

    assert reply.stop_reason == "max_tokens"
    assert reply.tool_calls == (ToolCall("bash", {INVALID_ARGS_KEY: TRUNCATED_ARGS_MESSAGE}, "call1"),)

    called = []

    @tool
    def bash(cwd: str = ".", command: str = "echo hi") -> str:
        """Fake bash for tests"""
        called.append((cwd, command))
        return "ran"

    content = _agent_runs_tool_and_reports_first_result(reply, bash)

    assert content.startswith("(input error:")
    assert called == [], "a truncated call must not run with command filled in from its default"


def test_anthropic_earlier_complete_tool_use_block_is_not_over_marked(monkeypatch):
    """Even when it ended on max_tokens, only the last block is suspect -- tool_use blocks received in full
    before it are left alone."""
    model = Anthropic("claude-sonnet-5")
    final = SimpleNamespace(
        content=[
            _anthropic_block("tool_use", id="call1", name="search", input={"q": "weather"}),
            _anthropic_block("tool_use", id="call2", name="bash", input={"cwd": "/"}),
        ],
        usage=_anthropic_usage(input_tokens=1, output_tokens=1),
        stop_reason="max_tokens",
    )
    _install_anthropic_client(monkeypatch, model, _AnthropicStreamCM(final=final))

    reply = model.respond(_req())

    assert reply.tool_calls[0] == ToolCall("search", {"q": "weather"}, "call1")
    assert reply.tool_calls[1] == ToolCall("bash", {INVALID_ARGS_KEY: TRUNCATED_ARGS_MESSAGE}, "call2")


def test_anthropic_tool_use_stop_reason_still_works(monkeypatch):
    """A call that ended normally (``stop_reason == "tool_use"``) runs as is -- the max_tokens handling must not
    kick in by mistake."""
    model = Anthropic("claude-sonnet-5")
    final = SimpleNamespace(
        content=[_anthropic_block("tool_use", id="call1", name="bash", input={"cwd": "/", "command": "ls"})],
        usage=_anthropic_usage(input_tokens=1, output_tokens=1),
        stop_reason="tool_use",
    )
    _install_anthropic_client(monkeypatch, model, _AnthropicStreamCM(final=final))

    reply = model.respond(_req())

    assert reply.tool_calls == (ToolCall("bash", {"cwd": "/", "command": "ls"}, "call1"),)

    called = []

    @tool
    def bash(cwd: str = ".", command: str = "echo hi") -> str:
        """Fake bash for tests"""
        called.append((cwd, command))
        return "ran"

    content = _agent_runs_tool_and_reports_first_result(reply, bash)

    assert content == "ran"
    assert called == [("/", "ls")]


def test_openai_compatible_length_truncated_tool_call_is_marked_invalid(monkeypatch):
    """With ``finish_reason == "length"``, the last tool call must not run even if its argument JSON happens to be
    valid (e.g. it ends at ``{"cwd": "/"}`` and ``command`` is missing entirely)."""
    model = OpenAICompatible("gpt-5", api_key="x")
    chunks = [
        _oai_chunk(tool_calls=[_oai_tool_delta(0, id="call_1", name="bash", arguments='{"cwd": "/"}')]),
        _oai_chunk(finish_reason="length"),
    ]
    _install_openai_client(monkeypatch, model, chunks=chunks)

    reply = model.respond(_req())

    assert reply.stop_reason == "length"
    assert reply.tool_calls == (ToolCall("bash", {INVALID_ARGS_KEY: TRUNCATED_ARGS_MESSAGE}, "call_1"),)

    called = []

    @tool
    def bash(cwd: str = ".", command: str = "echo hi") -> str:
        """Fake bash for tests"""
        called.append((cwd, command))
        return "ran"

    content = _agent_runs_tool_and_reports_first_result(reply, bash)

    assert content.startswith("(input error:")
    assert called == [], "a truncated call must not run with command filled in from its default"


def test_openai_compatible_earlier_complete_tool_call_is_not_over_marked(monkeypatch):
    """Even when it ended on length, only the last call (highest index) is suspect -- calls received in full
    before it are left alone."""
    model = OpenAICompatible("gpt-5", api_key="x")
    chunks = [
        _oai_chunk(tool_calls=[_oai_tool_delta(0, id="call_1", name="search", arguments='{"q": "weather"}')]),
        _oai_chunk(tool_calls=[_oai_tool_delta(1, id="call_2", name="bash", arguments='{"cwd": "/"}')]),
        _oai_chunk(finish_reason="length"),
    ]
    _install_openai_client(monkeypatch, model, chunks=chunks)

    reply = model.respond(_req())

    assert reply.tool_calls[0] == ToolCall("search", {"q": "weather"}, "call_1")
    assert reply.tool_calls[1] == ToolCall("bash", {INVALID_ARGS_KEY: TRUNCATED_ARGS_MESSAGE}, "call_2")


def test_openai_compatible_tool_calls_stop_reason_still_works(monkeypatch):
    """A call that ended normally (``finish_reason == "tool_calls"``) runs as is."""
    model = OpenAICompatible("gpt-5", api_key="x")
    chunks = [
        _oai_chunk(tool_calls=[_oai_tool_delta(0, id="call_1", name="bash", arguments='{"cwd": "/", "command": "ls"}')]),
        _oai_chunk(finish_reason="tool_calls"),
    ]
    _install_openai_client(monkeypatch, model, chunks=chunks)

    reply = model.respond(_req())

    assert reply.tool_calls == (ToolCall("bash", {"cwd": "/", "command": "ls"}, "call_1"),)

    called = []

    @tool
    def bash(cwd: str = ".", command: str = "echo hi") -> str:
        """Fake bash for tests"""
        called.append((cwd, command))
        return "ran"

    content = _agent_runs_tool_and_reports_first_result(reply, bash)

    assert content == "ran"
    assert called == [("/", "ls")]
