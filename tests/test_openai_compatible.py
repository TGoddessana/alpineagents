"""``models.openai_compatible.OpenAICompatible`` tests for failures while the reply streams. No network: the real
openai SDK runs over an ``httpx2.MockTransport`` whose response body fails mid-stream.

The openai SDK wraps a transport error raised while the stream is read (``APITimeoutError`` for a read timeout,
``APIConnectionError`` for a dropped connection) and raises ``APIError`` for an error sent in the stream, so all of
them reach the adapter as ``OpenAIError`` and become ``ProviderError``.
"""

import asyncio

import httpx2
import openai
import pytest

from alpineagents.errors import ProviderError
from alpineagents.models.openai_compatible import OpenAICompatible
from alpineagents.types import Message, Request, TextBlock

_SSE_START = (
    b'data: {"id":"c1","object":"chat.completion.chunk","created":0,"model":"gpt-5",'
    b'"choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}]}\n\n'
)

_SSE_END = (
    b'data: {"id":"c1","object":"chat.completion.chunk","created":0,"model":"gpt-5",'
    b'"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
    b'data: {"id":"c1","object":"chat.completion.chunk","created":0,"model":"gpt-5","choices":[],'
    b'"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n'
    b"data: [DONE]\n\n"
)

_SSE_ERROR = b'data: {"error":{"message":"The server is overloaded","type":"server_error"}}\n\n'


class _SyncBody(httpx2.SyncByteStream):
    """A response body: yields the bytes parts and raises the exception parts."""

    def __init__(self, parts):
        self.parts = parts

    def __iter__(self):
        for part in self.parts:
            if isinstance(part, BaseException):
                raise part
            yield part


class _AsyncBody(httpx2.AsyncByteStream):
    def __init__(self, parts):
        self.parts = parts

    async def __aiter__(self):
        for part in self.parts:
            if isinstance(part, BaseException):
                raise part
            yield part


def _install_sdk_over_transport(monkeypatch, model, parts, *, use_async):
    def transport(body_cls):
        def handler(request):
            return httpx2.Response(
                200, headers={"content-type": "text/event-stream"}, stream=body_cls(parts), request=request
            )

        return httpx2.MockTransport(handler)

    kwargs = {"api_key": "test-key", "base_url": "http://gateway.test/v1", "max_retries": 0}
    if use_async:
        client = openai.AsyncOpenAI(**kwargs, http_client=httpx2.AsyncClient(transport=transport(_AsyncBody)))
        monkeypatch.setattr(model, "_async_client", lambda: client)
    else:
        client = openai.OpenAI(**kwargs, http_client=httpx2.Client(transport=transport(_SyncBody)))
        monkeypatch.setattr(model, "_client", lambda: client)


async def _respond(model, use_async, on_text=None):
    request = Request(system=None, messages=(Message("user", (TextBlock("hi"),)),))
    if use_async:
        return await model.arespond(request, on_text=on_text)
    return model.respond(request, on_text=on_text)


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "fail, sdk_error",
    [
        (httpx2.ReadTimeout("The read operation timed out"), openai.APITimeoutError),
        (
            httpx2.RemoteProtocolError("peer closed connection without sending complete message body"),
            openai.APIConnectionError,
        ),
        (httpx2.ReadError("connection reset by peer"), openai.APIConnectionError),
    ],
    ids=["ReadTimeout", "RemoteProtocolError", "ReadError"],
)
async def test_transport_error_mid_stream_becomes_provider_error(monkeypatch, use_async, fail, sdk_error):
    model = OpenAICompatible("gpt-5")
    _install_sdk_over_transport(monkeypatch, model, [_SSE_START, fail], use_async=use_async)
    seen = []
    with pytest.raises(ProviderError) as exc_info:
        await _respond(model, use_async, seen.append)
    assert type(exc_info.value) is ProviderError
    cause = exc_info.value.__cause__
    assert type(cause) is sdk_error
    assert cause.__cause__ is fail  # the SDK wrapped the raw httpx2 error itself
    assert seen == ["Hel"]


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_error_event_mid_stream_becomes_provider_error(monkeypatch, use_async):
    model = OpenAICompatible("gpt-5")
    _install_sdk_over_transport(monkeypatch, model, [_SSE_START, _SSE_ERROR], use_async=use_async)
    seen = []
    with pytest.raises(ProviderError) as exc_info:
        await _respond(model, use_async, seen.append)
    assert type(exc_info.value) is ProviderError
    assert isinstance(exc_info.value.__cause__, openai.APIError)
    assert "overloaded" in str(exc_info.value)
    assert seen == ["Hel"]


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("fail_type", [KeyboardInterrupt, asyncio.CancelledError])
async def test_interrupt_or_cancel_mid_stream_propagates_unwrapped(monkeypatch, use_async, fail_type):
    model = OpenAICompatible("gpt-5")
    fail = fail_type()
    _install_sdk_over_transport(monkeypatch, model, [_SSE_START, fail], use_async=use_async)
    with pytest.raises(fail_type) as exc_info:
        await _respond(model, use_async)
    assert exc_info.value is fail


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_on_text_exception_propagates_unwrapped(monkeypatch, use_async):
    model = OpenAICompatible("gpt-5")
    _install_sdk_over_transport(monkeypatch, model, [_SSE_START, _SSE_END], use_async=use_async)
    mine = httpx2.ReadError("my own webhook failed")

    def on_text(_chunk):
        raise mine

    with pytest.raises(httpx2.ReadError) as exc_info:
        await _respond(model, use_async, on_text)
    assert exc_info.value is mine


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_complete_stream_over_the_fake_transport(monkeypatch, use_async):
    model = OpenAICompatible("gpt-5")
    _install_sdk_over_transport(monkeypatch, model, [_SSE_START, _SSE_END], use_async=use_async)
    seen = []
    reply = await _respond(model, use_async, seen.append)
    assert seen == ["Hel", "lo"]
    assert reply.message.content == (TextBlock("Hello"),)
    assert reply.usage.input_tokens == 5
