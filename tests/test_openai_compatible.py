"""``models.openai_compatible.OpenAICompatible`` tests for failures while the reply streams. No network: the real
openai SDK runs over an ``httpx2.MockTransport`` whose response body fails mid-stream.

The openai SDK wraps a transport error raised while the stream is read (``APITimeoutError`` for a read timeout,
``APIConnectionError`` for a dropped connection) and raises ``APIError`` for an error sent in the stream, so all of
them reach the adapter as ``OpenAIError`` and become ``ProviderError``.
"""

import asyncio
import json

import httpx2
import openai
import pytest

from alpineagents.errors import ProviderError
from alpineagents.models.openai_compatible import OpenAICompatible
from alpineagents.types import Image, Message, RawBlock, Request, TextBlock, ToolCall, ToolResultBlock

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


# ---------------------------------------------------------------- images in user messages

PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(40))
JPEG = b"\xff\xd8\xff\xe0" + bytes(range(40))
GIF_ = b"GIF89a" + bytes(40)


def _image_part(image):
    return {"type": "image_url", "image_url": {"url": f"data:{image.media_type};base64,{image.base64}"}}


def _messages(*messages):
    return OpenAICompatible("gpt-5", api_key="k")._request_kwargs(Request(system=None, messages=messages))["messages"]


def test_a_user_message_image_becomes_an_image_url_part():
    (sent,) = _messages(Message.user("What is wrong in this chart?", Image(PNG)))
    assert sent == {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is wrong in this chart?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{Image(PNG).base64}"}},
        ],
    }


def test_user_message_images_keep_block_order_and_each_media_type():
    message = Message("user", (Image(PNG), TextBlock("between"), Image(JPEG), TextBlock("after")))
    (sent,) = _messages(message)
    assert sent["content"] == [
        _image_part(Image(PNG)),
        {"type": "text", "text": "between"},
        _image_part(Image(JPEG)),
        {"type": "text", "text": "after"},
    ]
    assert sent["content"][2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_adjacent_text_blocks_are_one_part_next_to_an_image():
    (sent,) = _messages(Message("user", (TextBlock("a"), TextBlock("b"), Image(PNG), TextBlock(""), TextBlock("c"))))
    assert sent["content"] == [{"type": "text", "text": "ab"}, _image_part(Image(PNG)), {"type": "text", "text": "c"}]


def test_an_image_only_user_message_has_no_placeholder_text():
    (sent,) = _messages(Message.user("", Image(PNG)))
    assert sent == {"role": "user", "content": [_image_part(Image(PNG))]}


def test_a_text_only_user_message_is_still_a_plain_string():
    assert _messages(Message.user("just words")) == [{"role": "user", "content": "just words"}]


def test_user_images_come_after_the_tool_result_images_in_the_same_user_message():
    call = ToolCall("shot", {"url": "x.dev"}, "c1")
    result = ToolResultBlock("c1", (TextBlock("Loaded"), Image(GIF_)), "shot")
    messages = _messages(
        Message.user("Task"),
        Message("assistant", (call,)),
        Message("user", (result, TextBlock("see this"), Image(PNG))),
    )
    tool_message, user_message = messages[-2:]
    assert tool_message["role"] == "tool"
    assert user_message["content"] == [
        {"type": "text", "text": '<tool_result tool_name="shot" tool_call_id="c1">'},
        _image_part(Image(GIF_)),
        {"type": "text", "text": "</tool_result>"},
        {"type": "text", "text": "see this"},
        _image_part(Image(PNG)),
    ]


def test_user_images_are_sent_even_when_vision_is_not_declared():
    model = OpenAICompatible("gpt-5", api_key="k", supports=frozenset())
    (sent,) = model._request_kwargs(Request(system=None, messages=(Message.user("x", Image(PNG)),)))["messages"]
    assert sent["content"][1] == _image_part(Image(PNG))


def test_nothing_frozen_reaches_the_sdk():
    # ToolCall.args and RawBlock.data are deep-frozen in history; the request holds plain containers (or the JSON text).
    call = ToolCall("look", {"opts": {"tags": ["a"]}}, "c1")
    raw = RawBlock("openai_compatible", {"reasoning_content": "hm", "extra": {"parts": ["a"]}})
    messages = _messages(
        Message.user("Task"),
        Message("assistant", (raw, call)),
        Message("user", (ToolResultBlock("c1", "ok", "look"),)),
    )
    assistant = messages[1]
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"opts": {"tags": ["a"]}}'
    assert type(assistant["extra"]["parts"]) is list and type(assistant["extra"]) is dict
    assistant["extra"]["parts"].append("b")
    assert raw.data == {"reasoning_content": "hm", "extra": {"parts": ["a"]}}


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_the_image_reaches_the_wire_as_a_data_url(monkeypatch, use_async):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_SyncBody([_SSE_START, _SSE_END]),
            request=request,
        )

    model = OpenAICompatible("gpt-5")
    kwargs = {"api_key": "test-key", "base_url": "http://gateway.test/v1", "max_retries": 0}
    request = Request(system=None, messages=(Message.user("What is this?", Image(PNG)),))
    if use_async:

        async def ahandler(request):
            sent.append(json.loads(request.content))
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_AsyncBody([_SSE_START, _SSE_END]),
                request=request,
            )

        client = openai.AsyncOpenAI(**kwargs, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(ahandler)))
        monkeypatch.setattr(model, "_async_client", lambda: client)
        await model.arespond(request)
    else:
        client = openai.OpenAI(**kwargs, http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
        monkeypatch.setattr(model, "_client", lambda: client)
        model.respond(request)
    (body,) = sent
    assert body["messages"][-1] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{Image(PNG).base64}"}},
        ],
    }
