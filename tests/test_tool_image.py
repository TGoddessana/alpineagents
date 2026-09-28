"""Image tool results: ``return Image(...)`` from a tool, and what the model, the Reporter, the context size estimate
and the store get."""

import asyncio
import base64
import io

import pytest

from alpineagents import Agent, FileStore, Image, Reporter, State, tool
from alpineagents._tokens import IMAGE_TOKENS, estimate_message_tokens
from alpineagents.models.anthropic import Anthropic
from alpineagents.models.openai_compatible import OpenAICompatible
from alpineagents.terminal import Terminal
from alpineagents.testing import FakeModel, tool_call
from alpineagents.tool import format_result
from alpineagents.types import (
    Message,
    Request,
    TextBlock,
    ToolCall,
    ToolOutcome,
    ToolResultBlock,
    result_text,
)

PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(100))
JPEG = b"\xff\xd8\xff\xe0" + bytes(100)
GIF = b"GIF89a" + bytes(100)
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + bytes(100)


def make_agent(fake, tools, **settings):
    settings.setdefault("reporter", None)
    return Agent(model=fake, tools=tools, human=None, **settings)


def result_block(state):
    return next(
        block for message in state.context for block in message.content if isinstance(block, ToolResultBlock)
    )


def result_entry(state):
    return next(entry for entry in state.history if entry.kind == "tool_result")


@tool
def screenshot(url: str):
    """Take a screenshot"""
    return [f"Loaded {url}", Image(PNG)]


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data, media_type", [(PNG, "image/png"), (JPEG, "image/jpeg"), (GIF, "image/gif"), (WEBP, "image/webp")]
)
def test_the_type_is_read_from_the_bytes(data, media_type):
    assert Image(data).media_type == media_type


def test_a_given_type_is_kept():
    assert Image(b"not sniffed", "image/jpeg").media_type == "image/jpeg"


def test_bytes_that_are_not_an_image_need_a_type():
    with pytest.raises(ValueError, match="not a PNG, JPEG, GIF or WebP image"):
        Image(b"hello")


def test_an_unsupported_type_is_refused():
    with pytest.raises(ValueError, match="'image/svg\\+xml' is not supported"):
        Image(b"<svg/>", "image/svg+xml")


def test_data_must_be_bytes():
    with pytest.raises(TypeError, match="Image.from_path"):
        Image(base64.b64encode(PNG).decode())


def test_bytearray_becomes_bytes():
    image = Image(bytearray(PNG))
    assert type(image.data) is bytes and image == Image(PNG)


def test_from_path_and_from_base64(tmp_path):
    path = tmp_path / "shot.bin"  # the name does not matter
    path.write_bytes(PNG)
    assert Image.from_path(path) == Image(PNG)
    assert Image.from_base64(base64.b64encode(PNG).decode()) == Image(PNG)
    assert Image(PNG).base64 == base64.b64encode(PNG).decode()


def test_from_base64_refuses_text_that_is_not_base64():
    with pytest.raises(ValueError):
        Image.from_base64("not base64!", "image/png")


def test_repr_does_not_print_the_bytes():
    assert repr(Image(PNG)) == "Image(image/png, 108B)"


# ---------------------------------------------------------------------------
# format_result
# ---------------------------------------------------------------------------


def test_an_image_becomes_one_block():
    assert format_result(Image(PNG)) == (Image(PNG),)


def test_a_list_with_an_image_becomes_blocks_in_order():
    content = format_result(["Loaded", "", None, {"status": 200}, Image(PNG), Image(GIF)])
    assert content == (TextBlock("Loaded"), TextBlock('{"status": 200}'), Image(PNG), Image(GIF))


def test_a_list_without_an_image_is_still_json():
    assert format_result(["a", "b"]) == '["a", "b"]'


def test_an_image_inside_a_dict_is_an_error_that_says_how_to_return_it():
    with pytest.raises(TypeError, match="Return the Image itself, or a list of text and Images"):
        format_result({"shot": Image(PNG)})


def test_result_text():
    assert result_text("plain") == "plain"
    assert result_text((TextBlock("Loaded"), Image(PNG))) == "Loaded\n(image/png, 108B)"


# ---------------------------------------------------------------------------
# A run
# ---------------------------------------------------------------------------


def test_the_image_reaches_the_context_history_and_next_request():
    fake = FakeModel([tool_call("screenshot", url="x.dev"), "done"])
    state = State("Task")
    make_agent(fake, [screenshot]).run(state)

    expected = (TextBlock("Loaded x.dev"), Image(PNG))
    assert result_block(state).content == expected
    assert result_entry(state).content == expected
    sent = [b for m in fake.requests[1].messages for b in m.content if isinstance(b, ToolResultBlock)]
    assert sent[0].content == expected
    assert state.answer == "done"


def test_an_async_tool_can_return_an_image():
    @tool
    async def shot() -> Image:
        """Take a screenshot"""
        return Image(PNG)

    state = State("Task")
    asyncio.run(make_agent(FakeModel([tool_call("shot"), "done"]), [shot]).arun(state))
    assert result_block(state).content == (Image(PNG),)


def test_the_reporter_gets_the_blocks():
    class Ends(Reporter):
        def __init__(self):
            self.results = []

        def on_tool_end(self, state, call, result, outcome):
            self.results.append((result, outcome.kind))

    ends = Ends()
    make_agent(FakeModel([tool_call("screenshot", url="x.dev"), "done"]), [screenshot], reporter=ends).run(
        State("Task")
    )
    assert ends.results == [((TextBlock("Loaded x.dev"), Image(PNG)), "done")]


def test_terminal_shows_the_size_and_image_count():
    out = io.StringIO()
    state = State("Task")
    Terminal(output=out).on_tool_end(state, ToolCall("shot", {}, "c1"), (Image(PNG), Image(GIF)), ToolOutcome("done"))
    assert out.getvalue().splitlines()[-1] == "  done 214B (2 images)"


def test_state_str_shows_the_size():
    state = State("Task")
    make_agent(FakeModel([tool_call("screenshot", url="x.dev"), "done"]), [screenshot]).run(state)
    assert "tool_result screenshot: 120B" in str(state)


def test_an_image_counts_as_image_tokens_not_its_base64_length():
    big = Image(b"\x89PNG\r\n\x1a\n" + bytes(300_000))
    with_image = Message("user", (ToolResultBlock("c1", (TextBlock("hi"), big)),))
    text_only = Message("user", (ToolResultBlock("c1", "hi"),))
    assert estimate_message_tokens(with_image) == estimate_message_tokens(text_only) + IMAGE_TOKENS


def test_clear_tool_results_clears_images_too():
    state = State("Task")
    make_agent(FakeModel([tool_call("screenshot", url="x.dev"), "done"]), [screenshot]).run(state)
    state.clear_tool_results(keep_last=0)
    assert result_block(state).content == "(cleared: kept in history)"
    assert result_entry(state).content == (TextBlock("Loaded x.dev"), Image(PNG))


def test_store_saves_and_loads_the_image(tmp_path):
    store = FileStore(tmp_path)
    state = State("Task")
    make_agent(FakeModel([tool_call("screenshot", url="x.dev"), "done"]), [screenshot], store=store).run(state)

    loaded = store.load(state.id)
    expected = (TextBlock("Loaded x.dev"), Image(PNG))
    assert result_entry(loaded).content == expected
    assert result_block(loaded).content == expected


# ---------------------------------------------------------------------------
# What the providers get
# ---------------------------------------------------------------------------


def image_request(content, *, is_error=False, text=None):
    call = ToolCall("screenshot", {"url": "x.dev"}, "c1")
    results = [ToolResultBlock("c1", content, "screenshot", is_error=is_error)]
    if text is not None:
        results.append(TextBlock(text))
    return Request(None, (Message.user("Task"), Message("assistant", (call,)), Message("user", tuple(results))), ())


def test_anthropic_sends_image_blocks_in_the_tool_result():
    request = image_request((TextBlock("Loaded"), TextBlock("  "), Image(PNG)))
    messages = Anthropic("claude-sonnet-5", api_key="k")._to_anthropic_messages(request.messages)
    (result,) = messages[-1]["content"]
    assert result["content"] == [
        {"type": "text", "text": "Loaded"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": Image(PNG).base64}},
    ]


def test_anthropic_still_sends_a_string_result_as_a_string():
    messages = Anthropic("claude-sonnet-5", api_key="k")._to_anthropic_messages(image_request("ok").messages)
    assert messages[-1]["content"][0]["content"] == "ok"


def test_openai_compatible_moves_images_to_a_user_message_with_tags():
    request = image_request((TextBlock("Loaded"), Image(PNG)), text="Also check the footer")
    messages = OpenAICompatible("gpt-5", api_key="k")._request_kwargs(request)["messages"]
    tool_message, user_message = messages[-2:]
    assert tool_message == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "Loaded\n(1 image in the next user message, inside <tool_result> tags)",
    }
    assert user_message == {
        "role": "user",
        "content": [
            {"type": "text", "text": '<tool_result tool_name="screenshot" tool_call_id="c1">'},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{Image(PNG).base64}"}},
            {"type": "text", "text": "</tool_result>"},
            {"type": "text", "text": "Also check the footer"},
        ],
    }


def test_openai_compatible_error_result_with_images():
    request = image_request((Image(PNG), Image(GIF)), is_error=True)
    messages = OpenAICompatible("gpt-5", api_key="k")._request_kwargs(request)["messages"]
    assert messages[-2]["content"] == "Error: (2 images in the next user message, inside <tool_result> tags)"
    assert [part["type"] for part in messages[-1]["content"]] == ["text", "image_url", "image_url", "text"]


def test_openai_compatible_text_results_are_unchanged():
    messages = OpenAICompatible("gpt-5", api_key="k")._request_kwargs(image_request("ok", text="next")).get("messages")
    assert messages[-2:] == [
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "user", "content": "next"},
    ]
