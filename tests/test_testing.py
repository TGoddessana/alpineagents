"""Tests for FakeModel, FakeHuman and tool_call."""

from __future__ import annotations

import threading

import pytest

from alpineagents.errors import ContextTooLongError, RateLimitError
from alpineagents.testing import FakeHuman, FakeModel, tool_call
from alpineagents.types import Message, Request, ToolCall


def _request(*texts: str) -> Request:
    return Request(system=None, messages=tuple(Message.user(t) for t in texts))


def test_tool_call_builds_unique_ids_with_prefix():
    a = tool_call("read_file", path="main.py")
    b = tool_call("read_file", path="main.py")
    assert isinstance(a, ToolCall)
    assert a.name == "read_file"
    assert a.args == {"path": "main.py"}
    assert a.id.startswith("call_")
    assert len(a.id) == len("call_") + 12
    assert a.id != b.id


def test_fake_model_text_reply():
    fake = FakeModel(["The bug is on line 3"])
    reply = fake.respond(_request("Find the bug"))
    assert reply.text == "The bug is on line 3"
    assert reply.tool_calls == ()
    assert reply.stop_reason == "end_turn"
    assert reply.usage.requests == 1
    assert reply.context_tokens == reply.usage.input_tokens + reply.usage.output_tokens
    assert fake.requests == [_request("Find the bug")]
    assert fake.remaining == 0


def test_fake_model_tool_call_reply():
    call = tool_call("read_file", path="main.py")
    fake = FakeModel([call])
    reply = fake.respond(_request("Find the bug"))
    assert reply.tool_calls == (call,)
    assert reply.stop_reason == "tool_use"


def test_fake_model_from_design_example():
    fake = FakeModel([tool_call("read_file", path="main.py"), "The bug is on line 3"])
    reply1 = fake.respond(_request("Find the bug"))
    assert reply1.tool_calls[0].name == "read_file"
    reply2 = fake.respond(_request("Find the bug"))
    assert reply2.text == "The bug is on line 3"


def test_fake_model_list_of_blocks():
    call = tool_call("web_search", query="x")
    fake = FakeModel([["Let me look first.", call]])
    reply = fake.respond(_request("question"))
    assert reply.text == "Let me look first."
    assert reply.tool_calls == (call,)


def test_fake_model_on_text_called_per_text_block():
    fake = FakeModel([["Hel", "lo"]])
    seen = []
    fake.respond(_request("x"), on_text=seen.append)
    assert seen == ["Hel", "lo"]


def test_fake_model_reply_instance_passthrough():
    from alpineagents.types import Reply, Usage

    custom = Reply(message=Message("assistant", ()), usage=Usage(requests=1), stop_reason="end_turn")
    fake = FakeModel([custom])
    reply = fake.respond(_request("x"))
    assert reply is custom


def test_fake_model_exception_item_is_raised():
    fake = FakeModel([RateLimitError("too fast")])
    with pytest.raises(RateLimitError):
        fake.respond(_request("x"))


def test_fake_model_callable_item():
    seen_requests = []

    def responder(request):
        seen_requests.append(request)
        return "dynamic reply"

    fake = FakeModel([responder])
    req = _request("x")
    reply = fake.respond(req)
    assert reply.text == "dynamic reply"
    assert seen_requests == [req]


def test_fake_model_context_too_long_does_not_consume_item():
    fake = FakeModel(["hi"], context_window=1)
    with pytest.raises(ContextTooLongError):
        fake.respond(_request("x" * 1000))
    assert fake.remaining == 1


def test_fake_model_exhausted_raises_runtime_error():
    fake = FakeModel(["hi"])
    fake.respond(_request("x"))
    with pytest.raises(RuntimeError, match="ran out of prepared replies"):
        fake.respond(_request("y"))


def test_fake_model_requests_recorded_even_on_error():
    fake = FakeModel([])
    with pytest.raises(RuntimeError):
        fake.respond(_request("x"))
    assert len(fake.requests) == 1


def test_fake_model_thread_safe_distinct_replies():
    fake = FakeModel([f"answer{i}" for i in range(20)])
    results: list[str] = []
    lock = threading.Lock()

    def worker():
        reply = fake.respond(_request("x"))
        with lock:
            results.append(reply.text)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert sorted(results) == sorted(f"answer{i}" for i in range(20))
    assert fake.remaining == 0


def test_fake_human_returns_answers_in_order():
    human = FakeHuman(["no", "yes"])
    assert human.ask(None, "Continue?", returns=bool) is False
    assert human.ask(None, "Continue?", returns=bool) is True
    assert human.questions == ["Continue?", "Continue?"]


def test_fake_human_reasks_itself_on_bad_answer():
    human = FakeHuman(["maybe", "yes"])
    assert human.ask(None, "Continue?", returns=bool) is True


def test_fake_human_literal_returns():
    from typing import Literal

    human = FakeHuman(["always"])
    assert human.ask(None, "Allow?", returns=Literal["yes", "no", "always"]) == "always"


def test_fake_human_exhausted_raises_runtime_error():
    human = FakeHuman([])
    with pytest.raises(RuntimeError, match="Continue\\?"):
        human.ask(None, "Continue?", returns=bool)
