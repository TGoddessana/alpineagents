"""Async API tests: ``arun``/``athink``/``ause_tools``/``aask``/``aask_human``/``acompact``, async ``@loop``, the
default ``Model.arespond``/``Human.aask``, and the async adapters. No network."""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time
from types import SimpleNamespace
from typing import Literal

import pytest
from pydantic import BaseModel

from alpineagents import (
    Agent,
    CompactIfFull,
    Human,
    Loop,
    ModelReplyEntry,
    ModelRequestEntry,
    Reporter,
    RunStartEntry,
    State,
    StopEntry,
    StoppedByFinish,
    StoppedByLimit,
    StoppedByUntil,
    acompact_if_full,
    adefault_loop,
    compact_if_full,
    default_loop,
    loop,
    tool,
)
from alpineagents.errors import OutputError, ProviderError, RateLimitError
from alpineagents.human import parse_answer
from alpineagents.models.anthropic import Anthropic
from alpineagents.models.base import Model
from alpineagents.models.openai_compatible import OpenAICompatible
from alpineagents.state import INTERRUPTED
from alpineagents.testing import FakeHuman, FakeModel, tool_call
from alpineagents.types import Message, Reply, Request, TextBlock, ToolCall, Usage


class Recorder(Reporter):
    """Records every notification with the thread it came from."""

    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []
        self.texts: list[str] = []
        self.ends: list[tuple[str, str, str]] = []
        self.run_error: BaseException | None = None

    def on_run_start(self, state):
        self.events.append(("run_start", threading.get_ident()))

    def on_run_end(self, state, error):
        self.events.append(("run_end", threading.get_ident()))
        self.run_error = error

    def on_think_start(self, state):
        self.events.append(("think_start", threading.get_ident()))

    def on_text(self, state, chunk):
        self.events.append(("text", threading.get_ident()))
        self.texts.append(chunk)

    def on_think_end(self, state, reply):
        self.events.append(("think_end", threading.get_ident()))

    def on_tool_start(self, state, call):
        self.events.append(("tool_start", threading.get_ident()))

    def on_tool_end(self, state, call, result, outcome):
        self.events.append(("tool_end", threading.get_ident()))
        self.ends.append((call.name, result, outcome.kind))


def make_agent(replies, **settings) -> Agent:
    settings.setdefault("reporter", None)
    settings.setdefault("human", None)
    return Agent(model=FakeModel(replies), **settings)


def waiting_for_user(state: State) -> bool:
    """The user-written until function of SPEC section 6."""
    if state.pending_calls or not state.messages:
        return False
    last = state.messages[-1]
    return last.role == "assistant" and not last.tool_calls


@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


# ================================================================ arun and the default loop


async def test_arun_default_loop_with_sync_tool():
    agent = make_agent([tool_call("add", a=1, b=2), "It is 3"], tools=[add])
    state = State(messages=[Message.user("What is 1 + 2?")])
    assert await agent.arun(state) == "It is 3"
    assert state.stopped == StoppedByUntil("waiting_for_user")
    assert [e.content for e in state.history if e.kind == "tool_result"] == ["3"]


async def test_arun_takes_a_task_string():
    assert await make_agent(["hi"]).arun("Say hi") == "hi"


async def test_arun_with_explicit_default_loop_uses_the_async_default():
    agent = make_agent(["done"], loop=default_loop)
    assert await agent.arun("Task") == "done"


async def test_async_tools_run_on_the_loop_thread_and_concurrently():
    loop_thread = threading.get_ident()
    seen: list[int] = []

    @tool
    async def wait_a() -> str:
        """Wait"""
        seen.append(threading.get_ident())
        await asyncio.sleep(0.2)
        return "a"

    @tool
    async def wait_b() -> str:
        """Wait"""
        seen.append(threading.get_ident())
        await asyncio.sleep(0.2)
        return "b"

    agent = make_agent([[tool_call("wait_a"), tool_call("wait_b")], "done"], tools=[wait_a, wait_b])
    started = time.monotonic()
    await agent.arun("Task")
    assert time.monotonic() - started < 0.35
    assert seen == [loop_thread, loop_thread]


async def test_sync_tools_run_on_worker_threads_and_see_context_variables():
    request_id = contextvars.ContextVar("request_id", default=None)
    seen: list[tuple[int, str | None]] = []

    @tool
    def where() -> str:
        """Where am I"""
        seen.append((threading.get_ident(), request_id.get()))
        return "here"

    request_id.set("req-1")
    agent = make_agent([tool_call("where"), "done"], tools=[where])
    await agent.arun("Task")
    assert seen[0][0] != threading.get_ident()
    assert seen[0][1] == "req-1"


async def test_reporter_notifications_all_come_from_the_loop_thread():
    recorder = Recorder()

    @tool
    def slow() -> str:
        """Slow"""
        time.sleep(0.05)
        return "ok"

    agent = make_agent([["Let me look", tool_call("slow")], "Found it"], tools=[slow], reporter=recorder)
    await agent.arun("Task")
    names = [name for name, _ in recorder.events]
    assert names == [
        "run_start", "think_start", "text", "think_end", "tool_start", "tool_end",
        "think_start", "text", "think_end", "run_end",
    ]
    assert {thread for _, thread in recorder.events} == {threading.get_ident()}
    assert recorder.texts == ["Let me look", "Found it"]


async def test_async_with_agent():
    async with make_agent(["ok"]) as agent:
        assert await agent.arun("Task") == "ok"


# ================================================================ async @loop


async def test_async_loop_body_and_limit():
    calls = []

    @loop(until=waiting_for_user, limit=2)
    async def looping(agent: Agent, state: State):
        calls.append(state.turn)
        await agent.athink(state)
        if state.pending_calls:
            await agent.ause_tools(state)

    assert looping.is_async
    agent = make_agent([tool_call("add", a=1, b=1), tool_call("add", a=2, b=2)], tools=[add], loop=looping)
    state = State(messages=[Message.user("Keep adding")])
    await agent.arun(state)
    assert state.stopped == StoppedByLimit(2)
    assert len(calls) == 2


async def test_async_loop_copy_keeps_async():
    copied = adefault_loop.copy(limit=3)
    assert copied.is_async and copied.limit == 3
    assert await make_agent(["ok"], loop=copied).arun("Task") == "ok"


async def test_async_loop_as_a_block_of_another_async_loop():
    @loop(until=waiting_for_user, limit=5)
    async def outer(agent: Agent, state: State):
        await adefault_loop(agent, state)

    assert await make_agent([tool_call("add", a=1, b=2), "3"], tools=[add], loop=outer).arun("Task") == "3"


async def test_async_loop_body_returning_a_value_is_an_error():
    @loop(until=waiting_for_user, limit=5)
    async def returns(agent: Agent, state: State):
        await agent.athink(state)
        return "answer"

    with pytest.raises(TypeError, match="returned a value"):
        await make_agent(["x"], loop=returns).arun("Task")


def test_sync_loop_body_returning_an_awaitable_is_an_error():
    @loop(until=waiting_for_user, limit=5)
    def forgot_async(agent: Agent, state: State):
        return agent.athink(state)

    with pytest.raises(TypeError, match="returned an awaitable"):
        make_agent(["x"], loop=forgot_async).run("Task")


def test_run_with_an_async_loop_is_an_error():
    with pytest.raises(TypeError, match="arun"):
        make_agent(["x"], loop=adefault_loop).run("Task")


async def test_arun_with_a_sync_custom_loop_is_an_error():
    @loop(until=waiting_for_user, limit=5)
    def sync_loop(agent: Agent, state: State):
        agent.think(state)

    state = State(messages=[Message.user("Task")])
    with pytest.raises(TypeError, match="async def"):
        await make_agent(["x"], loop=sync_loop).arun(state)
    assert [e.kind for e in state.history] == ["context_change"]  # only the import: nothing ran, no run_start
    assert state.turn == 0


async def test_plain_async_function_as_loop():
    async def my_loop(agent: Agent, state: State):
        await agent.athink(state)
        return state.answer

    assert await make_agent(["ok"], loop=my_loop).arun("Task") == "ok"


# ================================================================ sync calls inside an async run


async def test_sync_think_inside_async_loop_is_an_error():
    @loop(until=waiting_for_user, limit=5)
    async def mixed(agent: Agent, state: State):
        agent.think(state)

    with pytest.raises(TypeError, match=r"await agent\.athink"):
        await make_agent(["x"], loop=mixed).arun("Task")


async def test_sync_compact_block_inside_async_loop_points_to_the_async_block():
    @loop(until=waiting_for_user, limit=5)
    async def mixed(agent: Agent, state: State):
        CompactIfFull(at=-1)(agent, state)

    with pytest.raises(TypeError, match="acompact_if_full"):
        await make_agent(["x"], loop=mixed).arun("Task")


async def test_sync_run_inside_a_sync_tool_of_an_async_run_works():
    helper = make_agent(["inner answer"])

    @tool
    def delegate() -> str:
        """Ask the helper"""
        return helper.run("Inner task")

    agent = make_agent([tool_call("delegate"), "outer answer"], tools=[delegate])
    assert await agent.arun("Task") == "outer answer"


async def test_sync_run_works_where_a_loop_runs_but_no_async_run_does():
    # Like a Jupyter cell: an event loop is running in this thread, but not an async run.
    assert make_agent(["ok"]).run("Task") == "ok"


# ================================================================ tools: exceptions, order


async def test_tool_exception_is_raised_with_a_note_after_the_others_finish():
    @tool
    async def boom() -> str:
        """Boom"""
        raise ValueError("bad input")

    @tool
    async def fine() -> str:
        """Fine"""
        await asyncio.sleep(0.05)
        return "fine"

    agent = make_agent([[tool_call("boom"), tool_call("fine")], "never"], tools=[boom, fine])
    state = State(messages=[Message.user("Task")])
    with pytest.raises(ValueError, match="bad input") as info:
        await agent.arun(state)
    assert any("exception raised in tool boom()" in note for note in info.value.__notes__)
    results = {e.call.name: e.content for e in state.history if e.kind == "tool_result"}
    assert results["fine"] == "fine"
    assert results["boom"] == "(aborted: ValueError)"
    assert not state.pending_calls


async def test_serial_tools_run_after_the_parallel_group_one_at_a_time():
    order: list[str] = []

    @tool(parallel=False)
    async def write(name: str) -> str:
        """Write"""
        order.append(f"start {name}")
        await asyncio.sleep(0.02)
        order.append(f"end {name}")
        return "ok"

    @tool
    async def read() -> str:
        """Read"""
        await asyncio.sleep(0.05)
        order.append("read")
        return "ok"

    calls = [tool_call("write", name="a"), tool_call("read"), tool_call("write", name="b")]
    await make_agent([calls, "done"], tools=[write, read]).arun("Task")
    assert order == ["read", "start a", "end a", "start b", "end b"]


async def test_unknown_tool_and_input_error_are_results():
    agent = make_agent([[tool_call("nope"), tool_call("add", a="x", b=1)], "done"], tools=[add])
    state = State(messages=[Message.user("Task")])
    await agent.arun(state)
    results = [e.content for e in state.history if e.kind == "tool_result"]
    assert results[0].startswith("(input error: unknown tool 'nope'")
    assert results[1].startswith("(input error:")


# ================================================================ cancellation


class WaitingModel(Model):
    """A Model whose ``arespond`` waits until released (for cancelling mid-think)."""

    provider = "waiting"
    name = "waiting"

    def __init__(self) -> None:
        self.started = asyncio.Event()

    @property
    def context_window(self) -> int:
        return 200_000

    def respond(self, request, on_text=None, on_event=None):  # pragma: no cover - async only
        raise AssertionError("sync respond should not be used")

    async def arespond(self, request, on_text=None, on_event=None):
        self.started.set()
        await asyncio.sleep(10)
        raise AssertionError("not cancelled")


async def test_cancel_during_athink_rolls_back_and_ends_the_run():
    model = WaitingModel()
    recorder = Recorder()
    agent = Agent(model=model, reporter=recorder, human=None)
    state = State(messages=[Message.user("Task")])
    task = asyncio.create_task(agent.arun(state))
    await model.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert state.turn == 0
    assert [m.role for m in state.messages] == ["user"]
    assert isinstance(recorder.run_error, asyncio.CancelledError)
    assert [e.kind for e in state.history[-2:]] == ["model_request", "error"]  # request, then the error: a rollback


async def test_asyncio_timeout_around_arun():
    model = WaitingModel()
    state = State(messages=[Message.user("Task")])
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            await Agent(model=model, reporter=None, human=None).arun(state)
    assert state.turn == 0
    assert state.history[-1].kind == "error"


async def test_cancel_during_ause_tools_closes_calls_and_records_late_results():
    tool_started = threading.Event()
    async_cancelled = asyncio.Event()

    @tool
    def slow_sync() -> str:
        """Finishes after the cancel"""
        tool_started.set()
        time.sleep(0.2)
        return "exit 0"

    @tool
    async def slow_async() -> str:
        """Gets cancelled"""
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            async_cancelled.set()
            raise
        return "never"

    recorder = Recorder()
    agent = make_agent(
        [[tool_call("slow_sync"), tool_call("slow_async")]], tools=[slow_sync, slow_async], reporter=recorder
    )
    state = State(messages=[Message.user("Task")])
    task = asyncio.create_task(agent.arun(state))
    await asyncio.to_thread(tool_started.wait)
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert async_cancelled.is_set()
    assert not state.pending_calls
    assert sorted((name, result, kind) for name, result, kind in recorder.ends) == [
        ("slow_async", INTERRUPTED, "interrupted"),
        ("slow_sync", INTERRUPTED, "interrupted"),
    ]
    results = [e.content for e in state.history if e.kind == "tool_result" and not e.late]
    assert results == [INTERRUPTED, INTERRUPTED]

    await asyncio.sleep(0.4)  # the sync tool finishes: its result is recorded as a late result
    late = [e for e in state.history if e.kind == "tool_result" and e.late]
    assert [(e.call.name, e.content) for e in late] == [("slow_sync", "exit 0")]


async def test_cancelled_default_arespond_stops_the_stream_early():
    chunks_sent: list[int] = []

    class SlowStream(FakeModel):
        def respond(self, request, on_text=None, on_event=None):
            for i in range(50):
                time.sleep(0.01)
                chunks_sent.append(i)
                if on_text is not None:
                    on_text(f"{i} ")
            return super().respond(request, on_text, on_event)

    agent = Agent(model=SlowStream(["done"]), reporter=Recorder(), human=None)
    task = asyncio.create_task(agent.arun("Task"))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.2)
    assert len(chunks_sent) < 30


async def test_on_text_exception_propagates_through_the_default_arespond():
    class Broken(Recorder):
        def on_text(self, state, chunk):
            raise RuntimeError("reporter broke")

    state = State(messages=[Message.user("Task")])
    agent = Agent(model=FakeModel(["hello"]), reporter=Broken(), human=None)
    with pytest.raises(RuntimeError, match="reporter broke"):
        await agent.athink(state)
    assert state.turn == 0 and [m.role for m in state.messages] == ["user"]


async def test_timeout_error_from_a_relayed_callback_propagates():
    class Broken(Recorder):
        def on_text(self, state, chunk):
            raise TimeoutError("reporter timed out")

    agent = Agent(model=FakeModel(["hello"]), reporter=Broken(), human=None)
    with pytest.raises(TimeoutError, match="reporter timed out"):
        await asyncio.wait_for(agent.athink(State(messages=[Message.user("Task")])), timeout=2)


async def test_relayed_callback_queued_before_cancel_does_not_run_after_it():
    """A worker's on_event that reaches the loop after the run was cancelled is dropped, not recorded."""
    release = threading.Event()
    recorder = Recorder()

    class LateEvent(FakeModel):
        def respond(self, request, on_text=None, on_event=None):
            release.wait(2)
            from alpineagents.types import ModelEvent

            on_event(ModelEvent("retry", "late event"))
            return super().respond(request, on_text, on_event)

    state = State(messages=[Message.user("Task")])
    agent = Agent(model=LateEvent(["x"]), reporter=recorder, human=None)
    task = asyncio.create_task(agent.arun(state))
    await asyncio.sleep(0.05)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.3)
    # run_start and the request were recorded; the cancelled request was taken back, the error stays
    assert [e.kind for e in state.history] == ["context_change", "run_start", "model_request", "error"]


async def test_async_tool_cancelling_itself_is_recorded_once():
    @tool
    async def gives_up() -> str:
        """Cancels itself"""
        raise asyncio.CancelledError()

    state = State(messages=[Message.user("Task")])
    with pytest.raises(asyncio.CancelledError):
        await make_agent([tool_call("gives_up")], tools=[gives_up]).arun(state)
    assert [e.kind for e in state.history].count("error") == 1


@pytest.mark.parametrize("use_async", [False, True])
async def test_anthropic_on_text_type_error_propagates_as_is(monkeypatch, use_async):
    model = Anthropic("claude-sonnet-5")
    message = _anthropic_message(SimpleNamespace(type="text", text="x"))

    def broken(chunk):
        raise TypeError("reporter bug")

    if use_async:
        cm = FakeAsyncStreamCM(["x"], message)
        client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: cm))
        monkeypatch.setattr(model, "_async_client", lambda: client)
        with pytest.raises(TypeError, match="reporter bug"):
            await model.arespond(Request(None, (Message.user("hi"),)), broken)
    else:
        from test_anthropic import FakeStreamCM, install_fake_client

        install_fake_client(monkeypatch, model, FakeStreamCM(["x"], message))
        with pytest.raises(TypeError, match="reporter bug"):
            model.respond(Request(None, (Message.user("hi"),)), broken)


# ================================================================ aask, aask_human, acompact


class Plan(BaseModel):
    steps: list[str]


async def test_aask_with_a_retry():
    agent = make_agent(["not json", '{"steps": ["read", "fix"]}'])
    state = State(messages=[Message.user("Task")])
    plan = await agent.aask(state, "Plan it", returns=Plan)
    assert plan.steps == ["read", "fix"]
    assert [m.role for m in state.messages] == ["user"]
    assert state.usage.requests == 2


async def test_aask_keeps_failing_raises_output_error():
    with pytest.raises(OutputError, match="aask"):
        await make_agent(["x", "y"]).aask(State(messages=[Message.user("Task")]), "Number?", returns=int, retries=1)


async def test_aask_human_with_a_sync_human_runs_on_a_worker_thread():
    human = FakeHuman(["yes"])
    state = State(messages=[Message.user("Task")])
    assert await make_agent([], human=human).aask_human(state, "Run it?", returns=bool) is True
    assert human.questions == ["Run it?"]
    assert state.history[-1].kind == "human"


class WebHuman(Human):
    """A Human reached only asynchronously."""

    def __init__(self, answer: str) -> None:
        self.answer = answer

    async def aask(self, state, prompt, returns=str):
        await asyncio.sleep(0)
        return parse_answer(self.answer, returns)


async def test_aask_human_with_an_async_only_human():
    agent = make_agent([], human=WebHuman("always"))
    answer = await agent.aask_human(State(messages=[Message.user("Task")]), "Allow?", returns=Literal["yes", "no", "always"])
    assert answer == "always"


def test_sync_ask_human_with_an_async_only_human_is_an_error():
    with pytest.raises(TypeError, match="aask_human"):
        make_agent([], human=WebHuman("yes")).ask_human(State(messages=[Message.user("Task")]), "Allow?")


def test_human_subclass_must_implement_ask_or_aask():
    class Silent(Human):
        pass

    with pytest.raises(TypeError, match="neither ask nor aask"):
        Silent()


def test_abstract_intermediate_human_class_is_allowed():
    from abc import ABC, abstractmethod

    class ChannelHuman(Human, ABC):
        @abstractmethod
        def send(self, text: str) -> None: ...

    class Console(ChannelHuman):
        def send(self, text: str) -> None:
            pass

        def ask(self, state, prompt, returns=str):
            return parse_answer("yes", returns)

    with pytest.raises(TypeError):
        ChannelHuman()
    assert Console().ask(State(messages=[Message.user("Task")]), "Run it?", bool) is True


async def test_acompact_if_full_compacts_when_over_the_threshold():
    agent = make_agent(["Summary: nothing done yet", "done"], loop=None)
    state = State(messages=[Message.user("A long task " * 20)])

    @loop(until=waiting_for_user, limit=3)
    async def compacting(agent: Agent, state: State):
        await CompactIfFull(at=0.0).acall(agent, state)
        await agent.athink(state)

    await agent.copy(loop=compacting).arun(state)
    changes = [e.content for e in state.history if e.kind == "context_change" and e.content.kind == "compact"]
    assert len(changes) == 1
    assert state.answer == "done"


async def test_acompact_if_full_does_nothing_under_the_threshold():
    agent = make_agent(["done"])
    state = State(messages=[Message.user("Task")])
    await acompact_if_full(agent, state)
    assert not any(e.kind == "context_change" and e.content.kind == "compact" for e in state.history)
    assert compact_if_full.acall == acompact_if_full


async def test_acompact_uses_an_adapter_s_own_compact():
    class ServerCompact(FakeModel):
        def compact(self, request, instructions=None, on_event=None):
            return Reply(Message("assistant", (TextBlock("server summary"),)), Usage(requests=1), None, "end_turn")

    agent = Agent(model=ServerCompact([]), reporter=None, human=None)
    state = State(messages=[Message.user("Task")])
    await agent.acompact(state)
    assert state.messages[-1].text.endswith("server summary")


# ================================================================ arun(prompt_or_state, /) and the run guard


async def test_arun_with_an_empty_or_blank_string_raises_value_error():
    fake = FakeModel(["never used"])
    for prompt in ("", "  \n"):
        with pytest.raises(ValueError, match="empty prompt"):
            await Agent(model=fake, reporter=None, human=None).arun(prompt)
    assert fake.requests == []


async def test_arun_rejects_something_that_is_neither_a_string_nor_a_state():
    with pytest.raises(TypeError):
        await make_agent([]).arun(123)  # type: ignore[arg-type]


async def test_arun_records_a_run_start_entry_per_run():
    agent = make_agent(["first", "second"])
    state = State(messages=[Message.user("Task")])
    await agent.arun(state)
    state.add_message(Message.user("More"))
    await agent.arun(state)
    assert [e.kind for e in state.history if e.kind in ("run_start", "stop")] == [
        "run_start",
        "stop",
        "run_start",
        "stop",
    ]


async def test_a_state_being_arun_refuses_a_second_arun_from_any_agent():
    # "One run at a time": the second run raises ValueError and leaves no trace; the first is not disturbed
    started = asyncio.Event()
    release = asyncio.Event()

    @tool
    async def block() -> str:
        """Waits until the test lets it go"""
        started.set()
        await release.wait()
        return "released"

    first = make_agent([tool_call("block"), "done"], tools=[block])
    rival = make_agent(["never used"])
    state = State(messages=[Message.user("Task")])
    task = asyncio.create_task(first.arun(state))
    try:
        await asyncio.wait_for(started.wait(), 5)
        before = state.history
        with pytest.raises(ValueError, match="already being run"):
            await first.arun(state)
        with pytest.raises(ValueError, match="already being run"):
            await rival.arun(state)
        assert state.history[: len(before)] == before
        assert len([e for e in state.history if isinstance(e, RunStartEntry)]) == 1
    finally:
        release.set()
    assert await task == "done"


async def test_arun_is_refused_while_a_sync_run_in_a_thread_holds_the_state():
    # the guard is on the State, so it holds between run (thread) and arun (event loop)
    started = threading.Event()
    release = threading.Event()

    @tool
    def block() -> str:
        """Waits until the test lets it go"""
        started.set()
        assert release.wait(5)
        return "ok"

    state = State(messages=[Message.user("Task")])
    worker = threading.Thread(target=lambda: make_agent([tool_call("block"), "done"], tools=[block]).run(state))
    worker.start()
    try:
        assert await asyncio.to_thread(started.wait, 5)
        with pytest.raises(ValueError, match="already being run"):
            await make_agent(["never used"]).arun(state)
    finally:
        release.set()
        worker.join(5)


async def test_the_run_guard_is_released_after_a_cancelled_arun():
    model = WaitingModel()
    state = State(messages=[Message.user("Task")])
    task = asyncio.create_task(Agent(model=model, reporter=None, human=None).arun(state))
    await model.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await make_agent(["Recovered"]).arun(state) == "Recovered"


async def test_agent_copy_with_another_model_continues_a_state_in_arun():
    first = make_agent(["one"])
    second = first.copy(model=FakeModel(["two"], name="other"))
    state = State(messages=[Message.user("Task")])
    await first.arun(state)
    state.add_message(Message.user("Again"))
    assert await second.arun(state) == "two"
    models = [e.content.model for e in state.history if isinstance(e, RunStartEntry)]
    assert models == ["fake/fake", "fake/other"]
    assert [e.content for e in state.history if isinstance(e, ModelRequestEntry)] == ["fake/fake", "fake/other"]


# ================================================================ athink: request/reply pairs and the rollback


async def test_athink_records_a_request_then_a_reply():
    state = State(messages=[Message.user("Task")])
    await make_agent(["Hello"]).athink(state)
    tail = state.history[-2:]
    assert isinstance(tail[0], ModelRequestEntry) and tail[0].turn == 1 and tail[0].content == "fake/fake"
    assert isinstance(tail[1], ModelReplyEntry) and tail[1].turn == 1


async def test_athink_failing_takes_the_request_back_and_flushes_deferred_messages():
    state = State(messages=[Message.user("Task")])

    def fail_after_a_message(request):
        state.add_message(Message.user("Added while waiting"))
        raise RuntimeError("model error")

    agent = make_agent([fail_after_a_message, "ok"])
    with pytest.raises(RuntimeError, match="model error"):
        await agent.athink(state)
    # request, the message added while waiting (deferred in messages, but on record), then the error
    assert [e.kind for e in state.history[-3:]] == ["model_request", "user", "error"]
    assert state.turn == 0
    assert [m.text for m in state.messages] == ["Task", "Added while waiting"]
    await agent.athink(state)  # the turn number is reused
    assert state.turn == 1


async def test_athink_on_an_empty_state_raises_value_error():
    fake = FakeModel(["never used"])
    state = State()
    with pytest.raises(ValueError, match="no messages"):
        await Agent(model=fake, reporter=None, human=None).athink(state)
    assert fake.requests == [] and state.history == ()


# ================================================================ aask / acompact record usage; context_used


async def test_aask_records_usage_and_json_while_returning_the_object():
    agent = make_agent(["not json", '{"steps": ["read", "fix"]}'])
    state = State(messages=[Message.user("Task")])
    plan = await agent.aask(state, "Plan it", returns=Plan)
    assert isinstance(plan, Plan)
    entry = state.history[-1]
    assert entry.kind == "ask"
    assert entry.content.answer == {"steps": ["read", "fix"]}
    assert not isinstance(entry.content.answer, Plan)
    assert entry.usage is not None and entry.usage.requests == 2
    assert state.usage == entry.usage


async def test_acompact_records_kept_and_the_usage_of_the_request():
    state = State(messages=[Message.user("Task")])

    def summarize_while_user_types(request):
        state.add_message(Message.user("One more thing"))
        return "The summary"

    agent = make_agent([summarize_while_user_types])
    await agent.acompact(state)
    change = state.history[-1].content
    assert change.kind == "compact" and change.kept == 1
    assert change.usage is not None and change.usage.requests == 1
    assert state.usage == change.usage
    assert [m.text for m in state.messages][0] == "Task"
    assert state.messages[-1].text == "One more thing"


async def test_context_used_and_tokens_work_inside_async_code():
    agent = Agent(model=FakeModel([], context_window=1_000), reporter=None, human=None)
    state = State(messages=[Message.user("word " * 100)])
    assert agent.context_tokens(state) > 0
    assert agent.context_used(state) == pytest.approx(agent.context_tokens(state) / 1_000)


# ================================================================ async loops record stops as StopEntry


async def test_async_default_loop_records_the_waiting_for_user_stop():
    state = State(messages=[Message.user("Task")])
    await make_agent(["answer"]).arun(state)
    assert [e.content for e in state.history if isinstance(e, StopEntry)] == [StoppedByUntil("waiting_for_user")]


async def test_async_finish_is_recorded_as_a_stop_entry_with_the_answer():
    @loop(until=waiting_for_user, limit=5)
    async def finishing(agent: Agent, state: State):
        await agent.athink(state)
        state.finish("all done")

    state = State(messages=[Message.user("Task")])
    assert await make_agent(["x"], loop=finishing).arun(state) == "all done"
    assert [e.content for e in state.history if isinstance(e, StopEntry)] == [StoppedByFinish("all done")]


async def test_async_nested_loop_stop_is_cleared_when_the_outer_loop_goes_on():
    def never(state):
        return False

    @loop(until=never, limit=1)
    async def inner(agent: Agent, state: State):
        await agent.athink(state)

    def after_two_turns(state):
        return state.turn >= 2

    @loop(until=after_two_turns, limit=5)
    async def outer(agent: Agent, state: State):
        await inner(agent, state)

    state = State(messages=[Message.user("Task")])
    await make_agent(["one", "two"], loop=outer).arun(state)
    assert [e.content for e in state.history if isinstance(e, StopEntry)] == [
        StoppedByLimit(1),
        None,
        StoppedByLimit(1),
        StoppedByUntil("after_two_turns"),
    ]


# ================================================================ adapters


class FakeAsyncStreamCM:
    def __init__(self, chunks, final_message=None, error=None):
        self.chunks = chunks
        self.final_message = final_message
        self.error = error

    async def __aenter__(self):
        if self.error is not None:
            raise self.error
        return self

    async def __aexit__(self, *exc_info):
        return False

    @property
    def text_stream(self):
        async def gen():
            for chunk in self.chunks:
                yield chunk

        return gen()

    async def get_final_message(self):
        return self.final_message


def _anthropic_message(*blocks, stop_reason="end_turn"):
    usage = SimpleNamespace(
        input_tokens=10, output_tokens=5, cache_read_input_tokens=None, cache_creation_input_tokens=None
    )
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, usage=usage)


async def test_anthropic_arespond_streams_and_converts(monkeypatch):
    model = Anthropic("claude-sonnet-5")
    text = SimpleNamespace(type="text", text="Hello")
    use = SimpleNamespace(type="tool_use", name="add", input={"a": 1, "b": 2}, id="toolu_1")
    cm = FakeAsyncStreamCM(["Hel", "lo"], _anthropic_message(text, use, stop_reason="tool_use"))
    sent = []
    client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: sent.append(kw) or cm))
    monkeypatch.setattr(model, "_async_client", lambda: client)

    chunks: list[str] = []
    reply = await model.arespond(Request("sys", (Message.user("hi"),)), chunks.append)
    assert chunks == ["Hel", "lo"]
    assert reply.text == "Hello"
    assert reply.tool_calls == (ToolCall("add", {"a": 1, "b": 2}, "toolu_1"),)
    assert reply.context_tokens == 15
    assert sent[0]["system"] == "sys"


async def test_anthropic_arespond_wraps_sdk_errors(monkeypatch):
    import anthropic
    import httpx2

    model = Anthropic("claude-sonnet-5")
    response = httpx2.Response(429, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    error = anthropic.RateLimitError("slow down", response=response, body=None)
    client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: FakeAsyncStreamCM([], error=error)))
    monkeypatch.setattr(model, "_async_client", lambda: client)
    with pytest.raises(RateLimitError) as info:
        await model.arespond(Request(None, (Message.user("hi"),)))
    assert info.value.__cause__ is error


def test_async_clients_are_per_event_loop():
    model = Anthropic("claude-sonnet-5", api_key="test-key")

    async def get_twice():
        return model._async_client(), model._async_client()

    first_a, first_b = asyncio.run(get_twice())
    second, _ = asyncio.run(get_twice())
    assert first_a is first_b
    assert second is not first_a
    assert model._sdk_client is None


class FakeAsyncChunks:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __aiter__(self):
        async def gen():
            for chunk in self.chunks:
                yield chunk

        return gen()

    async def close(self):
        self.closed = True


def _openai_chunk(content=None, tool_calls=None, finish_reason=None, usage=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


async def test_openai_compatible_arespond_streams_and_closes(monkeypatch):
    model = OpenAICompatible("llama3", base_url="http://localhost:11434/v1")
    call = SimpleNamespace(index=0, id="c1", function=SimpleNamespace(name="add", arguments='{"a": 1, "b": 2}'))
    usage = SimpleNamespace(
        prompt_tokens=12, completion_tokens=3, prompt_tokens_details=SimpleNamespace(cached_tokens=2)
    )
    stream = FakeAsyncChunks([
        _openai_chunk("Hel"),
        _openai_chunk("lo", tool_calls=[call]),
        _openai_chunk(finish_reason="tool_calls"),
        SimpleNamespace(choices=[], usage=usage),
    ])

    async def create(**kwargs):
        return stream

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(model, "_async_client", lambda: client)

    chunks: list[str] = []
    reply = await model.arespond(Request(None, (Message.user("hi"),)), chunks.append)
    assert chunks == ["Hel", "lo"]
    assert reply.text == "Hello"
    assert reply.tool_calls == (ToolCall("add", {"a": 1, "b": 2}, "c1"),)
    assert reply.usage.input_tokens == 10 and reply.usage.cache_read_tokens == 2
    assert reply.stop_reason == "tool_calls"
    assert stream.closed


async def test_openai_compatible_arespond_wraps_sdk_errors(monkeypatch):
    import openai

    model = OpenAICompatible("llama3", base_url="http://localhost:11434/v1")
    error = openai.APIConnectionError(request=None)

    async def create(**kwargs):
        raise error

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(model, "_async_client", lambda: client)
    with pytest.raises(ProviderError) as info:
        await model.arespond(Request(None, (Message.user("hi"),)))
    assert info.value.__cause__ is error


def test_loop_is_async_flag():
    @loop(until=waiting_for_user, limit=1)
    def sync_body(agent, state):
        pass

    assert isinstance(adefault_loop, Loop) and adefault_loop.is_async
    assert not sync_body.is_async
    assert not default_loop.is_async
