import asyncio
import time
from threading import Event

import pytest

from tensorcode.ops import llm
from tensorcode.tools.agents import Chatbot, JsonMemory, ObjectiveRevision


def encode_messages(messages):
    return [{"role": message.role, "content": message.content} for message in messages]


def decode_messages(payload):
    return tuple(llm.Message(item["role"], item["content"]) for item in payload)


def most_recent(request):
    return request.candidates[-request.limit :]


def test_failed_response_rolls_back_objective_memory_and_history(tmp_path):
    memory = JsonMemory(
        tmp_path / "memory.json",
        retrieve=most_recent,
        encode_value=encode_messages,
        decode_value=decode_messages,
    )
    initial = (llm.Message("system", "initial objective"),)

    def update(revision, *, context=None):
        assert isinstance(revision, ObjectiveRevision)
        return (llm.Message("system", "changed objective"),)

    def fail(messages, *, context=None):
        raise RuntimeError("provider offline")

    bot = Chatbot(
        respond=fail,
        objective=initial,
        update_objective=update,
        memory=memory,
    )

    with pytest.raises(RuntimeError, match="offline"):
        bot("do not commit")

    assert bot.objective == initial
    assert bot.history == ()
    assert memory.records == ()


def test_failed_response_rolls_back_in_place_objective_mutation():
    objective = {"status": "initial"}

    def mutate(revision, *, context=None):
        revision.current["status"] = "changed"
        return revision.current

    bot = Chatbot(
        respond=lambda messages, *, context=None: (_ for _ in ()).throw(
            RuntimeError("offline")
        ),
        objective=objective,
        update_objective=mutate,
    )

    with pytest.raises(RuntimeError, match="offline"):
        bot("question")

    assert bot.objective == {"status": "initial"}


def test_failed_memory_commit_rolls_back_successful_response_and_objective(tmp_path):
    def reject_assistant(messages):
        if any(message.role == "assistant" for message in messages):
            raise TypeError("response cannot be persisted")
        return encode_messages(messages)

    memory = JsonMemory(
        tmp_path / "memory.json",
        retrieve=most_recent,
        encode_value=reject_assistant,
        decode_value=decode_messages,
    )
    initial = (llm.Message("system", "initial"),)
    bot = Chatbot(
        respond=lambda messages, *, context=None: messages
        + (llm.Message("assistant", "answer"),),
        objective=initial,
        update_objective=lambda revision, *, context=None: (
            llm.Message("system", "changed"),
        ),
        memory=memory,
    )

    with pytest.raises(TypeError, match="persisted"):
        bot("question")

    assert bot.objective == initial
    assert bot.history == ()
    assert memory.records == ()


def test_chatbot_retrieves_memory_and_restores_state_after_restart(tmp_path):
    path = tmp_path / "memory.json"
    memory = JsonMemory(
        path,
        retrieve=most_recent,
        encode_value=encode_messages,
        decode_value=decode_messages,
    )
    seen = []

    def respond(messages, *, context=None):
        seen.append((messages, context))
        return messages + (llm.Message("assistant", "answer"),)

    def update(revision, *, context=None):
        return (llm.Message("system", f"objective-{len(revision.observation)}"),)

    bot = Chatbot(
        respond=respond,
        objective=(llm.Message("system", "initial"),),
        update_objective=update,
        memory=memory,
        memory_limit=2,
    )
    assert bot("first") == "answer"

    restarted_memory = JsonMemory(
        path,
        retrieve=most_recent,
        encode_value=encode_messages,
        decode_value=decode_messages,
    )
    restarted = Chatbot(
        respond=respond,
        objective=(llm.Message("system", "ignored when restored"),),
        update_objective=update,
        memory=restarted_memory,
        memory_limit=2,
    )

    assert [message.content for message in restarted.history] == ["first", "answer"]
    assert restarted.objective == (llm.Message("system", "objective-1"),)
    assert restarted("second") == "answer"
    messages, context = seen[-1]
    assert [message.content for message in messages] == ["first", "answer", "second"]
    assert "memory" in context
    assert [record.kind for record in restarted_memory.records] == [
        "observation",
        "response",
        "objective",
        "observation",
        "response",
        "objective",
    ]


def test_suffix_only_response_has_identical_live_and_restarted_history(tmp_path):
    path = tmp_path / "memory.json"
    memory = JsonMemory(
        path,
        retrieve=most_recent,
        encode_value=encode_messages,
        decode_value=decode_messages,
    )
    respond = lambda messages, *, context=None: (
        llm.Message("assistant", "answer"),
    )
    bot = Chatbot(respond=respond, memory=memory)

    bot("question")
    live_history = bot.history

    restarted = Chatbot(
        respond=respond,
        memory=JsonMemory(
            path,
            retrieve=most_recent,
            encode_value=encode_messages,
            decode_value=decode_messages,
        ),
    )
    assert restarted.history == live_history
    assert [message.content for message in live_history] == ["question", "answer"]


def test_concurrent_async_turns_have_a_serial_state_order():
    class AsyncResponder:
        def __call__(self, messages, *, context=None):
            raise AssertionError("sync path should not be used")

        async def acall(self, messages, *, context=None):
            await asyncio.sleep(0)
            return messages + (llm.Message("assistant", f"seen-{len(messages)}"),)

    bot = Chatbot(respond=AsyncResponder())

    async def run():
        return await asyncio.gather(bot.acall("one"), bot.acall("two"))

    answers = asyncio.run(run())

    assert answers == ["seen-1", "seen-3"]
    assert [message.content for message in bot.history] == [
        "one",
        "seen-1",
        "two",
        "seen-3",
    ]


def test_cancelled_async_lock_waiter_does_not_deadlock_later_turns():
    class ControlledResponder:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def acall(self, messages, *, context=None):
            if messages[-1].content == "first":
                self.started.set()
                await self.release.wait()
            return messages + (llm.Message("assistant", "ok"),)

    async def run():
        responder = ControlledResponder()
        bot = Chatbot(respond=responder)
        first = asyncio.create_task(bot.acall("first"))
        await responder.started.wait()
        cancelled = asyncio.create_task(bot.acall("cancelled"))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        responder.release.set()
        assert await first == "ok"
        assert await asyncio.wait_for(bot.acall("later"), timeout=0.2) == "ok"

    asyncio.run(run())


def test_async_turn_does_not_run_sync_model_callback_on_event_loop():
    def slow_response(messages, *, context=None):
        time.sleep(0.05)
        return messages + (llm.Message("assistant", "ok"),)

    async def run():
        bot = Chatbot(respond=slow_response)
        ticked = asyncio.Event()

        async def tick():
            await asyncio.sleep(0.005)
            ticked.set()

        turn = asyncio.create_task(bot.acall("hello"))
        ticker = asyncio.create_task(tick())
        await ticker
        assert ticked.is_set()
        assert not turn.done()
        assert await turn == "ok"

    asyncio.run(run())


def test_cancellation_waits_for_in_flight_commit_before_releasing_state_lock():
    class BlockingCommitChatbot(Chatbot):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.commit_started = Event()
            self.release_commit = Event()
            self.block_once = True

        def _commit(self, observation, response, objective):
            if self.block_once:
                self.block_once = False
                self.commit_started.set()
                self.release_commit.wait()
            super()._commit(observation, response, objective)

    async def run():
        respond = lambda messages, *, context=None: messages + (
            llm.Message("assistant", "answer"),
        )
        bot = BlockingCommitChatbot(respond=respond)
        first = asyncio.create_task(bot.acall("first"))
        await asyncio.to_thread(bot.commit_started.wait)
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()
        second = asyncio.create_task(bot.acall("second"))
        try:
            await asyncio.sleep(0.01)
            assert not second.done()
        finally:
            bot.release_commit.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert await asyncio.wait_for(second, timeout=0.2) == "answer"
        assert [message.content for message in bot.history] == [
            "first",
            "answer",
            "second",
            "answer",
        ]

    asyncio.run(run())
