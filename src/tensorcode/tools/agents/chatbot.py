"""A transactional stateful chatbot composed from public message operations."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
import inspect
from threading import Lock
from typing import Any

from ...ops import llm
from .memory import JsonMemory, MemoryRecord


@dataclass(frozen=True)
class ObjectiveRevision:
    """Required operands for a caller-configured objective transform."""

    current: Any
    observation: tuple[Any, ...]


class Chatbot:
    """Compose message operations while committing each turn atomically.

    ``update_objective`` and ``memory`` are opt-in. The tool supplies no hidden
    intent detector, objective policy, or retrieval semantics.
    """

    def __init__(
        self,
        *,
        model=None,
        encode=None,
        encode_image=None,
        respond=None,
        decode=None,
        objective=None,
        update_objective=None,
        memory: JsonMemory | None = None,
        memory_limit: int = 5,
        state_id: str = "default",
        restore: bool = True,
    ):
        if (model is None) == (respond is None):
            raise ValueError("Supply exactly one of model or respond")
        if not isinstance(memory_limit, int) or memory_limit < 0:
            raise ValueError("memory_limit must be a non-negative integer")
        if not isinstance(state_id, str) or not state_id:
            raise ValueError("state_id must be a non-empty string")
        self.encode = encode if encode is not None else llm.TextEncoder()
        self.encode_image = encode_image
        self.respond = respond if respond is not None else llm.Transform(model)
        self.decode = decode if decode is not None else llm.TextDecoder()
        self.update_objective = update_objective
        self.memory = memory
        self.memory_limit = memory_limit
        self.state_id = state_id
        self.objective = deepcopy(objective)
        self._history: tuple[Any, ...] = ()
        self._turn = 0
        self._lock = Lock()
        if restore and memory is not None:
            self._restore(memory.records)

    @property
    def history(self):
        return self._history

    def __call__(self, value, *, images=(), context=None):
        with self._lock:
            return self._turn_sync(value, images=images, context=context)

    async def acall(self, value, *, images=(), context=None):
        await self._acquire_state_lock()
        try:
            return await self._turn_async(value, images=images, context=context)
        finally:
            self._lock.release()

    async def _acquire_state_lock(self) -> None:
        while not self._lock.acquire(blocking=False):
            # No background waiter survives cancellation and acquires the lock
            # after this task has stopped.
            await asyncio.sleep(0)

    def _encode_observation(self, value: Any, images: Sequence[Any]) -> tuple[Any, ...]:
        observation = tuple(self.encode(value))
        if images:
            image_encoder = self.encode_image
            if image_encoder is None:
                image_encoder = llm.ImageEncoder()
                self.encode_image = image_encoder
            for image in images:
                observation += tuple(image_encoder(image))
        return observation

    def _turn_sync(self, value, *, images, context):
        observation = self._encode_observation(value, images)
        objective = self._updated_objective_sync(observation, context)
        memories = self._retrieve(observation)
        response_context = self._response_context(context, objective, memories)
        messages = self._history + observation
        response = self.respond(messages, context=response_context)
        if inspect.isawaitable(response):
            raise TypeError("sync Chatbot call received an awaitable; use acall")
        response = self._normalize_response(messages, tuple(response))
        answer = self.decode(response)
        if inspect.isawaitable(answer):
            raise TypeError("sync Chatbot decoder returned an awaitable; use acall")
        self._commit(observation, response, objective)
        return answer

    async def _turn_async(self, value, *, images, context):
        observation = await asyncio.to_thread(self._encode_observation, value, images)
        objective = await self._updated_objective_async(observation, context)
        memories = await asyncio.to_thread(self._retrieve, observation)
        response_context = self._response_context(context, objective, memories)
        messages = self._history + observation
        response = self._normalize_response(
            messages,
            tuple(
                await self._invoke_async(
                    self.respond, messages, context=response_context
                )
            ),
        )
        answer = await self._invoke_async(self.decode, response, context=None)
        commit = asyncio.create_task(
            asyncio.to_thread(self._commit, observation, response, objective)
        )
        cancelled = False
        while not commit.done():
            try:
                await asyncio.shield(commit)
            except asyncio.CancelledError:
                # Repeated cancellation must not release the state lock while
                # the worker thread can still mutate committed state.
                cancelled = True
        commit.result()
        if cancelled:
            raise asyncio.CancelledError
        return answer

    def _updated_objective_sync(self, observation, context):
        if self.update_objective is None:
            return deepcopy(self.objective)
        result = self.update_objective(
            ObjectiveRevision(deepcopy(self.objective), observation), context=context
        )
        if inspect.isawaitable(result):
            raise TypeError("sync objective update returned an awaitable; use acall")
        return deepcopy(result)

    async def _updated_objective_async(self, observation, context):
        if self.update_objective is None:
            return deepcopy(self.objective)
        result = await self._invoke_async(
            self.update_objective,
            ObjectiveRevision(deepcopy(self.objective), observation),
            context=context,
        )
        return deepcopy(result)

    @staticmethod
    async def _invoke_async(operation, value, *, context):
        acall = getattr(operation, "acall", None)
        if callable(acall):
            result = acall(value, context=context)
        else:
            result = await asyncio.to_thread(operation, value, context=context)
        if inspect.isawaitable(result):
            return await result
        return result

    def _retrieve(self, observation) -> tuple[MemoryRecord, ...]:
        if self.memory is None or self.memory_limit == 0:
            return ()
        return self.memory.search(
            observation,
            limit=self.memory_limit,
            kinds=("observation", "response"),
        )

    @staticmethod
    def _response_context(context, objective, memories):
        combined = dict(context or {})
        if objective is not None:
            if "objective" in combined:
                raise ValueError("context key 'objective' is managed by Chatbot")
            combined["objective"] = deepcopy(objective)
        if memories:
            if "memory" in combined:
                raise ValueError("context key 'memory' is managed by Chatbot")
            combined["memory"] = tuple(
                message for record in memories for message in record.value
            )
        return combined or None

    @staticmethod
    def _normalize_response(messages, response):
        if response[: len(messages)] == messages:
            return response
        if response and all(
            getattr(message, "role", None) in ("assistant", "tool")
            for message in response
        ):
            return messages + response
        raise ValueError(
            "respond must return the input transcript plus a response, "
            "or an assistant/tool response suffix"
        )

    def _commit(self, observation, response, objective) -> None:
        messages_before = self._history + observation
        generated = (
            response[len(messages_before) :]
            if response[: len(messages_before)] == messages_before
            else response
        )
        turn = self._turn + 1
        if self.memory is not None:
            metadata = {
                "tensorcode_chatbot": True,
                "state_id": self.state_id,
                "turn": turn,
            }
            with self.memory.transaction() as transaction:
                observation_record = transaction.append(
                    observation, kind="observation", metadata=metadata
                )
                response_metadata = dict(metadata)
                response_metadata["observation_source_id"] = observation_record.source_id
                transaction.append(generated, kind="response", metadata=response_metadata)
                if self.update_objective is not None:
                    transaction.append(objective, kind="objective", metadata=metadata)
        self._history = response
        self.objective = deepcopy(objective)
        self._turn = turn

    def _restore(self, records: Sequence[MemoryRecord]) -> None:
        state_records = tuple(
            record
            for record in records
            if record.metadata.get("tensorcode_chatbot") is True
            and record.metadata.get("state_id") == self.state_id
        )
        history: list[Any] = []
        latest_objective = self.objective
        turn = 0
        for record in state_records:
            record_turn = record.metadata.get("turn")
            if isinstance(record_turn, int):
                turn = max(turn, record_turn)
            if record.kind in ("observation", "response"):
                history.extend(record.value)
            elif record.kind == "objective":
                latest_objective = record.value
        self._history = tuple(history)
        self.objective = latest_objective
        self._turn = turn
