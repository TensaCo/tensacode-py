from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Callable, Optional, Any, Coroutine
from functools import wraps
import re

import anyio
from pydantic import BaseModel, Field
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.internal.utils.misc import call_with_appropriate_args


class OpExample(BaseModel):
    input: str
    output: str


class BaseOp(BaseModel):
    name: str
    description: str
    examples: list[OpExample] = Field(default_factory=list)
    latent_type: type[LatentType] = LatentType
    engine_type: type[BaseEngine] = BaseEngine

    @abstractmethod
    def match_score_fn(self, engine: BaseEngine, *args, **kwargs) -> int: ...

    @abstractmethod
    def run(self, engine: BaseEngine, *args, **kwargs): ...

    @abstractmethod
    async def arun(self, engine: BaseEngine, *args, **kwargs): ...


class Op(BaseOp):
    def match_score_fn(self, engine: BaseEngine, *args, **kwargs) -> int:
        return 0

    def run(self, engine: BaseEngine, *args, _trace=True, _new_scope=True, **kwargs):
        match _trace, _new_scope:
            case (True, True):
                return engine.trace_execution(
                    fn=self._run,
                    args=(engine, *args),
                    kwargs=kwargs,
                    new_scope=_new_scope,
                )
            case (True, False):
                return engine.trace_execution(
                    fn=self._run,
                    args=(engine, *args),
                    kwargs=kwargs,
                )
            case (False, True):
                with engine.scope():
                    return self._run(engine, *args, **kwargs)
            case (False, False):
                return self._run(engine, *args, **kwargs)

    async def arun(self, engine: BaseEngine, *args, **kwargs):
        engine.trace_execution(self.name, self._arun, engine, *args, **kwargs)

    @abstractmethod
    def _run(self, engine: BaseEngine, *args, **kwargs): ...

    async def _arun(self, engine: BaseEngine, *args, **kwargs):
        return await anyio.to_thread.run_sync(self._run, engine, *args, **kwargs)

    def __call__(self, engine: BaseEngine, *args, **kwargs):
        return self.run(engine, *args, **kwargs)

    @classmethod
    def create_subclass(
        cls,
        name: str,
        latent_type: type[LatentType] | None = None,
        engine_type: type[BaseEngine] | None = None,
        match_score_fn: Callable[..., int] | None = None,
        run_fn: Callable[..., Any] | None = None,
        arun_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None,
    ) -> type[Self]:
        class_name = re.sub(r"(?:^|_)([a-z])", lambda x: x.group(1).upper(), name)

        _latent_type = latent_type
        _engine_type = engine_type

        class OpSubclass(Op):
            __name__ = class_name
            name: ClassVar[str] = name
            latent_type: ClassVar[type[LatentType]] = _latent_type or Op.latent_type
            engine_type: ClassVar[type[BaseEngine]] = _engine_type or Op.engine_type

            def match_score_fn(self, *args, **kwargs) -> int:
                if match_score_fn:
                    return call_with_appropriate_args(match_score_fn, *args, **kwargs)
                return 0

            def _run(self, engine: BaseEngine, *args, **kwargs):
                if run_fn:
                    return run_fn(engine, *args, **kwargs)
                else:
                    raise NotImplementedError("run method not implemented")

            async def _arun(self, engine: BaseEngine, *args, **kwargs):
                if arun_fn:
                    # developer has freedom to await or not inside their arun_fn
                    return arun_fn(engine, *args, **kwargs)
                elif run_fn:
                    return await anyio.to_thread.run_sync(
                        self.run, engine, *args, **kwargs
                    )
                else:
                    raise NotImplementedError("arun method not implemented")

        return OpSubclass
