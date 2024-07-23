from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Callable, Optional, Any
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
    def match_score_fn(self, *args, **kwargs) -> int: ...

    @abstractmethod
    def run(self, *args, **kwargs): ...

    @abstractmethod
    async def arun(self, *args, **kwargs): ...


class Op(BaseOp):
    def match_score_fn(self, *args, **kwargs) -> int:
        return 0

    @abstractmethod
    def run(self, *args, **kwargs): ...

    async def arun(self, *args, **kwargs):
        return await anyio.to_thread.run_sync(self.run, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @classmethod
    def create_subclass(
        cls,
        name: str,
        latent_type: type[LatentType] | None = None,
        engine_type: type[BaseEngine] | None = None,
        match_score_fn: Callable[..., int] | None = None,
        run_fn: Callable[..., Any] | None = None,
        arun_fn: Callable[..., Promise[Any]] | None = None,
    ) -> type[Self]:
        class_name = re.sub(r"(?:^|_)([a-z])", lambda x: x.group(1).upper(), name)

        class OpSubclass(Op):
            __name__ = class_name
            name: ClassVar[str] = name
            latent_type: ClassVar[type[LatentType]] = latent_type or Op.latent_type
            engine_type: ClassVar[type[BaseEngine]] = engine_type or Op.engine_type

            def match_score_fn(self, *args, **kwargs) -> int:
                if match_score_fn:
                    return call_with_appropriate_args(match_score_fn, *args, **kwargs)
                return 0

            def run(self, *args, **kwargs):
                if run_fn:
                    return run_fn(*args, **kwargs)
                else:
                    raise NotImplementedError("run method not implemented")

            async def arun(self, *args, **kwargs):
                if arun_fn:
                    # developer has freedom to await or not inside their arun_fn
                    return arun_fn(*args, **kwargs)
                elif run_fn:
                    return await anyio.to_thread.run_sync(self.run, *args, **kwargs)
                else:
                    raise NotImplementedError("arun method not implemented")

        return OpSubclass

    # this is bad practice. register from the engine
    # @classmethod
    # def register(
    #     cls,
    #     name: str,
    #     latent_type: type[LatentType] | None = None,
    #     engine_type: type[BaseEngine] | None = None,
    #     match_score_fn: Callable[..., int] | None = None,
    #     register_with_engine: Optional[type[BaseEngine] | BaseEngine] = None,
    # ) -> type[Self]:
    #     def decorator(run_fn: Callable[..., Any]):
    #         return cls.create_subclass(
    #             name,
    #             latent_type,
    #             engine_type,
    #             match_score_fn,
    #             run_fn,
    #             register_with_engine,
    #         )

    #     return decorator
