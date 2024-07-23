from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Callable
from functools import wraps
import re

import anyio
from pydantic import BaseModel

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.protocols.latent import LatentType


class BaseOp(BaseModel):
    op_name: str
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

    @classmethod
    def create_subclass(
        cls,
        op_name: str,
        latent_type: type[LatentType] | None = None,
        engine_type: type[BaseEngine] | None = None,
        match_score_fn: Callable[..., int] | None = None,
        run_fn: Callable[..., Any] | None = None,
        arun_fn: Callable[..., Promise[Any]] | None = None,
    ) -> type[Self]:
        class_name = re.sub(r"(?:^|_)([a-z])", lambda x: x.group(1).upper(), op_name)

        class OpSubclass(Op):
            __name__ = class_name
            op_name: ClassVar[str] = op_name
            latent_type: ClassVar[type[LatentType]] = latent_type or Op.latent_type
            engine_type: ClassVar[type[BaseEngine]] = engine_type or Op.engine_type

            def match_score_fn(self, *args, **kwargs) -> int:
                return match_score_fn(*args, **kwargs) if match_score_fn else 0

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
