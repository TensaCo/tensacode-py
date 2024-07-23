from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.protocols.latent import LatentType


class BaseOp(BaseModel, ABC):
    op_name: ClassVar[str]
    latent_type: ClassVar[LatentType]
    engine_type: ClassVar[BaseEngine]

    def execute(
        self,
        *args,
        engine: BaseEngine,
        context: dict = None,
        config: dict = None,
        **kwargs,
    ):
        return engine.trace_execution(
            self._execute,
            args,
            kwargs,
            fn_name_override=self.op_name,
            config_overrides=config,
            context_overrides=context,
        )

    def __call__(
        self,
        *args,
        engine: BaseEngine,
        context: dict = None,
        config: dict = None,
        **kwargs,
    ):
        return self.execute(
            *args,
            engine=engine,
            context=context,
            config=config,
            **kwargs,
        )

    @abstractmethod
    def _execute(self, *args, engine: BaseEngine, **kwargs):
        pass

    def handler_match_score(
        self,
        *args,
        latent_type: LatentType,
        operator_type: type,
        operator_name: str,
        engine_type: BaseEngine,
        **kwargs,
    ) -> int:
        # TODO: implement this
        return 0
