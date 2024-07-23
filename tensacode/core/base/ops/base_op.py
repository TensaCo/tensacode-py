from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.protocols.latent import (
    LatentType,
    are_latent_subtypes,
    latent_type_subtype_distance,
)
from tensacode.internal.utils.misc import inheritance_distance


class BaseOp(BaseModel, ABC):
    op_name: ClassVar[str]
    latent_type: ClassVar[LatentType]
    engine_type: ClassVar[BaseEngine]

    def execute(
        self,
        *args,
        engine: BaseEngine,
        **kwargs,
    ):
        return engine.trace_execution(
            fn=self._execute,
            fn_name_override=self.op_name,
            args=args,
            kwargs=kwargs,
        )

    def __call__(
        self,
        *args,
        engine: BaseEngine,
        **kwargs,
    ):
        return self.execute(
            *args,
            engine=engine,
            **kwargs,
        )

    @abstractmethod
    def _execute(self, *args, engine: BaseEngine, **kwargs):
        pass

    def handler_match_score(
        self,
        *args,
        engine_type: BaseEngine,
        latent_type: LatentType,
        operator_type: type | None = None,
        operator_name: str | None = None,
        **kwargs,
    ) -> int:
        """
        Calculate a score for how well this op matches the given operator and engine.

        Higher is better. Use -inf for no match.
        """
        if not issubclass(engine_type, self.engine_type):
            return -float("inf")

        if not are_latent_subtypes(sub=latent_type, parent=self.latent_type):
            return -float("inf")

        if operator_name is not None:
            if operator_name != self.op_name:
                return -float("inf")
        elif operator_type is not None:
            if not issubclass(operator_type, type(self)):
                return -float("inf")
        else:
            raise ValueError("Either operator_name or operator_type must be provided")

        dist = 0

        # no distance calculations in the base class. it's simple down here

        return dist
