from abc import ABC

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional

from pydantic import BaseModel
from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.protocols.latent import LatentType
from tensacode.internal.utils.rergistry import Registry, HasRegistry


class BaseOp(BaseModel, ABC):
    op_name: ClassVar[str]
    object_type: ClassVar[type[object]]
    latent_type: ClassVar[LatentType]
    engine_type: ClassVar[BaseEngine]

    prompt: str

    def execute(
        self,
        *args,
        engine: BaseEngine,
        context: dict = None,
        config: dict = None,
        **kwargs,
    ):
        with engine.scope(config_overrides=config, context_overrides=context):
            return self._execute(*args, engine=engine, **kwargs)

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

    @abstractmethod
    @classmethod
    def from_engine(cls, engine: BaseEngine) -> Self:
        # DO NOT STORE THE ENGINE IN THE CLASS. This is a bad practice because it will bloat the serialization.
        raise NotImplementedError(
            f"Subclasses must implement {cls.__name__}.from_engine(engine: BaseEngine) -> Self"
        )
