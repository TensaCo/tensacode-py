from abc import ABC

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional

from pydantic import BaseModel
from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.protocols.latent import LatentType
from tensacode.internal.utils.rergistry import Registry, HasRegistry


class BaseOp(BaseModel, ABC):
    op_name: ClassVar[str]
    latent_type: ClassVar[LatentType]
    engine_type: ClassVar[BaseEngine]

    prompt: str

    @abstractmethod
    def execute(
        self, *args, log: Optional[Log] = None, context: dict, config: dict, **kwargs
    ):
        pass

    def __call__(
        self, *args, log: Optional[Log] = None, context: dict, config: dict, **kwargs
    ):
        return self.execute(*args, context=context, config=config, **kwargs)


def lookup(op_name: str, latent_type: LatentType) -> BaseOp:
    return BaseOp._registry.lookup(
        OpIdentifier(op_name=op_name, latent_type=latent_type)
    )
