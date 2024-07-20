from abc import ABC

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.protocols.latent import LatentType
from tensacode.internal.utils.rergistry import Registry, HasRegistry


class OpIdentifier(BaseModel):
    op_name: str
    latent_type: LatentType


class BaseOp(HasRegistry, ABC):
    op_name: ClassVar[str]
    latent_type: ClassVar[LatentType]

    _registry: ClassVar[Registry["BaseOp"]] = Registry()
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

    ALL_OPS: ClassVar[Registry[BaseOp]] = Registry()

    @classmethod
    def register_op(cls, op: BaseOp):
        cls.ALL_OPS.register(op.op_name, op)
