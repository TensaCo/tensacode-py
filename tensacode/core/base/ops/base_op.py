from abc import ABC

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.protocols.latent import LatentType


class BaseOp(ABC):
    op_name: ClassVar[str]
    latent_type: ClassVar[LatentType]

    log: Log = Field(default_factory=Log)
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
