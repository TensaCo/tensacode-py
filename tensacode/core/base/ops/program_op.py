from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op


class BaseProgramOp(Op):
    name: ClassVar[str] = "program"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine


@BaseEngine.register_op_class_for_all_class_instances
@BaseProgramOp.create_subclass(name="program")
def Program(
    engine: BaseEngine,
    *inputs: list[Any],
    **kwargs: Any,
) -> Any:
    """Existing docstring moved here"""
    raise NotImplementedError("Subclasses must implement this method")
