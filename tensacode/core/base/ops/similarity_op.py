from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op


class BaseSimilarityOp(Op):
    """Docstring for BaseSimilarityOp"""

    name: ClassVar[str] = "similarity"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine


@BaseEngine.register_op_class_for_all_class_instances
@BaseSimilarityOp.create_subclass(name="similarity")
def Similarity(
    engine: BaseEngine,
    *inputs: list[Any],
    **kwargs: Any,
) -> float:
    """Existing docstring moved here"""
    if not inputs:
        return True  # Empty list, all elements are equal (vacuously true)

    first_input = inputs[0]
    for input_item in inputs[1:]:
        if input_item != first_input:
            return False

    return True
