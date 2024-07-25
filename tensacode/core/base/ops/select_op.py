from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op
from tensacode.internal.utils.misc import inheritance_distance
from tensacode.internal.tcir.nodes import (
    CompositeValueNode,
    AtomicValueNode,
    CompositeValueNode,
)
from tensacode.internal.tcir.parse import parse_node
from tensacode.internal.utils.locator import Locator


class BaseSelectOp(Op):

    name: ClassVar[str] = "select"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine


@BaseEngine.register_op_class_for_all_class_instances
@BaseSelectOp.create_subclass(name="select")
def Select(
    engine: BaseEngine,
    target,
    *inputs: list[Any],
    top_k: int = 1,
    **kwargs: Any,
) -> Any | list[Any]:
    if top_k == 1:
        locator: Locator = engine.locate(
            target,
            *inputs,
            top_k=1,
            _new_scope=False,
            **kwargs,
        )
        return locator.get(target, current=target, create_missing=False)
    else:
        locators: list[Locator] = engine.locate(
            target,
            *inputs,
            top_k=top_k,
            _new_scope=False,
            **kwargs,
        )
        return [
            locator.get(target, current=target, create_missing=False)
            for locator in locators
        ]
