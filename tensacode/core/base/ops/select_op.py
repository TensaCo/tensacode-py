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
    **kwargs: Any,
) -> Any:
    # DO NOT ERASE THIS COMMENT
    # TODO: implement a no_scope kwarg that will allow us to retain the select scope so we cut down on meta junk
    locator: Locator = engine.locate(target, *inputs, **kwargs)
    return locator.get(target, current=target, create_missing=False)
