from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op
from tensacode.internal.utils.misc import (
    inheritance_distance,
    LocatorStr,
    get_using_locator,
    set_using_locator,
)
from tensacode.internal.tcir.nodes import (
    CompositeValueNode,
    AtomicValueNode,
    CompositeValueNode,
)
from tensacode.internal.tcir.parse import parse_node


class BaseUpdateOp(Op):
    name: ClassVar[str] = "update"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine


@BaseEngine.register_op_class_for_all_class_instances
@BaseUpdateOp.create_subclass(name="update")
def Update(
    engine: BaseEngine,
    target: object,
    value: Any,
    *args,
    **kwargs: Any,
) -> Any:
    """Update an object"""
    locator = engine.locate(target, *args, **kwargs)
    if locator is None:
        return None
    return set_using_locator(target, locator, value)
