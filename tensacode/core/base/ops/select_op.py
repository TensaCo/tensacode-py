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


@BaseEngine.register_op()
def Select(
    engine: BaseEngine,
    target,
    *inputs: list[Any],
    **kwargs: Any,
) -> Any:
    locator: Locator = engine.locate(
        target,
        *inputs,
        top_k=1,
        _new_scope=False,
        **kwargs,
    )
    return locator.get(target, current=target, create_missing=False)
