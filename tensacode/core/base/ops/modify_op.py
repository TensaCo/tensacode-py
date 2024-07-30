from typing import Any, ClassVar, Optional, Mapping, Sequence
import inspect
from itertools import count
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import BaseOp
from tensacode.core.base.ops.select_op import SelectOp
from tensacode.core.base.ops.decode_op import DecodeOp
from tensacode.core.base.ops.decide_op import DecideOp
from tensacode.internal.protocols.encoded import Encoded
from tensacode.internal.tcir.nodes import (
    SequenceNode,
    MappingNode,
    AtomicValueNode,
    Node,
    CompositeValueNode,
)
from tensacode.internal.tcir.parse import parse_node
from tensacode.internal.utils.misc import inheritance_distance, greatest_common_type
from tensacode.core.base.ops.base_op import Op
from tensacode.internal.utils.locator import Locator
from tensacode.internal.utils.tc import loop_until_done


@BaseEngine.register_op()
def modify(
    engine: BaseEngine,
    input: Any,
    /,
    max_steps: int = 10,
    **kwargs,
) -> Any:

    current_value = input
    for step in loop_until_done(
        max_steps,
        engine=engine,
        continue_prompt="Continue modifying object?",
        stop_prompt="Done modifying object?",
    ):
        with engine.scope(step=step, max_steps=max_steps):
            engine.info(current_value=current_value)
            attr_loc: Locator = engine.locate("Pick the next value to modify")
            old_value = attr_loc.get(current_value, current_value, create_missing=True)
            new_value = engine.decode(
                type=type(old_value),
                prompt="Modify atomic value",
                current_value=old_value,
                **kwargs,
            )
            attr_loc.set(current_value, old_value, new_value)
            engine.info(new_value=new_value)
            current_value = new_value
    return current_value
