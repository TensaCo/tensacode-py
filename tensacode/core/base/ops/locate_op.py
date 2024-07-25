from typing import Any, ClassVar, Sequence, Mapping
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op
from tensacode.internal.utils.misc import inheritance_distance
from tensacode.internal.utils.locator import (
    CompositeLocator,
    TerminalLocator,
    DotAccessStep,
    IndexAccessStep,
    Locator,
)
from tensacode.internal.tcir.nodes import (
    CompositeValueNode,
    SequenceNode,
    MappingNode,
    AtomicValueNode,
    CompositeValueNode,
)
from tensacode.internal.tcir.parse import parse_node


class BaseLocateOp(Op):
    name: ClassVar[str] = "locate"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine


@BaseEngine.register_op_class_for_all_class_instances
@BaseLocateOp.create_subclass(
    name="locate",
    match_score_fn=(lambda top_k: float("inf") if top_k != 1 else -float("inf")),
)
def LocateMultiple(
    engine: BaseEngine,
    input: Any,
    /,
    max_depth: int = -1,
    top_k: int = 1,
    **kwargs: Any,
) -> Locator:
    """Locate an object"""
    return [
        engine.locate(input, max_depth=max_depth, top_k=1, **kwargs)
        for _ in range(top_k)
    ]


@BaseEngine.register_op_class_for_all_class_instances
@BaseLocateOp.create_subclass(
    name="locate",
    match_score_fn=(
        lambda engine, input_composite: 0
        - inheritance_distance(parse_node(input_composite), AtomicValueNode)
    ),
)
def LocateAtomic(
    engine: BaseEngine,
    input: Any,
    /,
    max_depth: int = -1,
    top_k: int = 1,
    **kwargs: Any,
) -> Locator:
    """Locate an object"""
    if max_depth == 0:
        return TerminalLocator()
    return TerminalLocator()


@BaseEngine.register_op_class_for_all_class_instances
@BaseLocateOp.create_subclass(
    name="locate",
    match_score_fn=(
        lambda engine, input_composite: 0
        - inheritance_distance(parse_node(input_composite), SequenceNode)
    ),
)
def LocateSequence(
    engine: BaseEngine,
    input: Sequence[Any],
    /,
    max_depth: int = -1,
    top_k: int = 1,
    **kwargs: Any,
) -> Locator:
    """Locate an object in a sequence"""
    if max_depth == 0:
        return TerminalLocator()

    if not engine.decide("Select deeper inside the sequence?"):
        return TerminalLocator()

    if not isinstance(input, list):
        input = list(input)

    selected_item = engine.select_step(input, **kwargs)
    selected_item_index = input.index(selected_item)
    selected_item_locator = IndexAccessStep(index=selected_item_index)
    next_locator = engine.locate(selected_item, max_depth=max_depth - 1, **kwargs)
    return CompositeLocator(steps=[selected_item_locator, next_locator])


@BaseEngine.register_op_class_for_all_class_instances
@BaseLocateOp.create_subclass(
    name="locate",
    match_score_fn=(
        lambda engine, input_composite: 0
        - inheritance_distance(parse_node(input_composite), MappingNode)
    ),
)
def LocateMapping(
    engine: BaseEngine,
    input: Mapping[str, Any],
    /,
    max_depth: int = -1,
    top_k: int = 1,
    **kwargs: Any,
) -> Locator:
    """Locate an object"""
    if max_depth == 0:
        return TerminalLocator()

    if not engine.decide("Select deeper inside the object?"):
        return TerminalLocator()

    if not isinstance(input, dict):
        input = dict(input)

    selected_item = engine.search(
        input,
        prompt="Select a key from the object",
    )
    selected_item_key = next(
        key for key, value in input.items() if value == selected_item
    )
    selected_item_locator = DotAccessStep(key=selected_item_key)
    next_locator = engine.locate(selected_item, max_depth=max_depth - 1, **kwargs)
    return CompositeLocator(steps=[selected_item_locator, next_locator])


@BaseEngine.register_op_class_for_all_class_instances
@BaseLocateOp.create_subclass(
    name="locate",
    match_score_fn=(
        lambda engine, input_composite: 0
        - inheritance_distance(parse_node(input_composite), CompositeValueNode)
    ),
)
def LocateComposite(
    engine: BaseEngine,
    input: object,
    /,
    max_depth: int = -1,
    top_k: int = 1,
    **kwargs: Any,
) -> Locator:
    """Locate an object"""
    if max_depth == 0:
        return TerminalLocator()

    if not engine.decide("Select deeper inside the object?"):
        return TerminalLocator()

    selected_attr = engine.search(
        input,
        prompt="Select an attribute from the object",
    )
    selected_attr_name = next(
        attr for attr in dir(input) if getattr(input, attr) == selected_attr
    )
    selected_attr_locator = DotAccessStep(key=selected_attr_name)
    next_locator = engine.locate(selected_attr, max_depth=max_depth - 1, **kwargs)
    return CompositeLocator(steps=[selected_attr_locator, next_locator])
