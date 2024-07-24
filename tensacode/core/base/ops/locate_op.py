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


class BaseLocateOp(Op):
    name: ClassVar[str] = "locate"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine


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
    root: Any,
    /,
    *args,
    **kwargs: Any,
) -> LocatorStr:
    """Locate an object"""
    return TerminalLocator()


@BaseEngine.register_op_class_for_all_class_instances
@BaseLocateOp.create_subclass(
    name="locate",
    match_score_fn=(
        lambda engine, input_composite: 0
        - inheritance_distance(parse_node(input_composite), SequenceValueNode)
    ),
)
def LocateSequence(
    engine: BaseEngine,
    root: Sequence[Any],
    /,
    *args,
    **kwargs: Any,
) -> LocatorStr:
    """Locate an object in a sequence"""
    if not engine.decide("Select deeper inside the sequence?"):
        return None

    indices = list(range(len(root)))
    # selected_index = engine.select(
    #     indices,
    #     prompt="Select an index from the sequence",
    #     object=root,
    # )
    child = root[selected_index]
    next_locator = engine.locate(child, *args, **kwargs)
    return f'[{selected_index}]{next_locator or ""}'


@BaseEngine.register_op_class_for_all_class_instances
@BaseLocateOp.create_subclass(
    name="locate",
    match_score_fn=(
        lambda engine, input_composite: 0
        - inheritance_distance(parse_node(input_composite), MappingValueNode)
    ),
)
def LocateMapping(
    engine: BaseEngine,
    root: Mapping[str, Any],
    /,
    *args,
    **kwargs: Any,
) -> LocatorStr:
    """Locate an object"""
    if not engine.decide("Select deeper inside the object?"):
        return None

    keys = list(root.keys())
    # selected_key = engine.select(
    #     keys,
    #     prompt="Select a key from the object",
    #     object=root,
    # )
    child = root[selected_key]
    next_key = engine.locate(child, *args, **kwargs)
    return f"['{selected_key}']{next_key or ''}"


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
    root: object,
    /,
    *args,
    **kwargs: Any,
) -> LocatorStr:
    """Locate an object"""

    if not engine.decide("Select deeper inside the object?"):
        return None

    fields = dir(root)
    # selected_field = engine.select(
    #     fields,
    #     prompt="Select a field from the object",
    #     object=root,
    # )
    child = getattr(root, selected_field)
    next_field = engine.locate(child, *args, **kwargs)
    return f"{root}.{selected_field}{next_field or ''}"
