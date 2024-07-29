from typing import Any, ClassVar, Sequence, Mapping, List, Tuple
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op
from tensacode.internal.utils.misc import inheritance_distance, get_annotation
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
        lambda engine, input_atom: 0
        - inheritance_distance(parse_node(input_atom), AtomicValueNode)
    ),
)
def LocateAtomic(
    engine: BaseEngine,
    input: Any,
    /,
    max_depth: int = -1,
    **kwargs: Any,
) -> Locator:
    return TerminalLocator()


@BaseEngine.register_op_class_for_all_class_instances
@BaseLocateOp.create_subclass(
    name="locate",
    match_score_fn=(
        lambda engine, input_sequence: 0
        - inheritance_distance(parse_node(input_sequence), SequenceNode)
    ),
)
def LocateSequence(
    engine: BaseEngine,
    input_sequence: Sequence[Any],
    /,
    max_depth: int = -1,
    **kwargs: Any,
) -> Locator:
    if max_depth == 0 or not engine.decide("Select deeper inside the sequence?"):
        return TerminalLocator()

    query = engine.latent
    query_latent = engine.encode(query)
    query_latent = engine.transform(query_latent, "transform this into a locate query")

    embeddings = [
        engine.transform(item, "transform this into a locate key")
        for item in input_sequence
    ]

    input_items_std = list(zip(embeddings, input_sequence, range(len(input_sequence))))

    selected_item = _locate_items(
        engine,
        input_items_std,
        query=query_latent,
        key_fn=lambda x: x[0],
        value_fn=lambda x: x[1],
        index_fn=lambda x: x[2],
        **kwargs,
    )

    if not selected_item:
        return TerminalLocator()

    selected_item, selected_index = selected_item
    selected_item_locator = IndexAccessStep(index=selected_index)
    next_locator = engine.locate(selected_item, max_depth=max_depth - 1, **kwargs)
    return CompositeLocator(steps=[selected_item_locator, next_locator])


@BaseEngine.register_op_class_for_all_class_instances
@BaseLocateOp.create_subclass(
    name="locate",
    match_score_fn=(
        lambda engine, input_mapping: 0
        - inheritance_distance(parse_node(input_mapping), MappingNode)
    ),
)
def LocateMapping(
    engine: BaseEngine,
    input_mapping: Mapping[Any, Any],
    /,
    max_depth: int = -1,
    **kwargs: Any,
) -> Locator:
    if max_depth == 0 or not engine.decide("Select deeper inside the object?"):
        return TerminalLocator()

    query = engine.latent
    query_latent = engine.encode(query)
    query_latent = engine.transform(query_latent, "transform this into a locate query")

    keys = list(input_mapping.keys())
    values = list(input_mapping.values())
    embeddings = [
        engine.transform(key, "transform this into a locate key") for key in keys
    ]

    input_items_std = list(zip(embeddings, values, keys))

    selected_item = _locate_items(
        engine,
        input_items_std,
        query=query_latent,
        key_fn=lambda x: x[0],
        value_fn=lambda x: x[1],
        index_fn=lambda x: x[2],
        **kwargs,
    )

    if not selected_item:
        return TerminalLocator()

    selected_item, selected_key = selected_item
    selected_item_locator = DotAccessStep(key=selected_key)
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
    input_obj: object,
    /,
    max_depth: int = -1,
    **kwargs: Any,
) -> Locator:
    if max_depth == 0 or not engine.decide("Select deeper inside the object?"):
        return TerminalLocator()

    query = engine.latent
    query_latent = engine.encode(query)
    query_latent = engine.transform(query_latent, "transform this into a locate query")

    keys = [
        (k, getattr(input_obj, k), get_annotation(input_obj, k, getattr(input_obj, k)))
        for k in dir(input_obj)
    ]
    values = [getattr(input_obj, k) for k in dir(input_obj)]
    embeddings = [
        engine.transform(key, "transform this into a locate key") for key in keys
    ]

    input_items_std = list(zip(embeddings, values, [k[0] for k in keys]))

    selected_item = _locate_items(
        engine,
        input_items_std,
        query=query_latent,
        key_fn=lambda x: x[0],
        value_fn=lambda x: x[1],
        index_fn=lambda x: x[2],
        **kwargs,
    )

    if not selected_item:
        return TerminalLocator()

    selected_item, selected_attr_name = selected_item
    selected_attr_locator = DotAccessStep(key=selected_attr_name)
    next_locator = engine.locate(selected_item, max_depth=max_depth - 1, **kwargs)
    return CompositeLocator(steps=[selected_attr_locator, next_locator])


def _locate_items(
    engine,
    items,
    query,
    key_fn=lambda x: x,
    value_fn=lambda x: x,
    index_fn=lambda x: x,
    **kwargs,
) -> Tuple[Any, Any]:
    query = query or engine.latent
    query_latent = engine.encode(query)
    similarities = [
        engine.similarity(key_fn(item), query_latent, **kwargs) for item in items
    ]

    # Find the index of the item with the highest similarity
    max_index = max(range(len(similarities)), key=lambda i: similarities[i])

    # Return the best match with its index/key
    return (value_fn(items[max_index]), index_fn(items[max_index]))
