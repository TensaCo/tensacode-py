from typing import Any, ClassVar, Sequence, Mapping, List
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op
from tensacode.internal.utils.misc import (
    inheritance_distance,
    greatest_common_type,
    get_annotation,
)
from tensacode.internal.tcir.nodes import (
    CompositeValueNode,
    AtomicValueNode,
    SequenceNode,
    MappingNode,
)
from tensacode.internal.tcir.parse import parse_node
from tensacode.internal.utils.locator import Locator


class BaseSelectStepOp(Op):

    name: ClassVar[str] = "select_step"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine


@BaseEngine.register_op_class_for_all_class_instances
@BaseSelectStepOp.create_subclass(
    name="select_step",
    match_score_fn=(
        lambda engine, input_atom: 0
        - inheritance_distance(parse_node(input_atom), AtomicValueNode)
    ),
)
def SelectStepAtomic(
    engine: BaseEngine,
    items: Sequence[Any],
    /,
    query=None,
    top_k: int = 1,
    **kwargs: Any,
) -> Any:
    query = query or engine.latent
    return next((item for item in items if item == query), None)


@BaseEngine.register_op_class_for_all_class_instances
@BaseSelectStepOp.create_subclass(
    name="select_step",
    match_score_fn=(
        lambda engine, input_sequence: 0
        - inheritance_distance(parse_node(input_sequence), SequenceNode)
    ),
)
def SelectStepSequence(
    engine: BaseEngine,
    input_sequence: Sequence[Any],
    /,
    query=None,
    top_k: int = 1,
    **kwargs: Any,
) -> Any:
    query = query or engine.latent
    query_latent = engine.encode(query)
    query_latent = engine.transform(
        query_latent, "transform this into a select_step query"
    )

    # Compute embeddings separately
    embeddings = [
        engine.transform(item, "transform this into a select_step key")
        for item in input_sequence
    ]

    # Create input_items_std using pre-computed embeddings
    input_items_std = list(zip(embeddings, input_sequence))

    return _select_step_items(
        engine,
        input_items_std,
        query=query_latent,
        key_fn=lambda x: x[0],
        value_fn=lambda x: x[1],
        top_k=top_k,
        **kwargs,
    )


@BaseEngine.register_op_class_for_all_class_instances
@BaseSelectStepOp.create_subclass(
    name="select_step",
    match_score_fn=(
        lambda engine, input_mapping: 0
        - inheritance_distance(parse_node(input_mapping), MappingNode)
    ),
)
def SelectStepMapping(
    engine: BaseEngine,
    input_mapping: Mapping[Any, Any],
    /,
    query=None,
    top_k: int = 1,
    **kwargs: Any,
) -> Any:
    query = query or engine.latent
    query_latent = engine.encode(query)
    query_latent = engine.transform(
        query_latent, "transform this into a select_step query"
    )

    # Compute embeddings separately
    keys = input_mapping
    values = [v for k, v in keys]
    embeddings = [
        engine.transform(key, "transform this into a select_step key") for key in keys
    ]

    # Create input_items_std using pre-computed embeddings
    input_items_std = list(zip(embeddings, values))

    return _select_step_items(
        engine,
        input_items_std,
        query=query_latent,
        key_fn=lambda x: x[0],
        value_fn=lambda x: x[1],
        top_k=top_k,
        **kwargs,
    )


@BaseEngine.register_op_class_for_all_class_instances
@BaseSelectStepOp.create_subclass(
    name="select_step",
    match_score_fn=(
        lambda engine, input_composite: 0
        - inheritance_distance(parse_node(input_composite), CompositeValueNode)
    ),
)
def SelectStepComposite(
    engine: BaseEngine,
    input_obj: object,
    /,
    query=None,
    top_k: int = 1,
    **kwargs: Any,
) -> Any:
    query = query or engine.latent
    query_latent = engine.encode(query)
    query_latent = engine.transform(
        query_latent, "transform this into a select_step query"
    )

    # Prepare keys and compute embeddings separately
    keys = [
        (k, getattr(input_obj, k), get_annotation(input_obj, k, getattr(input_obj, k)))
        for k in dir(input_obj)
    ]
    values = [getattr(input_obj, k) for k in dir(input_obj)]
    embeddings = [
        engine.transform(key, "transform this into a select_step key") for key in keys
    ]

    # Create input_items_std using pre-computed embeddings
    input_items_std = list(zip(embeddings, values))

    return _select_step_items(
        engine,
        input_items_std,
        query=query_latent,
        key_fn=lambda x: x[0],
        value_fn=lambda x: x[1],
        top_k=top_k,
        **kwargs,
    )


def _select_step_items(
    engine,
    items,
    query,
    key_fn=lambda x: x,
    value_fn=lambda x: x,
    top_k: int = 1,
    **kwargs,
) -> List[Any]:
    query = query or engine.latent
    query_latent = engine.encode(query)
    similarities = [
        engine.similarity(key_fn(item), query_latent, **kwargs) for item in items
    ]

    # Sort items by similarity and get the top k
    sorted_indices = sorted(
        range(len(similarities)), key=lambda i: similarities[i], reverse=True
    )
    top_k = min(top_k, len(items))
    top_indices = sorted_indices[:top_k]

    # Return the top k matches
    return [value_fn(items[i]) for i in top_indices]
