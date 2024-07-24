from typing import Any, ClassVar, Sequence, Mapping
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


class BaseSearchOp(Op):

    name: ClassVar[str] = "search"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine


@BaseEngine.register_op_class_for_all_class_instances
@BaseSearchOp.create_subclass(
    name="search",
    match_score_fn=(
        lambda engine, input_atom: 0
        - inheritance_distance(parse_node(input_atom), AtomicValueNode)
    ),
)
def SearchAtomic(
    engine: BaseEngine,
    items: Sequence[Any],
    /,
    query=None,
    **kwargs: Any,
) -> Any:
    query = query or engine.latent
    return next((item for item in items if item == query), None)


@BaseEngine.register_op_class_for_all_class_instances
@BaseSearchOp.create_subclass(
    name="search",
    match_score_fn=(
        lambda engine, input_sequence: 0
        - inheritance_distance(parse_node(input_sequence), SequenceNode)
    ),
)
def SearchSequence(
    engine: BaseEngine,
    input_sequence: Sequence[Any],
    /,
    query=None,
    **kwargs: Any,
) -> Any:
    query = query or engine.latent
    query_latent = engine.encode(query)
    query_latent = engine.transform(query_latent, "transform this into a search query")

    # Compute embeddings separately
    embeddings = [
        engine.transform(item, "transform this into a search key")
        for item in input_sequence
    ]

    # Create input_items_std using pre-computed embeddings
    input_items_std = list(zip(embeddings, input_sequence))

    return _search_items(
        engine,
        input_items_std,
        query=query_latent,
        key_fn=lambda x: x[0],
        value_fn=lambda x: x[1],
        **kwargs,
    )


@BaseEngine.register_op_class_for_all_class_instances
@BaseSearchOp.create_subclass(
    name="search",
    match_score_fn=(
        lambda engine, input_mapping: 0
        - inheritance_distance(parse_node(input_mapping), MappingNode)
    ),
)
def SearchMapping(
    engine: BaseEngine,
    input_mapping: Mapping[Any, Any],
    /,
    query=None,
    **kwargs: Any,
) -> Any:
    query = query or engine.latent
    query_latent = engine.encode(query)
    query_latent = engine.transform(query_latent, "transform this into a search query")

    # Compute embeddings separately
    keys = input_mapping
    values = [v for k, v in keys]
    embeddings = [
        engine.transform(key, "transform this into a search key") for key in keys
    ]

    # Create input_items_std using pre-computed embeddings
    input_items_std = list(zip(embeddings, values))

    return _search_items(
        engine,
        input_items_std,
        query=query_latent,
        key_fn=lambda x: x[0],
        value_fn=lambda x: x[1],
        **kwargs,
    )


@BaseEngine.register_op_class_for_all_class_instances
@BaseSearchOp.create_subclass(
    name="search",
    match_score_fn=(
        lambda engine, input_composite: 0
        - inheritance_distance(parse_node(input_composite), CompositeValueNode)
    ),
)
def SearchComposite(
    engine: BaseEngine,
    input_obj: object,
    /,
    query=None,
    **kwargs: Any,
) -> Any:
    query = query or engine.latent
    query_latent = engine.encode(query)
    query_latent = engine.transform(query_latent, "transform this into a search query")

    # Prepare keys and compute embeddings separately
    keys = [
        (k, getattr(input_obj, k), get_annotation(input_obj, k, getattr(input_obj, k)))
        for k in dir(input_obj)
    ]
    values = [getattr(input_obj, k) for k in dir(input_obj)]
    embeddings = [
        engine.transform(key, "transform this into a search key") for key in keys
    ]

    # Create input_items_std using pre-computed embeddings
    input_items_std = list(zip(embeddings, values))

    return _search_items(
        engine,
        input_items_std,
        query=query_latent,
        key_fn=lambda x: x[0],
        value_fn=lambda x: x[1],
        **kwargs,
    )


def _search_items(
    engine,
    items,
    query,
    key_fn=lambda x: x,
    value_fn=lambda x: x,
    **kwargs,
) -> Any:
    query = query or engine.latent
    query_latent = engine.encode(query)
    similarities = [
        engine.similarity(key_fn(item), query_latent, **kwargs) for item in items
    ]
    best_match_index = similarities.index(max(similarities))
    best_match = items[best_match_index]
    return value_fn(best_match)
