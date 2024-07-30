from typing import Any, ClassVar, List, Mapping, Sequence, get_type_hints
from inspect import signature
from typing_extensions import Self

from tensacode.internal.tcir.nodes import CompositeValueNode, SequenceNode, MappingNode
from tensacode.internal.tcir.parse import parse_node
from tensacode.internal.utils.misc import (
    inheritance_distance,
    score_node_inheritance_distance,
    get_type_arg,
)
from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op
from tensacode.internal.utils.tc import loop_until_done


@BaseEngine.register_op(score_fn=score_node_inheritance_distance(type_=AtomicValueNode))
def decode_atomic(
    engine: BaseEngine,
    /,
    type_: type[Any] = Any,
    latent: LatentType = None,
    prompt: Optional[Encoded[str]] = None,
    **kwargs: Any,
) -> Any:
    raise NotImplementedError("Subclass must implement atomic decoding")


@BaseEngine.register_op(score_fn=score_node_inheritance_distance(type_=SequenceNode))
def decode_list(
    engine: BaseEngine,
    /,
    type_: type[list[Any]] = list,
    latent: LatentType = None,
    count: int | None = None,
    max_items: int | None = None,
    min_items: int | None = None,
    prompt: Optional[Encoded[str]] = None,
    **kwargs: Any,
) -> list[Any]:
    elem_type = get_type_arg(type_, 0, Any)
    sequence = list()
    for _ in loop_until_done(
        count=count,
        min=min_items,
        max=max_items,
        engine=engine,
        continue_prompt="Continue adding items?",
    ):
        sequence.append(
            engine.decode(latent=latent, type=elem_type, prompt=prompt, **kwargs)
        )
    return sequence


@BaseEngine.register_op(score_fn=score_node_inheritance_distance(type_=MappingNode))
def decode_mapping(
    engine: BaseEngine,
    /,
    type_: type[Mapping[Any, Any]] = Mapping[str, Any],
    latent: LatentType = None,
    count: int | None = None,
    max_items: int | None = None,
    min_items: int | None = None,
    prompt: Optional[Encoded[str]] = None,
    **kwargs: Any,
) -> Mapping[Any, Any]:
    key_type = get_type_arg(type_, 0, Any)
    value_type = get_type_arg(type_, 1, Any)
    mapping = {}
    for _ in loop_until_done(
        count=count,
        min=min_items,
        max=max_items,
        engine=engine,
        continue_prompt="Continue adding items?",
    ):
        key = engine.decode(latent=latent, type=key_type, prompt=prompt, **kwargs)
        value = engine.decode(latent=latent, type=value_type, prompt=prompt, **kwargs)
        mapping[key] = value
    return mapping


@BaseEngine.register_op(
    score_fn=score_node_inheritance_distance(type_=CompositeValueNode)
)
def decode_composite(
    engine: BaseEngine,
    /,
    type_: type[object] = object,
    latent: LatentType = None,
    prompt: Optional[Encoded[str]] = None,
    **kwargs: Any,
) -> object:

    # Get type hints for the class
    type_hints = get_type_hints(type_)

    # Get __init__ parameters
    init_params = (
        signature(type_.__init__).parameters if hasattr(type_, "__init__") else {}
    )

    # Prepare arguments for instantiation
    init_args = {}
    for param_name, _ in list(init_params.items())[1:]:  # Skip the first parameter
        param_type = type_hints.get(param_name, Any)
        init_args[param_name] = engine.decode(
            latent=latent, type=param_type, prompt=prompt, **kwargs
        )

    # Create an instance of the composite type
    instance = type_(**init_args)

    # Handle any remaining annotated attributes not in __init__
    for attr, attr_type in type_hints.items():
        if attr not in init_args and not attr.startswith("_"):
            value = engine.decode(
                latent=latent, type=attr_type, prompt=prompt, **kwargs
            )
            try:
                setattr(instance, attr, value)
            except AttributeError:
                pass  # the object might not support setting attributes

    return instance
