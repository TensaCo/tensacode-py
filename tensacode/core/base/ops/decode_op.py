from typing import Any, ClassVar, List, Mapping, Sequence, get_type_hints
from inspect import signature
from typing_extensions import Self

from tensacode.internal.tcir.nodes import CompositeValueNode, SequenceNode, MappingNode
from tensacode.internal.tcir.parse import parse_node
from tensacode.internal.utils.misc import inheritance_distance
from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op
from tensacode.internal.utils.misc import get_type_arg
from tensacode.internal.utils.tc import loop_until_done


class BaseDecodeOp(Op):
    name: ClassVar[str] = "decode"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine


@BaseEngine.register_op_class_for_all_class_instances
@BaseDecodeOp.create_subclass(name="decode")
def Decode(
    engine: BaseEngine,
    *inputs: list[Any],
    **kwargs: Any,
) -> Any:
    """Decode operation"""
    raise NotImplementedError("Subclass must implement this method")


@BaseEngine.register_op_class_for_all_class_instances
@BaseDecodeOp.create_subclass(
    name="decode",
    match_score_fn=(
        lambda engine, input_atom: 0
        - inheritance_distance(parse_node(input_atom), AtomicValueNode)
    ),
)
def DecodeAtomic(
    engine: BaseEngine,
    /,
    type_: type[Any] = Any,
    latent: LatentType = None,
    **kwargs: Any,
) -> Any:
    raise NotImplementedError("Subclass must implement atomic decoding")


@BaseEngine.register_op_class_for_all_class_instances
@BaseDecodeOp.create_subclass(
    name="decode",
    match_score_fn=(
        lambda engine, input_sequence: 0
        - inheritance_distance(parse_node(input_sequence), SequenceNode)
    ),
)
def DecodeList(
    engine: BaseEngine,
    /,
    type_: type[list[Any]] = list,
    latent: LatentType = None,
    count: int | None = None,
    max_items: int | None = None,
    min_items: int | None = None,
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
        sequence.append(engine.decode(latent=latent, type=elem_type, **kwargs))
    return sequence


@BaseEngine.register_op_class_for_all_class_instances
@BaseDecodeOp.create_subclass(
    name="decode",
    match_score_fn=(
        lambda engine, input_mapping: 0
        - inheritance_distance(parse_node(input_mapping), MappingNode)
    ),
)
def DecodeMapping(
    engine: BaseEngine,
    /,
    type_: type[Mapping[Any, Any]] = Mapping[str, Any],
    latent: LatentType = None,
    count: int | None = None,
    max_items: int | None = None,
    min_items: int | None = None,
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
        key = engine.decode(latent=latent, type=key_type, **kwargs)
        value = engine.decode(latent=latent, type=value_type, **kwargs)
        mapping[key] = value
    return mapping


@BaseEngine.register_op_class_for_all_class_instances
@BaseDecodeOp.create_subclass(
    name="decode",
    match_score_fn=(
        lambda engine, input_composite: 0
        - inheritance_distance(parse_node(input_composite), CompositeValueNode)
    ),
)
def DecodeComposite(
    engine: BaseEngine,
    /,
    type_: type[object] = object,
    latent: LatentType = None,
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
        init_args[param_name] = engine.decode(latent=latent, type=param_type, **kwargs)

    # Create an instance of the composite type
    instance = type_(**init_args)

    # Handle any remaining annotated attributes not in __init__
    for attr, attr_type in type_hints.items():
        if attr not in init_args and not attr.startswith("_"):
            value = engine.decode(latent=latent, type=attr_type, **kwargs)
            try:
                setattr(instance, attr, value)
            except AttributeError:
                pass  # the object might not support setting attributes

    return instance
