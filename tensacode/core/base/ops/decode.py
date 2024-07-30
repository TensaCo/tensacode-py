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
    """
    Decode a latent representation into an atomic value of the specified type.

    This operation uses the engine to convert a latent representation into an atomic value.

    Args:
        engine (BaseEngine): The engine used for decoding.
        type_ (type[Any], optional): The target type for decoding. Defaults to Any.
        latent (LatentType, optional): The latent representation to decode. Defaults to None.
        prompt (Optional[Encoded[str]], optional): A prompt to guide the decoding process. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        Any: The decoded atomic value.

    Raises:
        NotImplementedError: This method must be implemented by a subclass.

    Examples:
        >>> latent = engine.encode("forty-two")
        >>> result = decode_atomic(engine, type_=int, latent=latent)
        >>> print(result)
        42

        >>> latent = engine.encode("true")
        >>> result = decode_atomic(engine, type_=bool, latent=latent)
        >>> print(result)
        True
    """
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
    """
    Decode a latent representation into a list of items of the specified type.

    This operation uses the engine to convert a latent representation into a list of items.

    Args:
        engine (BaseEngine): The engine used for decoding.
        type_ (type[list[Any]], optional): The target list type for decoding. Defaults to list.
        latent (LatentType, optional): The latent representation to decode. Defaults to None.
        count (int | None, optional): The exact number of items to decode. Defaults to None.
        max_items (int | None, optional): The maximum number of items to decode. Defaults to None.
        min_items (int | None, optional): The minimum number of items to decode. Defaults to None.
        prompt (Optional[Encoded[str]], optional): A prompt to guide the decoding process. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        list[Any]: The decoded list of items.

    Examples:
        >>> latent = engine.encode("A list of three prime numbers")
        >>> result = decode_list(engine, type_=list[int], latent=latent, count=3)
        >>> print(result)
        [2, 3, 5]

        >>> latent = engine.encode("A list of fruits")
        >>> result = decode_list(engine, type_=list[str], latent=latent, min_items=2, max_items=4)
        >>> print(result)
        ['apple', 'banana', 'orange']
    """
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
    """
    Decode a latent representation into a mapping of key-value pairs of the specified types.

    This operation uses the engine to convert a latent representation into a mapping.

    Args:
        engine (BaseEngine): The engine used for decoding.
        type_ (type[Mapping[Any, Any]], optional): The target mapping type for decoding. Defaults to Mapping[str, Any].
        latent (LatentType, optional): The latent representation to decode. Defaults to None.
        count (int | None, optional): The exact number of key-value pairs to decode. Defaults to None.
        max_items (int | None, optional): The maximum number of key-value pairs to decode. Defaults to None.
        min_items (int | None, optional): The minimum number of key-value pairs to decode. Defaults to None.
        prompt (Optional[Encoded[str]], optional): A prompt to guide the decoding process. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        Mapping[Any, Any]: The decoded mapping of key-value pairs.

    Examples:
        >>> latent = engine.encode("A mapping of three countries to their capitals")
        >>> result = decode_mapping(engine, type_=Mapping[str, str], latent=latent, count=3)
        >>> print(result)
        {'France': 'Paris', 'Japan': 'Tokyo', 'Brazil': 'Brasília'}

        >>> latent = engine.encode("A mapping of items to their prices")
        >>> result = decode_mapping(engine, type_=Mapping[str, float], latent=latent, min_items=2, max_items=4)
        >>> print(result)
        {'apple': 0.5, 'banana': 0.75, 'orange': 0.6}
    """
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
    """
    Decode a latent representation into a composite object of the specified type.

    This operation uses the engine to convert a latent representation into a composite object,
    instantiating the object and populating its attributes based on type hints and __init__ parameters.

    Args:
        engine (BaseEngine): The engine used for decoding.
        type_ (type[object], optional): The target composite type for decoding. Defaults to object.
        latent (LatentType, optional): The latent representation to decode. Defaults to None.
        prompt (Optional[Encoded[str]], optional): A prompt to guide the decoding process. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        object: The decoded composite object.

    Examples:
        >>> class Person:
        ...     def __init__(self, name: str, age: int):
        ...         self.name = name
        ...         self.age = age
        >>> latent = engine.encode("A 30-year-old named Alice")
        >>> result = decode_composite(engine, type_=Person, latent=latent)
        >>> print(f"{result.name}, {result.age}")
        Alice, 30

        >>> class Car:
        ...     def __init__(self, make: str, model: str, year: int):
        ...         self.make = make
        ...         self.model = model
        ...         self.year = year
        >>> latent = engine.encode("A 2022 Tesla Model 3")
        >>> result = decode_composite(engine, type_=Car, latent=latent)
        >>> print(f"{result.year} {result.make} {result.model}")
        2022 Tesla Model 3
    """

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
