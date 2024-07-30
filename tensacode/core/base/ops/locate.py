from typing import Any, ClassVar, Sequence, Mapping, List, Tuple, Optional
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType, Encoded
from tensacode.core.base.ops.base_op import Op
from tensacode.internal.utils.misc import (
    inheritance_distance,
    get_annotation,
    score_node_inheritance_distance,
)
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


@BaseEngine.register_op(score_fn=score_node_inheritance_distance(input=AtomicValueNode))
def locate_atomic(
    engine: BaseEngine,
    input: Any,
    /,
    prompt: Optional[Encoded[str]] = None,
    max_depth: int = -1,
    **kwargs: Any,
) -> Locator:
    """
    Locate a specific part of an atomic value.

    This operation is used for atomic values and returns a TerminalLocator.

    Args:
        engine (BaseEngine): The engine used for locating.
        input (Any): The atomic input to locate within.
        prompt (Optional[Encoded[str]], optional): A prompt to guide the locating process. Defaults to None.
        max_depth (int, optional): The maximum depth to search. Defaults to -1 (no limit).
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        Locator: A TerminalLocator for the atomic value.

    Examples:
        >>> value = 42
        >>> locator = locate_atomic(engine, value)
        >>> locator.get(value, value)
        42
    """
    return TerminalLocator()


@BaseEngine.register_op(
    score_fn=score_node_inheritance_distance(input_sequence=SequenceNode)
)
def locate_sequence(
    engine: BaseEngine,
    input_sequence: Sequence[Any],
    /,
    max_depth: int = -1,
    prompt: Optional[Encoded[str]] = None,
    **kwargs: Any,
) -> Locator:
    """
    Locate a specific element or subset within a sequence.

    This operation uses the engine to find a specific part of the input sequence.

    Args:
        engine (BaseEngine): The engine used for locating.
        input_sequence (Sequence[Any]): The input sequence to locate within.
        max_depth (int, optional): The maximum depth to search. Defaults to -1 (no limit).
        prompt (Optional[Encoded[str]], optional): A prompt to guide the locating process. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        Locator: A Locator for the located element or subset in the sequence.

    Examples:
        >>> sequence = [1, 2, 3, 4, 5]
        >>> locator = locate_sequence(engine, sequence, prompt="Locate the middle element")
        >>> locator.get(sequence, sequence)
        3

        >>> sequence = ["apple", "banana", "orange"]
        >>> locator = locate_sequence(engine, sequence, prompt="Locate the fruit that starts with 'b'")
        >>> locator.get(sequence, sequence)
        'banana'
    """
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

    selected_item = _locate(
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


@BaseEngine.register_op(
    score_fn=score_node_inheritance_distance(input_mapping=MappingNode)
)
def locate_mapping(
    engine: BaseEngine,
    input_mapping: Mapping[Any, Any],
    /,
    max_depth: int = -1,
    prompt: Optional[Encoded[str]] = None,
    **kwargs: Any,
) -> Locator:
    """
    Locate a specific key-value pair or subset within a mapping.

    This operation uses the engine to find a specific part of the input mapping.

    Args:
        engine (BaseEngine): The engine used for locating.
        input_mapping (Mapping[Any, Any]): The input mapping to locate within.
        max_depth (int, optional): The maximum depth to search. Defaults to -1 (no limit).
        prompt (Optional[Encoded[str]], optional): A prompt to guide the locating process. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        Locator: A Locator for the located key-value pair or subset in the mapping.

    Examples:
        >>> mapping = {"name": "Alice", "age": 30, "city": "New York"}
        >>> locator = locate_mapping(engine, mapping, prompt="Locate the key-value pair with the name")
        >>> locator.get(mapping, mapping)
        'Alice'

        >>> mapping = {"apple": 1, "banana": 2, "orange": 3}
        >>> locator = locate_mapping(engine, mapping, prompt="Locate the key-value pair with the value 2")
        >>> locator.get(mapping, mapping)
        2
    """
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

    selected_item = _locate(
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


@BaseEngine.register_op(
    score_fn=score_node_inheritance_distance(input_obj=CompositeValueNode)
)
def locate_composite(
    engine: BaseEngine,
    input_obj: object,
    /,
    max_depth: int = -1,
    prompt: Optional[Encoded[str]] = None,
    **kwargs: Any,
) -> Locator:
    """
    Locate a specific attribute or subset within a composite object.

    This operation uses the engine to find a specific part of the input composite object.

    Args:
        engine (BaseEngine): The engine used for locating.
        input_obj (object): The input composite object to locate within.
        max_depth (int, optional): The maximum depth to search. Defaults to -1 (no limit).
        prompt (Optional[Encoded[str]], optional): A prompt to guide the locating process. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        Locator: A Locator for the located attribute or subset in the composite object.

    Examples:
        >>> class Person:
        ...     def __init__(self, name: str, age: int):
        ...         self.name = name
        ...         self.age = age
        >>> person = Person("Alice", 30)
        >>> locator = locate_composite(engine, person, prompt="Locate the name attribute")
        >>> locator.get(person, person)
        'Alice'

        >>> class Car:
        ...     def __init__(self, color: str, speed: int):
        ...         self.color = color
        ...         self.speed = speed
        >>> car = Car("red", 200)
        >>> locator = locate_composite(engine, car, prompt="Locate the speed attribute")
        >>> locator.get(car, car)
        200
    """
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

    selected_item = _locate(
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


def _locate(
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
