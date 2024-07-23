from functools import reduce
import inspect
from collections.abc import Mapping, Sequence
from typing import Any, Hashable


def advanced_equality_check(*objects):
    """
    Perform an advanced equality check on multiple objects.

    This function compares objects by their attributes or items, ignoring None values.
    It supports Pydantic BaseModels, dictionaries, lists, and atomic values.

    None is a wildcard that matches any value. If one of the objects has a None value for an attribute or item,
    the equality check will ignore that attribute/item for that object, but still consider it on the other objects.

    Args:
        *objects: Variable number of objects to compare.

    Returns:
        bool: True if all non-None attributes/items are equal across objects, False otherwise.

    Examples:
        >>> from pydantic import BaseModel
        >>> class TestModel(BaseModel):
        ...     a: int
        ...     b: str
        >>> obj1 = TestModel(a=1, b='test')
        >>> obj2 = TestModel(a=1, b='test')
        >>> obj3 = TestModel(a=1, b=None)
        >>> advanced_equality_check(obj1, obj2, obj3)
        True
        >>> advanced_equality_check({'a': 1, 'b': 'test'}, {'a': 1, 'b': None})
        True
        >>> advanced_equality_check([1, 2, None], [1, 2, 3], [1, 2, 4])
        True
        >>> advanced_equality_check(1, 1, 1)
        True
        >>> advanced_equality_check(1, '1', 1.0)
        False
    """
    if not objects:
        return True

    def get_items(obj):
        if hasattr(obj, "__fields__"):  # Pydantic BaseModel
            return {field: getattr(obj, field) for field in obj.__fields__}
        elif isinstance(obj, dict):
            return obj
        elif isinstance(obj, list):
            return {i: v for i, v in enumerate(obj)}
        else:  # Atomic value
            return {"value": obj}

    first = get_items(objects[0])

    for key in first:
        values = [get_items(obj).get(key) for obj in objects]
        non_none_values = [v for v in values if v is not None]

        # If all values are None, skip this attribute/item
        if not non_none_values:
            continue

        # If non-None values are not all equal, return False
        if not all(v == non_none_values[0] for v in non_none_values):
            return False

    # All attributes/items are equal (ignoring None values)
    return True


def inheritance_distance(sub, parent) -> int | None:
    """
    Calculate the inheritance distance between a subclass and a parent class.

    Args:
        sub: The subclass to check.
        parent: The potential parent class.

    Returns:
        int: The number of inheritance levels between sub and parent.
        None: If sub is not a subclass of parent.
    """
    if not issubclass(sub, parent):
        return None

    distance = 0
    current_class = sub

    while current_class != parent:
        distance += 1
        current_class = current_class.__base__

    return distance


def stack_dicts(*dicts: dict) -> dict:
    """
    Stack multiple dictionaries, with later dictionaries overriding earlier ones.

    Args:
        *dicts: Variable number of dictionaries to stack.

    Returns:
        A new dictionary with all input dictionaries stacked.
    """
    return reduce(lambda acc, d: {**acc, **d}, dicts, {})


def generate_callstack(skip_frames=1):
    return [
        {
            "fn_name": frame.function,
            "params": {
                **dict(
                    zip(
                        inspect.getargvalues(frame[0]).args,
                        inspect.getargvalues(frame[0]).locals.values(),
                    )
                ),
                **{
                    k: v
                    for k, v in inspect.getargvalues(frame[0]).locals.items()
                    if k not in inspect.getargvalues(frame[0]).args
                },
            },
        }
        for frame in inspect.stack()[skip_frames:]
    ]


def hash_mutable(obj: Any) -> int:
    """
    Hash a potentially mutable object by first flattening it to immutable structures.

    Args:
        obj: The object to be hashed.

    Returns:
        int: A hash value for the object.
    """

    def flatten(item: Any) -> Hashable:
        if isinstance(item, Mapping):
            return tuple(sorted((k, flatten(v)) for k, v in item.items()))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            return tuple(flatten(i) for i in item)
        elif isinstance(item, set):
            return tuple(sorted(flatten(i) for i in item))
        elif hasattr(item, "__dict__"):
            return flatten(item.__dict__)
        else:
            return item

    flattened = flatten(obj)
    return hash(flattened)
