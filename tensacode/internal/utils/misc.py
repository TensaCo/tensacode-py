from functools import reduce
import inspect
from collections.abc import Mapping, MutableSequence, Sequence
from typing import Any, Hashable
from contextlib import contextmanager

from typing import Any, ClassVar, Sequence, Mapping, get_type_hints
from typing_extensions import Self
import pydantic
from typing import Any, get_args, get_origin

from tensacode.internal.utils.misc import inheritance_distance
from tensacode.internal.tcir.parse import parse_node

from dataclasses import dataclass

from typing import get_type_hints, get_origin, get_args, Annotated
from functools import wraps
from inspect import signature, Parameter
from dataclasses import dataclass
from typing import Any, Callable
import inspect

@dataclass
class Score:
    coefficient: float = 1.0  # Default coefficient is 1.0

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

    if sub == parent:
        return 0

    distance = 0
    current_class = sub

    while current_class != parent:
        distance += 1
        current_class = current_class.__base__

    return distance

def score_inheritance_distance(func: Callable):
    """
    Decorator that creates and attaches a scoring function to the function,
    based on its parameters annotated with 'Score'.

    The generated scoring function evaluates arguments based on their inheritance
    distance from the expected types, considering only parameters annotated with 'Score'.

    The score is computed as follows:
    - Add coefficient * (inheritance distance) for each parameter.
    - Add infinity (+inf) if the argument's type is not a subclass of the parameter's type.
    - Add infinity (+inf) if the argument fails to satisfy any constraints of the parameter annotation.
    - Do not divide by the number of arguments; return the total score.

    Args:
        func: The function to decorate.

    Returns:
        The original function with an attached '_score_fn' attribute.
    """
    sig = signature(func)
    type_hints = get_type_hints(func, include_extras=True)

    # Collect parameters to consider, along with their expected types and coefficients
    scoring_params = {}  # param_name -> (expected_type, coefficient)
    for param in sig.parameters.values():
        param_name = param.name
        if param_name in type_hints:
            annotated_type = type_hints[param_name]
            origin = get_origin(annotated_type)
            metadata = ()
            if origin is Annotated:
                args = get_args(annotated_type)
                expected_type = args[0]
                metadata = args[1:]
            else:
                expected_type = annotated_type

            # Check if parameter is annotated with 'Score'
            score_annotations = [meta for meta in metadata if isinstance(meta, Score)]
            if score_annotations:
                # Use the first Score annotation (in case of multiple)
                score_annotation = score_annotations[0]
                coefficient = score_annotation.coefficient
                scoring_params[param_name] = (expected_type, coefficient)

    # Define the scoring function
    def score_fn(*args, **kwargs):
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()

        total_score = 0

        for param_name, (expected_type, coefficient) in scoring_params.items():
            if param_name not in bound_args.arguments:
                # Missing argument
                return float('inf')

            arg_value = bound_args.arguments[param_name]
            arg_type = type(arg_value)

            # Handle Annotated types and constraints
            origin_type = get_origin(expected_type)
            constraints = []
            if origin_type is Annotated:
                typing_args = get_args(expected_type)
                expected_type = typing_args[0]
                constraints = typing_args[1:]
            elif hasattr(expected_type, '__metadata__'):
                # For Python < 3.9 compatibility
                typing_args = get_args(expected_type)
                expected_type = typing_args[0]
                constraints = expected_type.__metadata__

            # Check if arg_type is a subclass of expected_type
            dist = inheritance_distance(sub=arg_type, parent=expected_type)
            if dist is None:
                # Argument does not fit inside its parameter annotation
                return float('inf')
            else:
                # Add coefficient * inheritance distance
                total_score += coefficient * dist

            # Check constraints
            for constraint in constraints:
                if not constraint(arg_value):
                    # Argument fails to satisfy the constraint
                    return float('inf')

        return total_score

    # Attach the scoring function to the decorated function
    func._score_fn = score_fn

    return func

def get_type_arg(type_hint: Any, index: int = 0, default: Any = Any) -> Any:
    origin = get_origin(type_hint)
    if origin is None:
        return default
    args = get_args(type_hint)
    return args[index] if len(args) > index else default


def get_annotation(obj: object, attr: str, value: Any) -> type:
    # Special case for Pydantic models
    if isinstance(obj, pydantic.BaseModel):
        model_fields = obj.model_fields
        if attr in model_fields:
            return model_fields[attr].annotation

    try:
        hints = get_type_hints(obj.__class__)
        return hints.get(attr, type(value))
    except TypeError:
        return type(value)


@contextmanager
def conditional_ctx_manager(condition, ctx_manager):
    """
    Conditionally employ a context manager if the condition is true,
    otherwise use a no-op context manager.

    Args:
        condition (bool): The condition to check.
        ctx_manager (callable): A function that returns a context manager.

    Yields:
        The result of the context manager if condition is True, otherwise None.
    """
    if condition:
        with ctx_manager() as result:
            yield result
    else:
        yield None


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
        elif isinstance(item, set_using_locator):
            return tuple(sorted(flatten(i) for i in item))
        elif hasattr(item, "__dict__"):
            return flatten(item.__dict__)
        else:
            return item

    flattened = flatten(obj)
    return hash(flattened)


def greatest_common_type(types: tuple[type, ...]) -> type:
    if not types:
        return Any  # Return Any for empty sequences

    def find_common_base(t1: type, t2: type) -> type:
        if issubclass(t1, t2):
            return t2
        if issubclass(t2, t1):
            return t1
        for base in t1.__mro__:
            if issubclass(t2, base):
                return base
        return object  # Fallback to object if no common base found

    common_type = types[0]
    for t in types[1:]:
        common_type = find_common_base(common_type, t)

    return common_type


from functools import wraps, cache
from typing import Callable, Any, TYPE_CHECKING
import inspect
import threading


def call_with_appropriate_args(fn, *args, **kwargs):
    """
    Call a function with only the arguments it can accept.

    This function inspects the signature of the given function and calls it with
    only the arguments that match its parameters. It filters out any excess
    arguments that are not part of the function's signature.

    Args:
        fn (callable): The function to be called.
        *args: Positional arguments to be passed to the function.
        **kwargs: Keyword arguments to be passed to the function.

    Returns:
        The result of calling the function with the filtered arguments.

    Example:
        def example_func(a, b):
            return a + b

        result = call_with_appropriate_args(example_func, a=1, b=2, c=3)
        # result will be 3, and 'c' will be ignored
    """
    sig = inspect.signature(fn)
    bound_args = sig.bind_partial(*args, **kwargs)
    bound_args.apply_defaults()

    # Filter out excess arguments
    filtered_args = {
        k: v for k, v in bound_args.arguments.items() if k in sig.parameters
    }
    return fn(**filtered_args)



class ReadWriteProxyDict(Mapping):
    read_dict_getter: Callable[[], dict]
    write_dict_getter: Callable[[], dict]

    def __init__(
        self,
        read_dict_getter: Callable[[], dict],
        write_dict_getter: Callable[[], dict],
    ):
        self.read_dict_getter = read_dict_getter
        self.write_dict_getter = write_dict_getter

    def __getitem__(self, key):
        return self.read_dict_getter()[key]

    def __setitem__(self, key, value):
        self.write_dict_getter()[key] = value

    def __delitem__(self, key):
        d = self.write_dict_getter()
        del d[key]

    def __len__(self):
        return len(self.read_dict_getter())

    def __keys__(self):
        return self.read_dict_getter().keys()

    def __values__(self):
        return self.read_dict_getter().values()

    def __items__(self):
        return self.read_dict_getter().items()

    def __iter__(self):
        return iter(self.read_dict_getter())

    def __contains__(self, key):
        return key in self.read_dict_getter()

    def __repr__(self):
        return repr(self.read_dict_getter())


class ReadWriteProxyList(MutableSequence):
    read_list_getter: Callable[[], list]
    write_list_getter: Callable[[], list]

    def __init__(
        self,
        read_list_getter: Callable[[], list],
        write_list_getter: Callable[[], list],
    ):
        self.read_list_getter = read_list_getter
        self.write_list_getter = write_list_getter

    def __getitem__(self, index):
        return self.read_list_getter()[index]

    def __setitem__(self, index, value):
        self.write_list_getter()[index] = value

    def __delitem__(self, index):
        del self.write_list_getter()[index]

    def __len__(self):
        return len(self.read_list_getter())

    def insert(self, index, value):
        self.write_list_getter().insert(index, value)

    def __iter__(self):
        return iter(self.read_list_getter())

    def __repr__(self):
        return repr(self.read_list_getter())


import re


from pydantic import BaseModel, validator, Annotated
import re


class LocatorValidator(BaseModel):
    locator: str

    @validator("locator")
    def validate_locator(cls, v):
        pattern = r'^(\.(\w+)|\[(\d+)\]|\[\'([\w\s]+)\'\]|\["([\w\s]+)"\])+$'
        if not re.match(pattern, v):
            raise ValueError("Invalid locator format")
        return v


LocatorStr = Annotated[str, LocatorValidator]


def parse_locator(locator: LocatorStr):
    """Parse the locator string into a list of access steps."""
    pattern = r'\.(\w+)|\[(\d+)\]|\[\'([\w\s]+)\'\]|\["([\w\s]+)"\]'
    return [
        next(group for group in match.groups() if group is not None)
        for match in re.finditer(pattern, locator)
    ]


def get_using_locator(root: object, locator: LocatorStr) -> Any:
    """
    Get a value from an object using a string locator.

    Args:
        root (object): The root object to start from.
        locator (LocatorStr): A string representing the path to the desired value.

    Returns:
        The value at the specified location.

    Raises:
        AttributeError: If an attribute doesn't exist.
        IndexError: If an index is out of range.
        KeyError: If a dictionary key doesn't exist.

    Examples:
        >>> class Person:
        ...     def __init__(self, name, age):
        ...         self.name = name
        ...         self.age = age
        >>> data = {
        ...     'people': [
        ...         Person('Alice', 30),
        ...         Person('Bob', 25)
        ...     ],
        ...     'numbers': [1, 2, 3]
        ... }
        >>> get(data, '.people[0].name')
        'Alice'
        >>> get(data, '.people[1].age')
        25
        >>> get(data, '.numbers[2]')
        3
        >>> get(data, '.missing')
        Traceback (most recent call last):
        ...
        KeyError: "Invalid access: missing"
    """
    current = root
    for step in parse_locator(locator):
        try:
            if step.isdigit():
                current = current[int(step)]
            elif isinstance(current, dict):
                current = current[step]
            else:
                current = getattr(current, step)
        except (AttributeError, IndexError, KeyError) as e:
            raise type(e)(f"Invalid access: {step}") from e
    return current


def set_using_locator(root: object, locator: LocatorStr, value: object):
    """
    Set a value in an object using a string locator.

    Args:
        root (object): The root object to start from.
        locator (str): A string representing the path to the desired location.
        value (object): The value to set at the specified location.

    Raises:
        AttributeError: If an attribute doesn't exist.
        IndexError: If an index is out of range.
        KeyError: If a dictionary key doesn't exist.

    Examples:
        >>> class Person:
        ...     def __init__(self, name, age):
        ...         self.name = name
        ...         self.age = age
        >>> data = {
        ...     'people': [
        ...         Person('Alice', 30),
        ...         Person('Bob', 25)
        ...     ],
        ...     'numbers': [1, 2, 3]
        ... }
        >>> set(data, '.people[0].name', 'Alicia')
        >>> get(data, '.people[0].name')
        'Alicia'
        >>> set(data, '.numbers[1]', 10)
        >>> get(data, '.numbers[1]')
        10
        >>> set(data, '.new_key', 'new value')
        >>> get(data, '.new_key')
        'new value'
        >>> set(data, '.missing.key', 'value')
        Traceback (most recent call last):
        ...
        KeyError: "Invalid access: missing"
    """
    steps = parse_locator(locator)
    current = root
    for step in steps[:-1]:
        try:
            if step.isdigit():
                current = current[int(step)]
            elif isinstance(current, dict):
                current = current[step]
            else:
                current = getattr(current, step)
        except (AttributeError, IndexError, KeyError) as e:
            raise type(e)(f"Invalid access: {step}") from e

    last_step = steps[-1]
    try:
        if last_step.isdigit():
            current[int(last_step)] = value
        elif isinstance(current, dict):
            current[last_step] = value
        else:
            setattr(current, last_step, value)
    except (AttributeError, IndexError, KeyError) as e:
        raise type(e)(f"Invalid access: {last_step}") from e
