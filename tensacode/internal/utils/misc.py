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


def polymorphic(fn):
    """
    A decorator for creating polymorphic functions.

    This decorator allows you to define a base function and register multiple
    implementations for different conditions. When the decorated function is called,
    it will execute the appropriate implementation based on the registered conditions.

    The decorator adds a 'register' method to the wrapped function, which can be used
    to register new implementations with their corresponding condition functions.

    Args:
        fn (callable): The base function to be decorated.

    Returns:
        callable: A wrapper function that handles the polymorphic behavior.

    Example:
        @polymorphic
        def process(obj):
            return "Default processing"

        @process.register(lambda obj: isinstance(obj, int))
        def process_int(obj):
            return f"Processing integer: {obj}"

        @process.register(lambda obj: isinstance(obj, str))
        def process_str(obj):
            return f"Processing string: {obj}"

        @process.register(lambda obj: isinstance(obj, list))
        def process_list(obj):
            return f"Processing list: {obj}"

        print(process(10))  # Output: "Processing integer: 10"
        print(process("hello"))  # Output: "Processing string: hello"
        print(process([1, 2, 3]))  # Output: "Default processing"
    """
    Override = tuple[int, Callable, Callable]
    overrides: list[Override] = []

    class PolymorphicDecorator:
        # just for typing
        overrides: list[Override]

        def register(self, condition_fn: Callable, /, priority: int = 0):
            def decorator(override_fn):
                overrides.append(Override(priority, condition_fn, override_fn))
                return override_fn

            return decorator

        __call__: Callable[..., Any]

    @cache
    def overrides_sorted_by_priority(overrides):
        # highest priority overrides are ordered at the beginning of the list.
        return sorted(overrides, key=lambda x: x[0], reverse=True)

    @wraps(fn, updated=["__annotations__"])
    def wrapper(cls, *args, **kwargs):
        for _, condition_fn, override_fn in overrides_sorted_by_priority(overrides):
            if call_with_appropriate_args(condition_fn, cls, *args, **kwargs):
                return override_fn(cls, *args, **kwargs)
        return fn(cls, *args, **kwargs)

    setattr(wrapper, "overrides", overrides)

    def register(condition_fn, /, priority: int = 0):
        def decorator(override_fn):
            overrides.append(Override(priority, condition_fn, override_fn))
            return override_fn

        return decorator

    setattr(wrapper, "register", register)

    typed_wrapper: PolymorphicDecorator = wrapper
    return typed_wrapper


def cached_with_key(key_func=lambda input: input):
    """
    A decorator that caches the result of a method based on a key function.

    This decorator is thread-safe and caches the result of the decorated method.
    The cache is invalidated when the key returned by key_func changes.

    Args:
        key_func (callable): A function that returns a cache key for the instance.

    Returns:
        callable: A decorator function.

    Example:
        >>> import time
        >>> class Example:
        ...     def __init__(self):
        ...         self.value = 0
        ...
        ...     @cached_with_key(lambda self: self.value)
        ...     def expensive_operation(self):
        ...         time.sleep(0.1)  # Simulate expensive operation
        ...         return f"Result: {self.value}"
        ...
        >>> obj = Example()
        >>> start = time.time()
        >>> print(obj.expensive_operation)
        Result: 0
        >>> print(f"Time taken: {time.time() - start:.2f} seconds")
        Time taken: 0.10 seconds
        >>> start = time.time()
        >>> print(obj.expensive_operation)
        Result: 0
        >>> print(f"Time taken: {time.time() - start:.2f} seconds")
        Time taken: 0.00 seconds
        >>> obj.value = 1
        >>> start = time.time()
        >>> print(obj.expensive_operation)
        Result: 1
        >>> print(f"Time taken: {time.time() - start:.2f} seconds")
        Time taken: 0.10 seconds
    """

    def decorator(func):
        cache_name = f"_cached_{func.__name__}"
        key_name = f"_cached_key_{func.__name__}"
        lock_name = f"_lock_{func.__name__}"

        @wraps(func)
        def wrapper(self):
            if not hasattr(self, lock_name):
                setattr(self, lock_name, threading.Lock())

            with getattr(self, lock_name):
                if not hasattr(self, cache_name) or getattr(self, key_name) != key_func(
                    self
                ):
                    setattr(self, cache_name, func(self))
                    setattr(self, key_name, key_func(self))
                return getattr(self, cache_name)

        return property(wrapper)

    return decorator


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
