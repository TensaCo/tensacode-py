"""
Flexible object locator system for nested data structures.

Provides classes for accessing and modifying nested attributes, indices, and function calls
in complex data structures. Supports composite locators for multi-step access.
"""

from functools import reduce
import inspect
from collections.abc import Mapping, Sequence, MutableSequence, MutableMapping
from typing import Any, Hashable, Union, Literal, Annotated
from abc import abstractmethod
from pydantic import BaseModel, Field, Discriminator


class Locator(BaseModel):
    """Base class for all locator types."""

    type: str = Field(discriminator=True)

    @abstractmethod
    def get(self, root: Any, current: Any, create_missing: bool = False) -> Any:
        """Get the value at the location specified by this locator."""
        pass

    @abstractmethod
    def set(self, root: Any, current: Any, value: Any, create_missing: bool = False):
        """Set the value at the location specified by this locator."""
        pass


class TerminalLocator(Locator):
    """Locator that does not have any further steps."""

    type: Literal["terminal"] = "terminal"

    def get(self, root: Any, current: Any, create_missing: bool = False) -> Any:
        """Get the value at the location specified by this locator."""
        return current

    def set(self, root: Any, current: Any, value: Any, create_missing: bool = False):
        """Set the value at the location specified by this locator."""
        current = value  # not sure if this actually does anything but lol


class LocatorStep(Locator):
    """Base class for individual steps in a composite locator."""

    pass


class IndexAccessStep(LocatorStep):
    """Locator step for accessing elements by index."""

    type: Literal["index"] = "index"
    index: int | None = None

    def get(self, root: Any, current: Any, create_missing: bool = False) -> Any:
        """
        Get the value at the specified index.

        >>> lst = [1, 2, 3]
        >>> step = IndexAccessStep(index=1)
        >>> step.get(lst, lst)
        2
        >>> step = IndexAccessStep(index=3)
        >>> step.get(lst, lst, create_missing=True)
        {}
        >>> lst
        [1, 2, 3, {}]
        >>> step = IndexAccessStep(index=5)
        >>> step.get(lst, lst, create_missing=True)
        {}
        >>> lst
        [1, 2, 3, {}, {}, {}]
        >>> step.get(lst, lst)
        {}
        >>> step = IndexAccessStep(index=10)
        >>> step.get(lst, lst)
        Traceback (most recent call last):
        ...
        IndexError: list index out of range
        """
        if (
            isinstance(current, Sequence)
            and create_missing
            and self.index >= len(current)
        ):
            if hasattr(current, "extend"):
                current.extend([{} for _ in range(self.index - len(current) + 1)])
            elif hasattr(current, "append"):
                current.append({})
            else:
                raise TypeError("Cannot extend non-list sequence")
        return current[self.index]

    def set(self, root: Any, current: Any, value: Any, create_missing: bool = False):
        """
        Set the value at the specified index.

        >>> lst = [1, 2, 3]
        >>> step = IndexAccessStep(index=1)
        >>> step.set(lst, lst, 4)
        >>> lst
        [1, 4, 3]
        >>> step = IndexAccessStep(index=3)
        >>> step.set(lst, lst, 5, create_missing=True)
        >>> lst
        [1, 4, 3, 5]
        >>> step = IndexAccessStep(index=6)
        >>> step.set(lst, lst, 7, create_missing=True)
        >>> lst
        [1, 4, 3, 5, {}, {}, 7]
        >>> step.set(lst, lst, 8)
        >>> lst
        [1, 4, 3, 5, {}, {}, 8]
        >>> step = IndexAccessStep(index=10)
        >>> step.set(lst, lst, 9)
        Traceback (most recent call last):
        ...
        IndexError: list assignment index out of range
        """
        if (
            isinstance(current, MutableSequence)
            and create_missing
            and self.index >= len(current)
        ):
            if hasattr(current, "extend"):
                current.extend([{} for _ in range(self.index - len(current) + 1)])
            elif hasattr(current, "append"):
                current.append({})
            else:
                raise TypeError("Cannot extend non-list sequence")
        current[self.index] = value


class DotAccessStep(LocatorStep):
    """Locator step for accessing attributes using dot notation."""

    type: Literal["dot"] = "dot"
    key: str | None = None

    def get(self, root: Any, current: Any, create_missing: bool = False) -> Any:
        """
        Get the value of the specified attribute.

        >>> class Example:
        ...     attr = 'value'
        ...     dict_attr = {'key': 'value'}
        >>> obj = Example()
        >>> step = DotAccessStep(key='attr')
        >>> step.get(obj, obj)
        'value'
        >>> step = DotAccessStep(key='dict_attr')
        >>> step.get(obj, obj)
        {'key': 'value'}
        >>> step = DotAccessStep(key='new_attr')
        >>> step.get(obj, obj, create_missing=True)
        {}
        >>> hasattr(obj, 'new_attr')
        True
        >>> step = DotAccessStep(key='nested')
        >>> step.get(obj.dict_attr, obj.dict_attr, create_missing=True)
        {}
        >>> obj.dict_attr
        {'key': 'value', 'nested': {}}
        >>> step.get({}, {}, create_missing=True)
        {}
        >>> step.get({}, {})
        Traceback (most recent call last):
        ...
        KeyError: 'nested'
        """
        if create_missing:
            if isinstance(current, Mapping) and self.key not in current:
                current[self.key] = {}
            elif not hasattr(current, self.key):
                setattr(current, self.key, {})
        return (
            getattr(current, self.key)
            if hasattr(current, self.key)
            else current[self.key]
        )

    def set(self, root: Any, current: Any, value: Any, create_missing: bool = False):
        """
        Set the value of the specified attribute.

        >>> class Example:
        ...     attr = 'value'
        ...     dict_attr = {'key': 'value'}
        >>> obj = Example()
        >>> step = DotAccessStep(key='attr')
        >>> step.set(obj, obj, 'new_value')
        >>> obj.attr
        'new_value'
        >>> step = DotAccessStep(key='new_attr')
        >>> step.set(obj, obj, 'created_value', create_missing=True)
        >>> obj.new_attr
        'created_value'
        >>> step = DotAccessStep(key='nested')
        >>> step.set(obj.dict_attr, obj.dict_attr, 'nested_value', create_missing=True)
        >>> obj.dict_attr
        {'key': 'value', 'nested': 'nested_value'}
        >>> step.set({}, {}, 'dict_value', create_missing=True)
        >>> step.set({}, {}, 'dict_value')
        Traceback (most recent call last):
        ...
        KeyError: 'nested'
        """
        if create_missing:
            if isinstance(current, Mapping) and self.key not in current:
                current[self.key] = value
            elif not hasattr(current, self.key):
                setattr(current, self.key, value)
            else:
                if hasattr(current, self.key):
                    setattr(current, self.key, value)
                else:
                    current[self.key] = value
        else:
            if hasattr(current, self.key):
                setattr(current, self.key, value)
            else:
                current[self.key] = value


class FunctionCallStep(LocatorStep):
    """Locator step for calling functions."""

    type: Literal["function"] = "function"
    fn_name: str | None = None
    args: list[Locator] | None = None
    kwargs: dict[str, Locator] | None = None

    def get(self, root: Any, current: Any, create_missing: bool = False) -> Any:
        """
        Call the function and return its result.

        >>> class Example:
        ...     def func(self, a, b=2):
        ...         return a + b
        >>> obj = Example()
        >>> step = FunctionCallStep(fn_name='func', args=[IndexAccessStep(index=0)], kwargs={'b': IndexAccessStep(index=1)})
        >>> step.get([1, 3], obj)
        4
        """
        arg_vals = [
            arg_locator.get(root, current, create_missing) for arg_locator in self.args
        ]
        kwarg_vals = {
            k: kwarg_locator.get(root, current, create_missing)
            for k, kwarg_locator in self.kwargs.items()
        }
        fn = getattr(current, self.fn_name)
        return fn(*arg_vals, **kwarg_vals)

    def set(self, root: Any, current: Any, value: Any, create_missing: bool = False):
        """
        Raise an error as function call results cannot be assigned to.

        >>> step = FunctionCallStep(fn_name='func')
        >>> step.set(None, None, None)
        Traceback (most recent call last):
        ...
        NotImplementedError: cannot assign to the result of a function call
        """
        raise NotImplementedError("cannot assign to the result of a function call")


class CompositeLocator(Locator):
    """Locator composed of multiple steps."""

    type: Literal["composite"] = "composite"
    steps: list[LocatorStep]

    def get(self, root: Any, current: Any, create_missing: bool = False) -> Any:
        """
        Get the value by applying all steps in sequence.

        >>> class Example:
        ...     def __init__(self):
        ...         self.lst = [{'a': 1}, {'a': 2}]
        ...         self.dict = {'x': {'y': 3}}
        >>> obj = Example()
        >>> locator = CompositeLocator(steps=[
        ...     DotAccessStep(key='lst'),
        ...     IndexAccessStep(index=1),
        ...     DotAccessStep(key='a')
        ... ])
        >>> locator.get(obj, obj)
        2
        >>> locator = CompositeLocator(steps=[
        ...     DotAccessStep(key='lst'),
        ...     IndexAccessStep(index=2),
        ...     DotAccessStep(key='a')
        ... ])
        >>> locator.get(obj, obj, create_missing=True)
        {}
        >>> obj.lst
        [{'a': 1}, {'a': 2}, {'a': {}}]
        >>> locator = CompositeLocator(steps=[
        ...     DotAccessStep(key='dict'),
        ...     DotAccessStep(key='x'),
        ...     DotAccessStep(key='z')
        ... ])
        >>> locator.get(obj, obj, create_missing=True)
        {}
        >>> obj.dict
        {'x': {'y': 3, 'z': {}}}
        >>> locator.get(obj, obj)
        {}
        """
        for step in self.steps:
            current = step.get(root, current, create_missing)
        return current

    def set(self, root: Any, current: Any, value: Any, create_missing: bool = False):
        """
        Set the value by applying all steps except the last, then setting the value at the last step.

        >>> class Example:
        ...     def __init__(self):
        ...         self.lst = [{'a': 1}, {'a': 2}]
        ...         self.dict = {'x': {'y': 3}}
        >>> obj = Example()
        >>> locator = CompositeLocator(steps=[
        ...     DotAccessStep(key='lst'),
        ...     IndexAccessStep(index=1),
        ...     DotAccessStep(key='a')
        ... ])
        >>> locator.set(obj, obj, 3)
        >>> obj.lst[1]['a']
        3
        >>> locator = CompositeLocator(steps=[
        ...     DotAccessStep(key='lst'),
        ...     IndexAccessStep(index=2),
        ...     DotAccessStep(key='b')
        ... ])
        >>> locator.set(obj, obj, 4, create_missing=True)
        >>> obj.lst
        [{'a': 1}, {'a': 3}, {'b': 4}]
        >>> locator = CompositeLocator(steps=[
        ...     DotAccessStep(key='dict'),
        ...     DotAccessStep(key='x'),
        ...     DotAccessStep(key='z')
        ... ])
        >>> locator.set(obj, obj, 5, create_missing=True)
        >>> obj.dict
        {'x': {'y': 3, 'z': 5}}
        >>> locator = CompositeLocator(steps=[
        ...     DotAccessStep(key='new_attr'),
        ...     DotAccessStep(key='nested'),
        ...     IndexAccessStep(index=0)
        ... ])
        >>> locator.set(obj, obj, 6, create_missing=True)
        >>> obj.new_attr
        {'nested': [6]}
        """
        for step in self.steps[:-1]:
            current = step.get(root, current, create_missing)
        final_step = self.steps[-1]
        final_step.set(root, current, value, create_missing)


def get_discriminator(v: Any) -> str:
    """
    Get the type discriminator from a dictionary or object.

    >>> get_discriminator({'type': 'test'})
    'test'
    >>> class Example:
    ...     type = 'example'
    >>> get_discriminator(Example())
    'example'
    >>> get_discriminator('not a dict or object')
    """
    if isinstance(v, Mapping):
        return v.get("type")
    return getattr(v, "type", None)


LocatorImpl = Annotated[
    Union[IndexAccessStep, DotAccessStep, FunctionCallStep, CompositeLocator],
    Discriminator(get_discriminator),
]
