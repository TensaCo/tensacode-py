from __future__ import annotations

from typing import Mapping
from typing_extensions import Self


class Namespaced:
    """Namespaced

    Seemlessly hijacks the `__new__` method of subclasses to ensure
    instances of `Namespaced` with identical `name`s are identical.

    Subclasses can implement desired behavior in
    ... __init__ for brand-new class instances
    ... __additional__ for another 'initialization' of the same instance
    """

    ALL: Mapping[str, object] = dict()  # {name, obj}

    def __new__(cls: type[Self], name: str, *args, **kwargs) -> Self:
        if name in Namespaced.ALL:
            self = Namespaced.ALL[name]
            self.__additional__(*args, **kwargs)
        else:
            self = super.__new__()
            Namespaced.ALL[name] = self
        return self

    def __pre_init__(self, *args, **kwargs):
        """Called before either __init__ or __additional__ is called"""
        pass

    def __post_init__(self, *args, **kwargs):
        """Called after either __init__ or __additional__ is called"""
        pass

    def __init__(self, *args, **kwargs):
        """Called when a class is being initialized the first time"""
        pass

    def __additional__(self, *args, **kwargs):
        """Called when a class already exists but is being 'initialized' again"""
        pass
