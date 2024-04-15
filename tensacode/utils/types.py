from typing import Protocol, runtime_checkable


@runtime_checkable
class Predicate(Protocol):
    def __call__(self, *args, **kwargs) -> bool:
        pass