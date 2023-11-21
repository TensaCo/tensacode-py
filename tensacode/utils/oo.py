from abc import ABC
from contextlib import contextmanager
import inspect
from threading import Lock
from typing import Any, ClassVar, Generic, Literal, Self
from uuid import uuid4
import attr
import glom

from tensacode.utils.internal_types import nested_dict
from tensacode.utils.types import R


class HasDefault(ABC):
    _current_stack: ClassVar[list[Self]] = []

    @classmethod
    def get_current(cls) -> Self:
        return cls._current_stack[-1]

    @contextmanager
    def as_default(self):
        self._current_stack.append(self)
        yield
        self._current_stack.pop()


@attr.s(auto_attribs=True)
class Namespace(ABC):
    _namespace_lock = attr.ib(factory=Lock, init=False)
    _namespace_mode: Literal["automatic", "manual"] = attr.ib("automatic", init=False)
    _manual_namespace_stack: list[str] = attr.ib(factory=list, init=False)

    @property
    def qualpath(self) -> str:
        with self._namespace_lock:
            match self._namespace_mode:
                case "automatic":
                    frames = inspect.stack()[2:]
                    return ".".join(
                        frame.f_locals.get("__qualname__", frame.f_code.co_name)
                        for frame in frames
                    )
                case "manual":
                    return ".".join(self._manual_namespace_stack)
                case _:
                    raise ValueError(f"Invalid namespace mode: {self._namespace_mode}")

    @contextmanager
    def namespace(self, name: str = None):
        with self._namespace_lock:
            name = name or uuid4().hex
            self._manual_namespace_stack.append(name)
            prev_namespace_mode = self._namespace_mode
        yield
        with self._namespace_lock:
            self._namespace_mode = prev_namespace_mode
            _name = self._manual_namespace_stack.pop()
            assert (
                _name == name
            ), f"Corrupted namespace stack: Expected {name}; got {_name}"
