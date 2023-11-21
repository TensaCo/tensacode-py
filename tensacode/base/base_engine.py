from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import _DataclassT, dataclass
import functools
from functools import singledispatchmethod
import inspect
from pathlib import Path
import pickle
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generator,
    Generic,
    Iterable,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    TypeVar,
)
from box import Box
from uuid import uuid4
import attr
from jinja2 import Template
import loguru
from glom import glom
from pydantic import Field

import typingx
import pydantic, sqlalchemy
from _typeshed import DataclassInstance


import tensacode as tc
from tensacode.utils.decorators import (
    Decorator,
    Default,
    dynamic_defaults,
    is_attrs_instance,
    is_attrs_type,
    is_dataclass_instance,
    is_dataclass_type,
    is_namedtuple_instance,
    is_namedtuple_type,
    is_object_instance,
    is_type,
    is_pydantic_model_instance,
    is_pydantic_model_type,
    is_sqlalchemy_instance,
    is_sqlalchemy_model_type,
    overloaded,
)
from tensacode.utils.oo import HasDefault, Namespace
from tensacode.utils.string0 import render_invocation, render_stacktrace
from tensacode.utils.types import (
    enc,
    T,
    R,
    atomic_types,
    container_types,
    composite_types,
    tree_types,
    tree,
)
from tensacode.utils.internal_types import nested_dict


class BaseEngine(Generic[T, R], ABC):
    #######################################
    ############### meta ##################
    #######################################

    T: ClassVar[type[T]] = T
    R: ClassVar[type[R]] = R

    class _HasThisEngine(ABC):
        _engine: ClassVar[BaseEngine]

    class _EngineDecorator(Decorator, _HasThisEngine, ABC):
        pass

    @attr.s(auto_attribs=True)
    class DefaultParam(Default, _HasThisEngine):
        initial_value: Any | None = attr.ib(default=None)
        initializer: Callable[[BaseEngine], Any] | None = attr.ib(default=None)

        def __init__(self, initializer_or_initial_value: Any = None, /, **kw):
            if typingx.isinstance(self.default, Callable[[BaseEngine], Any]):
                self.initializer = initializer_or_initial_value
            else:
                self.initial_value = initializer_or_initial_value
            self.kw = kw
            super().__init__(get=self.get)

        def get(self, *a, **kw) -> Any:
            initial_val: Any
            if self.initial_value is not None:
                initial_val = self.initial_value
            elif self.initializer is not None:
                initial_val = self.initializer(self._engine)
            else:
                initial_val = None
            return self._engine.param(initial_val, **self.kw)

    @attr.s(auto_attribs=True)
    class trace(_EngineDecorator):
        trace_args = attr.ib(default=True)
        trace_result = attr.ib(default=True)

        def prologue(self, *a, **kw):
            if self.trace_args:
                stacktrace = render_stacktrace(
                    skip_frames=3,
                    depth=self._engine.DefaultParam(qualname="hparams.trace.depth"),
                )
                self._engine.inform(stacktrace)
            return super().prologue(*a, **kw)

        def epilogue(self, retval, *a, **kw):
            if self.trace_result:
                stacktrace = render_stacktrace(
                    skip_frames=3,
                    depth=self._engine.DefaultParam(qualname="hparams.trace.depth"),
                )
                self._engine.inform(stacktrace)
            return super().epilogue(retval, *a, **kw)

    @attr.s(auto_attribs=True)
    class encoded_args(_EngineDecorator):
        encode_args: bool = attr.ib(True)
        decode_retval: bool = attr.ib(True)

        def prologue(self, *a, **kw):
            if self.encode_args:
                # bind params to their values
                signature = inspect.signature(self.fn)
                bound_args = signature.bind_partial(*a, **kw)
                bound_args.apply_defaults()
                bound_args = bound_args.arguments
                # encode the params that are annotated with `enc[...]`
                for param_name, param in signature.parameters.items():
                    if param.annotation is not param.empty and typingx.issubclassx(
                        param.annotation, enc
                    ):
                        if param_name in bound_args:
                            bound_args[param_name] = self._engine.encode(
                                bound_args[param_name]
                            )
                # unpack the bound args
                a, kw = [], {}
                for arg, value in bound_args.items():
                    if arg in signature.parameters:
                        if signature.parameters[arg].kind in (
                            signature.parameters[arg].POSITIONAL_ONLY,
                            signature.parameters[arg].POSITIONAL_OR_KEYWORD,
                        ):
                            a.append(value)
                        elif signature.parameters[arg].kind in (
                            signature.parameters[arg].VAR_POSITIONAL,
                            signature.parameters[arg].KEYWORD_ONLY,
                            signature.parameters[arg].VAR_KEYWORD,
                        ):
                            kw[arg] = value
                a = tuple(a)

            return super().prologue(*a, **kw)

        def epilogue(self, retval, *a, **kw):
            if self.decode_retval:
                # get the return annotation from the function signature
                signature = inspect.signature(self.fn)
                return_annotation = signature.return_annotation

                # check if the return value is annotated with `enc[...]`
                if return_annotation is not signature.empty and typingx.issubclassx(
                    return_annotation, enc
                ):
                    # decode the retval
                    retval = self._engine.decode(retval)

            return super().epilogue(retval, *a, **kw)

    def is_encoded(self, object: T | R) -> bool:
        if TYPE_CHECKING:
            return isinstance(object, R)
        return self._is_encoded(object)

    #######################################
    ############### config ################
    #######################################

    PARAM_DEFAULTS = {
        "hparams": {
            "defaults": (defaults := {"depth_limit": 10}),
            "trace": {"depth": 5},
            "encode": {"depth_limit": defaults["depth_limit"], "instructions": None},
            "decode": {"depth_limit": defaults["depth_limit"], "instructions": None},
            "retrieve": {
                "count": 5,
                "depth_limit": defaults["depth_limit"],
                "instructions": None,
            },
            "store": {"depth_limit": defaults["depth_limit"], "instructions": None},
            "query": {"depth_limit": defaults["depth_limit"], "instructions": None},
            "modify": {"depth_limit": defaults["depth_limit"], "instructions": None},
            "combine": {"depth_limit": defaults["depth_limit"], "instructions": None},
            "split": {
                "num_splits": 2,
                "depth_limit": defaults["depth_limit"],
                "instructions": None,
            },
            "choice": {
                "threshold": 0.5,
                "randomness": 0.1,
                "depth_limit": defaults["depth_limit"],
                "instructions": None,
            },
            "decide": {
                "threshold": 0.5,
                "randomness": 0.1,
                "depth_limit": defaults["depth_limit"],
                "instructions": None,
            },
            "run": {"instructions": None, "budget": 1.0},
            "similarity": {
                "depth_limit": defaults["depth_limit"],
                "instructions": None,
            },
            "predict": {"depth_limit": defaults["depth_limit"], "instructions": None},
            "correct": {
                "threshold": 0.5,
                "depth_limit": defaults["depth_limit"],
                "instructions": None,
            },
        }
    }

    params: nested_dict[str, Any]

    #######################################
    ######## intelligence methods #########
    #######################################

    def __init_subclass__(cls):
        # cls._engine = Engine() # TODO: figure this out
        super().__init_subclass__()

    def __init__(self, *args, **kwargs):
        self.params = {}
        for base in reversed(self.__class__.__mro__):
            self.params.update(deepcopy(base.PARAM_DEFAULTS))
        self._HasThisEngine._engine = self  # TODO: this is wrong! It doesn't work with multiple instances or with subclassing
        super().__init__(*args, **kwargs)

    @encoded_args()
    def param(
        self, initial_value: enc[T] = None, name: str = None, qualname: str = None
    ) -> Any:
        """
        Hook that returns a parameter value. (Aka, like react.use_state, but for parameters.)

        Args:
            initial_value: The initial value of the parameter.
                If the parameter already exists, this argument is ignored.
                If the parameter does not exist, it is initialized to this value.
                If no initial_value is provided, the parameter is initialized to the default value.
            name: The name of the parameter.
                Identifies the parameter in the local namespace.
                If no name is provided, the parameter is given a unique name.
                If a name is provided, and the parameter already exists, `param` returns the parameter value.
            qualname: The qualified name of the parameter.
                Identifies the parameter in the global namespace.
                If no qualname is provided, the qualified name is given by "{engine.qualpath}.{name}".
                If a qualname is provided, and the parameter already exists, `param` returns the parameter value.

        Returns:
            The parameter value.

        Notes:
            For named variables, the key is already known, but for unnamed variables, the key must be idempotently generated.
            `param` achieves this by tracking the order of calling in a given namespace.
            Each unnamed parameter is named in the order it is called like so `f"param_{i}"`
            Functions, classes, and modules all have their own namespace, given by their name relative to the module root.
            You can also manually create a namespace with `engine.namespace`.
            If you want to use param calls inside conditional blocks, you should declare a namespace for each block like so:

            ```
            if condition:
                with engine.namespace(True):
                    x = engine.param()
                    ...
            else:
                with engine.namespace(False):
                    a = engine.param()
                    b = engine.param()
                    c = engine.param()
                    ...
            ```

            This enables your anonymous `engine.param()` will be able to re-run and still use the same param values.

        """

        match name, qualname:
            case None, None:
                # `use_state`-like mechanism that tracks the stack hierarchy and order of calling to make param calls idempotent. (Tracking can be overriden with the `.namespace(str)` method).
                name = self._anonymous_params_in_qualpath.setdefault(self.qualpath, 0)
                self._anonymous_params_in_qualpath[self.qualpath] += 1
                qualname = self.qualpath + "." + name
            case None, _:
                # keep qualname as is
                pass
            case _, None:
                qualname = self.qualpath + "." + name
            case _, _:
                # qualname overrides name
                pass
        if glom(self.params, qualname) is None:
            glom(self.params, qualname, default=initial_value)
        return glom(self.params, qualname)

    _anonymous_params_in_qualpath: dict[str, int] = attr.ib(factory=dict, init=False)

    messages: list[str] = attr.ib(factory=list, init=False)

    @encoded_args()
    @trace()
    @functools.singledispatchmethod
    def inform(self, message: enc[T]):
        self.messages.append(message)

    @inform.register
    def _(self, messages: list[T | enc[T]]):
        # don't worry about pre-encoding your list. self.inform(single T) will do that for you.
        for message in messages:
            self.inform(message)

    @encoded_args()
    @trace()
    def chat(self, message: enc[T]) -> enc[T]:
        raise NotImplementedError('Subclass must implement "chat"')

    @trace()
    def self_reflect(self):
        raise NotImplementedError('Subclass must implement "self_reflect"')

    @encoded_args()
    @trace()
    def reward(self, reward: enc[float]):
        raise NotImplementedError('Subclass must implement "reward"')

    @trace()
    def train(self):
        raise NotImplementedError('Subclass must implement "train"')

    @trace()
    def save(self, path: str | Path):
        path = Path(path)
        match path.suffix:
            case "yaml", "yml":
                Box(self.params).to_yaml(filename=path)
            case "json":
                Box(self.params).to_json(filename=path)
            case "toml":
                Box(self.params).to_toml(filename=path)
            case "pickle", "pkl":
                with path.open("wb") as f:
                    pickle.dump(self.params, f)
            case _:
                raise ValueError(f"Invalid file extension: {path.suffix}")

    @trace()
    def load(self, path: str | Path):
        path = Path(path)
        match path.suffix:
            case "yaml", "yml":
                new_params = Box.from_yaml(filename=path).to_dict()
                self.params.update(new_params)
            case "json":
                new_params = Box.from_json(filename=path).to_dict()
                self.params.update(new_params)
            case "toml":
                new_params = Box.from_toml(filename=path).to_dict()
                self.params.update(new_params)
            case "pickle", "pkl":
                with path.open("rb") as f:
                    new_params = pickle.load(f)
                self.params.update(new_params)
            case _:
                raise ValueError(f"Invalid file extension: {path.suffix}")

    def _is_encoded(self, object: T | R) -> bool:
        return typingx.isinstancex(object, (self.R, self.enc[T]))

    # TODO: move this to a separate mixin. Also move the llm_engine.base.BaseLLMEngine.combine to a mixin.
    def combine(self, *objects: enc[T]) -> enc[T]:
        return self.encode(objects)
