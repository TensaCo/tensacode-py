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
from pydantic import BaseModel, Field
from attrs import field, define

import typingx
import pydantic, sqlalchemy
from _typeshed import DataclassInstance

from tensacode._internal.python import ARG_IDENTIFIER
from tensacode._internal.typing import Action

T, R = TypeVar("T"), TypeVar("R")

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

    root_action: Action
    currnet_action: Action

    @contextmanager
    def with_action(self, action: Action):
        self.actions.append(action)
        yield
        self.actions.pop()

    @define
    class trace(_EngineDecorator):
        args = field(default=True)
        retval = field(default=True)

        def prologue(self, *a, **kw):
            invokation = (fn, a, kw)
            # TODO: think about: what is the difference between a trace and a with_action?
            # I know: a trace is a decorator. with_action is the meat of the oepration!
            self._engine.log(invokation) # override this in lm subclass to format it as an invokation string
            self._engine.namespace(fn.__name__)
            
            
            if self.args:
                stacktrace = render_stacktrace(
                    skip_frames=3,  # this frame, Decorator.__call__'s wrapper, and Decorator.__call__ (_EngineDecorator parent)
                    depth=self._engine.DefaultParam(qualname="hparams.trace.depth"),
                )
                self._engine.log(stacktrace)
            return super().prologue(*a, **kw)

        def epilogue(self, retval, *a, **kw):
            if self.retval:
                stacktrace = render_stacktrace(
                    skip_frames=3,  # this frame, Decorator.__call__'s wrapper, and Decorator.__call__ (_EngineDecorator parent)
                    depth=self._engine.DefaultParam(qualname="hparams.trace.depth"),
                )
                self._engine.log(stacktrace)
            return super().epilogue(retval, *a, **kw)

    @attr.s(auto_attribs=True)
    class autoconvert(_EngineDecorator):
        args: bool = field(True)
        only_args: Optional[list[ARG_IDENTIFIER]] = field(default=None)
        exclude_args: Optional[list[ARG_IDENTIFIER]] = field(default=None)
        retval: bool = field(True)

        def prologue(self, *a, **kw):
            if self.args:
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
            if self.retval:
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

    defaults_depth_limit=10,
    defaults_instructions=None

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

    @autoconvert()
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

    _anonymous_params_in_qualpath: dict[str, int] = field(factory=dict, init=False)


    messages: list[Message] = field(factory=lambda: [], init=False)

    @autoconvert()
    @trace()
    @functools.singledispatchmethod
    def log(self, content: R, metadata: dict[str, Any] = None, /):
        self.messages.append(self.Message(content=content, metadata=metadata))

    @autoconvert()
    @trace()
    def instruct(self, instructions: enc[T]=None, *, examples: list[Traj]=None):
        if instructions:
            self.defaults_instructions = instructions
        if examples:
            self.defaults_examples = examples

    @autoconvert()
    @trace()
    def chat(self, message: enc[T]) -> enc[T]:
        raise NotImplementedError('Subclass must implement "chat"')

    @trace()
    def self_reflect(self):
        raise NotImplementedError('Subclass must implement "self_reflect"')

    @autoconvert()
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
