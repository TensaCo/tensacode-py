from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import _DataclassT, dataclass
import functools
from functools import singledispatchmethod
import inspect
from itertools import count
from pathlib import Path
import pickle
import threading
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
    Self,
    Sequence,
    Set,
    TypeVar,
    get_origin,
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
import pydantic
from _typeshed import DataclassInstance
from tensacode._internal.code2str import render_invocation
from tensacode._internal.decorators import Decorator
from tensacode._internal.misc import HasDefault, Namespaced, Scoped

from tensacode._internal.python import ARG_IDENTIFIER
from tensacode._internal.typing import Action, Invocation, Message

T, R = TypeVar("T"), TypeVar("R")


class BaseEngine(Generic[T, R], Scoped[Action], HasDefault, ABC):
    
    #######################################
    ############### config ################
    #######################################
    
    T: ClassVar[type[T]] = T
    R: ClassVar[type[R]] = R

    params: dict[str, Any] = field(factory=dict)
    DEFAULT_PARAM_NAME_TEMPLATE = "param_{i}"
    
    default_depth_limit: int = field(default=10)
    default_instructions: list[R] = field(factory=list)
    default_examples: list[Action] = field(factory=list)
    default_trace_frame_depth: int = field(default=3)

    base_instructions: list[R] = field(factory=list)
    base_examples: list[Action] = field(factory=list)
    
    #######################################
    ############### meta ##################
    #######################################

    class _EngineDecorator(Decorator, ABC):
        _engine: ClassVar[BaseEngine]

    @property
    def root_action(self) -> Action:
        return self._scope_stack[0]

    @property
    def current_action_scope(self) -> list[Action]:
        return self._scope_stack

    @property
    def current_action(self) -> Action:
        return self.current_action_scope[-1]

    @contextmanager
    def with_action(self, action: Action):
        # no need to trace-wrap this ctx manager because it handles its own logging
        self._log(f"<starting {action}>")
        with self.with_instructions(self.base_instructions):
            yield self.scoped(action, keys_to_detach=["base_instructions", "base_examples"])
        self._log(f"<finished {action}>")

    @define
    class trace(_EngineDecorator):
        inputs: bool | str = field(default=True)
        result: bool = field(default=True)
        frame_depth: Optional[int] = field()
        _with_action_context_manager = field(init=False)
        _action = field(init=False)

        def __post_init__(self):
            if self.frame_depth is None:
                self.frame_depth = self._engine.default_trace_frame_depth

        def prologue(self, *a, **kw):
            if self.inputs:
                self._action = Invocation(fn=self.fn, args=a, kwargs=kw)
                self._with_action_context_manager = self._engine.with_action(
                    self._action
                )
                self._with_action_context_manager.__enter__()

        def epilogue(self, result, *a, **kw):
            if self.result:
                self._action.result = result
                self._with_action_context_manager.__exit__()

    @attr.s(auto_attribs=True)
    class autoconvert(_EngineDecorator):
        """
        - converts args into the python parameter type if different (or if encoded)
        - converts the result into the pythonreturn type if different (or if encoded)
        """

        args: bool = field(True)
        only_args: Optional[list[ARG_IDENTIFIER]] = field(default=None)
        exclude_args: Optional[list[ARG_IDENTIFIER]] = field(default=None)
        result: bool = field(True)

        def prologue(self, *a, **kw):
            assert typingx.isinstancex(self._engine, "SupportsConvert"), (
                "autoconvert can only be used with an engine that supports conversion. "
                "Did you forget to subclass the appropriate SupportsConvert aspect?"
            )

            if self.args:
                # bind params to their values
                signature = inspect.signature(self.fn)
                bound_args = signature.bind_partial(*a, **kw)
                bound_args.apply_defaults()
            # convert any args that don't match their expected type
            for param_name, param in signature.parameters.items():
                # check if the value provided doesn't match the annotation type expected
                if (
                    param.annotation is not param.empty
                    and param_name in bound_args
                    and not typingx.isinstancex(
                        bound_args[param_name], param.annotation
                    )
                ):
                    bound_args[param_name] = self._engine._convert(
                        bound_args[param_name], param.annotation
                    )

        def epilogue(self, result, *a, **kw):
            if self.result:
                # get the return annotation from the function signature
                signature = inspect.signature(self.fn)

                # check if the return value doesn't match the expected type
                if (
                    signature.return_annotation is not signature.empty
                    and not typingx.isinstancex(result, signature.return_annotation)
                ):
                    # convert the result
                    result = self._engine._convert(result, signature.return_annotation)

            return super().epilogue(result, *a, **kw)

    def is_encoded(self, object: T | R) -> bool:
        if TYPE_CHECKING:
            return isinstance(object, R)
        return self._is_encoded(object)

    #######################################
    ######## intelligence methods #########
    #######################################

    def __init_subclass__(cls):
        # cls._engine = Engine() # TODO: figure this out
        super().__init_subclass__()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # update this class' decorators to refer to this engine
        for k, v in self.__class__.__dict__.items():
            if isinstance(v, self._EngineDecorator):
                v_for_instance = copy(v)
                v_for_instance._engine = self
                setattr(self, k, v_for_instance)


    @autoconvert()
    def param(
        self,
        initial_value: Optional[R] = None,
        /,
        *,
        name: Optional[str] = None,
        qualname: Optional[str] = None,
    ) -> Any:
        """Hook that returns a parameter value. (Aka, like react.use_state, but for parameters.)

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
                lowest_available_idx = next(
                    i
                    for i in count()
                    if (
                        self.full_scope
                        + self.SCOPE_QUALPATH_SEPARATOR
                        + self.DEFAULT_PARAM_NAME_TEMPLATE.format(i)
                    )
                    not in self.params
                )
                name = self.DEFAULT_PARAM_NAME_TEMPLATE.format(lowest_available_idx)
                qualname = self.scope_qualpath + self.SCOPE_QUALPATH_SEPARATOR + name
            case _, None:
                qualname = self.scope_qualpath + self.SCOPE_QUALPATH_SEPARATOR + name
            case None, _:
                name = qualname.split(self.SCOPE_QUALPATH_SEPARATOR)[-1]
            case _, _:
                raise ValueError(f"Only one of name or qualname can be provided.")

        if glom(self.params, qualname) is None:
            glom(self.params, qualname, default=initial_value)
        return glom(self.params, qualname)

    @property
    def messages(self) -> list[Message]:
        return self.current_action.children

    @messages.setter
    def messages(self, value: list[Message]):
        raise RuntimeError(
            "This property is read-only. If you want to remove all messages use the .empty() method."
        )

    @autoconvert()
    @trace()
    def log(self, content: T, metadata: dict[str, Any] = None, /):
        self._log(content, metadata)

    @functools.singledispatchmethod
    def _log(self, content: T, metadata: dict[str, Any] = None, /):
        self.messages.append(Message(content=content, metadata=metadata))

    @autoconvert()
    @trace()
    def instruct(self, instructions: list[R] = None, *, examples: list[Action] = None):
        if instructions:
            self.base_instructions = instructions
        if examples:
            self.default_examples = examples

    @autoconvert()
    @trace()
    @contextmanager
    def with_instructions(self, instructions: list[R]):
        self.base_instructions = instructions
        yield
        self.base_instructions = []

    @autoconvert()
    @trace()
    def chat(self, message: T) -> T:
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

    def _is_encoded(self, object: R | Any) -> bool:
        return typingx.isinstancex(object, self.R)
