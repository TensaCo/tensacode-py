from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Any, Literal, TypedDict, ClassVar
from functools import cached_property, contextmanager
from pydantic import BaseModel, Field
from datetime import datetime
from functools import reduce
import traceback
import inspect
from functools import wraps
from tensacode.internal.utils.consts import VERSION
from tensacode.internal.utils.language import Language
from tensacode.internal.protocols.tagged_object import HasID
from tensacode.core.base.ops.ops import BaseOp
from tensacode.internal.utils.functional import cached_with_key
from contextlib import contextmanager
from pathlib import Path
from typing import Callable
from tensacode.internal.utils.python_str import render_function_call
from pydantic import Annotated
from typing import Optional
from tensacode.core.base.latents.latents import LatentType
from tensacode.core.base.ops.base_op import BaseOp
from typing import Hashable
from typing import Mapping


class BaseEngine(HasID, BaseModel):

    tensacode_version: ClassVar[str] = VERSION
    render_language: Language = "python"
    latent_type: type[LatentType]
    add_default_log_meta = True

    """
    ### Operations ###
    
    This system implements a flexible and extensible framework for managing different types of 
    engines and operations. It allows for dynamic registration, inheritance-based matching, 
    and type-safe operation execution across various engine types.

    Key features:
    1. Multiple engine types: Different BaseEngine subclasses can have their own set of operations.
    2. Multiple operation types: Various BaseOp subclasses can be created for different functionalities.
    3. Dynamic registration: Operations can be registered at runtime, both as instances and classes.
    4. Inheritance-based matching: The system finds the most specific operation based on 
       inheritance hierarchies of engine types, operation types, and object types.
    5. Type safety: Operations are bound to specific engine types, ensuring type-safe execution.

    Registration methods:
    - register_op_instance_for_this_object: Register an operation instance for a specific engine object.
    - register_op_class_for_this_object: Register an operation class for a specific engine object.
    - register_op_instance_for_all_class_instances: Register an operation instance for all instances of the engine class.
    - register_op_class_for_all_class_instances: Register an operation class for all instances of the engine class.

    Getter methods:
    - get_all_registered_op_instances: Get all registered operation instances for an engine object.
    - get_all_registered_op_classes: Get all registered operation classes for an engine object.
    - get_class_level_registered_op_instances: Get all operation instances registered at the class level.
    - get_class_level_registered_op_classes: Get all operation classes registered at the class level.

    Example usage:

    >>> from typing import Any
    >>> from tensacode.core.base.latents.latents import LatentType
    >>> from tensacode.core.base.ops.base_op import BaseOp

    >>> class TextLatent(LatentType):
    ...     pass

    >>> class ImageLatent(LatentType):
    ...     pass

    >>> class TextEngine(BaseEngine):
    ...     latent_type = TextLatent

    >>> class ImageEngine(BaseEngine):
    ...     latent_type = ImageLatent

    >>> class SummarizeOp(BaseOp):
    ...     op_name: str = "summarize"
    ...     latent_type: type[LatentType] = TextLatent
    ...     engine_type: type[BaseEngine] = TextEngine
    ...     object_type: type[Any] = str
    ...
    ...     def execute(self, engine: TextEngine, text: str, **kwargs):
    ...         return f"Summary of: {text[:10]}..."

    >>> class GenerateImageOp(BaseOp):
    ...     op_name: str = "generate_image"
    ...     latent_type: type[LatentType] = ImageLatent
    ...     engine_type: type[BaseEngine] = ImageEngine
    ...     object_type: type[Any] = str
    ...
    ...     def execute(self, engine: ImageEngine, prompt: str, **kwargs):
    ...         return f"Image generated from: {prompt[:10]}..."

    # Register ops
    >>> TextEngine.register_op_class_for_all_class_instances(SummarizeOp)
    >>> ImageEngine.register_op_class_for_all_class_instances(GenerateImageOp)

    # Usage with TextEngine
    >>> text_engine = TextEngine()
    >>> summarize_op = text_engine.get_op(SummarizeOp)
    >>> summary = summarize_op.execute(text_engine, text="Long text to summarize")
    >>> print(summary)
    Summary of: Long text ...

    # Usage with ImageEngine
    >>> image_engine = ImageEngine()
    >>> generate_image_op = image_engine.get_op(GenerateImageOp)
    >>> image = generate_image_op.execute(image_engine, prompt="A beautiful sunset")
    >>> print(image)
    Image generated from: A beautifu...

    This structure allows for easy extension of the system with new engine types and operations 
    while maintaining type safety and allowing for specialized implementations. The registration 
    system ensures that operations are correctly associated with their respective engine types, 
    and the get_op method provides a convenient way to retrieve the appropriate operation 
    instance for a given engine.
    """

    _op_instances_for_this_object: list[BaseOp] = Field(default_factory=list)
    _op_classes_for_this_object: list[type[BaseOp]] = Field(default_factory=list)
    _op_instances_for_all_class_instances: ClassVar[list[BaseOp]] = []
    _op_classes_for_all_class_instances: ClassVar[list[type[BaseOp]]] = []

    def register_op_instance_for_this_object(self, op: BaseOp):
        """
        Register an operation instance for this specific engine object.

        Args:
            op (BaseOp): The operation instance to register.

        Example:
        >>> engine = BaseEngine()
        >>> op = BaseOp()
        >>> engine.register_op_instance_for_this_object(op)
        >>> assert op in engine.get_all_registered_op_instances()
        """
        self._op_instances_for_this_object.append(op)

    def register_op_class_for_this_object(self, op_class: type[BaseOp]):
        """
        Register an operation class for this specific engine object.

        Args:
            op_class (type[BaseOp]): The operation class to register.

        Example:
        >>> engine = BaseEngine()
        >>> class MyOp(BaseOp):
        ...     pass
        >>> engine.register_op_class_for_this_object(MyOp)
        >>> assert MyOp in engine.get_all_registered_op_classes()
        """
        self._op_classes_for_this_object.append(op_class)

    @classmethod
    def register_op_instance_for_all_class_instances(cls, op: BaseOp):
        """
        Register an operation instance for all instances of this engine class.

        Args:
            op (BaseOp): The operation instance to register.

        Example:
        >>> class MyEngine(BaseEngine):
        ...     pass
        >>> op = BaseOp()
        >>> MyEngine.register_op_instance_for_all_class_instances(op)
        >>> engine = MyEngine()
        >>> assert op in engine.get_all_registered_op_instances()
        """
        cls._op_instances_for_all_class_instances.append(op)

    @classmethod
    def register_op_class_for_all_class_instances(cls, op_class: type[BaseOp]):
        """
        Register an operation class for all instances of this engine class.

        Args:
            op_class (type[BaseOp]): The operation class to register.

        Example:
        >>> class MyEngine(BaseEngine):
        ...     pass
        >>> class MyOp(BaseOp):
        ...     pass
        >>> MyEngine.register_op_class_for_all_class_instances(MyOp)
        >>> engine = MyEngine()
        >>> assert MyOp in engine.get_all_registered_op_classes()
        """
        cls._op_classes_for_all_class_instances.append(op_class)

    def get_all_registered_op_instances(self):
        """
        Get all registered operation instances for this engine object.

        Returns:
            list[BaseOp]: A list of registered operation instances.

        Example:
        >>> engine = BaseEngine()
        >>> op = BaseOp()
        >>> engine.register_op_instance_for_this_object(op)
        >>> assert op in engine.get_all_registered_op_instances()
        """
        return (
            self._op_instances_for_this_object
            + self.get_class_level_registered_op_instances()
        )

    def get_all_registered_op_classes(self):
        """
        Get all registered operation classes for this engine object.

        Returns:
            list[type[BaseOp]]: A list of registered operation classes.

        Example:
        >>> engine = BaseEngine()
        >>> class MyOp(BaseOp):
        ...     pass
        >>> engine.register_op_class_for_this_object(MyOp)
        >>> assert MyOp in engine.get_all_registered_op_classes()
        """
        return (
            self._op_classes_for_this_object
            + self.get_class_level_registered_op_classes()
        )

    @classmethod
    def get_class_level_registered_op_instances(cls):
        """
        Get all operation instances registered at the class level.

        Returns:
            list[BaseOp]: A list of registered operation instances.

        Example:
        >>> class MyEngine(BaseEngine):
        ...     pass
        >>> op = BaseOp()
        >>> MyEngine.register_op_instance_for_all_class_instances(op)
        >>> engine = MyEngine()
        >>> assert op in engine.get_class_level_registered_op_instances()
        """
        return cls._op_instances_for_all_class_instances

    @classmethod
    def get_class_level_registered_op_classes(cls):
        """
        Get all operation classes registered at the class level.

        Returns:
            list[type[BaseOp]]: A list of registered operation classes.

        Example:
        >>> class MyEngine(BaseEngine):
        ...     pass
        >>> class MyOp(BaseOp):
        ...     pass
        >>> MyEngine.register_op_class_for_all_class_instances(MyOp)
        >>> engine = MyEngine()
        >>> assert MyOp in engine.get_class_level_registered_op_classes()
        """
        return cls._op_classes_for_all_class_instances

    def get_op(
        self,
        operator_type: type[BaseOp],
        object_type: type[Any] | None = Any,
        latent_type: type[LatentType] | None = None,
    ) -> BaseOp:
        """
        Get the most specific operation instance that matches the given criteria.

        Args:
            operator_type (type[BaseOp]): The desired operation type.
            object_type (type[Any] | None, optional): The desired object type. Defaults to Any.
            latent_type (type[LatentType] | None, optional): The desired latent type. Defaults to None.

        Returns:
            BaseOp: The most specific operation instance that matches the criteria.

        Raises:
            ValueError: If no matching operation is found.

        Example:
        >>> class MyEngine(BaseEngine):
        ...     pass
        >>> class MyOp(BaseOp):
        ...     op_name: str = "my_op"
        ...     latent_type: type[LatentType] = LatentType
        ...     engine_type: type[BaseEngine] = MyEngine
        ...     object_type: type[Any] = str
        ...
        ...     def execute(self, engine: MyEngine, text: str, **kwargs):
        ...         return f"Result of MyOp: {text[:10]}..."
        >>> MyEngine.register_op_class_for_all_class_instances(MyOp)
        >>> engine = MyEngine()
        >>> op = engine.get_op(MyOp)
        >>> result = op.execute(engine, text="Long text to process")
        >>> print(result)
        Result of MyOp: Long text ...
        """
        if latent_type is None:
            latent_type = self.latent_type

        matching_ops = [
            op
            for op in self.get_all_registered_op_instances()
            if isinstance(op, operator_type)
            and issubclass(op.object_type, object_type)
            and issubclass(op.latent_type, latent_type)
        ]

        if not matching_ops:
            op_cls = self.get_op_cls(latent_type, operator_type, object_type)
            op_instance = op_cls.from_engine(engine=self)
            self.register_op_instance_for_this_object(op_instance)
            return op_instance

        return min(
            matching_ops,
            key=lambda op: (
                inheritance_distance(op, operator_type),
                inheritance_distance(op.latent_type, latent_type),
                inheritance_distance(op.object_type, object_type),
            ),
        )

    def get_op_cls(
        self,
        latent_type: type[LatentType],
        operator_type: type[BaseOp],
        object_type: type[Any],
    ) -> type[BaseOp]:
        """
        Get the most specific operation class that matches the given criteria.

        Args:
            latent_type (type[LatentType]): The desired latent type.
            operator_type (type[BaseOp]): The desired operation type.
            object_type (type[Any]): The desired object type.

        Returns:
            type[BaseOp]: The most specific operation class that matches the criteria.

        Raises:
            ValueError: If no matching operation class is found.

        Example:
        >>> class MyEngine(BaseEngine):
        ...     pass
        >>> class MyOp(BaseOp):
        ...     op_name: str = "my_op"
        ...     latent_type: type[LatentType] = LatentType
        ...     engine_type: type[BaseEngine] = MyEngine
        ...     object_type: type[Any] = str
        ...
        ...     def execute(self, engine: MyEngine, text: str, **kwargs):
        ...         return f"Result of MyOp: {text[:10]}..."
        >>> MyEngine.register_op_class_for_all_class_instances(MyOp)
        >>> engine = MyEngine()
        >>> op_cls = engine.get_op_cls(LatentType, MyOp, str)
        >>> op = op_cls.from_engine(engine)
        >>> result = op.execute(engine, text="Long text to process")
        >>> print(result)
        Result of MyOp: Long text ...
        """
        matching_ops = [
            op
            for op in self.get_all_registered_op_classes()
            if issubclass(op, operator_type)
            and issubclass(op.latent_type, latent_type)
            and issubclass(op.object_type, object_type)
        ]
        if not matching_ops:
            raise ValueError(
                f"No matching operator found for latent_type={latent_type}, operator_type={operator_type}, object_type={object_type}"
            )

        return min(
            matching_ops,
            key=lambda op: (
                inheritance_distance(op, operator_type),
                inheritance_distance(op.latent_type, latent_type),
                inheritance_distance(op.object_type, object_type),
            ),
        )

    ### Context Management ###
    """
    Manages a hierarchical context system for the BaseEngine class.

    This system allows for creating nested scopes of context, where each scope can
    contain multiple updates. The context is organized as a list of scopes, where
    each scope is a list of update dictionaries.

    The context management system provides the following features:
    1. Hierarchical scoping: Create nested scopes for organizing context updates.
    2. Read-write access: Access the current context state and make updates.
    3. Automatic scope management: Use context managers to handle scope entry and exit.
    4. Stack-based updates: New updates are added to the most recent scope.

    The context can be accessed and modified using the `context` property, which
    returns a ReadWriteProxyDict. This allows for easy reading of the entire context
    stack and writing to the current scope.

    Example usage:
    >>> engine = BaseEngine()
    >>> with engine.scope(initial_value=1):
    ...     engine.context['new_value'] = 2
    ...     with engine.scope(nested_value=3):
    ...         engine.context['deep_value'] = 4
    ...         print(engine.context)
    ...     print(engine.context)
    {'initial_value': 1, 'new_value': 2, 'nested_value': 3, 'deep_value': 4}
    {'initial_value': 1, 'new_value': 2}
    >>> print(engine.context)
    {}
    """

    # [scope_i: [update_i: {[key: str]: value}]]
    _all_updates: list[list[dict]] = Field(default_factory=list)

    @property
    def context(self):
        """
        Provides read-write access to the current context.

        Returns a ReadWriteProxyDict that allows reading from the entire context stack
        and writing to the most recent scope.

        Returns:
            ReadWriteProxyDict: A proxy object for accessing and modifying the context.

        Example:
        >>> engine = BaseEngine()
        >>> with engine.scope(initial_value=1):
        ...     engine.context['new_value'] = 2
        ...     print(engine.context)
        {'initial_value': 1, 'new_value': 2}
        """

        @cached_with_key(lambda: hash_mutable(self._all_updates))
        def get_read_context():
            all_updates = []
            for scope_updates in self._all_updates:
                all_updates.extend(scope_updates)
            return stack_dicts(all_updates)

        def get_write_context():
            new_update = dict()
            self._all_updates[-1].append(new_update)
            return new_update

        return ReadWriteProxyDict(get_read_context, get_write_context)

    @contextmanager
    def scope(self, config_overrides: dict = None, context_overrides: dict = None):
        """
        Creates a new context scope with optional initial overrides.

        This context manager ensures that the scope is properly entered and exited,
        maintaining the integrity of the context stack.

        Args:
            config (dict, optional): The configuration to use for the scope. Defaults to None.
            context_overrides (dict, optional): Initial key-value pairs to add to the new scope. Defaults to None.

        Yields:
            self: The current BaseEngine instance.

        Example:
        >>> engine = BaseEngine()
        >>> with engine.scope(config={'initial_value': 1}, context_overrides={'new_value': 2}):
        ...     engine.context['new_value'] = 2
        ...     print(engine.context)
        ...     with engine.scope(config={'nested_value': 3}, context_overrides={'deep_value': 4}):
        ...         print(engine.context)
        {'initial_value': 1, 'new_value': 2}
        {'initial_value': 1, 'new_value': 2, 'nested_value': 3}
        >>> print(engine.context)
        {}
        """
        prev_context = self.context
        orig_config = self.enter_scope(
            config_overrides=config_overrides,
            context_overrides=context_overrides,
        )
        try:
            yield self
        finally:
            self.exit_scope(origonal_config=orig_config)
            assert (
                prev_context == self.context
            ), "Stack imbalance: context should be the same before and after the scope"

    def enter_scope(
        self, config_overrides: dict = None, context_overrides: dict = None
    ):
        """
        Enters a new scope and adds initial overrides.

        This method is typically called by the `scope` context manager.

        Args:
            config_overides (dict, optional): The configuration to use for the scope. Defaults to None.
            context_overrides (dict, optional): Initial key-value pairs to add to the new scope. Defaults to None.

        Example:
        >>> engine = BaseEngine()
        >>> engine.enter_scope(initial_value=1)
        >>> print(engine.context)
        {'initial_value': 1}
        """
        new_scope_updates = list()
        self._all_updates.append(new_scope_updates)

        orig_config = {}
        if config_overrides is not None:
            orig_config = {key: getattr(self, key) for key in config_overrides}
            for k in config_overrides:
                setattr(self, k, config_overrides[k])

        if context_overrides is not None:
            self.log(**context_overrides)

        return orig_config

    def exit_scope(self, origonal_config: dict = None):
        """
        Exits the current scope and returns its updates.

        This method is typically called by the `scope` context manager.

        Args:
            origonal_config_settings (dict, optional): The original configuration settings to restore. Defaults to None.

        Returns:
            list[dict]: The list of update dictionaries from the exited scope.

        Example:
        >>> engine = BaseEngine()
        >>> engine.enter_scope(initial_value=1)
        >>> engine.context['new_value'] = 2
        >>> updates = engine.exit_scope()
        >>> print(updates)
        [{'initial_value': 1}, {'new_value': 2}]
        >>> print(engine.context)
        {}
        """
        if origonal_config is not None:
            for k in origonal_config:
                setattr(self, k, origonal_config[k])
        return self._all_updates.pop()

    #### Logging ####

    def command(
        self,
        command: Any,
        importance: float = 1.0,
        callstack_skip_frames=1,
        **updates,
    ):
        """
        Add a command update to the current context.

        Args:
            command (Any): The command to be added.
            importance (float, optional): The importance of the command. Defaults to 1.0.
            callstack_skip_frames (int, optional): The number of frames to skip in the callstack. Defaults to 1.
            **updates: Additional updates to be added alongside the command.

        Example:
        >>> engine = BaseEngine()
        >>> engine.command("do something", importance=0.8, extra_info="additional info")
        >>> print(engine.context)
        {'command': 'do something', 'importance': 0.8, 'extra_info': 'additional info', 'timestamp': <datetime>, 'callstack': <callstack>}
        """
        self.log(
            command=command,
            importance=importance,
            callstack_skip_frames=callstack_skip_frames,
            **updates,
        )

    def notes(
        self,
        notes: Any,
        importance: float = 1.0,
        *,
        callstack_skip_frames=1,
        **updates,
    ):
        """
        Add notes update to the current context.

        Args:
            notes (Any): The notes to be added.
            importance (float, optional): The importance of the notes. Defaults to 1.0.
            callstack_skip_frames (int, optional): The number of frames to skip in the callstack. Defaults to 1.
            **updates: Additional updates to be added alongside the notes.

        Example:
        >>> engine = BaseEngine()
        >>> engine.notes("some notes", importance=0.5, extra_info="additional info")
        >>> print(engine.context)
        {'notes': 'some notes', 'importance': 0.5, 'extra_info': 'additional info', 'timestamp': <datetime>, 'callstack': <callstack>}
        """
        self.log(
            notes=notes,
            importance=importance,
            callstack_skip_frames=callstack_skip_frames,
            **updates,
        )

    def feedback(
        self,
        feedback: Any,
        reward: float | None = None,
        importance: float = 1.0,
        *,
        callstack_skip_frames=1,
        **updates,
    ):
        """
        Add feedback update to the current context.

        Args:
            feedback (Any): The feedback to be added.
            reward (float | None, optional): The reward associated with the feedback. Defaults to None.
            importance (float, optional): The importance of the feedback. Defaults to 1.0.
            callstack_skip_frames (int, optional): The number of frames to skip in the callstack. Defaults to 1.
            **updates: Additional updates to be added alongside the feedback.

        Example:
        >>> engine = BaseEngine()
        >>> engine.feedback("good job", reward=1.0, importance=0.9, extra_info="additional info")
        >>> print(engine.context)
        {'feedback': 'good job', 'reward': 1.0, 'importance': 0.9, 'extra_info': 'additional info', 'timestamp': <datetime>, 'callstack': <callstack>}
        """
        if reward is not None:
            self.reward(reward)
        self.log(
            feedback=feedback,
            importance=importance,
            callstack_skip_frames=callstack_skip_frames,
            **updates,
        )

    def info(
        self,
        info: Any,
        importance: float = 1.0,
        *,
        callstack_skip_frames=1,
        **updates,
    ):
        """
        Add info update to the current context.

        Args:
            info (Any): The info to be added.
            importance (float, optional): The importance of the info. Defaults to 1.0.
            callstack_skip_frames (int, optional): The number of frames to skip in the callstack. Defaults to 1.
            **updates: Additional updates to be added alongside the info.

        Example:
        >>> engine = BaseEngine()
        >>> engine.info("some info", importance=0.7, extra_info="additional info")
        >>> print(engine.context)
        {'info': 'some info', 'importance': 0.7, 'extra_info': 'additional info', 'timestamp': <datetime>, 'callstack': <callstack>}
        """
        self.log(
            info=info,
            importance=importance,
            callstack_skip_frames=callstack_skip_frames,
            **updates,
        )

    def log(self, *, callstack_skip_frames=1, **updates):
        """
        Add arbitrary updates to the current context.

        Args:
            callstack_skip_frames (int, optional): The number of frames to skip in the callstack. Defaults to 1.
            **updates: The updates to be added to the current context.

        Example:
        >>> engine = BaseEngine()
        >>> engine.log(custom_update="some value", extra_info="additional info")
        >>> print(engine.context)
        {'custom_update': 'some value', 'extra_info': 'additional info', 'timestamp': <datetime>, 'callstack': <callstack>}
        """
        if self.add_default_log_meta:
            updates = {
                "timestamp": datetime.now(),
                "callstack": generate_callstack(skip_frames=callstack_skip_frames),
                **updates,
            }
        self.context.update(updates)

    ### Training and model management ###

    @abstractmethod
    def reward(self, reward: float):
        """
        Provide a reward signal to the engine.

        Args:
            reward (float): The reward value.
        """
        pass

    @abstractmethod
    def train(self):
        """
        Start the training process for the engine.
        """
        pass

    @abstractmethod
    def eval(self):
        """
        Set the engine to evaluation mode.
        """
        pass

    @classmethod
    def load(cls, path: str | Path):
        """
        Load the engine state from a file.

        Args:
            path (str | Path): The path to the file containing the engine state.

        Returns:
            cls: An instance of the engine with the loaded state.

        Example:
        >>> class MyEngine(BaseEngine):
        ...     pass
        >>> engine = MyEngine()
        >>> engine.save("engine_state.json")
        >>> loaded_engine = MyEngine.load("engine_state.json")
        >>> assert engine == loaded_engine
        """
        with open(path, "r") as f:
            return cls.model_validate_json(f.read())

    @abstractmethod
    def save(self, path: str | Path):
        """
        Save the current engine state to a file.

        Args:
            path (str | Path): The path where the engine state will be saved.

        Example:
        >>> class MyEngine(BaseEngine):
        ...     def save(self, path: str | Path):
        ...         with open(path, "w") as f:
        ...             f.write(self.model_dump_json())
        >>> engine = MyEngine()
        >>> engine.save("engine_state.json")
        """
        with open(path, "w") as f:
            f.write(self.model_dump_json())

    ### Decorators ###
    def trace(self):
        """
        Decorator for tracing function calls with detailed logging.

        This decorator provides comprehensive tracing capabilities for functions within the BaseEngine class.
        When applied to a method, it captures and logs detailed information about the function call, including:
        - Function name
        - Input arguments (both positional and keyword)
        - Return value
        - Timestamp of the function call
        - Callstack at the time of invocation

        The trace information is stored in the engine's context, allowing for easy access and analysis.

        Key features:
        1. Automatic argument binding: Captures all arguments, including default values.
        2. Scope management: Uses the engine's scope context manager to ensure proper context handling.
        3. Detailed logging: Logs function name, arguments, and result.
        4. Function call rendering: Provides a human-readable representation of the function call.
        5. Preserves function metadata: Uses @wraps to maintain the original function's metadata.

        Usage:
        Apply this decorator to any method in a BaseEngine subclass that you want to trace.

        Example:
        >>> class MyEngine(BaseEngine):
        ...     @trace()
        ...     def my_function(self, arg1, arg2):
        ...         return arg1 + arg2
        >>> engine = MyEngine()
        >>> result = engine.my_function(3, 4)
        >>> print(result)
        7
        >>> print(engine.context)
        {'__function__': 'my_function', 'arg1': 3, 'arg2': 4, '__result__': 7, 'timestamp': <datetime>, 'callstack': <callstack>}

        Note:
        The traced information is stored in the engine's context, which can be accessed and analyzed
        for debugging, profiling, or auditing purposes. The context includes the function name,
        arguments, result, timestamp, and callstack.
        """

        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                with self.scope():
                    # Capture function signature and bind arguments
                    signature = inspect.signature(fn)
                    bound_args = signature.bind_partial(*args, **kwargs)
                    bound_args.apply_defaults()

                    # Create a dictionary of named arguments
                    named_args = {k: v for k, v in bound_args.arguments.items()}

                    # Log function name and arguments
                    self.log(__function__=fn.__name__, **named_args)

                    # Render and log the function call
                    fn_call = render_function_call(fn, args=args, kwargs=kwargs)
                    self.info(fn_call)

                    # Execute the function and capture the result
                    result = fn(*args, **kwargs)

                    # Log the function result
                    self.log(__result__=result)

                    return result

            return wrapper

        return decorator
