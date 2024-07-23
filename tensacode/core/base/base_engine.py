from __future__ import annotations

# Standard library imports
from abc import abstractmethod, ABC
from contextlib import contextmanager
from datetime import datetime
from functools import cached_property, reduce, wraps
from pathlib import Path
import inspect
import traceback


# Typing imports
from typing import (
    Any,
    Callable,
    ClassVar,
    Hashable,
    Literal,
    Mapping,
    Optional,
    TypedDict,
)

# Third-party imports
from pydantic import Annotated, BaseModel, Field

# Local imports
from tensacode.core.base.latents.latents import LatentType
from tensacode.core.base.ops.base_op import BaseOp
from tensacode.internal.utils.consts import VERSION
from tensacode.internal.utils.functional import cached_with_key
from tensacode.internal.utils.language import Language
from tensacode.internal.utils.python_str import render_function_call
from tensacode.internal.utils.misc import (
    hash_mutable,
    generate_callstack,
    inheritance_distance,
    ReadWriteProxyDict,
    ReadWriteProxyList,
)
from tensacode.internal.protocols.latent import (
    are_latent_subtypes,
    latent_type_subtype_distance,
)


class BaseEngine(BaseModel):
    """
    BaseEngine: A comprehensive framework for AI engine management and operation execution.

    This class implements a sophisticated system for managing different types of AI engines
    and their associated operations. It provides a robust foundation for building complex AI
    systems with dynamic operation registration, inheritance-based matching, type-safe execution,
    and extensive context management and logging capabilities.

    Key Features:
    1. Flexible Engine Types: Supports multiple BaseEngine subclasses, each with unique operations.
    2. Dynamic Operation Management: Allows runtime registration of operation instances and classes.
    3. Inheritance-Based Matching: Finds the most specific operation based on engine, operation, and object type hierarchies.
    4. Type-Safe Execution: Ensures operations are bound to specific engine types for safe execution.
    5. Hierarchical Context Management: Provides a scoped context system for managing state and logging.
    6. Comprehensive Logging: Offers various logging methods for commands, notes, feedback, and general information.
    7. Abstract Training Interface: Defines abstract methods for reward handling, training, and evaluation.
    8. Serialization Support: Includes methods for saving and loading engine states.
    9. Tracing Capabilities: Provides a decorator for detailed function call tracing.

    Main Components:

    1. Operation Registration and Retrieval:
    - register_op_instance_for_this_object: Register an operation instance for a specific engine object.
    - register_op_class_for_this_object: Register an operation class for a specific engine object.
    - register_op_instance_for_all_class_instances: Register an operation instance for all instances of the engine class.
    - register_op_class_for_all_class_instances: Register an operation class for all instances of the engine class.
    - get_op: Retrieve the most specific operation instance matching given criteria.
    - get_op_cls: Retrieve the most specific operation class matching given criteria.
    - get_op_static: Class method to retrieve an operation instance at the class level.
    - get_op_cls_static: Class method to retrieve an operation class at the class level.

    2. Context Management:
    - context: Property providing read-write access to the current context.
    - scope: Context manager for creating nested scopes with optional overrides.
    - enter_scope: Method to enter a new scope with initial overrides.
    - exit_scope: Method to exit the current scope and return its updates.

    3. Logging System:
    - command: Log a command with associated metadata.
    - notes: Log notes with associated metadata.
    - feedback: Log feedback, optionally with a reward signal.
    - info: Log general information with associated metadata.
    - log: Base method for adding arbitrary updates to the current context.

    4. Training and Model Management:
    - reward: Abstract method to provide a reward signal to the engine.
    - train: Abstract method to start the training process for the engine.
    - eval: Abstract method to set the engine to evaluation mode.

    5. Serialization:
    - load: Class method to load the engine state from a file.
    - save: Abstract method to save the current engine state to a file.

    6. Tracing:
    - trace: Method to create a decorator for detailed function call tracing.

    Usage Example:
    >>> class TextEngine(BaseEngine):
    ...     latent_type = TextLatent
    ...
    >>> class SummarizeOp(BaseOp):
    ...     name: str = "summarize"
    ...     latent_type: type[LatentType] = TextLatent
    ...     engine_type: type[BaseEngine] = TextEngine
    ...     object_type: type[Any] = str
    ...
    ...     def execute(self, engine: TextEngine, text: str, **kwargs):
    ...         return f"Summary of: {text[:10]}..."
    ...
    >>> TextEngine.register_op_class_for_all_class_instances(SummarizeOp)
    >>> engine = TextEngine()
    >>> summarize_op = engine.get_op(SummarizeOp)
    >>> summary = summarize_op.execute(engine, text="Long text to summarize")
    >>> print(summary)
    Summary of: Long text ...

    >>> with engine.scope(context_overrides={"task": "summarization"}):
    ...     engine.command("Summarize text", importance=0.8)
    ...     result = summarize_op.execute(engine, text="Another long text")
    ...     engine.feedback("Good summary", reward=0.9)
    ...
    >>> print(engine.context)
    {'task': 'summarization', 'command': 'Summarize text', 'importance': 0.8, 'feedback': 'Good summary', 'reward': 0.9}

    Inheritance and Customization:
    To create a custom engine, inherit from BaseEngine and implement the abstract methods:

    >>> class MyCustomEngine(BaseEngine):
    ...     def reward(self, reward: float):
    ...         # Implement reward handling
    ...         pass
    ...
    ...     def train(self):
    ...         # Implement training process
    ...         pass
    ...
    ...     def eval(self):
    ...         # Implement evaluation mode setting
    ...         pass
    ...
    ...     def save(self, path: str | Path):
    ...         # Implement state saving
    ...         super().save(path)

    This BaseEngine class serves as a powerful foundation for building complex AI systems,
    providing a structured approach to managing operations, contexts, and engine-specific
    functionalities while maintaining flexibility and extensibility. It enables the creation
    of sophisticated AI engines with rich logging, context management, and operation execution
    capabilities.
    """

    tensacode_version: ClassVar[str] = VERSION
    render_language: Language = "python"
    latent_type: type[LatentType]
    add_default_log_meta = True

    ### Operations ###

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
        return op_class

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
        return op_class

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
        return ReadWriteProxyList(
            read_list_getter=lambda: (
                self._op_instances_for_this_object
                + self.get_class_level_registered_op_instances()
            ),
            write_list_getter=lambda: self._op_instances_for_this_object,
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
        return ReadWriteProxyList(
            read_list_getter=lambda: (
                self._op_classes_for_this_object
                + self.get_class_level_registered_op_classes()
            ),
            write_list_getter=lambda: self._op_classes_for_this_object,
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
        operator_name: str | None = None,
        operator_type: type[BaseOp] | None = None,
        latent_type: type[LatentType] | None = None,
        register_if_new: bool = True,
        op_args: tuple = None,
        op_kwargs: dict = None,
    ) -> BaseOp:
        """
        Get an operation instance based on the provided criteria.

        This method searches for an existing operation instance that matches the given criteria.
        If no matching instance is found, it creates a new one and optionally registers it.

        Args:
            operator_name (str | None): The name of the operator to retrieve.
            operator_type (type[BaseOp] | None): The type of the operator to retrieve.
            latent_type (type[LatentType] | None): The latent type associated with the operator.
            register_if_new (bool): Whether to register the new instance if created.
            op_args (tuple): Additional positional arguments for the operator.
            op_kwargs (dict): Additional keyword arguments for the operator.

        Returns:
            BaseOp: An instance of the requested operation.

        Raises:
            ValueError: If no matching operation is found and a new one cannot be created.

        Example:
        >>> class MyEngine(BaseEngine):
        ...     pass
        >>> class MyOp(BaseOp):
        ...     name = "my_op"
        ...     latent_type = str
        >>> engine = MyEngine()
        >>> engine.register_op_class_for_this_object(MyOp)
        >>> op = engine.get_op(operator_name="my_op")
        >>> isinstance(op, MyOp)
        True
        >>> op2 = engine.get_op(operator_type=MyOp)
        >>> isinstance(op2, MyOp)
        True
        >>> op3 = engine.get_op(latent_type=str)
        >>> isinstance(op3, MyOp)
        True
        """
        return self._get_op(
            self.get_all_registered_op_instances,
            self.get_op_cls,
            self.register_op_instance_for_this_object,
            operator_name,
            operator_type,
            latent_type,
            register_if_new,
            op_args,
            op_kwargs,
        )

    def get_op_cls(
        self,
        operator_name: str | None = None,
        operator_type: type[BaseOp] | None = None,
        latent_type: type[LatentType] | None = None,
        op_args: tuple = None,
        op_kwargs: dict = None,
    ) -> type[BaseOp]:
        """
        Get an operation class based on the provided criteria.

        This method searches for an existing operation class that matches the given criteria.

        Args:
            operator_name (str | None): The name of the operator class to retrieve.
            operator_type (type[BaseOp] | None): The type of the operator class to retrieve.
            latent_type (type[LatentType] | None): The latent type associated with the operator class.
            op_args (tuple): Additional positional arguments for matching.
            op_kwargs (dict): Additional keyword arguments for matching.

        Returns:
            type[BaseOp]: The class of the requested operation.

        Raises:
            ValueError: If no matching operation class is found.

        Example:
        >>> class MyEngine(BaseEngine):
        ...     pass
        >>> class MyOp(BaseOp):
        ...     name = "my_op"
        ...     latent_type = str
        >>> engine = MyEngine()
        >>> engine.register_op_class_for_this_object(MyOp)
        >>> op_cls = engine.get_op_cls(operator_name="my_op")
        >>> op_cls == MyOp
        True
        >>> op_cls2 = engine.get_op_cls(operator_type=MyOp)
        >>> op_cls2 == MyOp
        True
        >>> op_cls3 = engine.get_op_cls(latent_type=str)
        >>> op_cls3 == MyOp
        True
        """
        return self._get_op_cls(
            self.get_all_registered_op_classes,
            operator_name,
            operator_type,
            latent_type,
            op_args,
            op_kwargs,
        )

    @classmethod
    def get_op_static(
        cls,
        operator_name: str | None = None,
        operator_type: type[BaseOp] | None = None,
        latent_type: type[LatentType] | None = None,
        register_if_new: bool = True,
        op_args: tuple = None,
        op_kwargs: dict = None,
    ) -> BaseOp:
        """
        Get an operation instance based on the provided criteria using class-level registrations.

        This class method searches for an existing operation instance that matches the given criteria
        using class-level registrations. If no matching instance is found, it creates a new one and
        optionally registers it at the class level.

        Args:
            operator_name (str | None): The name of the operator to retrieve.
            operator_type (type[BaseOp] | None): The type of the operator to retrieve.
            latent_type (type[LatentType] | None): The latent type associated with the operator.
            register_if_new (bool): Whether to register the new instance if created.
            op_args (tuple): Additional positional arguments for the operator.
            op_kwargs (dict): Additional keyword arguments for the operator.

        Returns:
            BaseOp: An instance of the requested operation.

        Raises:
            ValueError: If no matching operation is found and a new one cannot be created.

        Example:
        >>> class MyEngine(BaseEngine):
        ...     pass
        >>> class MyOp(BaseOp):
        ...     name = "my_op"
        ...     latent_type = str
        >>> MyEngine.register_op_class_for_all_class_instances(MyOp)
        >>> op = MyEngine.get_op_static(operator_name="my_op")
        >>> isinstance(op, MyOp)
        True
        >>> op2 = MyEngine.get_op_static(operator_type=MyOp)
        >>> isinstance(op2, MyOp)
        True
        >>> op3 = MyEngine.get_op_static(latent_type=str)
        >>> isinstance(op3, MyOp)
        True
        """
        return cls._get_op(
            cls.get_class_level_registered_op_instances,
            cls.get_op_cls_static,
            cls.register_op_instance_for_all_class_instances,
            operator_name,
            operator_type,
            latent_type,
            register_if_new,
            op_args,
            op_kwargs,
        )

    @classmethod
    def get_op_cls_static(
        cls,
        operator_name: str | None = None,
        operator_type: type[BaseOp] | None = None,
        latent_type: type[LatentType] | None = None,
        op_args: tuple = None,
        op_kwargs: dict = None,
    ) -> type[BaseOp]:
        """
        Get an operation class based on the provided criteria using class-level registrations.

        This class method searches for an existing operation class that matches the given criteria
        using class-level registrations.

        Args:
            operator_name (str | None): The name of the operator class to retrieve.
            operator_type (type[BaseOp] | None): The type of the operator class to retrieve.
            latent_type (type[LatentType] | None): The latent type associated with the operator class.
            op_args (tuple): Additional positional arguments for matching.
            op_kwargs (dict): Additional keyword arguments for matching.

        Returns:
            type[BaseOp]: The class of the requested operation.

        Raises:
            ValueError: If no matching operation class is found.

        Example:
        >>> class MyEngine(BaseEngine):
        ...     pass
        >>> class MyOp(BaseOp):
        ...     name = "my_op"
        ...     latent_type = str
        >>> MyEngine.register_op_class_for_all_class_instances(MyOp)
        >>> op_cls = MyEngine.get_op_cls_static(operator_name="my_op")
        >>> op_cls == MyOp
        True
        >>> op_cls2 = MyEngine.get_op_cls_static(operator_type=MyOp)
        >>> op_cls2 == MyOp
        True
        >>> op_cls3 = MyEngine.get_op_cls_static(latent_type=str)
        >>> op_cls3 == MyOp
        True
        """
        return cls._get_op_cls(
            cls.get_class_level_registered_op_classes,
            operator_name,
            operator_type,
            latent_type,
            op_args,
            op_kwargs,
        )

    @classmethod
    def _get_op(
        cls,
        get_registered_ops,
        get_op_cls,
        register_op,
        operator_name: str | None = None,
        operator_type: type[BaseOp] | None = None,
        latent_type: type[LatentType] | None = None,
        register_if_new: bool = True,
        op_args: tuple = None,
        op_kwargs: dict = None,
    ) -> BaseOp:
        assert (
            operator_name is not None or operator_type is not None
        ), "Either operator_name or operator_type must be provided"
        assert not (
            operator_name and operator_type
        ), "Only one of operator_name or operator_type can be provided"

        if latent_type is None:
            latent_type = cls.latent_type
        if op_args is None:
            op_args = ()
        if op_kwargs is None:
            op_kwargs = {}

        op_instances = get_registered_ops()
        op_instances = cls._filter_ops(
            op_instances, operator_type, operator_name, latent_type
        )

        if not op_instances:
            op_cls = get_op_cls(
                operator_name=operator_name,
                operator_type=operator_type,
                latent_type=latent_type,
                op_args=op_args,
                op_kwargs=op_kwargs,
            )
            op_instance = op_cls()
            if register_if_new:
                register_op(op_instance)
            return op_instance

        return cls._sort_ops(op_instances, op_args, op_kwargs)[0]

    @classmethod
    def _get_op_cls(
        cls,
        get_registered_op_classes,
        operator_name: str | None = None,
        operator_type: type[BaseOp] | None = None,
        latent_type: type[LatentType] | None = None,
        op_args: tuple = None,
        op_kwargs: dict = None,
    ) -> type[BaseOp]:
        assert (
            operator_name is not None or operator_type is not None
        ), "Either operator_name or operator_type must be provided"
        assert not (
            operator_name and operator_type
        ), "Only one of operator_name or operator_type can be provided"

        if latent_type is None:
            latent_type = cls.latent_type
        if op_args is None:
            op_args = ()
        if op_kwargs is None:
            op_kwargs = {}

        op_classes = get_registered_op_classes()
        op_classes = cls._filter_ops(
            op_classes, operator_type, operator_name, latent_type, is_class=True
        )

        if not op_classes:
            raise ValueError(
                f"No matching operation class found for {operator_type or operator_name} and {latent_type}"
            )

        return cls._sort_ops(op_classes, op_args, op_kwargs)[0]

    @classmethod
    def _filter_ops(
        cls, ops, operator_type, operator_name, latent_type, is_class=False
    ):
        if operator_type is not None:
            ops = filter(
                lambda op: (
                    issubclass(op, operator_type)
                    if is_class
                    else isinstance(op, operator_type)
                ),
                ops,
            )
        if operator_name is not None:
            ops = filter(lambda op: op.name == operator_name, ops)
        if latent_type is not None:
            ops = filter(lambda op: op.latent_type == latent_type, ops)
        return list(ops)

    @classmethod
    def _sort_ops(cls, ops, op_args, op_kwargs):
        return sorted(
            ops,
            key=lambda op: op.match_score_fn(*op_args, **op_kwargs),
            reverse=True,
        )

    ### Context Management ###

    # [scope_i: [update_i: {[key: str]: value}]]
    _all_updates: list[list[dict]] = Field(
        default_factory=list,
        description=(
            "List of dict updates for each scope. The outer list is the list of "
            "scopes, and the inner list is the list of updates for each scope."
        ),
        exclude=False,
        examples=[
            [
                # example 1
                [{"c": 3, "d": 4}],
                [],
                [{"e": 5, "f": 6}, {"g": 7, "h": 8}],
                [{"i": 9, "j": 10}],
            ],
            [
                # example 2
                [{"g": 7}, {}, {"a": 1, "b": 2}],
                [{"c": 3, "d": 4}],
                [{"e": 5, "f": 6}, {"g": 7, "h": 8}],
            ],
        ],
    )

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
    def scope(
        self,
        config_overrides: dict = None,
        context_overrides: dict = None,
        **more_context_overrides,
    ):
        """
        Creates a new context scope with optional initial overrides.

        This context manager ensures that the scope is properly entered and exited,
        maintaining the integrity of the context stack.

        Args:
            config (dict, optional): The configuration to use for the scope. Defaults to None.
            context_overrides (dict, optional): Initial key-value pairs to add to the new scope. Defaults to None.
            **more_context_overrides: Additional key-value pairs to add to the new scope.

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
            **more_context_overrides,
            _callstack_skip_frames=3,
        )
        try:
            yield self
        finally:
            self.exit_scope(origonal_config=orig_config)
            assert (
                prev_context == self.context
            ), "Stack imbalance: context should be the same before and after the scope"

    def enter_scope(
        self,
        config_overrides: dict = None,
        context_overrides: dict = None,
        *,
        _callstack_skip_frames=2,
        **more_context_overrides,
    ):
        """
        Enters a new scope and adds initial overrides.

        This method is typically called by the `scope` context manager.

        Args:
            config_overides (dict, optional): The configuration to use for the scope. Defaults to None.
            context_overrides (dict, optional): Initial key-value pairs to add to the new scope. Defaults to None.
            _callstack_skip_frames (int, optional): The number of frames to skip in the callstack. Defaults to 2.
            **more_context_overrides: Additional key-value pairs to add to the new scope.

        Example:
        >>> engine = BaseEngine()
        >>> engine.enter_scope(initial_value=1)
        >>> print(engine.context)
        {'initial_value': 1}
        """

        new_scope_updates = list()
        self._all_updates.append(new_scope_updates)

        orig_config = {}
        if config_overrides:
            orig_config = {key: getattr(self, key) for key in config_overrides}
            for k in config_overrides:
                setattr(self, k, config_overrides[k])

        if context_overrides:
            context_overrides = {**context_overrides, **more_context_overrides}
            self.log(_callstack_skip_frames=_callstack_skip_frames, **context_overrides)

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
    def prompt(
        self,
        prompt: Any,
        importance: float = 1.0,
        _callstack_skip_frames=1,
        **updates,
    ):
        """
        Add a prompt update to the current context.

        Args:
            prompt (Any): The prompt to be added.
            importance (float, optional): The importance of the prompt. Defaults to 1.0.
            _callstack_skip_frames (int, optional): The number of frames to skip in the callstack. Defaults to 1.
            **updates: Additional updates to be added alongside the prompt.

        Example:
        >>> engine = BaseEngine()
        >>> engine.prompt("What should I do?", importance=0.8, extra_info="additional info")
        >>> print(engine.context)
        {'prompt': 'What should I do?', 'importance': 0.8, 'extra_info': 'additional info', 'timestamp': <datetime>, 'callstack': <callstack>}
        """
        self.log(
            prompt=prompt,
            importance=importance,
            **updates,
            _callstack_skip_frames=_callstack_skip_frames,
        )

    def feedback(
        self,
        feedback: Any,
        reward: float | None = None,
        importance: float = 1.0,
        *,
        _callstack_skip_frames=1,
        **updates,
    ):
        """
        Add feedback update to the current context.

        Args:
            feedback (Any): The feedback to be added.
            reward (float | None, optional): The reward associated with the feedback. Defaults to None.
            importance (float, optional): The importance of the feedback. Defaults to 1.0.
            _callstack_skip_frames (int, optional): The number of frames to skip in the callstack. Defaults to 1.
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
            **updates,
            _callstack_skip_frames=_callstack_skip_frames,
        )

    def info(
        self,
        info: Any,
        importance: float = 1.0,
        *,
        _callstack_skip_frames=1,
        **updates,
    ):
        """
        Add info update to the current context.

        Args:
            info (Any): The info to be added.
            importance (float, optional): The importance of the info. Defaults to 1.0.
            _callstack_skip_frames (int, optional): The number of frames to skip in the callstack. Defaults to 1.
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
            **updates,
            _callstack_skip_frames=_callstack_skip_frames,
        )

    def log(self, *, _callstack_skip_frames=1, **updates):
        """
        Add arbitrary updates to the current context.

        Args:
            _callstack_skip_frames (int, optional): The number of frames to skip in the callstack. Defaults to 1.
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
                "callstack": generate_callstack(skip_frames=_callstack_skip_frames),
                **updates,
            }
        self.context.update(updates)

    ### Training and model management ###

    @property
    def latent(self) -> LatentType:
        """
        Get the current latent state of the engine, calculated by encoding the context.
        """
        return self.encode(self.context)

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
    def trace(self, *, context_overrides=None, config_overrides=None):
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
                return self.trace_execution(
                    fn,
                    args,
                    kwargs,
                    context_overrides,
                    config_overrides,
                )

            return wrapper

        return decorator

    def trace_execution(
        self,
        fn,
        args,
        kwargs,
        fn_name_override=None,
        context_overrides=None,
        config_overrides=None,
    ):
        """
        Trace the execution of a function.
        """
        with self.scope(
            context_overrides=context_overrides, config_overrides=config_overrides
        ):
            # Capture function signature and bind arguments
            signature = inspect.signature(fn)
            bound_args = signature.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()

            # Create a dictionary of named arguments
            named_args = {k: v for k, v in bound_args.arguments.items()}

            # Log function name and arguments
            self.log(__function__=fn_name_override or fn.__name__, **named_args)

            # Render and log the function call
            fn_call = render_function_call(fn, args=args, kwargs=kwargs)
            self.info(fn_call)

            # Execute the function and capture the result
            result = fn(*args, **kwargs)

            # Log the function result
            self.log(__result__=result)

            return result

    ## Specific Operations

    def call(self, *args, **kwargs):
        """Call an operation"""
        call_op = self.get_op(
            operator_name="call",
            op_args=args,
            op_kwargs=kwargs,
        )
        return call_op._execute(*args, **kwargs)

    def choice(self, *args, **kwargs):
        """Choose one of the options"""
        choose_op = self.get_op(
            operator_name="choice",
            op_args=args,
            op_kwargs=kwargs,
        )
        return choose_op._execute(*args, **kwargs)

    def convert(self, *args, **kwargs):
        """Convert an object"""
        convert_op = self.get_op(
            operator_name="convert",
            op_args=args,
            op_kwargs=kwargs,
        )
        return convert_op._execute(*args, **kwargs)

    def correct(self, *args, **kwargs):
        """Correct an object"""
        correct_op = self.get_op(
            operator_name="correct",
            op_args=args,
            op_kwargs=kwargs,
        )
        return correct_op._execute(*args, **kwargs)

    def decide(self, *args, **kwargs):
        """Make a decision"""
        decide_op = self.get_op(
            operator_name="decide",
            op_args=args,
            op_kwargs=kwargs,
        )
        return decide_op._execute(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Decode an object"""
        decode_op = self.get_op(
            operator_name="decode",
            op_args=args,
            op_kwargs=kwargs,
        )
        return decode_op._execute(*args, **kwargs)

    def encode(self, *args, **kwargs):
        """Encode an object"""
        encode_op = self.get_op(
            operator_name="encode",
            op_args=args,
            op_kwargs=kwargs,
        )
        return encode_op._execute(*args, **kwargs)

    def merge(self, *args, **kwargs):
        """Merge objects"""
        merge_op = self.get_op(
            operator_name="merge",
            op_args=args,
            op_kwargs=kwargs,
        )
        return merge_op._execute(*args, **kwargs)

    def modify(self, *args, **kwargs):
        """Modify an object"""
        modify_op = self.get_op(
            operator_name="modify",
            op_args=args,
            op_kwargs=kwargs,
        )
        return modify_op._execute(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """Make a prediction"""
        predict_op = self.get_op(
            operator_name="predict",
            op_args=args,
            op_kwargs=kwargs,
        )
        return predict_op._execute(*args, **kwargs)

    def program(self, *args, **kwargs):
        """Execute a program"""
        program_op = self.get_op(
            operator_name="program",
            op_args=args,
            op_kwargs=kwargs,
        )
        return program_op._execute(*args, **kwargs)

    def query(self, *args, **kwargs):
        """Query an object"""
        query_op = self.get_op(
            operator_name="query",
            op_args=args,
            op_kwargs=kwargs,
        )
        return query_op._execute(*args, **kwargs)

    def run(self, *args, **kwargs):
        """Run an operation"""
        run_op = self.get_op(
            operator_name="run",
            op_args=args,
            op_kwargs=kwargs,
        )
        return run_op._execute(*args, **kwargs)

    def select(self, *args, **kwargs):
        """Select an object"""
        select_op = self.get_op(
            operator_name="select",
            op_args=args,
            op_kwargs=kwargs,
        )
        return select_op._execute(*args, **kwargs)

    def semantic_transfer(self, *args, **kwargs):
        """Perform semantic transfer"""
        semantic_transfer_op = self.get_op(
            operator_name="semantic_transfer",
            op_args=args,
            op_kwargs=kwargs,
        )
        return semantic_transfer_op._execute(*args, **kwargs)

    def similarity(self, *args, **kwargs):
        """Calculate similarity"""
        similarity_op = self.get_op(
            operator_name="similarity",
            op_args=args,
            op_kwargs=kwargs,
        )
        return similarity_op._execute(*args, **kwargs)

    def split(self, *args, **kwargs):
        """Split an object"""
        split_op = self.get_op(
            operator_name="split",
            op_args=args,
            op_kwargs=kwargs,
        )
        return split_op._execute(*args, **kwargs)

    def store(self, *args, **kwargs):
        """Store an object"""
        store_op = self.get_op(
            operator_name="store",
            op_args=args,
            op_kwargs=kwargs,
        )
        return store_op._execute(*args, **kwargs)
