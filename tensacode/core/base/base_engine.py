from __future__ import annotations

# Standard library imports
from abc import abstractmethod, ABC
from contextlib import contextmanager
from datetime import datetime
from functools import cached_property, reduce, wraps, partial
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
    conditional_ctx_manager,
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

    latent_type: ClassVar[type[LatentType]]
    add_default_log_meta = True

    ### Operations ###

    _ops: ClassVar[dict[str, list[tuple[Callable, Callable]]]] = {}

    @classmethod
    def register_op(
        cls,
        name: str = None,
        score_fn: Callable = lambda: 0,
        fn: Callable = None,
    ):
        """
        Register an operation with the engine class.

        Args:
            name (str, optional): The name of the operation. If not supplied, the function being called will be used as the name.
            score_fn (Callable, optional): A function that returns a score for how well this operation matches the current context.
            fn (Callable, optional): The function to be executed when this operation is called. If not supplied,
                this method turns into a decorator that can be used to register a function as an operation.

        Example:
        >>> def always_match(engine, *args, **kwargs):
        ...     return 1.0
        >>> def echo(engine, *args, **kwargs):
        ...     return args, kwargs
        >>> BaseEngine.register_op("echo", always_match, echo)
        """
        if fn is None:
            return partial(cls.register_op, name, score_fn)

        if name not in cls._ops:
            cls._ops[name] = []
        cls._ops[name].append((score_fn, fn))

    @classmethod
    def get_op(cls, name: str, *args, **kwargs) -> Callable:
        """
        Get the best matching operation for the given name and arguments.

        Args:
            name (str): The name of the operation to retrieve.
            *args: Positional arguments to pass to the score function.
            **kwargs: Keyword arguments to pass to the score function.

        Returns:
            Callable: The best matching operation function.

        Raises:
            ValueError: If no matching operation is found.

        Example:
        >>> def always_match(engine, *args, **kwargs):
        ...     return 1.0
        >>> def echo(engine, *args, **kwargs):
        ...     return args, kwargs
        >>> BaseEngine.register_op("echo", always_match, echo)
        >>> op = BaseEngine.get_op("echo")
        >>> op(BaseEngine(), "test")
        (('test',), {})
        """
        if name not in cls._ops:
            raise ValueError(f"No operation named '{name}' found")

        best_score = float("-inf")
        best_fn = None

        for score_fn, fn in cls._ops[name]:
            score = call_with_appropriate_args(score_fn, self, *args, **kwargs)
            if score and score >= best_score:
                best_score = score
                best_fn = fn

        if best_fn is None:
            raise ValueError(f"No matching operation found for '{name}'")

        return best_fn

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

    def __init__(self, **data):
        super().__init__(**data)
        self._all_updates = []

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
            state = {}
            for scope_updates in self._all_updates:
                for update in scope_updates:
                    state.update(update)
            return state

        def get_write_context():
            new_update = dict()
            self._all_updates[-1].append(new_update)
            return new_update

        return ReadWriteProxyDict(get_read_context, get_write_context)

    @contextmanager
    def scope(
        self,
        context_overrides: dict = None,
        **more_context_overrides,
    ):
        """
        Creates a new context scope with optional initial overrides.

        This context manager ensures that the scope is properly entered and exited,
        maintaining the integrity of the context stack.

        Args:
            context_overrides (dict, optional): Initial key-value pairs to add to the new scope. Defaults to None.
            **more_context_overrides: Additional key-value pairs to add to the new scope.

        Yields:
            self: The current BaseEngine instance.

        Example:
        >>> engine = BaseEngine()
        >>> with engine.scope(context_overrides={'new_value': 2}):
        ...     engine.context['new_value'] = 2
        ...     print(engine.context)
        ...     with engine.scope(context_overrides={'deep_value': 4}):
        ...         print(engine.context)
        {'new_value': 2}
        {'new_value': 2, 'deep_value': 4}
        >>> print(engine.context)
        {}
        """
        prev_context = self.context
        self.enter_scope(
            context_overrides=context_overrides,
            **more_context_overrides,
            _callstack_skip_frames=3,
        )
        try:
            yield self
        finally:
            self.exit_scope()
            assert (
                prev_context == self.context
            ), "Stack imbalance: context should be the same before and after the scope"

    def enter_scope(
        self,
        *,
        _callstack_skip_frames=2,
        **more_context_overrides,
    ):
        """
        Enters a new scope and adds initial overrides.

        This method is typically called by the `scope` context manager.

        Args:
            _callstack_skip_frames (int, optional): The number of frames to skip in the callstack. Defaults to 2.
            **more_context_overrides: Additional key-value pairs to add to the new scope.

        Example:
        >>> engine = BaseEngine()
        >>> engine.enter_scope(initial_value=1)
        >>> print(engine.context)
        {'initial_value': 1}
        """

        self._all_updates.append([])
        self.log(
            _callstack_skip_frames=_callstack_skip_frames,
            **more_context_overrides,
        )

    def exit_scope(self):
        """
        Exits the current scope and returns its updates.

        This method is typically called by the `scope` context manager.

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
        return self._all_updates.pop()

    #### Logging ####
    def prompt(
        self,
        prompt: Any = None,
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
        feedback: Any = None,
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
            self.reward(reward=reward, importance=importance)
        self.log(
            feedback=feedback,
            importance=importance,
            **updates,
            _callstack_skip_frames=_callstack_skip_frames,
        )

    def info(
        self,
        info: Any = None,
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
        self._all_updates[-1].append(updates)

    ### Training and model management ###

    @property
    def latent(self) -> LatentType:
        """
        Get the current latent state of the engine, calculated by encoding the context.
        """
        return self.encode(self.context)

    @abstractmethod
    def reward(self, reward: float, importance: float = 1.0):
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
        with open(path, "r", encoding="utf-8") as f:
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
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json())

    ### Decorators ###
    def trace_wrapper(self, *, context_overrides=None, config_overrides=None):
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
        ...     @trace_wrapper()
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
        new_scope=True,
    ):
        """
        Trace the execution of a function.
        """
        with conditional_ctx_manager(
            new_scope,
            lambda: self.scope(
                context_overrides=context_overrides,
                config_overrides=config_overrides,
            ),
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

    def _execute_op(self, name: str, *args, **kwargs):
        """Helper method to execute a specific operation"""
        op = self.get_op(name, *args, **kwargs)
        if op is None:
            raise ValueError(f"Operation {name} not found")
        return op(self, *args, **kwargs)

    def blend(self, *args, **kwargs):
        """Blend objects"""
        return self._execute_op("blend", *args, **kwargs)

    def call(self, *args, **kwargs):
        """Call an operation"""
        return self._execute_op("call", *args, **kwargs)

    def convert(self, *args, **kwargs):
        """Convert an object"""
        return self._execute_op("convert", *args, **kwargs)

    def correct(self, *args, **kwargs):
        """Correct an object"""
        return self._execute_op("correct", *args, **kwargs)

    def decide(self, *args, **kwargs):
        """Make a decision"""
        return self._execute_op("decide", *args, **kwargs)

    def decode(self, *args, **kwargs):
        """Decode an object"""
        return self._execute_op("decode", *args, **kwargs)

    def encode(self, *args, **kwargs):
        """Encode an object"""
        return self._execute_op("encode", *args, **kwargs)

    def locate(self, *args, **kwargs):
        """Locate an object"""
        return self._execute_op("locate", *args, **kwargs)

    def loop(self, *args, **kwargs):
        """Execute a loop operation"""
        return self._execute_op("loop", *args, **kwargs)

    def modify(self, *args, **kwargs):
        """Modify an object"""
        return self._execute_op("modify", *args, **kwargs)

    def plan(self, *args, **kwargs):
        """Execute a planning operation"""
        return self._execute_op("plan", *args, **kwargs)

    def predict(self, *args, **kwargs):
        """Make a prediction"""
        return self._execute_op("predict", *args, **kwargs)

    def program(self, *args, **kwargs):
        """Execute a program"""
        return self._execute_op("program", *args, **kwargs)

    def query(self, *args, **kwargs):
        """Query an object"""
        return self._execute_op("query", *args, **kwargs)

    def query_or_create(self, *args, **kwargs):
        """Execute a query or create operation"""
        return self._execute_op("query_or_create", *args, **kwargs)

    def select(self, *args, **kwargs):
        """Select an object"""
        return self._execute_op("select", *args, **kwargs)

    def similarity(self, *args, **kwargs):
        """Calculate similarity"""
        return self._execute_op("similarity", *args, **kwargs)

    def split(self, *args, **kwargs):
        """Split an object"""
        return self._execute_op("split", *args, **kwargs)

    def transform(self, *args, **kwargs):
        """Transform an object"""
        return self._execute_op("transform", *args, **kwargs)
