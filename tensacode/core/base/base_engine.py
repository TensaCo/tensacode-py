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
from tensacode.internal.utils.python_str import render_function_call


def generate_callstack(skip_frames=1):
    return [
        {
            "fn_name": frame.function,
            "params": {
                **dict(
                    zip(
                        inspect.getargvalues(frame[0]).args,
                        inspect.getargvalues(frame[0]).locals.values(),
                    )
                ),
                **{
                    k: v
                    for k, v in inspect.getargvalues(frame[0]).locals.items()
                    if k not in inspect.getargvalues(frame[0]).args
                },
            },
        }
        for frame in inspect.stack()[skip_frames:]
    ]


class Update(BaseModel, ABC):
    type: ClassVar[str]
    timestamp: datetime = Field(default_factory=datetime.now)
    callstack: list[dict] = Field(default_factory=generate_callstack)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True


class EngineContext(BaseModel):
    updates: list[dict[str, Any]] = Field(default_factory=list)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    callstack: list[dict] = Field()

    @property
    @cached_with_key(lambda self: hash(tuple(self.updates)))
    def cumulative_updates(self):
        update_data = {}
        for update in self.updates:
            update_data.update(update)
        return Update(**update_data)

    def add_update(self, arg=None, /, **update: dict[str, Any] | BaseModel):
        if arg is not None:
            update = arg
        if isinstance(update, dict):
            pass
        elif isinstance(update, BaseModel):
            update = update.model_dump()
        elif isinstance(update, object):
            update = {k: getattr(update, k) for k in dir(update)}
        else:
            raise ValueError(f"Invalid update type: {type(update)}")
        self.updates.append(update)

    def extend_updates(self, updates: list[dict[str, Any]]):
        for update in updates:
            self.add_update(update)

    @cached_with_key(lambda self: hash((self.started_at, self.ended_at)))
    def duration(self):
        if self.started_at is None:
            return 0
        if self.ended_at is None:
            return datetime.now() - self.started_at
        return self.ended_at - self.started_at

    def __enter__(self):
        self.started_at = datetime.now()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.ended_at = datetime.now()


class BaseEngine(HasID, BaseModel):

    tensacode_version: ClassVar[str] = VERSION
    render_language: Language = "python"
    ops_registry: dict[str, BaseOp] = Field(default_factory=dict)

    ### Context Management ###

    context_stack: list[EngineContext] = Field(default_factory=list)

    @cached_with_key(
        # recursively hash the context stack up to the index that way even if
        # upper contexts change, the cache remains the same for lower contexts
        lambda self, index: hash(
            tuple(context.cumulative_updates for context in self.context_stack[:index])
        )
    )
    def cumulative_context(self, index=-1) -> EngineContext:
        # Convert negative index to positive index
        if index < 0:
            index = len(self.context_stack) + index

        # Check if index is out of range
        if index > len(self.context_stack) - 1:
            raise IndexError(f"Index {index} out of range")

        cumulative_context = EngineContext()
        current_index = 0

        while current_index <= index:
            context = self.context_stack[current_index]
            cumulative_context.extend_updates(context.updates)
            current_index += 1

        return cumulative_context

    @contextmanager
    def subcontext(self, **overrides):
        new_context = self._context_start(**overrides)
        try:
            yield self
        finally:
            old_context = self._context_stop()
            assert (
                new_context == old_context
            ), "Context mismatch. Did you modify `engine.context_stack` while the `engine.subcontext` context manager was active?"

    def _context_start(self, **overrides):
        new_context = EngineContext(updates=overrides, callstack=generate_callstack(2))
        self.context_stack.append(new_context)
        return new_context

    def _context_stop(self):
        old_context = self.context_stack.pop()
        old_context.ended_at = datetime.now()
        return old_context

    #### Logging ####

    def command(self, command: Any, **updates):
        """
        Add a command update to the current context.

        Args:
            command (Any): The command to be added.
            **updates: Additional updates to be added alongside the command.
        """
        self.context_stack[-1].add_update(command=command, **updates)

    def notes(self, notes: Any, **updates):
        """
        Add notes update to the current context.

        Args:
            notes (Any): The notes to be added.
            **updates: Additional updates to be added alongside the notes.
        """
        self.context_stack[-1].add_update(notes=notes, **updates)

    def feedback(self, feedback: Any, reward: float | None = None, **updates):
        """
        Add feedback update to the current context.

        Args:
            feedback (Any): The feedback to be added.
            **updates: Additional updates to be added alongside the feedback.
        """
        if reward is not None:
            self.reward(reward)
        self.context_stack[-1].add_update(feedback=feedback, **updates)

    def info(self, info: Any, **updates):
        """
        Add info update to the current context.

        Args:
            info (Any): The info to be added.
            **updates: Additional updates to be added alongside the info.
        """
        self.context_stack[-1].add_update(info=info, **updates)

    def log(self, **updates):
        """
        Add arbitrary updates to the current context.

        Args:
            **updates: The updates to be added to the current context.
        """
        self.context_stack[-1].add_update(**updates)

    ### Execution ###

    def execute(self, op: BaseOp, *args, **kwargs: Any) -> Any:
        """
        Execute the given operation with the provided arguments.

        Args:
            op (BaseOp): The operation to execute.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the operation execution.
        """
        return op.execute(self, *args, **kwargs)

    def register_op(self, op: BaseOp):
        self.ops_registry[op.id] = op

    def get_op(
        self,
        # TODO: I left off here
        # engine_type: SubclassField(BaseEngine) | None = None,
        # engine_id: str | None = None,
        # run_id: UUID4 | None = None,
        # operator_type: SubclassField(BaseOp) | None = None,
        # operator_id: str | None = None,
        # object_type: SubclassField(Any) | None = None,
        # object: Any | None = None,
        # latent_type: SubclassField(LatentType) | None = None,
    ) -> BaseOp:
        return self.ops_registry[op_selector]

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
        """
        with open(path, "r") as f:
            return cls.model_validate_json(f.read())

    @abstractmethod
    def save(self, path: str | Path):
        """
        Save the current engine state to a file.

        Args:
            path (str | Path): The path where the engine state will be saved.
        """
        with open(path, "w") as f:
            f.write(self.model_dump_json())

    ### Decorators ###

    def trace(self):
        """
        Decorator for tracing function calls with detailed logging.
        """

        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                with self.subcontext():
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
