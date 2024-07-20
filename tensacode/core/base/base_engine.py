from abc import abstractmethod
from typing import Any, Literal, TypedDict, ClassVar
from functools import cached_property, contextmanager
from pydantic import BaseModel, Field
from datetime import datetime
from functools import reduce
import traceback
import inspect
from tensacode.internal.utils.consts import VERSION
from tensacode.internal.utils.language import Language
from tensacode.internal.protocols.tagged_object import HasID
from tensacode.core.base.ops.ops import BaseOp


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


class LogStatement(BaseModel):
    mode: Literal["command", "notes", "feedback", "info"] = Field(default="info")
    timestamp: datetime = Field(default_factory=datetime.now)
    content: str


class EngineContext(TypedDict, total=False):
    logs: list[LogStatement]
    callstack: list[dict] = Field(default_factory=generate_callstack)


class BaseEngine(HasID, BaseModel):

    tensacode_version: ClassVar[str] = VERSION
    render_language: Language = "python"
    config: dict[str, Any] = Field(default_factory=dict)
    ops_registry: dict[str, BaseOp] = Field(default_factory=dict)

    ### Context Management ###

    ## TODO: combine context and logs. Just amke statements that go down and go sideways and linked them togeher

    context_stack: list[EngineContext] = Field(default_factory=list)

    @cached_property
    def cumulative_context(self) -> EngineContext:
        context = {}
        for context in self.context_stack:
            context.update(context)
        return context

    @contextmanager
    def with_context(self, **overrides):
        new_context = self.context_start(**overrides)
        yield self
        old_context = self.context_stop()
        assert (
            new_context == old_context
        ), "Context mismatch. Did you modify `engine.context_stack` while the `engine.with_context` contextmanager was active?"

    def context_start(self, **overrides):
        new_context = {"logs": [], **overrides}
        self.context_stack.append(new_context)
        return new_context

    def context_stop(self):
        old_context = self.context_stack.pop()
        return old_context

    #### Logging ####

    @property
    def logs(self) -> list[LogStatement]:
        return reduce(
            lambda acc, context: acc + context.setdefault("logs", []),
            self.context_stack,
            [],
        )

    def log(self, content: Any, **kwargs):
        self.context_stack[-1].setdefault("logs", []).append(
            LogStatement(content=content, **kwargs)
        )

    def command(self, content: Any, **kwargs):
        self.log(content, mode="command", **kwargs)

    def notes(self, content: Any, **kwargs):
        self.log(content, mode="notes", **kwargs)

    def feedback(self, content: Any, **kwargs):
        self.log(content, mode="feedback", **kwargs)

    def info(self, content: Any, **kwargs):
        self.log(content, mode="info", **kwargs)

    ### Execution ###

    def execute(self, op: BaseOp, input: Any, **kwargs: Any) -> Any:
        return op.execute(self, input, **kwargs)

    def reward(self, reward: float): ...

    def start(self): ...

    def stop(self): ...

    ### Decorators ###

    def trace(self):
        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                with self.with_context():
                    args_dict = {n: v for n, v in zip(fn.__annotations__.keys(), args)}
                    self.log(
                        {
                            "command": f"Executing {fn.__name__}",
                            "args": args_dict,
                        }
                    )
                    result = fn(*args, **kwargs)
                    self.log(
                        {
                            "command": f"{fn.__name__} returned {result}",
                            "args": args_dict,
                            "result": result,
                        }
                    )
                    return result

            return wrapper

        return decorator
