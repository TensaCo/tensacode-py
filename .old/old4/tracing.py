import functools, loguru
from pydantic import BaseModel, Field
from tensacode.utils.string import render_invocation


class Trace(BaseModel):
    rendered_frames: list[str] = []


class Tracer(BaseModel):
    N_frames: int = 2
    logger: loguru.Logger | None = Field(factory=lambda: loguru.logger.bind())

    traces: list[str] = []

    def trace(self, record_input=True, record_output=False):
        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                if record_input:
                    self.logger.info(render_invocation(fn, args, kwargs))
                with self.logger.catch():
                    # TODO: trace stack leading to here. This is vital for the tensacode agent to do anything

                    result = fn(*args, **kwargs)
                if record_output:
                    self.logger.info(render_invocation(fn, args, kwargs, result))
                return result

            return wrapper

        return decorator
