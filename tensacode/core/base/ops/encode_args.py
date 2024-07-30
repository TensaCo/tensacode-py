from typing import Any, Optional, TypeVar, Callable, Union, Generic
from functools import wraps
from tensacode.internal.latent import LatentType
from tensacode.internal.protocols.encode import EncodeOp
from tensacode.internal.utils.misc import advanced_equality_check
from tensacode.core.base.ops.base_op import BaseOp
from tensacode.core.base.log import Log
import inspect


# TODO

T = TypeVar("T")


class Encoded(Generic[T]):
    """
    A generic type to represent encoded values.
    It serves as both a type hint and a marker for the @encode_args decorator.
    """

    def __class_getitem__(cls, params):
        return Union[Any, LatentType]


def encode_args(
    latent_type: Optional[LatentType] = None,
    encoder: Optional[EncodeOp] = None,
):
    """
    Decorator to automatically encode arguments of a method that are not encoded but are annotated with `Encoded`.

    @encode_args()
    def execute(
        self,
        input: Any,
        input_encoded: Optional[Encoded[Any, LatentType]] = None,
        prompt: Encoded[Any, LatentType] = None,
        context: dict = {},
        log: Optional[Log] = None,
        config: dict = {},
        **kwargs,
    ):
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            sig = inspect.signature(func)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()

            def encode_if_needed(arg_name, arg_value):
                param = sig.parameters[arg_name]
                if Encoded in getattr(param.annotation, "__origin__", ()):
                    if isinstance(arg_value, LatentType):
                        return arg_value
                    if encoder:
                        return encoder.encode(arg_value, latent_type)
                    else:
                        encoder = BaseOp.get.lookup((arg_value, latent_type))
                        if encoder:
                            return encoder.encode(arg_value, latent_type)
                    raise ValueError(
                        f"No encoder found for {type(arg_value)} to {latent_type}"
                    )
                return arg_value

            encoded_args = {
                name: encode_if_needed(name, value)
                for name, value in bound_args.arguments.items()
            }

            return func(**encoded_args)

        return wrapper

    return decorator
