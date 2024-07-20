from typing import Any, Optional, TypeVar, Callable, Union
from functools import wraps
from tensacode.internal.protocols.latent import LatentType
from tensacode.internal.protocols.encode import EncodeOp
from tensacode.internal.utils.misc import advanced_equality_check
from tensacode.core.base.ops.base_op import BaseOp
from tensacode.core.base.log import Log
import inspect


T = TypeVar("T")


def encode_args(
    latent_type: Optional[LatentType] = None, encoder: Optional[EncodeOp] = None
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
                if isinstance(param.annotation, type(Encoded)):
                    if isinstance(arg_value, Encoded):
                        return arg_value
                    arg_latent_type = param.annotation.__args__[1]
                    if encoder:
                        return encoder.encode(arg_value, arg_latent_type)
                    else:
                        encoder = op_registry.lookup((type(arg_value), arg_latent_type))
                        if encoder:
                            return encoder.encode(arg_value, arg_latent_type)
                    raise ValueError(
                        f"No encoder found for {type(arg_value)} to {arg_latent_type}"
                    )
                return arg_value

            encoded_args = {
                name: encode_if_needed(name, value)
                for name, value in bound_args.arguments.items()
            }

            return func(**encoded_args)

        return wrapper

    return decorator
