import inspect
from typing import Any, Callable, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op


class BaseCallOp(Op):
    """
    Get or create values to call a function,
    conditioned by curried and direct invocation arguments
    """

    name: ClassVar[str] = "call"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine


@BaseEngine.register_op_class_for_all_class_instances
@BaseCallOp.create_subclass(name="call")
def Call(
    engine: BaseEngine,
    func: Callable,
    **kwargs: Any,
) -> Any:
    """
    Get or create values to call a function,
    conditioned by curried and direct invocation arguments
    """
    # Get the function signature
    signature = inspect.signature(func)

    # Prepare arguments for the function call
    all_args = {}
    arg_count = 0

    for param_name, param in signature.parameters.items():
        engine.info(**{param_name: param})
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            # Handle *args
            var_args = engine.query_or_create(
                query=f"Provide values for *{param_name}",
                type=(
                    param.annotation
                    if param.annotation != inspect.Parameter.empty
                    else None
                ),
            )
            for arg in var_args:
                all_args[str(arg_count)] = arg
                arg_count += 1
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            # Handle **kwargs
            var_kwargs = engine.query_or_create(
                query=f"Provide key-value pairs for **{param_name}",
                type=(
                    param.annotation
                    if param.annotation != inspect.Parameter.empty
                    else None
                ),
            )
            all_args.update(var_kwargs)
        else:
            # Handle regular parameters
            param_type = (
                param.annotation
                if param.annotation != inspect.Parameter.empty
                else None
            )
            if param.default == inspect.Parameter.empty:
                value = engine.query_or_create(
                    query=f"Provide a value for parameter '{param_name}'",
                    type=param_type,
                )
            else:
                value = engine.query_or_create(
                    query=f"Provide a value for parameter '{param_name}' (default: {param.default})",
                    type=param_type,
                )

            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                all_args[str(arg_count)] = value
                arg_count += 1
            else:
                all_args[param_name] = value

    # Call the function with the prepared arguments
    args = [all_args[str(i)] for i in range(len(all_args))]
    kwargs = {k: v for k, v in all_args.items() if k not in args}
    return func(*args, **kwargs)
