from typing import Any, Optional, Annotated, List
from tensacode.core.base_engine import Engine, operator
from tensacode.internal.meta.param_tags import Encoded
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op
from tensacode.internal.utils.misc import inheritance_distance, score_inheritance_distance, Score
from tensacode.internal.tcir.nodes import (
    CompositeValueNode,
    AtomicValueNode,
    CompositeValueNode,
)
from tensacode.internal.tcir.parse import parse_node
from tensacode.internal.utils.locator import Locator


@Engine.register_op_on_class
@score_inheritance_distance
def select(
    engine: Engine,
    target: Annotated[Any, Score(coefficient=1)],
    *inputs: Annotated[List[Any], Score(coefficient=1)],
    prompt: Optional[Encoded[str]] = None,
    **kwargs: Any,
) -> Any:
    """
    Select a specific element or subset from the target based on the inputs and prompt.

    This operation uses the engine to locate and retrieve a specific part of the target object.

    Args:
        engine (Engine): The engine used for selection.
        target: The object to select from.
        *inputs (list[Any]): Additional inputs to guide the selection process.
        prompt (Optional[Encoded[str]], optional): A prompt to guide the selection. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        Any: The selected element or subset from the target.

    Examples:
        >>> data = {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}
        >>> result = select(engine, data, prompt="Select the name of the user who is 30 years old")
        >>> print(result)
        Alice

        >>> text = "The quick brown fox jumps over the lazy dog"
        >>> result = select(engine, text, prompt="Select all words with more than 4 letters")
        >>> print(result)
        ['quick', 'brown', 'jumps', 'lazy']
    """
    locator: Locator = engine.locate(
        target,
        *inputs,
        top_k=1,
        _new_scope=False,
        prompt=prompt,
        **kwargs,
    )
    return locator.get(target, current=target, create_missing=False)