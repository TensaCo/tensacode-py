from typing import ClassVar, Any
from tensacode.core.base_engine import Engine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op

from tensacode.internal.utils.tc import loop_until_done
from typing import Annotated
from tensacode.internal.utils.misc import score_inheritance_distance, Score


@Engine.register_op_on_class
@score_inheritance_distance
def blend(
    engine: Engine,
    *objects: Annotated[list[object], Score(coefficient=1)],
    prompt: Optional[Encoded[str]] = None,
    total_steps: int = 10,
    **kwargs: Any,
) -> Any:
    """
    Blend multiple objects together.

    This operation iteratively combines properties from the input objects to create a new blended object.

    Args:
        engine (Engine): The engine used for blending.
        *objects (list[object]): The objects to be blended.
        prompt (Optional[Encoded[str]], optional): A prompt to guide the blending process. Defaults to None.
        total_steps (int, optional): The total number of blending steps. Defaults to 10.
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        Any: The resulting blended object.

    Examples:
        >>> class Car:
        ...     def __init__(self, color: str, speed: int):
        ...         self.color = color
        ...         self.speed = speed
        >>> car1 = Car("red", 200)
        >>> car2 = Car("blue", 180)
        >>> result = blend(engine, car1, car2)
        >>> print(f"Blended car: {result.color}, {result.speed} km/h")
        Blended car: purple, 190 km/h

        >>> dict1 = {"a": 1, "b": 2}
        >>> dict2 = {"b": 3, "c": 4}
        >>> result = blend(engine, dict1, dict2)
        >>> print(result)
        {'a': 1, 'b': 2.5, 'c': 4}
    """
    for step in loop_until_done(
        total_steps,
        engine=engine,
        continue_prompt="Continue blending objects?",
        stop_prompt="Done blending objects?",
    ):
        with engine.scope(step=step, total_steps=total_steps):
            source_loc = engine.locate(objects)
            source_val = source_loc.value
            engine.info(source_loc=source_loc, source_val=source_val)
            dest_loc = engine.locate(objects)
            engine.info(dest_loc=dest_loc)
            dest_val_before = dest_loc.get(objects)
            dest_loc.set(objects, value=source_val, create_missing=True)
            dest_val_after = dest_loc.get(objects)
            engine.info(
                "updated",
                dest_val_before=dest_val_before,
                dest_val_after=dest_val_after,
            )