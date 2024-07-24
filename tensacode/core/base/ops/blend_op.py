from typing import ClassVar, Any
from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op

from tensacode.internal.utils.locator import Locator


class BaseBlendOp(Op):
    """Docstring for BaseBlendOp"""

    name: ClassVar[str] = "blend"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine


@BaseEngine.register_op_class_for_all_class_instances
@BaseBlendOp.create_subclass(name="blend")
def Blend(
    engine: BaseEngine,
    *objects: list[object],
    total_rounds: int = 10,
    **kwargs: Any,
) -> Any:
    """Blends objects"""

    step = 0
    while engine.decide("Continue blending objects?", objects=objects) and (
        total_rounds is None or step < total_rounds
    ):
        step += 1

        with engine.scope(step=step):
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
