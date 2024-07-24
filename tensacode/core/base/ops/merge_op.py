from typing import ClassVar, Any
from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op


class BaseMergeOp(Op):
    """Docstring for BaseMergeOp"""

    name: ClassVar[str] = "merge"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine


@BaseEngine.register_op_class_for_all_class_instances
@BaseMergeOp.create_subclass(name="merge")
def Merge(
    engine: BaseEngine,
    *objects: list[object],
    total_rounds: int = 10,
    **kwargs: Any,
) -> Any:
    """Merges objects"""

    objects_set = set(objects)

    step = 0
    while engine.decide("Continue merging objects?") and (
        total_rounds is None or step < total_rounds
    ):
        step += 1

        with engine.scope(
            step=step,
            total_steps=total_rounds,
            objects=objects,
        ):
            receiver = engine.select(objects_set)
            sender = engine.select(objects_set - {receiver})
            engine.info("transferring information", sender=sender, receiver=receiver)
            selected_value = engine.select("select the value to send")
            engine.update(receiver, value=selected_value)
            engine.info("updated object", final_receiver=receiver)
