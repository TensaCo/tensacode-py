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
def MergeComposite(
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
            # TODO: i need tro switch to an index based system so that i can update the origonal python container
            receiver = engine.choice(objects)
            sender = engine.choice(objects - receiver)
            engine.info(f"transferring information", sender=sender, receiver=receiver)
            sender_value = engine.select("select the value to send")
            receiver = engine.modify(
                "update using the given data",
                value=sender_value,
                total_rounds=1,
            )
            engine.info("updated object", final_receiver=receiver)
