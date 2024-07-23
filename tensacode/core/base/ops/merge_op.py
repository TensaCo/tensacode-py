from typing import Any, ClassVar, Optional
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.tcir.parse import parse_node
from tensacode.internal.protocols.latent import LatentType
from tensacode.core.base.ops.base_op import BaseOp
from tensacode.internal.protocols.encoded import Encoded


@BaseEngine.register_op_class_for_all_class_instances
class MergeOp(BaseOp):
    """Merges objects"""

    op_name: ClassVar[str] = "merge"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine

    def _execute(
        self,
        inputs: list[Any],
        *,
        engine: BaseEngine,
        **kwargs: Any,
    ) -> Any:
        """Merges objects"""

        i = 0
        final_fields = dict()
        while self.decide_continue_op.execute(prompt="Continue merging objects?"):
            engine.log.info(f"Pass {i}")
            fields = engine.select(input=inputs[i], prompt="Select fields to merge")
            # TODO: this algorithm doesn't evn amek sense
            engine.log.info(f"Selected fields: {fields}")
            final_fields = {**final_fields, **fields}
            i += 1
        engine.log.info(f"{final_fields}")
        final_object = engine.create(final_fields)
        engine.log.info(f"Final object: {final_object}")
        return final_object
