from typing import Any, ClassVar, Optional
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.protocols.latent import LatentType
from tensacode.core.base.ops.base_op import BaseOp
from tensacode.core.base.ops.select_op import SelectOp
from tensacode.core.base.ops.decode_op import DecodeOp
from tensacode.core.base.ops.decide_op import DecideOp
from tensacode.internal.protocols.encoded import Encoded


@BaseEngine.register_op_class_for_all_class_instances
class ModifyOp(BaseOp):
    """Modify an object"""

    op_name: ClassVar[str] = "modify"
    object_type: ClassVar[type[object]] = Any
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine

    @BaseEngine.trace
    def _execute(
        self,
        input: Any,
        *,
        engine: BaseEngine,
        **kwargs,
    ) -> Any:
        """Modify an object"""
        engine.log.info("Modifying object", object=input)

        while engine.decide("should we continue?"):
            engine.log.info("Selecting field from input")
            field = engine.select("field", input)
            engine.log.info(f"Selected field: {field}", selected_field=field)
            if isinstance(input, dict):
                old_value = input.get(field, None)
            else:
                old_value = getattr(input, field, None)
            if old_value is None:
                engine.log.warn(f"Field {field} not found in input")
                continue
            value = engine.decode(
                context={"input": input, "field": field, "old_value": old_value},
                **kwargs,
            )
            engine.log.info(f"Modifying {field} to {value}")
            if isinstance(input, dict):
                input[field] = value
            else:
                setattr(input, field, value)
        return input
