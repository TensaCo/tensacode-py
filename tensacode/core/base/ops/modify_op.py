from typing import Any, ClassVar, Optional
import inspect
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
        input: object,
        *,
        engine: BaseEngine,
        **kwargs,
    ) -> Any:
        # I left off here and i need to change the log statements to use native values rather than formatted strings
        raise NotImplementedError

        while engine.decide("continue modifying?"):
            field = engine.select("field", input)
            field_annotation: type
            try:
                # try dict access first
                old_value = input.get(field)
                field_annotation = type(old_value)
            except AttributeError:
                old_value = getattr(input, field)
                # Use inspect to get field_type from annotations
                annotations = inspect.get_annotations(type(input))
                field_annotation = annotations.get(field, type(old_value))
            if old_value is None:
                engine.log.warn(f"Field {field} not found in input")
                continue

            value = engine.decode(
                type=field_annotation,
                prompt=f"Modify {field} value",
                **kwargs,
            )
            engine.log.info(f"Modifying {field} to {value}")
            if isinstance(input, dict):
                input[field] = value
            else:
                setattr(input, field, value)
        return input
