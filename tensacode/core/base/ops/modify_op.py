from typing import Any, ClassVar, Optional
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.protocols.latent import LatentType
from tensacode.core.base.ops.base_op import BaseOp
from tensacode.core.base.ops.select_op import SelectOp
from tensacode.core.base.ops.decode_op import DecodeOp
from tensacode.core.base.ops.decide_op import DecideOp
from tensacode.internal.protocols.encoded import Encoded


class ModifyOp(BaseOp):
    """Modify an object"""

    op_name: ClassVar[str] = "modify"
    object_type: ClassVar[type[object]] = Any
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine

    field_select_op: SelectOp
    value_decode_op: DecodeOp
    decide_continue_op: DecideOp

    def _execute(
        self,
        input: Any,
        input_encoded: Optional[Encoded[Any, LatentType]] = None,
        prompt: Optional[Encoded[Any, LatentType]] = None,
        context: dict = {},
        engine: BaseEngine,
        **kwargs,
    ) -> Any:
        """Modify an object"""
        input_encoded = input_encoded or engine.common_encoder.execute(
            input, context=context, config=kwargs.get('config', {}), **kwargs
        )
        prompt = prompt or self.prompt

        while self.decide_continue_op.execute(
            prompt, context=context, engine=engine, config=kwargs.get('config', {})
        ):
            engine.log.info("Selecting field from input")
            field = self.field_select_op.execute(
                input, context=context, engine=engine, config=kwargs.get('config', {})
            )
            engine.log.info(f"Selected field: {field}")
            if isinstance(input, dict):
                old_value = input.get(field, None)
            else:
                old_value = getattr(input, field, None)
            if old_value is None:
                engine.log.warn(f"Field {field} not found in input")
                continue
            value = self.value_decode_op.execute(
                {"input": input, "field": field, "old_value": old_value},
                context=context,
                engine=engine,
                config=kwargs.get('config', {}),
            )
            engine.log.info(f"Modifying {field} to {value}")
            if isinstance(input, dict):
                input[field] = value
            else:
                setattr(input, field, value)
        return input

    @classmethod
    def from_engine(cls, engine: BaseEngine) -> Self:
        return cls(
            prompt="Modify the object",
            field_select_op=SelectOp.from_engine(engine),
            value_decode_op=DecodeOp.from_engine(engine),
            decide_continue_op=DecideOp.from_engine(engine)
        )
