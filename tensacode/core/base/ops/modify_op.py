from abc import ABC

from abc import ABC, abstractmethod
from typing import Any

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.tcir.nodes import Node
from tensacode.internal.protocols.latent import LatentType
from tensacode.core.base.ops.base_op import BaseOp
from tensacode.core.base.ops.select_op import SelectOp
from tensacode.core.base.ops.decode_op import DecodeOp
from tensacode.core.base.ops.decide_op import DecideOp


class ModifyOp(BaseOp):
    """Modify an object"""

    DEFAULT_PROMPT: str = "Modify the object"
    prompt: str = Field(default=DEFAULT_PROMPT)

    field_select_op: FieldSelectOp
    value_decode_op: DecodeOp
    decide_continue_op: DecideOp

    @encode_args
    def execute(
        self,
        input: Any,
        input_encoded: Optional[Encoded[Any, LatentType]] = None,
        prompt: Encoded[Any, LatentType] = None,
        context: dict = {},
        log: Optional[Log] = None,
        config: dict = {},
        **kwargs,
    ):
        """Modify an object"""
        prompt = prompt or self.prompt
        log = log or self.log

        while self.decide_continue_op.execute(
            prompt, context=context, log=log, config=config
        ):
            log.info("Selecting field from input")
            field = self.field_select_op.execute(
                input, context=context, log=log, config=config
            )
            log.info(f"Selected field: {field}")
            if isinstance(input, dict):
                old_value = input.get(field, None)
            else:
                old_value = getattr(input, field, None)
            if old_value is None:
                log.warn(f"Field {field} not found in input")
                continue
            value = self.value_decode_op.execute(
                {"input": input, "field": field, "old_value": old_value},
                context=context,
                log=log,
                config=config,
            )
            log.info(f"Modifying {field} to {value}")
            if isinstance(input, dict):
                input[field] = value
            else:
                setattr(input, field, value)
        return input
