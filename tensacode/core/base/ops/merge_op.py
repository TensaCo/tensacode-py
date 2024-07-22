from typing import Any, ClassVar, Optional
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.tcir.parse import parse_node
from tensacode.internal.protocols.latent import LatentType
from tensacode.core.base.ops.base_op import BaseOp
from tensacode.core.base.ops.select_op import SelectOp
from tensacode.core.base.ops.create_op import CreateOp
from tensacode.core.base.ops.decide_op import DecideOp
from tensacode.internal.protocols.encoded import Encoded


class MergeOp(BaseOp):
    """Merges objects"""

    op_name: ClassVar[str] = "merge"
    object_type: ClassVar[type[object]] = Any
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine

    field_select_op: SelectOp
    decide_continue_op: DecideOp
    create_op: CreateOp

    def _execute(
        self,
        inputs: list[Any],
        enc_inputs: Optional[list[Encoded[Any, LatentType]]] = None,
        prompt: Optional[Encoded[Any, LatentType]] = None,
        context: dict = {},
        config: dict = {},
        engine: BaseEngine,
        **kwargs: Any,
    ) -> Any:
        """Merges objects"""
        # standardize inputs to nodes
        nodes = [parse_node(input) for input in inputs]
        if not enc_inputs:
            enc_inputs = [
                engine.common_encoder.execute(
                    input, context=context, config=config, **kwargs
                )
                for input in inputs
            ]
        context = {**context, "inputs": enc_inputs}
        prompt = prompt or self.prompt

        final_fields = {}
        engine.log.command(prompt)
        engine.log.info(f"Inputs: {inputs}")
        engine.log.info(f"Enc inputs: {enc_inputs}")
        engine.log.info(f"Context: {context}")
        engine.log.info(f"Config: {config}")
        i = 0
        while self.decide_continue_op.execute(
            prompt, context=context, engine=engine, config=config, **kwargs
        ):
            engine.log.info(f"Pass {i}")
            fields = self.field_select_op(
                nodes, context=context, engine=engine, config=config, **kwargs
            )
            engine.log.info(f"Selected fields: {fields}")
            final_fields = {**final_fields, **fields}
            i += 1
        engine.log.info(f"Final fields: {final_fields}")
        final_object = self.create_op(
            final_fields, context=context, engine=engine, config=config, **kwargs
        )
        engine.log.info(f"Final object: {final_object}")
        return final_object

    @classmethod
    def from_engine(cls, engine: BaseEngine) -> Self:
        return cls(
            prompt="",
            field_select_op=SelectOp.from_engine(engine),
            decide_continue_op=DecideOp.from_engine(engine),
            create_op=CreateOp.from_engine(engine)
        )
