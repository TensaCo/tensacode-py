from abc import ABC

from abc import ABC, abstractmethod
from typing import Any, Optional

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.tcir.nodes import Node
from tensacode.internal.tcir.parse import parse_node
from tensacode.internal.protocols.latent import LatentType
from tensacode.core.base.ops.base_op import BaseOp
from tensacode.core.base.ops.select_op import SelectOp
from tensacode.core.base.ops.create_op import CreateOp
from tensacode.core.base.ops.decide_op import DecideOp
from functools import cached_property
from tensacode.core.base.log import Log
from tensacode.internal.protocols.encoded import Encoded


class MergeOp(BaseOp):
    """Merges objects"""

    field_select_op: FieldSelectOp
    decide_continue_op: DecideOp
    create_op: CreateOp

    def execute(
        self,
        inputs: list[Any],
        enc_inputs: Optional[list[Encoded[Any, LatentType]]] = None,
        prompt: Optional[Encoded[Any, LatentType]] = None,
        context: dict = {},
        log: Optional[Log] = None,
        config: dict = {},
        **kwargs: Any,
    ) -> Any:
        """Merges objects"""
        # standardize inputs to nodes
        nodes = [parse_node(input) for input in inputs]
        if not enc_inputs:
            enc_inputs = [
                common_encoder.execute(
                    input, context=context, log=log, config=config, **kwargs
                )
                for input in inputs
            ]
        context = {**context, "inputs": enc_inputs}
        prompt = prompt or self.prompt
        log = log or self.log

        final_fields = {}
        log.command(prompt)
        log.info(f"Inputs: {inputs}")
        log.info(f"Enc inputs: {enc_inputs}")
        log.info(f"Context: {context}")
        log.info(f"Config: {config}")
        i = 0
        while self.decide_continue_op.execute(
            prompt, context=context, log=log, config=config, **kwargs
        ):
            log.info(f"Pass {i}")
            fields = self.field_select_op(
                nodes, context=context, log=log, config=config, **kwargs
            )
            log.info(f"Selected fields: {fields}")
            final_fields = {**final_fields, **fields}
            i += 1
        log.info(f"Final fields: {final_fields}")
        final_object = self.create_op(
            final_fields, context=context, log=log, config=config, **kwargs
        )
        log.info(f"Final object: {final_object}")
        return final_object
