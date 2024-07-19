from abc import ABC

from abc import ABC, abstractmethod
from typing import Any

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.tcir.nodes import Node
from tensacode.internal.tcir.parse import parse_node
from tensacode.internal.protocols.latent import LatentType
from tensacode.core.base.ops.base_op import BaseOp
from tensacode.core.base.ops.select_op import SelectOp
from tensacode.core.base.ops.create_op import CreateOp
from functools import cached_property


class MergeOp(BaseOp):
    """Merges objects"""

    field_select_op: FieldSelectOp
    create_op: CreateOp

    def execute(
        self,
        inputs: list[Any],
        rounds: int,
        log: Optional[Log]=None,
        context: dict,
        params: dict,
        **kwargs: Any,
    ) -> Any:
        """Merges objects"""
        # standardize inputs to nodes
        nodes = [parse_node(input) for input in inputs]

        final_fields = {}
        for _ in range(rounds):
            fields = self.field_select_op(
                nodes, context=context, params=params, **kwargs
            )
            final_fields = {**final_fields, **fields}
        return self.create_op(final_fields, context=context, params=params, **kwargs)
