from abc import ABC

from abc import ABC, abstractmethod
from typing import Any

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.tcir.nodes import Node
from tensacode.internal.protocols.latent import LatentType
from tensacode.core.base.ops.base_op import BaseOp


class CallOp(BaseOp):
    def execute(
        self, *args, log: Optional[Log] = None, context: dict, config: dict, **kwargs
    ):
        """Call a function by creating or obtaining all necessary args"""
        # Implementation goes here
        pass
