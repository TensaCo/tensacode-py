from abc import ABC

from abc import ABC, abstractmethod
from typing import Any

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.tcir.nodes import Node
from tensacode.internal.protocols.latent import LatentType
from tensacode.core.base.ops.base_op import BaseOp


class SimilarityOp(BaseOp):
    def execute(
        self, *args, log: Optional[Log] = None, context: dict, config: dict, **kwargs
    ):
        """Compute similarity between objects"""
        # Implementation goes here
        pass
