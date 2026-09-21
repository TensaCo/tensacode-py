"""A trainable decision model using the shared investigation architecture."""
from ..investigator import Investigator


class Decision(Investigator):
    """Evaluate explicit candidates through an owned cognitive workspace.

    Construct from configuration or load weights with ``from_pretrained``.
    For supplied-provider composition use ``runtime.DecisionPipeline``.
    """


__all__ = ["Decision"]
