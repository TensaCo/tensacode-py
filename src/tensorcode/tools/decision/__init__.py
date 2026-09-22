"""A trainable decision model using the shared investigation architecture."""
from ..investigator import Investigator


class Decision(Investigator):
    """Evaluate explicit candidates through an owned cognitive workspace.

    Decision shares the Investigator architecture and interface under a
    distinct persisted tool identity. Construct from configuration or load
    weights with ``from_pretrained``; application policy stays in caller code.
    """


__all__ = ["Decision"]
