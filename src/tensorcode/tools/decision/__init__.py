"""A trainable decision model using the shared investigation architecture."""
from ..investigator import Investigator


class Decision(Investigator):
    """Evaluate explicit candidates through an owned cognitive workspace.

    Construct from configuration or load weights with ``from_pretrained``.
    Compose supplied-provider operations with ordinary Python calls.
    """


__all__ = ["Decision"]
