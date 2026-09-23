"""Owned vector transformations with explicit architecture and space contracts."""
from tensorcode._internal.vec.owned import OwnedMap


class Transform(OwnedMap):
    """Construct a linear, MLP, or supported native transformer from JSON config."""


__all__ = ['Transform']
