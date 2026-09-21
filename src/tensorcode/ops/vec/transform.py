"""Tensor-native operations: preserve nn.Module hooks and autograd."""
from torch import nn
from ...tracing import invoke


class Transform(nn.Module):
    replayable = True

    def __init__(self, module: nn.Module, *, combine=None):
        super().__init__()
        self.module = module
        self.combine = combine

    def __call__(self, value, *, context=None):
        return invoke(self, value, context, super().__call__)

    def forward(self, value, *, context=None):
        if self.combine is not None:
            value = self.combine(value, context or {})
        elif context:
            raise ValueError('Context requires an explicit combine function')
        return self.module(value)
