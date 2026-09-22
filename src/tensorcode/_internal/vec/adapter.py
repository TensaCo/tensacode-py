"""Tensor-native operations: preserve nn.Module hooks and autograd."""
from torch import nn
from tensorcode._internal.tracing import invoke
from tensorcode.ops.vec._configuration import (
    callable_identity,
    module_configuration,
    qualified_name,
    space_configuration,
)
from tensorcode.ops.vec.latent import Latent, Space, require_compatible


class TensorAdapter(nn.Module):
    replayable = True

    def __init__(
        self,
        module: nn.Module,
        *,
        combine=None,
        input_space: Space | None = None,
        output_space: Space | None = None,
    ):
        super().__init__()
        if not isinstance(module, nn.Module):
            raise TypeError('Transform module must be a torch.nn.Module')
        self.module = module
        self.combine = combine
        self.input_space = input_space
        self.output_space = output_space

    def __call__(self, value, *, context=None):
        return invoke(self, value, context, super().__call__)

    def forward(self, value, *, context=None):
        source = value if isinstance(value, Latent) else None
        if source is not None:
            if self.input_space is not None:
                require_compatible(self.input_space, source.space)
            value = source.tensor
        elif self.input_space is not None:
            raise TypeError('A configured input_space requires a Latent input')
        if self.combine is not None:
            value = self.combine(value, context or {})
        elif context:
            raise ValueError('Context requires an explicit combine function')
        result = self.module(value)
        if isinstance(result, Latent):
            if self.output_space is not None:
                require_compatible(self.output_space, result.space, role='output')
            return result
        if self.output_space is None:
            return result
        if source is None:
            return Latent(result, self.output_space)
        retains_organization = tuple(result.shape[:-1]) == tuple(source.tensor.shape[:-1])
        return source.with_tensor(
            result,
            space=self.output_space,
            mask=source.mask if retains_organization else None,
            coordinates=source.coordinates if retains_organization else None,
        )

    def configuration(self):
        return {
            'operation': qualified_name(self),
            'module': module_configuration(self.module),
            'combine': callable_identity(self.combine),
            'input_space': space_configuration(self.input_space),
            'output_space': space_configuration(self.output_space),
        }
