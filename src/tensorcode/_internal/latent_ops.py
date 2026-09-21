"""Shared ownership and sequence contracts for pretrained latent operations."""
from __future__ import annotations

import torch

from .pretrained import PretrainedTool
from ..tracing import invoke
from ..ops.vec.latent import Latent, Space, require_compatible


class LatentOperation(PretrainedTool):
    """A complete owned model whose public call is an operation boundary.

    Replay opts out of external effects; it does not promise identical stochastic
    training samples. Generation randomness belongs in explicit context/config,
    while exact training continuation requires restored RNG state.
    Configurations describe architecture; weights
    remain registered native tensors in the complete pretrained artifact.
    """
    replayable = True

    def __call__(self, value, *, context=None):
        return invoke(self, value, context, super().__call__)

    def operation_bindings(self):
        return {'operation': self, **super().operation_bindings()}


def space_from_config(value):
    if isinstance(value, Space):
        return value
    if not isinstance(value, dict):
        raise TypeError('space must be a Space or its configuration object')
    return Space(**value)


def as_sequence(value, expected_space):
    """Normalize explicit feature/sequence/spatial batches without detaching.

    Feature shapes are D or B,D; sequence shapes L,D or B,L,D. Spatial input is
    explicitly batched B,H,W,D. Invalid tokens are zeroed and remain masked.
    """
    if not isinstance(value, Latent):
        raise TypeError('Expected a space-tagged Latent')
    require_compatible(space_from_config(expected_space), value.space)
    x=value.tensor
    if not x.dtype.is_floating_point or not torch.isfinite(x).all():
        raise ValueError('Latent conditioning must contain finite floating tensors')
    mask=value.mask
    if mask is not None and mask.dtype != torch.bool:
        raise ValueError('Latent conditioning mask must be boolean')
    if mask is None:
        mask=torch.ones(x.shape[:-1],dtype=torch.bool,device=x.device)
    organization=value.space.organization
    if organization=='feature' and x.ndim in (1,2):
        if x.ndim==1:
            x=x[None,None,:];mask=mask.reshape(1,1)
        else:
            x=x[:,None,:];mask=mask[:,None]
    elif organization=='sequence' and x.ndim in (2,3):
        if x.ndim==2:
            x=x[None,:,:];mask=mask[None,:]
    elif organization=='spatial' and x.ndim==4:
        x=x.flatten(1,2);mask=mask.flatten(1,2)
    else:
        raise ValueError('Conditioning organization/shape must be feature D/B,D, sequence L,D/B,L,D or spatial B,H,W,D')
    if not x.shape[0] or not x.shape[1] or not mask.any(dim=1).all():
        raise ValueError('Every conditioning sample requires at least one valid token')
    return x.masked_fill(~mask[...,None],0),mask
