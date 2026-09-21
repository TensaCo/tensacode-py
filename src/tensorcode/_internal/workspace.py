"""Differentiable evidence workspace shared by owned cognitive models.

Attention links index source positions; they are learned routing weights, not
claims of factual support. Slots carry no authored semantic roles.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ..ops.vec import Transform


class Workspace(nn.Module):
    """Refine learned slots using encoded evidence and inter-slot relations.

    All parameters exist at construction time. Inputs and source attention use
    ``[batch, tokens, dimensions]`` and ``[batch, slots, tokens]`` respectively.
    The caller owns evidence identities and session lifetime.
    """

    def __init__(self, dimensions: int, slots: int = 8, steps: int = 2):
        super().__init__()
        for name, value in (("dimensions", dimensions), ("slots", slots), ("steps", steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.dimensions = dimensions
        self.slots = slots
        self.steps = steps
        self.queries = nn.Parameter(torch.randn(slots, dimensions) / math.sqrt(dimensions))
        self.key = Transform(nn.Linear(dimensions, dimensions, bias=False))
        self.value = Transform(nn.Linear(dimensions, dimensions, bias=False))
        self.query = Transform(nn.Linear(dimensions, dimensions, bias=False))
        self.refine = nn.GRUCell(dimensions, dimensions)
        self.relation_query = Transform(nn.Linear(dimensions, dimensions, bias=False))
        self.relation_key = Transform(nn.Linear(dimensions, dimensions, bias=False))
        self.relation_value = Transform(nn.Linear(dimensions, dimensions, bias=False))
        self.update = Transform(nn.Sequential(
            nn.Linear(dimensions, dimensions * 2), nn.GELU(),
            nn.Linear(dimensions * 2, dimensions),
        ))

    def configuration(self) -> dict:
        return {
            "architecture": "tensorcode._internal.workspace.Workspace",
            "version": 1,
            "dimensions": self.dimensions,
            "slots": self.slots,
            "steps": self.steps,
        }

    def forward(self, encoded: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if not isinstance(encoded, torch.Tensor) or encoded.ndim != 3:
            raise ValueError("encoded must be a tensor with shape [batch, tokens, dimensions]")
        batch, tokens, dimensions = encoded.shape
        if batch < 1 or tokens < 1 or dimensions != self.dimensions:
            raise ValueError("encoded needs nonempty batch/tokens and the configured dimensions")
        if not encoded.is_floating_point() or not torch.isfinite(encoded).all():
            raise ValueError("encoded must contain finite floating point values")
        if mask is None:
            mask = torch.ones((batch, tokens), dtype=torch.bool, device=encoded.device)
        elif not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (batch, tokens):
            raise ValueError("mask must be boolean with shape [batch, tokens]")
        if mask.device != encoded.device:
            raise ValueError("mask and encoded must be on the same device")
        if not mask.any(dim=1).all():
            raise ValueError("every input must contain at least one unmasked token")
        # Mask before projections as well as softmax: excluded sources contribute
        # neither values nor input gradients, even when very large but finite.
        evidence = encoded.masked_fill(~mask.unsqueeze(-1), 0)
        keys = self.key(evidence)
        values = self.value(evidence)
        state = self.queries.unsqueeze(0).expand(batch, -1, -1)
        scale = math.sqrt(self.dimensions)
        for _ in range(self.steps):
            logits = torch.matmul(self.query(state), keys.transpose(-2, -1)) / scale
            attention = logits.masked_fill(~mask.unsqueeze(1), -torch.inf).softmax(dim=-1)
            received = torch.matmul(attention, values)
            state = self.refine(received.reshape(-1, dimensions), state.reshape(-1, dimensions)).reshape(batch, self.slots, dimensions)
            relations = (torch.matmul(self.relation_query(state), self.relation_key(state).transpose(-2, -1)) / scale).softmax(dim=-1)
            related = torch.matmul(relations, self.relation_value(state))
            state = state + self.update(related)
        return {
            "conditioning": state,
            "mask": torch.ones((batch, self.slots), dtype=torch.bool, device=encoded.device),
            "attention": attention,
            "relations": relations,
        }
