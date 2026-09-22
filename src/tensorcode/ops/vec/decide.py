"""Select one existing option from explicit vector scores without executing it."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tensorcode._internal.tracing import invoke
from ..._internal.operation_config import ConfigOperationMixin
from .candidates import Scores, gather_latent, select_python
from .latent import Latent


@dataclass(frozen=True)
class Decision:
    indices: torch.Tensor
    scores: torch.Tensor
    items: Latent
    scored: Scores

    @property
    def identity(self) -> str:
        if self.indices.ndim != 0:
            raise ValueError("identity is unavailable for a batched decision; use identities")
        return select_python(self.scored.candidates.identities, self.indices)

    @property
    def identities(self):
        if self.indices.ndim == 0:
            raise ValueError("identities requires a batched decision; use identity")
        return select_python(self.scored.candidates.identities, self.indices)


class Decide(ConfigOperationMixin, nn.Module):
    """Select the highest (or, with ``largest=False``, lowest) masked score."""
    replayable = True

    config_keys = frozenset({'largest'})
    config_defaults = {'largest': True}

    def __init__(self, config=None) -> None:
        nn.Module.__init__(self)
        ConfigOperationMixin.__init__(self, config)
        if not isinstance(self.config['largest'], bool):
            raise ValueError('Decide largest must be a boolean')
        self.largest = self.config['largest']

    def __call__(self, value, *, context=None):
        return invoke(self, value, context, super().__call__)

    def forward(self, value, *, context=None):
        if context:
            raise ValueError("Decide does not consume context")
        if not isinstance(value, Scores):
            raise TypeError("Decide expects Scores")
        selectable = value.values
        if value.candidates.candidates.mask is not None:
            fill = -torch.inf if self.largest else torch.inf
            selectable = selectable.masked_fill(~value.candidates.candidates.mask, fill)
        indices = selectable.argmax(dim=-1) if self.largest else selectable.argmin(dim=-1)
        selected_scores = torch.gather(value.values, -1, indices.unsqueeze(-1)).squeeze(-1)
        items = gather_latent(value.candidates.candidates, indices.unsqueeze(-1))
        items = items.with_tensor(
            items.tensor.squeeze(-2),
            mask=None if items.mask is None else items.mask.squeeze(-1),
            coordinates=None if items.coordinates is None else items.coordinates.squeeze(-2),
        )
        return Decision(indices, selected_scores, items, value)

    def configuration(self):
        return {"largest": self.largest}
