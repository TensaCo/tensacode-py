"""Return top-ranked existing vector candidates without generating new items."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ...tracing import invoke
from ..._internal.operation_config import ConfigOperationMixin
from .candidates import Scores, gather_latent, select_python
from .latent import Latent


@dataclass(frozen=True)
class Retrieval:
    indices: torch.Tensor
    scores: torch.Tensor
    items: Latent
    scored: Scores

    @property
    def identities(self):
        return select_python(self.scored.candidates.identities, self.indices)

    @property
    def metadata(self):
        return select_python(self.scored.candidates.metadata, self.indices)


class Retrieve(ConfigOperationMixin, nn.Module):
    replayable = True

    config_keys = frozenset({'k', 'largest'})
    config_defaults = {'largest': True}

    def __init__(self, config=None) -> None:
        nn.Module.__init__(self)
        ConfigOperationMixin.__init__(self, config)
        k = self.config.get('k')
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError('Retrieve k must be a positive integer')
        if not isinstance(self.config['largest'], bool):
            raise ValueError('Retrieve largest must be a boolean')
        self.k = k
        self.largest = self.config['largest']

    def __call__(self, value, *, context=None):
        return invoke(self, value, context, super().__call__)

    def forward(self, value, *, context=None):
        if context:
            raise ValueError("Retrieve does not consume context")
        if not isinstance(value, Scores):
            raise TypeError("Retrieve expects Scores")
        count = value.candidates.count
        if self.k > count:
            raise ValueError(f"Cannot retrieve {self.k} items from only {count} candidates")
        selectable = value.values
        mask = value.candidates.candidates.mask
        if mask is not None:
            if bool((mask.sum(dim=-1) < self.k).any()):
                raise ValueError(f"Retrieve k={self.k} exceeds the valid candidates for a query")
            fill = -torch.inf if self.largest else torch.inf
            selectable = selectable.masked_fill(~mask, fill)
        _, indices = torch.topk(selectable, self.k, dim=-1, largest=self.largest, sorted=True)
        scores = torch.gather(value.values, -1, indices)
        items = gather_latent(value.candidates.candidates, indices)
        return Retrieval(indices, scores, items, value)

    def configuration(self):
        return {
            "k": self.k,
            "largest": self.largest,
        }
