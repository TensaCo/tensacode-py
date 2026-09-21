"""Score explicit vector candidates with caller-supplied tensor modules."""
from __future__ import annotations

from torch import nn

from ...tracing import invoke
from ._configuration import module_configuration, qualified_name
from .candidates import CandidateSet, Scores
from .latent import Space, require_compatible


class Score(nn.Module):
    replayable = True

    def __init__(
        self,
        module: nn.Module,
        *,
        query_space: Space,
        candidate_space: Space,
        meaning: str,
    ) -> None:
        super().__init__()
        if not isinstance(module, nn.Module):
            raise TypeError("Score module must be a torch.nn.Module")
        if not isinstance(meaning, str) or not meaning.strip():
            raise ValueError("Score meaning must be a nonempty string")
        self.module = module
        self.query_space = query_space
        self.candidate_space = candidate_space
        self.meaning = meaning

    def __call__(self, value, *, context=None):
        return invoke(self, value, context, super().__call__)

    def forward(self, value, *, context=None):
        if context:
            raise ValueError("Score does not consume context")
        if not isinstance(value, CandidateSet):
            raise TypeError("Score expects a CandidateSet")
        require_compatible(self.query_space, value.query.space, role="query")
        require_compatible(self.candidate_space, value.candidates.space, role="candidates")
        scores = self.module(value.query.tensor, value.candidates.tensor)
        return Scores(scores, self.meaning, value)

    def configuration(self):
        return {
            "operation": qualified_name(self),
            "query_space": self.query_space.configuration(),
            "candidate_space": self.candidate_space.configuration(),
            "meaning": self.meaning,
            "module": module_configuration(self.module),
        }
