"""Explicit operands and tensor-valued results for vector candidate operations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .latent import Latent


@dataclass(frozen=True)
class CandidateSet:
    """A query and existing candidates shaped ``(..., N, features)``."""

    query: Latent
    candidates: Latent
    identities: tuple[str, ...]
    metadata: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.query, Latent) or not isinstance(self.candidates, Latent):
            raise TypeError("CandidateSet query and candidates must be Latent objects")
        if self.candidates.tensor.ndim < 2:
            raise ValueError("Candidate tensor needs a candidate and feature dimension")
        count = self.candidates.tensor.shape[-2]
        if count == 0:
            raise ValueError("CandidateSet requires at least one candidate")
        if tuple(self.query.tensor.shape[:-1]) != tuple(self.candidates.tensor.shape[:-2]):
            raise ValueError("Query and candidates must have the same batch shape")
        if self.candidates.mask is not None:
            if self.candidates.mask.dtype != torch.bool:
                raise ValueError("Candidate availability mask must be boolean")
            if not bool(self.candidates.mask.any(dim=-1).all()):
                raise ValueError("Every query requires at least one valid candidate")
        identities = tuple(self.identities)
        if len(identities) != count or not all(isinstance(value, str) for value in identities):
            raise ValueError("Candidate identities must be strings matching the candidate count")
        if any(not value for value in identities) or len(set(identities)) != len(identities):
            raise ValueError("Candidate identities must be unique nonempty strings")
        object.__setattr__(self, "identities", identities)
        metadata = tuple(self.metadata) if self.metadata else tuple({} for _ in range(count))
        if len(metadata) != count or not all(isinstance(value, Mapping) for value in metadata):
            raise ValueError("Candidate metadata must match the candidate count")
        object.__setattr__(self, "metadata", tuple(dict(value) for value in metadata))

    @property
    def count(self) -> int:
        return self.candidates.tensor.shape[-2]


@dataclass(frozen=True)
class Scores:
    values: torch.Tensor
    meaning: str
    candidates: CandidateSet

    def __post_init__(self) -> None:
        expected = self.candidates.candidates.tensor.shape[:-1]
        if not isinstance(self.values, torch.Tensor) or tuple(self.values.shape) != tuple(expected):
            raise ValueError(f"Scores must have shape {tuple(expected)}")
        if not self.values.is_floating_point():
            raise ValueError("Scores must be floating-point tensors")
        if not isinstance(self.meaning, str) or not self.meaning.strip():
            raise ValueError("Score meaning must be a nonempty string")


def gather_latent(candidates: Latent, indices: torch.Tensor) -> Latent:
    """Gather candidate-axis items while retaining gradient and provenance."""

    tensor_index = indices.unsqueeze(-1).expand(*indices.shape, candidates.tensor.shape[-1])
    tensor = torch.gather(candidates.tensor, -2, tensor_index)
    mask = None
    if candidates.mask is not None:
        mask = torch.gather(candidates.mask, -1, indices)
    coordinates = None
    if candidates.coordinates is not None:
        coordinate_index = indices.unsqueeze(-1).expand(
            *indices.shape, candidates.coordinates.shape[-1]
        )
        coordinates = torch.gather(candidates.coordinates, -2, coordinate_index)
    return candidates.with_tensor(
        tensor,
        mask=mask,
        coordinates=coordinates,
    )


def select_python(values: tuple[Any, ...], indices: torch.Tensor):
    """Explicit device-to-Python conversion used by result convenience properties."""

    selected = indices.detach().cpu().tolist()

    def convert(value):
        if isinstance(value, list):
            return tuple(convert(item) for item in value)
        return values[value]

    return convert(selected)
