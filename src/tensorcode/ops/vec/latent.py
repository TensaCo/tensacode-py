"""Tensor-backed vector representations with explicit semantic space identity."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class Space:
    """Identity and shape contract for a vector representation.

    A matching dimension is insufficient for compatibility: name, version, and
    organization must also match.  ``organization`` describes mechanics such as
    ``"feature"``, ``"sequence"``, or ``"spatial"``; it does not assign learned
    semantics to the vectors.
    """

    name: str
    dimensions: int
    version: str = "1"
    organization: str = "feature"
    dtype: str | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Space name must be a nonempty string")
        if not isinstance(self.dimensions, int) or isinstance(self.dimensions, bool) or self.dimensions <= 0:
            raise ValueError("Space dimensions must be a positive integer")
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("Space version must be a nonempty string")
        if not isinstance(self.organization, str) or not self.organization:
            raise ValueError("Space organization must be a nonempty string")
        if self.dtype is not None and (not isinstance(self.dtype, str) or not self.dtype):
            raise ValueError("Space dtype must be a nonempty string or None")
        if self.device is not None and (not isinstance(self.device, str) or not self.device):
            raise ValueError("Space device must be a nonempty string or None")

    def configuration(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dimensions": self.dimensions,
            "version": self.version,
            "organization": self.organization,
            "dtype": self.dtype,
            "device": self.device,
        }


@dataclass(frozen=True)
class Latent:
    """A native tensor plus the information required to consume it safely."""

    tensor: torch.Tensor
    space: Space
    mask: torch.Tensor | None = None
    coordinates: torch.Tensor | None = None
    sources: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError("Latent tensor must be a torch.Tensor")
        if not isinstance(self.space, Space):
            raise TypeError("Latent space must be a Space")
        if self.tensor.ndim < 1 or self.tensor.shape[-1] != self.space.dimensions:
            raise ValueError(
                f"Latent feature dimension must be {self.space.dimensions}, "
                f"received shape {tuple(self.tensor.shape)}"
            )
        if self.space.dtype is not None and str(self.tensor.dtype) != self.space.dtype:
            raise ValueError(
                f"Latent dtype must be {self.space.dtype}, received {self.tensor.dtype}"
            )
        if self.space.device is not None and str(self.tensor.device) != self.space.device:
            raise ValueError(
                f"Latent device must be {self.space.device}, received {self.tensor.device}"
            )
        prefix = tuple(self.tensor.shape[:-1])
        if self.mask is not None:
            if not isinstance(self.mask, torch.Tensor) or tuple(self.mask.shape) != prefix:
                raise ValueError(f"Latent mask shape must be {prefix}")
            if self.mask.device != self.tensor.device:
                raise ValueError("Latent mask and tensor must use the same device")
        if self.coordinates is not None:
            if (
                not isinstance(self.coordinates, torch.Tensor)
                or self.coordinates.ndim != self.tensor.ndim
                or tuple(self.coordinates.shape[:-1]) != prefix
                or self.coordinates.shape[-1] < 1
            ):
                raise ValueError(
                    "Latent coordinates must have the tensor's leading shape "
                    "and a final coordinate dimension"
                )
            if self.coordinates.device != self.tensor.device:
                raise ValueError("Latent coordinates and tensor must use the same device")
        sources = tuple(self.sources)
        if not all(isinstance(source, str) for source in sources):
            raise TypeError("Latent sources must be strings")
        object.__setattr__(self, "sources", sources)
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        elif not isinstance(self.metadata, Mapping):
            raise TypeError("Latent metadata must be a mapping")
        else:
            object.__setattr__(self, "metadata", dict(self.metadata))

    def with_tensor(
        self,
        tensor: torch.Tensor,
        *,
        space: Space | None = None,
        mask: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> "Latent":
        """Carry provenance onto a new tensor without detaching autograd."""

        return Latent(
            tensor,
            self.space if space is None else space,
            mask=mask,
            coordinates=coordinates,
            sources=self.sources,
            metadata=self.metadata,
        )


def require_compatible(expected: Space, actual: Space, *, role: str = "input") -> None:
    if expected != actual:
        raise ValueError(
            f"{role} has incompatible vector space: expected {expected!r}, "
            f"received {actual!r}"
        )
