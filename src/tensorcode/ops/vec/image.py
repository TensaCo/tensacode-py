"""Trainable image patch projection that retains patch-grid coordinates.

The default convolution starts from ordinary random PyTorch initialization.  It
is a trainable projection mechanism, not a pretrained image understanding model.
"""
from __future__ import annotations

import torch
from torch import nn

from ...tracing import invoke
from ._configuration import module_configuration, qualified_name
from .latent import Latent, Space


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    result = (value, value) if isinstance(value, int) else tuple(value)
    if len(result) != 2 or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in result):
        raise ValueError("patch_size must be a positive integer or pair")
    return result


def _coordinate_pair(value, *, name):
    result = (value, value) if isinstance(value, (int, float)) else tuple(value)
    if len(result) != 2 or any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in result):
        raise ValueError(f"{name} must be a number or pair")
    return tuple(float(v) for v in result)


class PatchEncoder(nn.Module):
    """Project CHW/BCHW images to channel-last spatial patch latents."""

    replayable = True

    def __init__(
        self,
        *,
        patch_size: int | tuple[int, int],
        space: Space,
        in_channels: int | None = None,
        dimensions: int | None = None,
        module: nn.Module | None = None,
        coordinate_stride=None,
        coordinate_offset=None,
    ) -> None:
        super().__init__()
        if not isinstance(space, Space) or space.organization != "spatial":
            raise ValueError("PatchEncoder requires a spatial Space")
        self.patch_size = _pair(patch_size)
        self.space = space
        if module is None:
            if not isinstance(in_channels, int) or in_channels <= 0:
                raise ValueError("in_channels is required for a newly initialized encoder")
            if dimensions is None:
                dimensions = space.dimensions
            if dimensions != space.dimensions:
                raise ValueError("PatchEncoder dimensions must match its space")
            module = nn.Conv2d(
                in_channels,
                dimensions,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
            self.initialization = "pytorch-random"
        else:
            if not isinstance(module, nn.Module):
                raise TypeError("PatchEncoder module must be a torch.nn.Module")
            if dimensions is not None and dimensions != space.dimensions:
                raise ValueError("PatchEncoder dimensions must match its space")
            self.initialization = "supplied"
        self.module = module
        if (coordinate_stride is None) != (coordinate_offset is None):
            raise ValueError("coordinate_stride and coordinate_offset must be supplied together")
        if coordinate_stride is not None:
            self.coordinate_stride = _coordinate_pair(coordinate_stride, name="coordinate_stride")
            self.coordinate_offset = _coordinate_pair(coordinate_offset, name="coordinate_offset")
        elif isinstance(module, nn.Conv2d):
            stride = tuple(float(value) for value in module.stride)
            effective_kernel = tuple(
                dilation * (kernel - 1) + 1
                for kernel, dilation in zip(module.kernel_size, module.dilation)
            )
            self.coordinate_stride = stride
            self.coordinate_offset = tuple(
                -padding + kernel / 2
                for padding, kernel in zip(module.padding, effective_kernel)
            )
        else:
            self.coordinate_stride = None
            self.coordinate_offset = None

    def __call__(self, value, *, context=None):
        return invoke(self, value, context, super().__call__)

    def forward(self, value, *, context=None):
        if context:
            raise ValueError("PatchEncoder does not consume context")
        if not isinstance(value, torch.Tensor) or value.ndim not in (3, 4):
            raise ValueError("PatchEncoder expects a CHW or BCHW tensor")
        single = value.ndim == 3
        batch = value.unsqueeze(0) if single else value
        encoded = self.module(batch)
        if not isinstance(encoded, torch.Tensor) or encoded.ndim != 4:
            raise ValueError("PatchEncoder module must return a BCHW tensor")
        if encoded.shape[0] != batch.shape[0]:
            raise ValueError("PatchEncoder module must preserve the input batch count")
        if encoded.shape[1] != self.space.dimensions:
            raise ValueError("PatchEncoder module output channels must match its space")

        rows, columns = encoded.shape[-2:]
        coordinates = None
        if self.coordinate_stride is not None:
            # Coordinates are actual receptive-field centers in source pixel space.
            row_centers = (
                torch.arange(rows, device=encoded.device, dtype=encoded.dtype)
                * self.coordinate_stride[0]
                + self.coordinate_offset[0]
            )
            column_centers = (
                torch.arange(columns, device=encoded.device, dtype=encoded.dtype)
                * self.coordinate_stride[1]
                + self.coordinate_offset[1]
            )
            row_grid, column_grid = torch.meshgrid(row_centers, column_centers, indexing="ij")
            coordinates = torch.stack((row_grid, column_grid), dim=-1)
            coordinates = coordinates.expand(encoded.shape[0], -1, -1, -1)
        result = encoded.permute(0, 2, 3, 1)
        if single:
            result = result[0]
            if coordinates is not None:
                coordinates = coordinates[0]
        return Latent(result, self.space, coordinates=coordinates)

    def configuration(self):
        return {
            "operation": qualified_name(self),
            "patch_size": list(self.patch_size),
            "space": self.space.configuration(),
            "initialization": self.initialization,
            "coordinate_stride": None if self.coordinate_stride is None else list(self.coordinate_stride),
            "coordinate_offset": None if self.coordinate_offset is None else list(self.coordinate_offset),
            "module": module_configuration(self.module),
        }


def __getattr__(name):
    if name in ('ImageEncoder', 'ImageEncode'):
        from .vision_model import ImageEncoder
        return ImageEncoder
    raise AttributeError(name)
