"""Trainable image patch projection that retains patch-grid coordinates.

The default convolution starts from ordinary random PyTorch initialization.  It
is a trainable projection mechanism, not a pretrained image understanding model.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from tensorcode._internal.latent_ops import LatentOperation
from tensorcode._internal.operation_config import validated_config
from tensorcode.ops.vec._configuration import module_configuration
from tensorcode.ops.vec.latent import Latent, Space


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    result = (value, value) if isinstance(value, int) else tuple(value)
    if len(result) != 2 or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in result):
        raise ValueError("patch_size must be a positive integer or pair")
    return result


def _coordinate_pair(value, *, name):
    result = (value, value) if isinstance(value, (int, float)) else tuple(value)
    if len(result) != 2 or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in result):
        raise ValueError(f"{name} must be a number or pair")
    return tuple(float(v) for v in result)


class PatchEncoder(LatentOperation):
    """Project CHW/BCHW images to channel-last spatial patch latents."""

    replayable = True

    def __init__(self, config):
        config = validated_config(config, {
            'patch_size', 'output_space', 'in_channels', 'dimensions',
            'coordinate_stride', 'coordinate_offset',
        })
        if 'patch_size' not in config or 'output_space' not in config:
            raise ValueError('PatchEncoder requires patch_size and output_space')
        if not isinstance(config['output_space'], dict):
            raise TypeError('output_space must be a Space configuration object')
        output_space = Space(**config['output_space'])
        if output_space.organization != 'spatial':
            raise ValueError('PatchEncoder requires a spatial Space')
        patch_size = _pair(config['patch_size'])
        in_channels = config.get('in_channels')
        if type(in_channels) is not int or in_channels <= 0:
            raise ValueError('in_channels is required for a newly initialized encoder')
        dimensions = config.get('dimensions', output_space.dimensions)
        if type(dimensions) is not int or dimensions != output_space.dimensions:
            raise ValueError('PatchEncoder dimensions must match its output_space')
        config.update(patch_size=list(patch_size), dimensions=dimensions)
        super().__init__(config)
        self.patch_size = patch_size
        self.output_space = output_space
        self.initialization = 'pytorch-random'
        self.module = nn.Conv2d(in_channels, dimensions, kernel_size=patch_size, stride=patch_size)
        self._set_geometry(config.get('coordinate_stride'), config.get('coordinate_offset'))
        self.config.update(
            coordinate_stride=list(self.coordinate_stride),
            coordinate_offset=list(self.coordinate_offset),
        )

    @classmethod
    def from_module(cls, module, *, patch_size, output_space, in_channels=None,
                    dimensions=None, coordinate_stride=None, coordinate_offset=None):
        if not isinstance(module, nn.Module):
            raise TypeError('PatchEncoder module must be a torch.nn.Module')
        if not isinstance(output_space, Space) or output_space.organization != 'spatial':
            raise ValueError('PatchEncoder requires a spatial Space')
        if dimensions is not None and (type(dimensions) is not int or dimensions != output_space.dimensions):
            raise ValueError('PatchEncoder dimensions must match its output_space')
        result = cls.__new__(cls)
        LatentOperation.__init__(result, {})
        result.patch_size = _pair(patch_size)
        result.output_space = output_space
        result.initialization = 'supplied'
        result.module = module
        result._set_geometry(coordinate_stride, coordinate_offset)
        result.config = {
            'patch_size': list(result.patch_size),
            'output_space': output_space.configuration(),
            'dimensions': output_space.dimensions,
        }
        if in_channels is not None:
            if type(in_channels) is not int or in_channels <= 0:
                raise ValueError('in_channels must be a positive integer')
            result.config['in_channels'] = in_channels
        return result

    def _set_geometry(self, coordinate_stride, coordinate_offset):
        if (coordinate_stride is None) != (coordinate_offset is None):
            raise ValueError('coordinate_stride and coordinate_offset must be supplied together')
        if coordinate_stride is not None:
            self.coordinate_stride = _coordinate_pair(coordinate_stride, name='coordinate_stride')
            self.coordinate_offset = _coordinate_pair(coordinate_offset, name='coordinate_offset')
        elif isinstance(self.module, nn.Conv2d):
            self.coordinate_stride = tuple(float(value) for value in self.module.stride)
            effective_kernel = tuple(
                dilation * (kernel - 1) + 1
                for kernel, dilation in zip(self.module.kernel_size, self.module.dilation)
            )
            self.coordinate_offset = tuple(
                -padding + kernel / 2
                for padding, kernel in zip(self.module.padding, effective_kernel)
            )
        else:
            self.coordinate_stride = self.coordinate_offset = None

    def save_pretrained(self, directory):
        if self.initialization == 'supplied':
            raise ValueError('Cannot save a supplied PatchEncoder module as a reconstructible artifact')
        return super().save_pretrained(directory)

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
        if encoded.shape[1] != self.output_space.dimensions:
            raise ValueError("PatchEncoder module output channels must match its output_space")

        rows, columns = encoded.shape[-2:]
        coordinates = None
        if self.coordinate_stride is not None:
            # Coordinates are actual receptive-field centers in source pixel output_space.
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
        return Latent(result, self.output_space, coordinates=coordinates)

    def configuration(self):
        result = super().configuration()
        result.update(
            coordinate_stride=None if self.coordinate_stride is None else list(self.coordinate_stride),
            coordinate_offset=None if self.coordinate_offset is None else list(self.coordinate_offset),
        )
        if self.initialization == 'supplied':
            result['module'] = module_configuration(self.module)
            result['initialization'] = 'supplied'
        return result
