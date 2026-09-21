"""Decode a known vector space through an explicit caller-supplied module."""
from __future__ import annotations

from torch import nn

from ._configuration import module_configuration, qualified_name
from .latent import Latent, Space
from .transform import Transform


class Decode(Transform):
    def __init__(self, module: nn.Module, *, input_space: Space, output: str) -> None:
        if not isinstance(output, str) or not output.strip():
            raise ValueError("Decode output description must be a nonempty string")
        super().__init__(module, input_space=input_space)
        self.output = output

    def configuration(self):
        return {
            "operation": qualified_name(self),
            "input_space": self.input_space.configuration(),
            "output": self.output,
            "module": module_configuration(self.module),
        }


Decoder = Decode


def __getattr__(name):
    if name in ('TextDecoder', 'TextDecode'):
        from .text_model import TextDecoder
        return TextDecoder
    if name in ('ImageDecoder', 'ImageDecode'):
        from .diffusion import ImageDecoder
        return ImageDecoder
    raise AttributeError(name)
