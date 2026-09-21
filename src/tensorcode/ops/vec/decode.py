"""Owned tensor readouts and concrete pretrained modality decoders."""
from .transform import Transform


class Decode(Transform):
    """Decode to a tensor with an explicit output width and description."""
    kind = 'decode'


Decoder = Decode


from tensorcode._internal.vec.text import TextDecoder as _TextDecoder
from tensorcode._internal.vec.diffusion import ImageDecoder as _ImageDecoder


class TextDecoder(_TextDecoder):
    """Generate text from vectors through an owned pretrained transformer."""


class ImageDecoder(_ImageDecoder):
    """Generate RGB images from vectors through owned latent diffusion."""


TextDecode = TextDecoder
ImageDecode = ImageDecoder

__all__ = ['Decode', 'Decoder', 'TextDecoder', 'ImageDecoder', 'TextDecode', 'ImageDecode']
