"""Encode external inputs into vector representations.

Public classes have stable operation identities. Native model implementations
are private and load their optional foundation dependencies on construction.
"""
from tensorcode._internal.vec.text import TextEncoder as _TextEncoder
from tensorcode._internal.vec.vision import ImageEncoder as _ImageEncoder
from tensorcode._internal.vec.vocabulary import VocabularyEncoder as _VocabularyEncoder
from tensorcode._internal.vec.patch import PatchEncoder as _PatchEncoder


class TextEncoder(_TextEncoder):
    """Owned text transformer with sequence or pooled vector readout."""


class ImageEncoder(_ImageEncoder):
    """Owned vision transformer with sequence or pooled vector readout."""


class VocabularyEncoder(_VocabularyEncoder):
    """Learn word embeddings from an explicit vocabulary using mean pooling.

    This specialized encoding operation starts from random weights and uses
    lowercase regex tokenization; it does not provide pretrained semantics.
    """


class PatchEncoder(_PatchEncoder):
    """Project image patches into a spatial output Space with source coordinates.

    The default convolution starts from random weights; supplied modules must
    satisfy the explicit geometry contract.
    """


TextEncode = TextEncoder
ImageEncode = ImageEncoder

__all__ = ['TextEncoder', 'ImageEncoder', 'VocabularyEncoder', 'PatchEncoder',
           'TextEncode', 'ImageEncode']
