"""Vector operations, organized by operation with lazy convenience exports.

Import concrete classes from ``encode`` or ``decode`` for canonical identities.
Foundation libraries are loaded only when a corresponding model is constructed.
"""
from importlib import import_module as _import_module

_EXPORTS = {
    'Latent': ('latent', 'Latent'), 'Space': ('latent', 'Space'),
    'Transform': ('transform', 'Transform'),
    'Classify': ('classify', 'Classify'), 'Prediction': ('classify', 'Prediction'),
    'CandidateSet': ('candidates', 'CandidateSet'), 'Scores': ('candidates', 'Scores'),
    'Score': ('score', 'Score'), 'Decide': ('decide', 'Decide'), 'Decision': ('decide', 'Decision'),
    'Retrieve': ('retrieve', 'Retrieve'), 'Retrieval': ('retrieve', 'Retrieval'),
    'TextEncoder': ('encode', 'TextEncoder'), 'TextEncode': ('encode', 'TextEncoder'),
    'ImageEncoder': ('encode', 'ImageEncoder'), 'ImageEncode': ('encode', 'ImageEncoder'),
    'VocabularyEncoder': ('encode', 'VocabularyEncoder'), 'PatchEncoder': ('encode', 'PatchEncoder'),
    'Decode': ('decode', 'Decode'), 'Decoder': ('decode', 'Decoder'),
    'TextDecoder': ('decode', 'TextDecoder'), 'TextDecode': ('decode', 'TextDecoder'),
    'ImageDecoder': ('decode', 'ImageDecoder'), 'ImageDecode': ('decode', 'ImageDecoder'),
}
__all__ = [*_EXPORTS, 'latent_codecs']


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(_import_module(f'{__name__}.{module}'), symbol)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


def latent_codecs():
    """Explicit trusted types for durable vector experience serialization."""
    from .latent import Latent, Space
    return {'tensorcode.Latent': Latent, 'tensorcode.Space': Space}
