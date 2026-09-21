"""Optional tensor operations; import requires the vec extra."""
from .latent import Latent, Space
from .image import PatchEncoder
from .transform import Transform
from .classify import Classify, Prediction
from .candidates import CandidateSet, Scores
from .decode import Decode, Decoder
from .score import Score
from .decide import Decide, Decision
from .retrieve import Retrieve, Retrieval

__all__ = [
    'Latent', 'Space', 'PatchEncoder', 'Transform', 'Classify', 'Prediction',
    'CandidateSet', 'Scores', 'Decode', 'Decoder', 'Score', 'Decide',
    'Decision', 'Retrieve', 'Retrieval',
]
from .encode import VocabularyEncoder
__all__.append('VocabularyEncoder')

# Foundation dependencies remain optional until a pretrained operation is used.
_PRETRAINED = {
    'TextEncoder': ('text_model', 'TextEncoder'),
    'TextEncode': ('text_model', 'TextEncoder'),
    'TextDecoder': ('text_model', 'TextDecoder'),
    'TextDecode': ('text_model', 'TextDecoder'),
    'ImageEncoder': ('vision_model', 'ImageEncoder'),
    'ImageEncode': ('vision_model', 'ImageEncoder'),
    'ImageDecoder': ('diffusion', 'ImageDecoder'),
    'ImageDecode': ('diffusion', 'ImageDecoder'),
}
__all__ += list(_PRETRAINED)

def __getattr__(name):
    if name not in _PRETRAINED:
        raise AttributeError(name)
    from importlib import import_module
    module, symbol = _PRETRAINED[name]
    result = getattr(import_module(f'{__name__}.{module}'), symbol)
    globals()[name] = result
    return result


def latent_codecs():
    """Explicit allowlist for durable Latent/Space experience serialization."""
    return {'tensorcode.Latent': Latent, 'tensorcode.Space': Space}

__all__.append('latent_codecs')
