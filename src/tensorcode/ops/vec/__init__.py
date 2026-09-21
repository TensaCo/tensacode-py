"""Optional tensor operations; import requires the vec extra."""
from .latent import Latent, Space
from .image import ImageEncoder
from .transform import Transform
from .classify import Classify, Prediction
from .candidates import CandidateSet, Scores
from .decode import Decode, Decoder
from .score import Score
from .decide import Decide, Decision
from .retrieve import Retrieve, Retrieval

__all__ = [
    'Latent', 'Space', 'ImageEncoder', 'Transform', 'Classify', 'Prediction',
    'CandidateSet', 'Scores', 'Decode', 'Decoder', 'Score', 'Decide',
    'Decision', 'Retrieve', 'Retrieval',
]
from .encode import TextEncoder
__all__.append('TextEncoder')
