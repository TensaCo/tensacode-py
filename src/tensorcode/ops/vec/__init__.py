"""Optional tensor operations; import requires the vec extra."""
from .transform import Transform
from .classify import Classify, Prediction

__all__ = ['Transform', 'Classify', 'Prediction']
from .encode import TextEncoder
__all__.append('TextEncoder')
