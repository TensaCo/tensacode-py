from ..base import Operation
from .messages import Message


class TextEncoder(Operation):
    replayable = True

    def forward(self, value, *, context=None):
        if context:
            raise ValueError('TextEncoder only serializes text; context belongs on a transform')
        return (Message('user', value),)
