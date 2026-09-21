from ..base import Operation


class TextDecoder(Operation):
    replayable = True

    def forward(self, value, *, context=None):
        if context:
            raise ValueError('TextDecoder does not consume context')
        if not value or value[-1].role != 'assistant':
            raise ValueError('Expected a final assistant message')
        return value[-1].content
