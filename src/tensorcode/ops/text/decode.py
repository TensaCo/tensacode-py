from ..base import Operation
from ..._internal.operation_config import ConfigOperationMixin
from .messages import ImagePart, TextPart


class TextDecoder(ConfigOperationMixin, Operation):
    """Return the text of a final assistant message; image content is rejected."""
    replayable = True

    def forward(self, value, *, context=None):
        if context:
            raise ValueError('TextDecoder does not consume context')
        if not value or value[-1].role != 'assistant':
            raise ValueError('Expected a final assistant message')
        content = value[-1].content
        if isinstance(content, str):
            return content
        if any(isinstance(part, ImagePart) for part in content):
            raise ValueError("Assistant message contains an image and cannot be decoded as text")
        return "".join(part.text for part in content if isinstance(part, TextPart))
