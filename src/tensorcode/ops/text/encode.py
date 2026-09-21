from ..base import Operation
from ..._internal.operation_config import ConfigOperationMixin
from .messages import ImagePart, Message


class TextEncoder(ConfigOperationMixin, Operation):
    replayable = True

    def forward(self, value, *, context=None):
        if context:
            raise ValueError('TextEncoder only serializes text; context belongs on a transform')
        return (Message('user', value),)


class ImageEncoder(ConfigOperationMixin, Operation):
    """Serialize image bytes or a URL without interpreting or fetching it."""

    replayable = True

    config_keys = frozenset({'media_type', 'source_ref', 'detail'})
    config_defaults = {'media_type': None, 'source_ref': None, 'detail': None}

    def __init__(self, config=None):
        super().__init__(config)
        settings = super().configuration()
        self.media_type = settings['media_type']
        self.source_ref = settings['source_ref']
        self.detail = settings['detail']
        ImagePart(url='https://example.invalid/image', media_type=self.media_type,
                  source_ref=self.source_ref, detail=self.detail)

    def configuration(self):
        from ..._internal.operation_config import validated_config
        return validated_config({
            'media_type': self.media_type,
            'source_ref': self.source_ref,
            'detail': self.detail,
        }, self.config_keys)

    def forward(self, value, *, context=None):
        if context:
            raise ValueError("ImageEncoder only serializes an image; context belongs on a transform")
        if isinstance(value, ImagePart):
            # An explicit part is authoritative; encoder defaults apply only to
            # raw bytes/URLs and never overwrite source metadata.
            part = value
        elif isinstance(value, bytes):
            part = ImagePart(
                data=value,
                media_type=self.media_type,
                source_ref=self.source_ref,
                detail=self.detail,
            )
        elif isinstance(value, str):
            part = ImagePart(
                url=value,
                media_type=self.media_type,
                source_ref=self.source_ref,
                detail=self.detail,
            )
        else:
            raise TypeError("ImageEncoder expects ImagePart, bytes or an image URL")
        return (Message("user", (part,)),)
