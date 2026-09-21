from ..base import Operation
from .messages import ImagePart, Message


class TextEncoder(Operation):
    replayable = True

    def forward(self, value, *, context=None):
        if context:
            raise ValueError('TextEncoder only serializes text; context belongs on a transform')
        return (Message('user', value),)


class ImageEncoder(Operation):
    """Serialize image bytes or a URL without interpreting or fetching it."""

    replayable = True

    def __init__(self, *, media_type=None, source_ref=None, detail=None):
        self.media_type = media_type
        self.source_ref = source_ref
        self.detail = detail

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

    def configuration(self):
        return {
            "type": "text_image_encoder",
            "media_type": self.media_type,
            "source_ref": self.source_ref,
            "detail": self.detail,
        }
