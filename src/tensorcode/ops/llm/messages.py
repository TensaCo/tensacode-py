from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit


@dataclass(frozen=True)
class TextPart:
    """An immutable text segment with an optional caller-owned source ID."""

    text: str
    source_ref: str | None = None

    def __post_init__(self):
        if not isinstance(self.text, str):
            raise TypeError("TextPart.text must be a string")
        if self.source_ref is not None and not isinstance(self.source_ref, str):
            raise TypeError("TextPart.source_ref must be a string or None")


@dataclass(frozen=True)
class ImagePart:
    """Image bytes or a remote reference; construction never dereferences URLs."""

    data: bytes | None = None
    url: str | None = None
    media_type: str | None = None
    source_ref: str | None = None
    detail: Literal["low", "high", "auto"] | None = None

    def __post_init__(self):
        if (self.data is None) == (self.url is None):
            raise ValueError("ImagePart requires exactly one of data or url")
        if self.data is not None and not isinstance(self.data, bytes):
            raise TypeError("ImagePart.data must be bytes")
        if self.url is not None:
            if not isinstance(self.url, str):
                raise TypeError("ImagePart.url must be a string")
            if urlsplit(self.url).scheme not in ("http", "https", "data"):
                raise ValueError("ImagePart URL must use http, https or data")
        if self.media_type is not None and not isinstance(self.media_type, str):
            raise TypeError("ImagePart.media_type must be a string or None")
        if self.source_ref is not None and not isinstance(self.source_ref, str):
            raise TypeError("ImagePart.source_ref must be a string or None")
        if self.detail not in (None, "low", "high", "auto"):
            raise ValueError("ImagePart.detail must be low, high, auto or None")


@dataclass(frozen=True)
class Message:
    role: str
    content: str | tuple[TextPart | ImagePart, ...]

    def __post_init__(self):
        if self.role not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError('Unsupported message role')
        if isinstance(self.content, str):
            return
        try:
            content = tuple(self.content)
        except TypeError as exc:
            raise TypeError("Message content must be text or a sequence of parts") from exc
        if not content or not all(isinstance(part, (TextPart, ImagePart)) for part in content):
            raise TypeError("Multipart message content requires TextPart or ImagePart values")
        object.__setattr__(self, "content", content)
