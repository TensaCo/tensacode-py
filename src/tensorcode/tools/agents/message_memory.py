"""JSON data codec for public LLM message sequences."""

from __future__ import annotations

import base64
from collections.abc import Sequence

from ...ops.llm import ImagePart, Message, TextPart


def encode_message_sequence(value):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("message memory values must be Message sequences")
    messages = []
    for message in value:
        if not isinstance(message, Message):
            raise TypeError("message memory values must contain Message objects")
        if isinstance(message.content, str):
            content = {"kind": "string", "text": message.content}
        else:
            parts = []
            for part in message.content:
                if isinstance(part, TextPart):
                    parts.append(
                        {
                            "kind": "text",
                            "text": part.text,
                            "source_ref": part.source_ref,
                        }
                    )
                elif isinstance(part, ImagePart):
                    parts.append(
                        {
                            "kind": "image",
                            "data": (
                                base64.b64encode(part.data).decode("ascii")
                                if part.data is not None
                                else None
                            ),
                            "url": part.url,
                            "media_type": part.media_type,
                            "source_ref": part.source_ref,
                            "detail": part.detail,
                        }
                    )
                else:  # pragma: no cover - Message validates this itself
                    raise TypeError("unsupported message part")
            content = {"kind": "parts", "parts": parts}
        messages.append({"role": message.role, "content": content})
    return {"format": "tensorcode-messages", "version": 1, "messages": messages}


def decode_message_sequence(payload):
    if not isinstance(payload, dict) or payload.get("format") != "tensorcode-messages":
        raise ValueError("invalid message memory value")
    if payload.get("version") != 1 or not isinstance(payload.get("messages"), list):
        raise ValueError("unsupported or malformed message memory value")
    messages = []
    for raw_message in payload["messages"]:
        if not isinstance(raw_message, dict):
            raise ValueError("malformed message memory entry")
        content = raw_message.get("content")
        if not isinstance(content, dict):
            raise ValueError("malformed message memory content")
        if content.get("kind") == "string":
            decoded_content = content.get("text")
        elif content.get("kind") == "parts":
            raw_parts = content.get("parts")
            if not isinstance(raw_parts, list):
                raise ValueError("malformed message parts")
            decoded_parts = []
            for raw_part in raw_parts:
                if not isinstance(raw_part, dict):
                    raise ValueError("malformed message part")
                if raw_part.get("kind") == "text":
                    decoded_parts.append(
                        TextPart(raw_part.get("text"), raw_part.get("source_ref"))
                    )
                elif raw_part.get("kind") == "image":
                    encoded_data = raw_part.get("data")
                    try:
                        data = (
                            base64.b64decode(encoded_data, validate=True)
                            if encoded_data is not None
                            else None
                        )
                    except (TypeError, ValueError) as error:
                        raise ValueError("malformed base64 image data") from error
                    decoded_parts.append(
                        ImagePart(
                            data=data,
                            url=raw_part.get("url"),
                            media_type=raw_part.get("media_type"),
                            source_ref=raw_part.get("source_ref"),
                            detail=raw_part.get("detail"),
                        )
                    )
                else:
                    raise ValueError("unsupported message part kind")
            decoded_content = tuple(decoded_parts)
        else:
            raise ValueError("unsupported message content kind")
        messages.append(Message(raw_message.get("role"), decoded_content))
    return tuple(messages)
