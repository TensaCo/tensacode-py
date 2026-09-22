"""The reusable message codec preserves multimodal source evidence."""
import json

import pytest

from tensorcode.ops.text import ImagePart, Message, TextPart
from tensorcode._internal.memory.messages import decode_message_sequence, encode_message_sequence


def test_runtime_message_codec_roundtrips_text_images_and_sources():
    messages = (
        Message('user', (
            TextPart('inspect this', source_ref='document:1'),
            ImagePart(data=b'\x00\xffimage', media_type='image/png', source_ref='image:1'),
        )),
        Message('assistant', 'Two possibilities remain.'),
    )
    encoded = encode_message_sequence(messages)
    assert decode_message_sequence(json.loads(json.dumps(encoded))) == messages


def test_runtime_message_codec_rejects_corrupt_image_bytes():
    payload = encode_message_sequence((Message('user', (
        ImagePart(data=b'image', media_type='image/png'),
    )),))
    payload['messages'][0]['content']['parts'][0]['data'] = '%%%'
    with pytest.raises(ValueError, match='base64'):
        decode_message_sequence(payload)
