import pytest

from tensorcode.ops import text as text_ops


def test_message_keeps_legacy_string_content_and_freezes_part_sequences():
    legacy = text_ops.Message("user", "hello")
    parts = [text_ops.TextPart("look", source_ref="ticket:1")]
    multipart = text_ops.Message("user", parts)
    parts.append(text_ops.TextPart("later"))

    assert legacy.content == "hello"
    assert multipart.content == (text_ops.TextPart("look", source_ref="ticket:1"),)


def test_image_encoder_preserves_bytes_and_urls_without_fetching():
    encoder = text_ops.ImageEncoder({"media_type":"image/png", "source_ref":"upload:7"})

    encoded_bytes = encoder(b"\x89PNG")
    encoded_url = encoder("https://example.test/image.png")

    byte_part = encoded_bytes[0].content[0]
    url_part = encoded_url[0].content[0]
    assert byte_part == text_ops.ImagePart(
        data=b"\x89PNG", media_type="image/png", source_ref="upload:7"
    )
    assert url_part == text_ops.ImagePart(
        url="https://example.test/image.png",
        media_type="image/png",
        source_ref="upload:7",
    )


def test_image_encoder_preserves_an_explicit_image_part_without_overrides():
    explicit = text_ops.ImagePart(
        data=b"image", media_type="image/jpeg", source_ref="camera:1", detail="high"
    )
    encoded = text_ops.ImageEncoder({
        "media_type":"image/png", "source_ref":"encoder-default", "detail":"low"
    })(explicit)
    assert encoded == (text_ops.Message("user", (explicit,)),)


def test_image_part_requires_exactly_one_source():
    with pytest.raises(ValueError, match="exactly one"):
        text_ops.ImagePart()
    with pytest.raises(ValueError, match="exactly one"):
        text_ops.ImagePart(data=b"x", url="https://example.test/x")
    with pytest.raises(ValueError, match="URL"):
        text_ops.ImagePart(url="file:///tmp/private")


def test_text_decoder_handles_multipart_assistant_text_only():
    messages = (
        text_ops.Message("user", "hello"),
        text_ops.Message("assistant", (text_ops.TextPart("one"), text_ops.TextPart("two"))),
    )
    assert text_ops.TextDecoder()(messages) == "onetwo"


def test_image_encoder_artifact_preserves_current_source_settings(tmp_path):
    from tensorcode.ops.text import ImageEncoder
    encoder = ImageEncoder({'source_ref': 'source:before'})
    encoder.source_ref = 'source:after'
    encoder.save_pretrained(tmp_path)
    restored = ImageEncoder.from_pretrained(tmp_path)
    assert restored(b'image') == encoder(b'image')
    assert restored.configuration()['source_ref'] == 'source:after'
