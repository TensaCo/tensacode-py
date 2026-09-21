import pytest

from tensorcode.ops import llm


def test_message_keeps_legacy_string_content_and_freezes_part_sequences():
    legacy = llm.Message("user", "hello")
    parts = [llm.TextPart("look", source_ref="ticket:1")]
    multipart = llm.Message("user", parts)
    parts.append(llm.TextPart("later"))

    assert legacy.content == "hello"
    assert multipart.content == (llm.TextPart("look", source_ref="ticket:1"),)


def test_image_encoder_preserves_bytes_and_urls_without_fetching():
    encoder = llm.ImageEncoder(media_type="image/png", source_ref="upload:7")

    encoded_bytes = encoder(b"\x89PNG")
    encoded_url = encoder("https://example.test/image.png")

    byte_part = encoded_bytes[0].content[0]
    url_part = encoded_url[0].content[0]
    assert byte_part == llm.ImagePart(
        data=b"\x89PNG", media_type="image/png", source_ref="upload:7"
    )
    assert url_part == llm.ImagePart(
        url="https://example.test/image.png",
        media_type="image/png",
        source_ref="upload:7",
    )


def test_image_encoder_preserves_an_explicit_image_part_without_overrides():
    explicit = llm.ImagePart(
        data=b"image", media_type="image/jpeg", source_ref="camera:1", detail="high"
    )
    encoded = llm.ImageEncoder(
        media_type="image/png", source_ref="encoder-default", detail="low"
    )(explicit)
    assert encoded == (llm.Message("user", (explicit,)),)


def test_image_part_requires_exactly_one_source():
    with pytest.raises(ValueError, match="exactly one"):
        llm.ImagePart()
    with pytest.raises(ValueError, match="exactly one"):
        llm.ImagePart(data=b"x", url="https://example.test/x")
    with pytest.raises(ValueError, match="URL"):
        llm.ImagePart(url="file:///tmp/private")


def test_text_decoder_handles_multipart_assistant_text_only():
    messages = (
        llm.Message("user", "hello"),
        llm.Message("assistant", (llm.TextPart("one"), llm.TextPart("two"))),
    )
    assert llm.TextDecoder()(messages) == "onetwo"
