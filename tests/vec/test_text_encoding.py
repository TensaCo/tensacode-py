import torch
from tensorcode.ops.vec import TextEncoder


def test_text_encoder_batches_strings_and_trains_embedding_parameters():
    encoder = TextEncoder(vocabulary=('hello', 'world'), dimensions=4)
    result = encoder(('hello world', 'unknown', ''))
    assert result.shape == (3, 4)
    assert torch.isfinite(result).all()
    result.sum().backward()
    assert encoder.embedding.weight.grad is not None
    assert encoder('hello').shape == (4,)


def test_text_encoder_rejects_nontext_without_stringifying_it():
    import pytest
    encoder = TextEncoder(vocabulary=('hello',), dimensions=4)
    with pytest.raises(TypeError):
        encoder((object(),))


def test_serialized_encoder_weights_produce_identical_encodings():
    a = TextEncoder(vocabulary=('hello', 'world'), dimensions=4)
    b = TextEncoder(vocabulary=('hello', 'world'), dimensions=4)
    b.load_state_dict(a.state_dict())
    assert torch.equal(a(('hello', 'world')), b(('hello', 'world')))
