import pytest
import torch

from tensorcode.ops.vec import Latent, Space, VocabularyEncoder, Transform


def test_latent_retains_tensor_gradient_and_metadata():
    space = Space("example/features", 3, version="2")
    tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    latent = Latent(
        tensor,
        space,
        sources=("sample:7",),
        metadata={"split": "train"},
    )

    latent.tensor.square().sum().backward()
    assert tensor.grad is not None
    assert tensor.grad.tolist() == [2.0, 4.0, 6.0]
    assert latent.sources == ("sample:7",)
    assert latent.metadata == {"split": "train"}


def test_transform_rejects_equal_shape_from_incompatible_space():
    expected = Space("model-a/text", 2)
    other = Space("model-b/text", 2)
    transform = Transform(
        torch.nn.Identity(), input_space=expected, output_space=expected
    )

    with pytest.raises(ValueError, match="incompatible.*space"):
        transform(Latent(torch.ones(2), other))


def test_space_aware_transform_preserves_provenance_and_gradients():
    input_space = Space("encoder/text", 2)
    output_space = Space("projector/shared", 3)
    source = torch.tensor([1.0, -1.0], requires_grad=True)
    transform = Transform(
        torch.nn.Linear(2, 3, bias=False),
        input_space=input_space,
        output_space=output_space,
    )

    result = transform(
        Latent(source, input_space, sources=("ticket:3",), metadata={"lang": "en"})
    )

    assert isinstance(result, Latent)
    assert result.space == output_space
    assert result.sources == ("ticket:3",)
    assert result.metadata == {"lang": "en"}
    result.tensor.sum().backward()
    assert source.grad is not None
    assert transform.module.weight.grad is not None


def test_text_encoder_space_is_opt_in_and_existing_tensor_api_remains():
    ordinary = VocabularyEncoder(vocabulary=("hello",), dimensions=4)
    configured = VocabularyEncoder(
        vocabulary=("hello",),
        dimensions=4,
        output_space=Space("local/text", 4),
    )

    assert isinstance(ordinary("hello"), torch.Tensor)
    encoded = configured(("hello", "unknown"))
    assert isinstance(encoded, Latent)
    assert encoded.tensor.shape == (2, 4)
    assert encoded.space == Space("local/text", 4)


def test_latent_validates_feature_mask_and_coordinate_shapes():
    space = Space("image/patches", 4, organization="spatial")

    with pytest.raises(ValueError, match="feature dimension"):
        Latent(torch.ones(2, 3), space)
    with pytest.raises(ValueError, match="mask"):
        Latent(torch.ones(2, 2, 4), space, mask=torch.ones(2))
    with pytest.raises(ValueError, match="coordinates"):
        Latent(torch.ones(2, 2, 4), space, coordinates=torch.ones(2, 2))


def test_space_can_declare_dtype_and_device_expectations():
    space = Space(
        "typed/features",
        2,
        dtype="torch.float32",
        device="cpu",
    )

    assert Latent(torch.ones(2), space).space == space
    with pytest.raises(ValueError, match="dtype"):
        Latent(torch.ones(2, dtype=torch.float64), space)


def test_latent_structural_tensors_share_the_data_device():
    space = Space("masked/features", 2)
    if not torch.cuda.is_available():
        pytest.skip("requires a second torch device")
    with pytest.raises(ValueError, match="same device"):
        Latent(torch.ones(2, device="cuda"), space, mask=torch.tensor(True))
