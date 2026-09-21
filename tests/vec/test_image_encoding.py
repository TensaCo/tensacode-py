import pytest
import torch

from tensorcode.ops.vec import PatchEncoder, Latent, Space, Transform


def test_image_encoder_returns_spatial_patch_grid_and_pixel_coordinates():
    encoder = PatchEncoder(
        in_channels=1,
        patch_size=(2, 2),
        dimensions=3,
        space=Space("random-image-patches", 3, organization="spatial"),
    )
    image = torch.arange(24.0).reshape(1, 4, 6)

    result = encoder(image)

    assert isinstance(result, Latent)
    assert result.tensor.shape == (2, 3, 3)
    assert result.coordinates.shape == (2, 3, 2)
    assert result.coordinates.tolist() == [
        [[1.0, 1.0], [1.0, 3.0], [1.0, 5.0]],
        [[3.0, 1.0], [3.0, 3.0], [3.0, 5.0]],
    ]


def test_default_patch_coordinates_do_not_rescale_over_unused_border_pixels():
    encoder = PatchEncoder(
        in_channels=1,
        patch_size=2,
        dimensions=1,
        space=Space("odd-image-patches", 1, organization="spatial"),
    )

    result = encoder(torch.ones(1, 5, 5))

    assert result.tensor.shape == (2, 2, 1)
    assert result.coordinates[..., 0].tolist() == [[1.0, 1.0], [3.0, 3.0]]
    assert result.coordinates[..., 1].tolist() == [[1.0, 3.0], [1.0, 3.0]]


def test_image_encoder_batches_images_and_updates_real_parameters():
    encoder = PatchEncoder(
        in_channels=1,
        patch_size=2,
        dimensions=2,
        space=Space("trainable-image-patches", 2, organization="spatial"),
    )
    images = torch.stack((torch.zeros(1, 4, 4), torch.ones(1, 4, 4)))
    before = encoder.module.weight.detach().clone()

    loss = encoder(images).tensor.square().mean()
    loss.backward()
    torch.optim.SGD(encoder.parameters(), lr=0.1).step()

    assert encoder(images).tensor.shape == (2, 2, 2, 2)
    assert encoder.module.weight.grad is not None
    assert not torch.equal(before, encoder.module.weight.detach())


def test_image_encoder_exposes_supplied_module_without_claiming_semantics():
    supplied = torch.nn.Conv2d(3, 5, kernel_size=4, stride=4, bias=False)
    encoder = PatchEncoder(
        patch_size=4,
        space=Space("caller-trained/model-x", 5, organization="spatial"),
        module=supplied,
    )

    assert encoder.module is supplied
    result = encoder(torch.ones(3, 8, 8))
    assert result.tensor.shape == (2, 2, 5)
    assert result.coordinates.tolist() == [
        [[2.0, 2.0], [2.0, 6.0]],
        [[6.0, 2.0], [6.0, 6.0]],
    ]
    assert encoder.initialization == "supplied"


def test_arbitrary_supplied_module_does_not_invent_patch_coordinates():
    class ArbitraryPatches(torch.nn.Module):
        def forward(self, value):
            return torch.nn.functional.avg_pool2d(value, 2)

    encoder = PatchEncoder(
        patch_size=2,
        space=Space("caller-module", 1, organization="spatial"),
        module=ArbitraryPatches(),
    )

    assert encoder(torch.ones(1, 4, 4)).coordinates is None


def test_arbitrary_module_can_declare_spatial_coordinate_geometry():
    class PoolPatches(torch.nn.Module):
        def forward(self, value):
            return torch.nn.functional.avg_pool2d(value, 2)

    encoder = PatchEncoder(
        patch_size=2,
        space=Space("caller-module", 1, organization="spatial"),
        module=PoolPatches(),
        coordinate_stride=2,
        coordinate_offset=1,
    )

    result = encoder(torch.ones(1, 5, 5))

    assert result.coordinates[..., 0].tolist() == [[1.0, 1.0], [3.0, 3.0]]
    assert encoder.configuration()["coordinate_stride"] == [2.0, 2.0]


def test_supplied_image_module_must_preserve_batch_count():
    class DropsBatch(torch.nn.Module):
        def forward(self, value):
            return value[:1]

    encoder = PatchEncoder(
        patch_size=1,
        space=Space("bad-module", 1, organization="spatial"),
        module=DropsBatch(),
    )

    with pytest.raises(ValueError, match="batch"):
        encoder(torch.ones(2, 1, 2, 2))


def test_image_encoder_rejects_non_spatial_space_and_bad_image_shape():
    with pytest.raises(ValueError, match="spatial"):
        PatchEncoder(
            in_channels=3,
            patch_size=2,
            dimensions=4,
            space=Space("not-spatial", 4),
        )
    encoder = PatchEncoder(
        in_channels=3,
        patch_size=2,
        dimensions=4,
        space=Space("patches", 4, organization="spatial"),
    )
    with pytest.raises(ValueError, match="CHW or BCHW"):
        encoder(torch.ones(8, 8))


def test_cross_modal_projection_requires_an_explicit_adapter_transform():
    image_space = Space("image-model/patches", 2, organization="spatial")
    shared_space = Space("paired-model/shared", 4, organization="spatial")
    image = PatchEncoder(
        in_channels=1,
        patch_size=2,
        dimensions=2,
        space=image_space,
    )(torch.ones(1, 4, 4))
    adapter = Transform(
        torch.nn.Linear(2, 4),
        input_space=image_space,
        output_space=shared_space,
    )

    projected = adapter(image)

    assert projected.space == shared_space
    assert projected.tensor.shape == (2, 2, 4)
    assert torch.equal(projected.coordinates, image.coordinates)
