import pytest
import torch

from tensorcode.ops.vec import PatchEncoder, Latent, Space, Transform


def test_image_encoder_returns_spatial_patch_grid_and_pixel_coordinates():
    encoder = PatchEncoder({
        'in_channels': 1,
        'patch_size': [2, 2],
        'dimensions': 3,
        'output_space': Space('random-image-patches', 3, organization='spatial').configuration(),
    })
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
    encoder = PatchEncoder({
        'in_channels': 1,
        'patch_size': 2,
        'dimensions': 1,
        'output_space': Space('odd-image-patches', 1, organization='spatial').configuration(),
    })

    result = encoder(torch.ones(1, 5, 5))

    assert result.tensor.shape == (2, 2, 1)
    assert result.coordinates[..., 0].tolist() == [[1.0, 1.0], [3.0, 3.0]]
    assert result.coordinates[..., 1].tolist() == [[1.0, 3.0], [1.0, 3.0]]


def test_image_encoder_batches_images_and_updates_real_parameters():
    encoder = PatchEncoder({
        'in_channels': 1,
        'patch_size': 2,
        'dimensions': 2,
        'output_space': Space('trainable-image-patches', 2, organization='spatial').configuration(),
    })
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
    encoder = PatchEncoder.from_module(supplied, patch_size=4, output_space=Space("caller-trained/model-x", 5, organization="spatial"))

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

    encoder = PatchEncoder.from_module(ArbitraryPatches(), patch_size=2, output_space=Space("caller-module", 1, organization="spatial"))

    assert encoder(torch.ones(1, 4, 4)).coordinates is None


def test_arbitrary_module_can_declare_spatial_coordinate_geometry():
    class PoolPatches(torch.nn.Module):
        def forward(self, value):
            return torch.nn.functional.avg_pool2d(value, 2)

    encoder = PatchEncoder.from_module(PoolPatches(), patch_size=2, output_space=Space("caller-module", 1, organization="spatial"), coordinate_stride=2, coordinate_offset=1)

    result = encoder(torch.ones(1, 5, 5))

    assert result.coordinates[..., 0].tolist() == [[1.0, 1.0], [3.0, 3.0]]
    assert encoder.configuration()["coordinate_stride"] == [2.0, 2.0]


def test_supplied_image_module_must_preserve_batch_count():
    class DropsBatch(torch.nn.Module):
        def forward(self, value):
            return value[:1]

    encoder = PatchEncoder.from_module(DropsBatch(), patch_size=1, output_space=Space("bad-module", 1, organization="spatial"))

    with pytest.raises(ValueError, match="batch"):
        encoder(torch.ones(2, 1, 2, 2))


def test_image_encoder_rejects_non_spatial_space_and_bad_image_shape():
    with pytest.raises(ValueError, match="spatial"):
        PatchEncoder({'in_channels': 3, 'patch_size': 2, 'dimensions': 4, 'output_space': Space("not-spatial", 4).configuration()})
    encoder = PatchEncoder({
        'in_channels': 3,
        'patch_size': 2,
        'dimensions': 4,
        'output_space': Space('patches', 4, organization='spatial').configuration(),
    })
    with pytest.raises(ValueError, match="CHW or BCHW"):
        encoder(torch.ones(8, 8))


def test_cross_modal_projection_requires_an_explicit_adapter_transform():
    image_space = Space("image-model/patches", 2, organization="spatial")
    shared_space = Space("paired-model/shared", 4, organization="spatial")
    image = PatchEncoder({
        'in_channels': 1,
        'patch_size': 2,
        'dimensions': 2,
        'output_space': image_space.configuration(),
    })(torch.ones(1, 4, 4))
    adapter = Transform.from_module(
        torch.nn.Linear(2, 4),
        input_space=image_space,
        output_space=shared_space,
    )

    projected = adapter(image)

    assert projected.space == shared_space
    assert projected.tensor.shape == (2, 2, 4)
    assert torch.equal(projected.coordinates, image.coordinates)


def test_owned_patch_artifact_restores_trained_weights_dtype_and_geometry(tmp_path):
    config = {
        'in_channels': 1, 'patch_size': [2, 3],
        'output_space': Space('owned-patches', 2, organization='spatial').configuration(),
        'coordinate_stride': [3, 4], 'coordinate_offset': [-1, 2],
    }
    encoder = PatchEncoder(config).double()
    images = torch.arange(48., dtype=torch.float64).reshape(2, 1, 4, 6) / 48
    encoder(images).tensor.square().mean().backward()
    torch.optim.SGD(encoder.parameters(), lr=.1).step()
    expected = encoder(images)
    encoder.save_pretrained(tmp_path / 'patches')
    restored = PatchEncoder.from_pretrained(tmp_path / 'patches')
    actual = restored(images)
    assert restored.configuration() == encoder.configuration()
    reconstructed = PatchEncoder(restored.configuration())
    assert reconstructed.configuration() == restored.configuration()
    assert restored.module.weight.dtype == torch.float64
    assert list(restored.state_dict()) == ['module.weight', 'module.bias']
    assert torch.equal(actual.tensor, expected.tensor)
    assert torch.equal(actual.coordinates, expected.coordinates)
    assert actual.space == expected.space
    # Construction and returned configuration cannot alias caller state.
    config['coordinate_offset'][0] = 99
    returned = restored.configuration()
    returned['coordinate_offset'][0] = 99
    assert restored.configuration()['coordinate_offset'] == [-1., 2.]


def test_supplied_patch_module_artifact_save_is_rejected_before_writing(tmp_path):
    encoder = PatchEncoder.from_module(
        torch.nn.Conv2d(1, 2, 2), patch_size=2,
        output_space=Space('supplied', 2, organization='spatial'),
    )
    destination = tmp_path / 'unsupported'
    with pytest.raises(ValueError, match='supplied'):
        encoder.save_pretrained(destination)
    assert not destination.exists()


@pytest.mark.parametrize('changes', [
    {'module': torch.nn.Identity()}, {'obsolete': True}, {'in_channels': True},
    {'dimensions': True}, {'coordinate_stride': 2},
    {'coordinate_stride': 2, 'coordinate_offset': float('nan')},
])
def test_owned_patch_config_rejects_unsupported_or_invalid_fields(changes):
    config = {'in_channels': 1, 'patch_size': 2,
              'output_space': Space('patches', 2, organization='spatial').configuration()}
    with pytest.raises((TypeError, ValueError)):
        PatchEncoder(config | changes)


def test_patch_constructor_rejects_legacy_keyword_arguments():
    with pytest.raises(TypeError):
        PatchEncoder(patch_size=2, in_channels=1,
                     output_space=Space('patches', 2, organization='spatial'))
