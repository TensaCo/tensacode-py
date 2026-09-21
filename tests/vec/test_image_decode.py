"""Real tiny diffusion models exercise conditioning, diffusion and artifacts."""
import pytest
import torch

from tensorcode.ops.vec.diffusion import ImageDecoder
from tensorcode.ops.vec.latent import Latent, Space


@pytest.fixture
def model():
    return ImageDecoder({
        'input_space': Space('image-conditioning', 6, organization='sequence').configuration(),
        'unet_config': dict(sample_size=4, in_channels=4, out_channels=4,
            down_block_types=['CrossAttnDownBlock2D'], up_block_types=['CrossAttnUpBlock2D'],
            block_out_channels=[8], layers_per_block=1, norm_num_groups=4,
            cross_attention_dim=8, attention_head_dim=2),
        'vae_config': dict(in_channels=3, out_channels=3, latent_channels=4,
            down_block_types=['DownEncoderBlock2D'], up_block_types=['UpDecoderBlock2D'],
            block_out_channels=[8], layers_per_block=1, norm_num_groups=4, sample_size=4),
        'scheduler_config': dict(num_train_timesteps=10, clip_sample=False),
        'num_inference_steps': 2,
    })


def test_real_reverse_diffusion_replay_masks_context_and_rng(model):
    value = Latent(torch.randn(1, 2, 6), model.input_space, mask=torch.tensor([[True, False]]))
    noise = torch.randn(1, 4, 4, 4)
    model.train()
    state = torch.random.get_rng_state().clone()
    image = model(value, context={'noise': noise})
    assert model.training
    assert torch.equal(state, torch.random.get_rng_state())
    assert image.shape == (1, 3, 4, 4)
    assert image.min() >= 0 and image.max() <= 1
    assert torch.equal(image, model(value, context={'noise': noise}))
    altered = value.tensor.clone(); altered[:, 1] += 100
    assert torch.equal(image, model(Latent(altered, value.space, mask=value.mask), context={'noise': noise}))
    context = {'noise': noise, 'latents': [Latent(torch.randn(1, 1, 6), value.space)]}
    assert not torch.allclose(image, model(value, context=context))
    assert torch.equal(model(value, context={'seed': 42}), model(value, context={'seed': 42}))
    with pytest.raises(ValueError, match='seed|noise'):
        model(value)


def test_diffusion_loss_trains_conditioning_and_denoiser(model):
    tensor = torch.randn(1, 2, 6, requires_grad=True)
    value = Latent(tensor, model.input_space)
    loss = model.loss(value, torch.rand(1, 3, 4, 4), noise=torch.randn(1, 4, 4, 4), timesteps=torch.tensor([4]))
    loss.backward()
    assert torch.isfinite(loss)
    assert tensor.grad is not None and tensor.grad.abs().sum() > 0
    assert model.projection.weight.grad.abs().sum() > 0
    assert model.unet.conv_in.weight.grad.abs().sum() > 0


def test_complete_diffusion_artifact_round_trip(model, tmp_path):
    value = Latent(torch.randn(1, 2, 6), model.input_space)
    expected = model(value, context={'seed': 3})
    model.save_pretrained(tmp_path / 'decoder')
    restored = ImageDecoder.from_pretrained(tmp_path / 'decoder')
    assert torch.equal(expected, restored(value, context={'seed': 3}))
    assert restored.config['conditioning_status'] == 'requires_training'


def test_diffusion_objective_trace_restart(model, tmp_path):
    from tensorcode.training import ToolTrainer, load
    inputs = {'value': Latent(torch.randn(1, 2, 6), model.input_space),
              'noise': torch.randn(1, 4, 4, 4), 'timesteps': torch.tensor([3])}
    target = torch.rand(1, 3, 4, 4)
    trainer = ToolTrainer(model)
    session = trainer.capture(inputs, target, source='tiny real RGB fixture')
    session.save(tmp_path / 'trace.json', operations=trainer.operations, codecs={'latent': Latent, 'space': Space})
    model.save_pretrained(tmp_path / 'model')
    restarted = ImageDecoder.from_pretrained(tmp_path / 'model')
    resumed = ToolTrainer(restarted)
    restored_session = load(tmp_path / 'trace.json', operations=resumed.operations, codecs={'latent': Latent, 'space': Space})
    previous = restarted.projection.weight.detach().clone()
    assert torch.isfinite(torch.tensor(resumed.step(restored_session)))
    assert not torch.equal(previous, restarted.projection.weight)


def test_local_foundation_owns_weights_and_native_identity(model, tmp_path):
    from diffusers import DDIMScheduler
    model.unet.save_pretrained(tmp_path / 'unet')
    model.vae.save_pretrained(tmp_path / 'vae')
    model.scheduler.save_pretrained(tmp_path / 'scheduler')
    native = ImageDecoder.from_foundation(tmp_path,
        input_space=Space('native', 8, organization='sequence'),
        conditioning_projection='identity', num_inference_steps=2, local_files_only=True)
    embedding = torch.randn(1, 2, 8)
    noise = torch.randn(1, 4, 4, 4)
    value = Latent(embedding, native.input_space)
    actual = native(value, context={'noise': noise})
    scheduler = DDIMScheduler.from_config(model.scheduler.config)
    scheduler.set_timesteps(2)
    sample = noise * scheduler.init_noise_sigma
    model.eval()
    with torch.no_grad():
        for step in scheduler.timesteps:
            prediction = model.unet(scheduler.scale_model_input(sample, step), step,
                                    encoder_hidden_states=embedding).sample
            sample = scheduler.step(prediction, step, sample).prev_sample
        expected = (model.vae.decode(sample / model.vae.config.scaling_factor).sample / 2 + .5).clamp(0, 1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert native.config['conditioning_status'] == 'caller_declared_native_identity'
    native.save_pretrained(tmp_path / 'complete')
    restored = ImageDecoder.from_pretrained(tmp_path / 'complete')
    assert torch.equal(actual, restored(value, context={'noise': noise}))


def test_reject_unsupported_and_malformed_diffusion(model):
    from copy import deepcopy
    config = deepcopy(model.config)
    config['unet_config']['addition_embed_type'] = 'text'
    with pytest.raises(ValueError, match='unsupported'):
        ImageDecoder(config)
    value = Latent(torch.randn(1, 2, 6), model.input_space)
    with pytest.raises(ValueError, match='noise shape'):
        model(value, context={'noise': torch.zeros(1, 4, 5, 5)})
    with pytest.raises(ValueError, match='compatible'):
        model(Latent(value.tensor, Space('wrong', 6, organization='sequence')), context={'seed': 2})
    with pytest.raises(ValueError, match='valid|unmasked'):
        model(Latent(value.tensor, value.space, mask=torch.zeros(1, 2, dtype=torch.bool)), context={'seed': 2})
    with pytest.raises(ValueError, match='target_pixels'):
        model.loss(value, torch.full((1, 3, 4, 4), 2.), noise=torch.zeros(1, 4, 4, 4), timesteps=torch.tensor([0]))


def test_foundation_rejects_missing_weights(model, tmp_path):
    from safetensors.torch import load_file, save_file
    model.unet.save_pretrained(tmp_path / 'unet')
    model.vae.save_pretrained(tmp_path / 'vae')
    model.scheduler.save_pretrained(tmp_path / 'scheduler')
    path = tmp_path / 'unet' / 'diffusion_pytorch_model.safetensors'
    weights = load_file(path)
    del weights['conv_in.weight']
    save_file(weights, path, metadata={'format': 'pt'})
    with pytest.raises(ValueError, match='missing|incomplete'):
        ImageDecoder.from_foundation(tmp_path, input_space=model.input_space,
                                     local_files_only=True, num_inference_steps=2)


def test_rejects_nonconditioning_context_and_unconditional_unet(model):
    from copy import deepcopy
    value = Latent(torch.randn(1, 2, 6), model.input_space)
    with pytest.raises(ValueError, match='context'):
        model(value, context={'seed': 0, 'targets': torch.rand(1, 3, 4, 4)})
    config = deepcopy(model.config)
    config['unet_config'].update(down_block_types=['DownBlock2D'],
                                up_block_types=['UpBlock2D'], mid_block_type='UNetMidBlock2D')
    with pytest.raises(ValueError, match='cross-attention'):
        ImageDecoder(config)
