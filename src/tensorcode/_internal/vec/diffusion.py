"""Owned latent diffusion with explicit conditioning and replayable sampling.

Only ordinary cross-attention UNet2DConditionModel + AutoencoderKL models are
supported. A linear adapter starts untrained; importing diffusion weights does
not establish alignment with an arbitrary input Space.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from tensorcode._internal.latent_ops import LatentOperation, as_sequence, space_from_config
from tensorcode.ops.vec.latent import Latent, Space
from tensorcode.ops.base import Operation
import weakref


def _diffusers():
    try:
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    except ImportError as exc:
        raise ImportError('ImageDecoder requires the tensorcode diffusion dependencies') from exc
    return UNet2DConditionModel, AutoencoderKL, DDIMScheduler


def _native_config(config):
    # Diffusers' private _use_default_values can silently discard explicit values
    # on from_config; retain the resolved public architecture only.
    return json.loads(json.dumps({key: value for key, value in dict(config).items()
                                  if not key.startswith('_')}))


class _DiffusionObjective(Operation):
    replayable = True

    def __init__(self, owner):
        self._owner = weakref.ref(owner)

    def forward(self, value, *, context=None):
        if context:
            raise ValueError('Pass diffusion objective context inside inputs')
        inputs = value['inputs']
        return self._owner().loss(inputs['value'], value['targets'],
            context=inputs.get('context'), noise=inputs['noise'], timesteps=inputs['timesteps'])

    def parameters(self, recurse=True):
        return self._owner().parameters(recurse=recurse)

    def _operation_identity(self):
        return self._owner()._tool_identity() + '.objective'

    def configuration(self):
        owner = self._owner()
        return {'operation': f'{type(owner).__module__}.{type(owner).__qualname__}',
                'role': 'objective', 'model': owner.configuration()}


@contextmanager
def _evaluation(module):
    modes = [(child, child.training) for child in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for child, training in modes:
            child.training = training


class ImageDecoder(LatentOperation):
    """Generate RGB tensors from compatible latent conditioning.

    Configuration owns ``input_space``, native ``unet_config``, ``vae_config``,
    ``scheduler_config``, ``bridge``, and ``num_inference_steps``. Construction initializes
    weights locally. ``from_foundation`` explicitly imports pretrained weights.
    Sampling requires ``context['noise']`` (unscaled standard Gaussian noise) or
    ``context['seed']``. ``context['latents']`` prefixes conditioning in order.
    DDIM sampling uses eta=0. No classifier-free guidance is applied.
    """

    def __init__(self, config):
        self._initialize(config)

    def _initialize(self, config, components=None):
        config = self._validated_config(config)
        if 'conditioning_projection' in config:
            raise ValueError('conditioning_projection is unsupported; configure bridge instead')
        allowed = {'input_space', 'unet_config', 'vae_config', 'scheduler_config',
                   'bridge', 'conditioning_status', 'num_inference_steps', 'foundation'}
        if set(config) - allowed:
            raise ValueError(f'Unknown configuration fields: {sorted(set(config) - allowed)}')
        for key in ('unet_config', 'vae_config', 'scheduler_config'):
            config[key] = _native_config(config[key])
        config.setdefault('bridge', 'linear')
        config.setdefault('conditioning_status', 'requires_training' if
                          config['bridge'] == 'linear' else 'caller_declared_native_identity')
        config.setdefault('num_inference_steps', 20)
        super().__init__(config)
        self.input_space = space_from_config(self.config['input_space'])
        UNet, VAE, Scheduler = _diffusers()
        if components is None:
            self.unet = UNet.from_config(self.config['unet_config'])
            self.vae = VAE.from_config(self.config['vae_config'])
            self.scheduler = Scheduler.from_config(self.config['scheduler_config'])
        else:
            self.unet, self.vae, self.scheduler = components
        uc, vc = self.unet.config, self.vae.config
        if (uc.addition_embed_type is not None or uc.class_embed_type is not None or
                uc.num_class_embeds is not None or uc.encoder_hid_dim_type is not None or
                uc.time_cond_proj_dim is not None or uc.dual_cross_attention or
                uc.attention_type != 'default'):
            raise ValueError('unsupported diffusion pipeline: only plain cross-attention UNets are supported')
        blocks = [*uc.down_block_types, *uc.up_block_types, uc.mid_block_type or '']
        if not any('CrossAttn' in block for block in blocks):
            raise ValueError('UNet must contain cross-attention conditioning blocks')
        width = uc.cross_attention_dim
        if not isinstance(width, int):
            raise ValueError('UNet cross_attention_dim must be a single integer')
        if uc.in_channels != vc.latent_channels or uc.out_channels != vc.latent_channels:
            raise ValueError('UNet channels must match VAE latent_channels')
        if vc.in_channels != 3 or vc.out_channels != 3:
            raise ValueError('VAE must encode and decode RGB images')
        if (getattr(vc, 'shift_factor', None) not in (None, 0) or
                getattr(vc, 'latents_mean', None) is not None or
                getattr(vc, 'latents_std', None) is not None):
            raise ValueError('unsupported VAE latent normalization')
        if not vc.scaling_factor or vc.scaling_factor <= 0:
            raise ValueError('VAE scaling_factor must be positive')
        if self.scheduler.config.prediction_type not in ('epsilon', 'v_prediction', 'sample'):
            raise ValueError('unsupported scheduler prediction_type')
        steps = self.config['num_inference_steps']
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= self.scheduler.config.num_train_timesteps:
            raise ValueError('num_inference_steps must be within the training timestep count')
        projection = self.config['bridge']
        if projection == 'identity':
            if self.input_space.dimensions != width:
                raise ValueError('identity conditioning requires native cross-attention dimensions')
            self.projection = nn.Identity()
        elif projection == 'linear':
            self.projection = nn.Linear(self.input_space.dimensions, width)
        else:
            raise ValueError('bridge must be linear or identity')
        size = uc.sample_size
        self.latent_size = (size, size) if isinstance(size, int) else tuple(size or ())
        if len(self.latent_size) != 2 or any(not isinstance(x, int) or x <= 0 for x in self.latent_size):
            raise ValueError('UNet sample_size must specify a positive height and width')
        self.vae_scale_factor = 2 ** (len(vc.block_out_channels) - 1)
        # Complete resolved native configurations are portable without Hub access.
        for key, component in [('unet_config', self.unet), ('vae_config', self.vae), ('scheduler_config', self.scheduler)]:
            self.config[key] = _native_config(component.config)

    @classmethod
    def from_foundation(cls, repo_id_or_path, *, input_space: Space, revision=None,
                        local_files_only=False, cache_dir=None, token=None,
                        bridge='linear', num_inference_steps=20):
        """Import known diffusion components; never execute repository Python.

        ``identity`` is an explicit caller assertion (not library-verified) that
        input embeddings already
        inhabit the foundation's native conditioning space, including position.
        Linear projections are initialized randomly and require paired training.
        """
        UNet, VAE, Scheduler = _diffusers()
        common = dict(revision=revision, local_files_only=local_files_only,
                      cache_dir=cache_dir, token=token)
        def load_component(component, subfolder):
            model, info = component.from_pretrained(str(repo_id_or_path), subfolder=subfolder,
                use_safetensors=True, output_loading_info=True, **common)
            if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
                raise ValueError(f'incomplete or incompatible foundation {subfolder} weights: {info}')
            return model
        unet = load_component(UNet, 'unet')
        vae = load_component(VAE, 'vae')
        scheduler_config = Scheduler.load_config(repo_id_or_path, subfolder='scheduler', **common)
        scheduler = Scheduler.from_config(scheduler_config)
        config = dict(input_space=input_space.configuration(),
                      unet_config=_native_config(unet.config),
                      vae_config=_native_config(vae.config),
                      scheduler_config=_native_config(scheduler.config),
                      bridge=bridge,
                      num_inference_steps=num_inference_steps,
                      foundation={'source': str(repo_id_or_path), 'revision': revision,
                                  'scheduler': 'DDIMScheduler'})
        model = cls.__new__(cls)
        model._initialize(config, (unet, vae, scheduler))
        return model

    training_inputs_include_targets = True

    @property
    def training_operation(self):
        if '_objective' not in self.__dict__:
            self.__dict__['_objective'] = _DiffusionObjective(self)
        return self.__dict__['_objective']

    def operation_bindings(self):
        return {**super().operation_bindings(), 'objective': self.training_operation}

    def _conditioning(self, value, context):
        if context is not None and not isinstance(context, Mapping):
            raise TypeError('context must be a mapping')
        context = {} if context is None else context
        if set(context) - {'latents', 'seed', 'noise'}:
            raise ValueError('unsupported diffusion context fields')
        extras = context.get('latents', [])
        if not isinstance(extras, (list, tuple)):
            raise TypeError("context['latents'] must be an ordered sequence of Latent values")
        pairs = [as_sequence(item, self.input_space) for item in [*extras, value]]
        batch = pairs[0][0].shape[0]
        parameter = next(self.unet.parameters())
        for tensor, mask in pairs:
            if tensor.shape[0] != batch:
                raise ValueError('conditioning batch sizes must match')
            if tensor.device != parameter.device or tensor.dtype != parameter.dtype:
                raise ValueError('conditioning dtype and device must match the decoder')
        sequence = torch.cat([pair[0] for pair in pairs], dim=1)
        mask = torch.cat([pair[1] for pair in pairs], dim=1)
        if not mask.any(dim=1).all():
            raise ValueError('each image requires at least one unmasked conditioning token')
        # Mask before projection as well: hidden NaNs cannot leak through attention.
        sequence = sequence.masked_fill(~mask.unsqueeze(-1), 0)
        return self.projection(sequence), mask

    def _noise_shape(self, batch):
        return (batch, self.vae.config.latent_channels, *self.latent_size)

    def _validate_noise(self, noise, batch, reference):
        if not isinstance(noise, torch.Tensor) or tuple(noise.shape) != self._noise_shape(batch):
            raise ValueError(f'noise shape must be {self._noise_shape(batch)}')
        if noise.dtype != reference.dtype or noise.device != reference.device or not torch.isfinite(noise).all():
            raise ValueError('noise must be finite and match conditioning dtype and device')
        return noise

    @torch.no_grad()
    def forward(self, value: Latent, *, context=None):
        embeddings, mask = self._conditioning(value, context)
        context = {} if context is None else context
        seed, noise = context.get('seed'), context.get('noise')
        if (seed is None) == (noise is None):
            raise ValueError("supply exactly one of context['seed'] or context['noise']")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError('seed must be an integer')
            generator = torch.Generator(device=embeddings.device).manual_seed(seed)
            noise = torch.randn(self._noise_shape(embeddings.shape[0]), generator=generator,
                                device=embeddings.device, dtype=embeddings.dtype)
        noise = self._validate_noise(noise, embeddings.shape[0], embeddings)
        # A local scheduler owns mutable timesteps; simultaneous calls do not race.
        scheduler = type(self.scheduler).from_config(self.scheduler.config)
        scheduler.set_timesteps(self.config['num_inference_steps'], device=embeddings.device)
        sample = noise * scheduler.init_noise_sigma
        with _evaluation(self):
            for timestep in scheduler.timesteps:
                prediction = self.unet(scheduler.scale_model_input(sample, timestep), timestep,
                    encoder_hidden_states=embeddings, encoder_attention_mask=mask).sample
                sample = scheduler.step(prediction, timestep, sample, eta=0).prev_sample
            pixels = self.vae.decode(sample / self.vae.config.scaling_factor).sample
        return (pixels / 2 + 0.5).clamp(0, 1)

    def loss(self, value: Latent, target_pixels: torch.Tensor, *, context=None,
             noise: torch.Tensor, timesteps: torch.Tensor):
        """Differentiable diffusion prediction loss using only actual targets.

        Targets are encoded with the VAE posterior mean (no hidden random draw).
        Noise and per-image timesteps are mandatory. Targets never enter the
        conditioning path. The caller controls train/eval mode for this objective.
        """
        embeddings, mask = self._conditioning(value, context)
        batch = embeddings.shape[0]
        expected = (batch, 3, *(size * self.vae_scale_factor for size in self.latent_size))
        if (not isinstance(target_pixels, torch.Tensor) or tuple(target_pixels.shape) != expected or
                target_pixels.dtype != embeddings.dtype or target_pixels.device != embeddings.device):
            raise ValueError(f'target_pixels must have shape {expected} and decoder dtype/device')
        if not torch.isfinite(target_pixels).all() or target_pixels.min() < 0 or target_pixels.max() > 1:
            raise ValueError('target_pixels must be finite RGB values in [0, 1]')
        if (not isinstance(timesteps, torch.Tensor) or timesteps.shape != (batch,) or
                timesteps.dtype != torch.long or timesteps.device != embeddings.device or
                (timesteps < 0).any() or (timesteps >= self.scheduler.config.num_train_timesteps).any()):
            raise ValueError('timesteps must contain one valid int64 training timestep per image')
        noise = self._validate_noise(noise, batch, embeddings)
        with torch.no_grad():
            target = self.vae.encode(target_pixels * 2 - 1).latent_dist.mode() * self.vae.config.scaling_factor
        # add_noise may move scheduler tensors, so isolate objective invocations too.
        scheduler = type(self.scheduler).from_config(self.scheduler.config)
        noisy = scheduler.add_noise(target, noise, timesteps)
        prediction = self.unet(noisy, timesteps, encoder_hidden_states=embeddings,
                               encoder_attention_mask=mask).sample
        prediction_type = scheduler.config.prediction_type
        desired = noise if prediction_type == 'epsilon' else (
            scheduler.get_velocity(target, noise, timesteps) if prediction_type == 'v_prediction' else target)
        return F.mse_loss(prediction.float(), desired.float())
