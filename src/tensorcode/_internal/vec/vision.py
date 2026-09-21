"""Owned ViT transformer with explicit preprocessing and embedding context."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from tensorcode._internal.latent_ops import LatentOperation, as_sequence
from tensorcode.ops.vec.latent import Latent, Space


class ImageEncoder(LatentOperation):
    """Encode images with a ViT (random on construction, loaded explicitly).

    Tensor inputs are CHW/BCHW floating pixels in [0,1], already at the model's
    spatial resolution. Normalization preserves gradients. Use ``preprocess``
    for PIL/other processor inputs; its resize/crop path is not differentiable.
    A mapping with ``pixel_values`` bypasses normalization, accepting finite
    processor-normalized BCHW tensors. Coordinates refer to processed pixels.

    Sequence output contains final patch states; pooled output is the native
    final CLS state, without a newly initialized projection. Ordered context
    latents are prepended to image embeddings before transformer attention.
    No task competence for that conditioning is implied without training.
    """

    def __init__(self, config):
        super().__init__(config)
        from transformers import ViTConfig, ViTImageProcessor, ViTModel
        if set(config) & {'space', 'output'}:
            raise ValueError('use output_space and readout')
        model_config = dict(self.config['model'])
        if model_config.get('model_type', 'vit') != 'vit':
            raise ValueError('ImageEncoder supports only ViTModel architecture')
        native_config = ViTConfig(**model_config)
        self.model = ViTModel(native_config, add_pooling_layer=False)
        if not hasattr(self.model, 'layers'):
            raise ValueError('ImageEncoder requires the transformers 5 ViT layers API')
        processor = dict(self.config['processor'])
        if processor.get('image_processor_type', 'ViTImageProcessor') != 'ViTImageProcessor':
            raise ValueError('ImageEncoder supports only ViTImageProcessor')
        self.processor = ViTImageProcessor(**processor)
        self.readout = self.config.get('readout', 'sequence')
        if self.readout not in ('sequence', 'pooled'):
            raise ValueError('readout must be sequence or pooled')
        self.output_space = Space(**self.config['output_space'])
        organization = 'sequence' if self.readout == 'sequence' else 'feature'
        if self.output_space.dimensions != native_config.hidden_size or self.output_space.organization != organization:
            raise ValueError('space must match native hidden size and output organization')
        context_config = self.config.get('context_space')
        self.context_space = Space(**context_config) if context_config else None
        if self.context_space and self.context_space.dimensions != native_config.hidden_size:
            raise ValueError('context_space must match ViT hidden size')
        size = native_config.image_size
        self.image_size = (size, size) if isinstance(size, int) else tuple(size)
        patch = native_config.patch_size
        self.patch_size = (patch, patch) if isinstance(patch, int) else tuple(patch)
        if any(s % p for s, p in zip(self.image_size, self.patch_size)):
            raise ValueError('image_size must be divisible by patch_size')
        if self.processor.do_normalize:
            mean, std = self.processor.image_mean, self.processor.image_std
            if len(mean) != native_config.num_channels or len(std) != native_config.num_channels or any(s <= 0 for s in std):
                raise ValueError('processor normalization must match channels with positive std')

    def preprocess(self, images):
        """Run owned processor assets; output uses processed-image coordinates."""
        return dict(self.processor(images=images, return_tensors='pt'))

    def _pixels(self, value):
        processed = isinstance(value, dict)
        if processed and set(value) - {'pixel_values', 'sources'}:
            raise ValueError('processed input supports only pixel_values and sources')
        pixels = value.get('pixel_values') if processed else value
        if not isinstance(pixels, torch.Tensor) or pixels.ndim not in (3, 4):
            raise ValueError('expected CHW/BCHW tensor or pixel_values mapping')
        single = pixels.ndim == 3
        pixels = pixels.unsqueeze(0) if single else pixels
        if not pixels.is_floating_point() or not torch.isfinite(pixels).all():
            raise ValueError('pixels must be finite floating point values')
        if pixels.shape[0] == 0 or pixels.shape[1] != self.model.config.num_channels or tuple(pixels.shape[-2:]) != self.image_size:
            raise ValueError('pixel batch, channels and size must match ViT configuration')
        if not processed and (pixels.min() < 0 or pixels.max() > 1):
            raise ValueError('raw floating pixels must be in [0,1]')
        parameter = self.model.embeddings.patch_embeddings.projection.weight
        pixels = pixels.to(device=parameter.device, dtype=parameter.dtype)
        if not processed and self.processor.do_normalize:
            mean = pixels.new_tensor(self.processor.image_mean)[None, :, None, None]
            std = pixels.new_tensor(self.processor.image_std)[None, :, None, None]
            pixels = (pixels - mean) / std
        return pixels, single

    def forward(self, value, *, context=None):
        from collections.abc import Mapping
        context = {} if context is None else context
        if not isinstance(context, Mapping):
            raise ValueError('context must be a mapping')
        if set(context) - {'latents'}:
            raise ValueError('ImageEncoder context supports only ordered latents')
        pixels, single = self._pixels(value)
        latents = context.get('latents', [])
        if not isinstance(latents, (list, tuple)):
            raise ValueError('context latents must be an ordered list or tuple')
        sources = list(value.get('sources', ())) if isinstance(value, dict) else []
        if not all(isinstance(source, str) for source in sources):
            raise ValueError('image sources must be strings')
        if latents:
            if self.context_space is None:
                raise ValueError('context requires an explicit context_space')
            embedded = self.model.embeddings(pixels)
            pieces, masks = [], []
            for latent in latents:
                seq, mask = as_sequence(latent, self.context_space)
                if seq.shape[0] != pixels.shape[0]:
                    raise ValueError('context batch must match image batch')
                pieces.append(seq.to(device=embedded.device, dtype=embedded.dtype))
                masks.append(mask.to(device=embedded.device))
                sources.extend(latent.sources)
            prefix_length = sum(piece.shape[1] for piece in pieces)
            pieces.append(embedded)
            masks.append(torch.ones(embedded.shape[:2], dtype=torch.bool, device=embedded.device))
            hidden = torch.cat(pieces, dim=1)
            mask = torch.cat(masks, dim=1)
            # Use exactly the mask construction used by the native ViT forward.
            from transformers.masking_utils import create_bidirectional_mask
            attention_mask = create_bidirectional_mask(config=self.model.config,
                inputs_embeds=hidden, attention_mask=mask)
            for layer in self.model.layers:
                hidden = layer(hidden, attention_mask)
            hidden = self.model.layernorm(hidden)[:, prefix_length:]
        else:
            hidden = self.model(pixel_values=pixels).last_hidden_state
        coordinates = None
        if self.readout == 'pooled':
            result = hidden[:, 0]
        else:
            result = hidden[:, 1:]
            ph, pw = self.patch_size
            ys = torch.arange(self.image_size[0] // ph, device=result.device, dtype=result.dtype) * ph + ph / 2
            xs = torch.arange(self.image_size[1] // pw, device=result.device, dtype=result.dtype) * pw + pw / 2
            y, x = torch.meshgrid(ys, xs, indexing='ij')
            coordinates = torch.stack((y, x), dim=-1).reshape(1, -1, 2).expand(result.shape[0], -1, -1)
        mask = torch.ones(result.shape[:-1], device=result.device, dtype=torch.bool)
        if single:
            result, mask = result[0], mask[0]
            coordinates = coordinates[0] if coordinates is not None else None
        return Latent(result, self.output_space, mask=mask, coordinates=coordinates,
            sources=tuple(sources), metadata={'readout': 'patch-states' if self.readout == 'sequence' else 'native-cls',
                'coordinate_space': 'processed-image-pixels',
                'initialization': 'foundation' if self.config.get('foundation') else 'random',
                'foundation': self.config.get('foundation'), 'context_conditioning': 'embedding-attention'})

    def configuration(self):
        """Fingerprint current owned processor semantics with fixed architecture."""
        config = super().configuration()
        config['processor'] = json.loads(self.processor.to_json_string())
        return self._validated_config(config)

    def _save_pretrained_assets(self, directory: Path):
        asset = directory / 'vision_processor.json'
        if asset.is_symlink():
            asset.unlink()
        asset.write_text(json.dumps(self.configuration()['processor'], sort_keys=True), encoding='utf-8')

    @classmethod
    def _load_pretrained_config(cls, config, directory):
        saved = json.loads((directory / 'vision_processor.json').read_text(encoding='utf-8'))
        if saved != config['processor']:
            raise ValueError('processor asset differs from model configuration')
        return config

    @classmethod
    def from_foundation(cls, repo_id_or_path, *, output_space, readout='sequence',
                        context_space=None, revision=None, local_files_only=False,
                        cache_dir=None, token=None, device='cpu'):
        """Load only known native ViT weights and processor, never Hub code."""
        from transformers import ViTConfig, ViTImageProcessor, ViTModel
        options = dict(revision=revision, local_files_only=local_files_only,
                       cache_dir=cache_dir, token=token)
        raw_config, _ = ViTConfig.get_config_dict(repo_id_or_path, **options)
        if raw_config.get('model_type') != 'vit':
            raise ValueError('foundation must be a ViT checkpoint')
        native, loading = ViTModel.from_pretrained(repo_id_or_path, add_pooling_layer=False,
                                          use_safetensors=True, output_loading_info=True, **options)
        if loading.get('missing_keys') or loading.get('mismatched_keys'):
            raise ValueError('foundation has missing or mismatched ViT weights')
        if native.config.model_type != 'vit':
            raise ValueError('foundation must be a ViT checkpoint')
        processor = ViTImageProcessor.from_pretrained(repo_id_or_path, **options)
        config = dict(model=json.loads(native.config.to_json_string()), processor=json.loads(processor.to_json_string()),
            readout=readout, output_space=output_space.configuration() if isinstance(output_space, Space) else output_space,
            context_space=context_space.configuration() if isinstance(context_space, Space) else context_space,
            foundation={'repo': str(repo_id_or_path), 'revision': revision,
                        'resolved_revision': getattr(native.config, '_commit_hash', None)})
        result = cls(config)
        result.model.load_state_dict(native.state_dict(), strict=True)
        return result.to(device).eval()
