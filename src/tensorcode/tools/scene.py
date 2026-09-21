"""Owned image-and-text candidate model with spatial workspace organization.

Candidates are supplied descriptions, not generated claims. Attention identifies
patch routing, not factual support. Fresh models have random visual weights.
"""
from __future__ import annotations

import copy
import re

import torch
from torch import nn
from torch.nn import functional as F

from .._internal.pretrained import PretrainedTool
from .._internal.ranking import RankingObjective, bindings
from .._internal.workspace import Workspace
from ..ops.vec import ImageEncoder, Space, Transform
from ..tracing import invoke


class SceneRank(nn.Module):
    replayable = True

    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        d = config['dimensions']
        self.vocabulary = {word: index + 1 for index, word in enumerate(config['vocabulary'])}
        self.image = ImageEncoder(patch_size=config['patch_size'], in_channels=config['in_channels'], space=Space('scene-patches', d, organization='spatial'))
        self.text = Transform(nn.Embedding(len(self.vocabulary) + 1, d))
        self.position = Transform(nn.Linear(2, d))
        self.text_position = nn.Embedding(config['max_tokens'], d)
        self.candidate_sequence = nn.GRU(d, d, batch_first=True)
        self.modality = nn.Parameter(torch.randn(2, d) * .02)
        self.workspace = Workspace(d, config['slots'], config['steps'])
        self.score = Transform(nn.Sequential(nn.Linear(d * 4, d), nn.GELU(), nn.Linear(d, 1)))

    def configuration(self):
        return {'operation': 'tensorcode.tools.scene.SceneRank', 'config': copy.deepcopy(self.config)}

    def __call__(self, value, *, context=None):
        return invoke(self, value, context, super().__call__)

    def tokens(self, text):
        words = re.findall(r'\w+|[^\w\s]', text.casefold())[:self.config['max_tokens']]
        return torch.tensor([self.vocabulary.get(word, 0) for word in words] or [0], device=self.modality.device)

    def validate(self, value):
        if not isinstance(value, dict):
            raise ValueError('scene inputs must be a dictionary')
        for key in ['question', 'source_id']:
            if not isinstance(value.get(key), str) or not value[key].strip():
                raise ValueError(f'{key} must be nonempty text')
        pixels = value.get('pixels')
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 3 or pixels.shape[0] != self.config['in_channels']:
            raise ValueError('pixels must be CHW with configured channels')
        if min(pixels.shape[1:]) < self.config['patch_size'] or max(pixels.shape[1:]) > self.config['max_image_size']:
            raise ValueError('image dimensions outside configured bounds')
        if not pixels.is_floating_point() or not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
            raise ValueError('pixels must be finite floating values in [0, 1]')
        candidates = value.get('candidates')
        if not isinstance(candidates, list) or not candidates or len(candidates) > self.config['max_candidates']:
            raise ValueError('candidates must be a nonempty bounded list')
        ids = []
        for item in candidates:
            if not isinstance(item, dict) or any(not isinstance(item.get(k), str) or not item[k].strip() for k in ['id', 'text']):
                raise ValueError('candidates require nonempty id and text')
            ids.append(item['id'])
        if len(set(ids)) != len(ids):
            raise ValueError('candidate IDs must be unique')
        return pixels, candidates

    def compute(self, value, *, workspace_ablation=None):
        pixels, candidates = self.validate(value)
        pixels = pixels.to(device=self.modality.device, dtype=self.modality.dtype)
        visual = self.image(pixels)
        coords = visual.coordinates.reshape(-1, 2)
        normalizer = coords.new_tensor(pixels.shape[-2:])
        patches = visual.tensor.reshape(-1, self.config['dimensions']) + self.position(coords / normalizer) + self.modality[0]
        question_tokens = self.tokens(value['question'])
        question = self.text(question_tokens) + self.text_position(torch.arange(len(question_tokens), device=self.modality.device)) + self.modality[1]
        encoded = torch.cat([patches, question]).unsqueeze(0)
        workspace = self.workspace(encoded)
        query = workspace['conditioning'].mean(1)
        if workspace_ablation == 'zero':
            query = torch.zeros_like(query)
        elif workspace_ablation == 'bypass':
            query = encoded.mean(1)
        elif workspace_ablation is not None:
            raise ValueError('workspace_ablation must be None, zero or bypass')
        candidate = torch.stack([self.candidate_sequence(self.text(self.tokens(item['text'])).unsqueeze(0))[1][0, 0] for item in candidates])
        query = query.expand_as(candidate)
        logits = self.score(torch.cat([query, candidate, query * candidate, (query - candidate).abs()], -1)).squeeze(-1)
        return logits, workspace, coords

    def forward(self, value, *, context=None):
        if context:
            raise ValueError('SceneRank does not accept context')
        return self.compute(value)[0]

class FoundationSceneRank(SceneRank):
    """Frozen owned CLIP perception with a trainable crossmodal workspace.

    CLIP's inherited representations supply perceptual competence. TensorCode
    learns the workspace and candidate ranking; cached features are session-free
    runtime optimizations and never appear in model artifacts.
    """

    def __init__(self, config, tokenizer_json):
        from collections import OrderedDict
        from tokenizers import Tokenizer
        from transformers import CLIPConfig, CLIPModel
        nn.Module.__init__(self)
        self.config = copy.deepcopy(config)
        self.tokenizer = Tokenizer.from_str(tokenizer_json)
        self.foundation = CLIPModel(CLIPConfig.from_dict(config['foundation_config']))
        self.foundation.requires_grad_(False)
        self.foundation.eval()
        d = config['dimensions']
        self.image_projection = Transform(nn.Linear(self.foundation.config.vision_config.hidden_size, d))
        self.text_projection = Transform(nn.Linear(self.foundation.config.text_config.hidden_size, d))
        self.candidate_projection = Transform(nn.Linear(self.foundation.config.projection_dim, d))
        self.global_projection = Transform(nn.Linear(2 * self.foundation.config.projection_dim + 1, d))
        self.modality = nn.Parameter(torch.randn(2, d) * .02)
        self.workspace = Workspace(d, config['slots'], config['steps'])
        self.score = Transform(nn.Sequential(nn.Linear(d * 4, d), nn.GELU(), nn.Linear(d, 1)))
        self._vision_cache, self._text_cache = OrderedDict(), OrderedDict()

    def configuration(self):
        return {'operation': 'tensorcode.tools.scene.FoundationSceneRank', 'config': copy.deepcopy(self.config)}

    def _clear_features(self):
        self._vision_cache.clear()
        self._text_cache.clear()

    def _apply(self, fn, recurse=True):
        self._clear_features()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self._clear_features()
        return super()._load_from_state_dict(*args, **kwargs)

    def train(self, mode=True):
        super().train(mode)
        self.foundation.eval()
        return self

    def tokens(self, text):
        ids = self.tokenizer.encode(text).ids
        limit = self.foundation.config.text_config.max_position_embeddings
        if len(ids) > limit:
            ids = ids[:limit - 1] + [self.foundation.config.text_config.eos_token_id]
        return torch.tensor(ids, device=self.modality.device, dtype=torch.long)

    @staticmethod
    def _remember(cache, key, value):
        with torch.inference_mode(False):
            cache[key] = tuple(t.detach().cpu().clone() for t in value)
        cache.move_to_end(key)
        if len(cache) > 1024:
            cache.popitem(last=False)

    def _encode_text(self, text):
        if text not in self._text_cache:
            with torch.no_grad():
                ids = self.tokens(text).unsqueeze(0)
                output = self.foundation.text_model(input_ids=ids, attention_mask=torch.ones_like(ids))
                pooled = self.foundation.text_projection(output.pooler_output)[0]
                self._remember(self._text_cache, text, (output.last_hidden_state[0], pooled))
        return tuple(t.to(device=self.modality.device, dtype=self.modality.dtype) for t in self._text_cache[text])

    def _encode_image(self, pixels):
        import hashlib
        # Include pixels rather than trusting a possibly reused external source ID.
        key = hashlib.sha256(pixels.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes() + str((tuple(pixels.shape), pixels.dtype)).encode()).hexdigest()
        if key not in self._vision_cache:
            with torch.no_grad():
                size = self.foundation.config.vision_config.image_size
                image = F.interpolate(pixels[None].to(device=self.modality.device, dtype=self.modality.dtype), size=(size, size), mode='bicubic', align_corners=False, antialias=True)
                mean = image.new_tensor(self.config['image_mean'])[None, :, None, None]
                std = image.new_tensor(self.config['image_std'])[None, :, None, None]
                output = self.foundation.vision_model(pixel_values=(image - mean) / std)
                pooled = self.foundation.visual_projection(output.pooler_output)[0]
                self._remember(self._vision_cache, key, (output.last_hidden_state[0, 1:], pooled))
        return tuple(t.to(device=self.modality.device, dtype=self.modality.dtype) for t in self._vision_cache[key])

    def compute(self, value, *, workspace_ablation=None):
        pixels, candidates = self.validate(value)
        patches, image_global = self._encode_image(pixels)
        question, question_global = self._encode_text(value['question'])
        encoded = torch.cat([self.image_projection(patches) + self.modality[0], self.text_projection(question) + self.modality[1]]).unsqueeze(0)
        workspace = self.workspace(encoded)
        query = workspace['conditioning'].mean(1)
        if workspace_ablation == 'zero':
            query = torch.zeros_like(query)
        elif workspace_ablation == 'bypass':
            query = encoded.mean(1)
        elif workspace_ablation is not None:
            raise ValueError('workspace_ablation must be None, zero or bypass')
        # Aligned pretrained globals are explicit foundation features. Their
        # relevance and interpretation are learned by the supplied-data objective.
        similarity = F.cosine_similarity(image_global, question_global, dim=0).reshape(1)
        query = query + self.global_projection(torch.cat([image_global, question_global, similarity])).unsqueeze(0)
        candidate = self.candidate_projection(torch.stack([self._encode_text(item['text'])[1] for item in candidates]))
        query = query.expand_as(candidate)
        logits = self.score(torch.cat([query, candidate, query * candidate, (query - candidate).abs()], -1)).squeeze(-1)
        side = self.foundation.config.vision_config.image_size // self.foundation.config.vision_config.patch_size
        row = (torch.arange(side, device=query.device, dtype=query.dtype) + .5) * self.foundation.config.vision_config.patch_size * pixels.shape[-2] / self.foundation.config.vision_config.image_size
        col = (torch.arange(side, device=query.device, dtype=query.dtype) + .5) * self.foundation.config.vision_config.patch_size * pixels.shape[-1] / self.foundation.config.vision_config.image_size
        coords = torch.stack(torch.meshgrid(row, col, indexing='ij'), -1).reshape(-1, 2)
        return logits, workspace, coords

class SceneLanguageObjective(nn.Module):
    replayable = True

    def __init__(self, tool):
        super().__init__()
        import weakref
        object.__setattr__(self, '_tool_ref', weakref.ref(tool))

    def parameters(self, recurse=True):
        return self._tool_ref().language.parameters(recurse=recurse)

    def configuration(self):
        return {'operation': 'tensorcode.tools.scene.SceneLanguageObjective', 'config': self._tool_ref().configuration()}

    def __call__(self, value, *, context=None):
        return invoke(self, value, context, super().__call__)

    def forward(self, value, *, context=None):
        if context:
            raise ValueError('Scene language objective does not accept context')
        return self._tool_ref().loss(value['inputs'], value['targets'])


class SceneLanguage(nn.Module):
    """Owned Idefics3 perception/realization with a trainable visual residual.

    The residual gate initializes to zero so importing a foundation preserves its
    behavior. A fresh workspace is not a learned improvement over that foundation.
    """

    def __init__(self, config, assets):
        super().__init__()
        import hashlib
        import tempfile
        from pathlib import Path
        from transformers import Idefics3Config, Idefics3ForConditionalGeneration, Idefics3Processor, GenerationConfig
        if not isinstance(assets, dict) or set(assets) != set(config['processor_hashes']):
            raise ValueError('language construction requires complete processor assets')
        for name, value in assets.items():
            if Path(name).is_absolute() or '..' in Path(name).parts or Path(name).suffix not in {'.json', '.jinja', '.txt'} or not isinstance(value, str):
                raise ValueError('invalid processor asset')
            if hashlib.sha256(value.encode()).hexdigest() != config['processor_hashes'][name]:
                raise ValueError('processor asset checksum mismatch')
        self.assets = dict(assets)
        with tempfile.TemporaryDirectory() as folder:
            for name, value in assets.items():
                (Path(folder) / name).parent.mkdir(parents=True, exist_ok=True)
                (Path(folder) / name).write_text(value, encoding='utf-8')
            self.processor = Idefics3Processor.from_pretrained(folder, local_files_only=True)
        self.model = Idefics3ForConditionalGeneration(Idefics3Config.from_dict(config['language_config'])).float()
        self.model.generation_config = GenerationConfig.from_dict(config['generation_config'])
        if config['freeze_foundation']:
            self.model.requires_grad_(False)
            self.model.eval()
        self.config = copy.deepcopy(config)
        width = self.model.config.text_config.hidden_size
        dimensions = config['workspace_dimensions']
        self.down = Transform(nn.Linear(width, dimensions))
        self.workspace = Workspace(dimensions, config['workspace_slots'], config['workspace_steps'])
        self.read = nn.MultiheadAttention(dimensions, 1, batch_first=True)
        self.up = Transform(nn.Linear(dimensions, width))
        self.gate = nn.Parameter(torch.zeros(()))

    def train(self, mode=True):
        super().train(mode)
        if self.config['freeze_foundation']:
            self.model.eval()
        return self

    def validate(self, value):
        if not isinstance(value, dict):
            raise ValueError('scene inputs must be a dictionary')
        for key in ['question', 'source_id']:
            if not isinstance(value.get(key), str) or not value[key].strip():
                raise ValueError(f'{key} must be nonempty text')
        if len(value['question']) > self.config['max_question_chars']:
            raise ValueError('question exceeds configured limit')
        pixels = value.get('pixels')
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 3 or pixels.shape[0] != 3 or min(pixels.shape[1:]) < 1 or max(pixels.shape[1:]) > self.config['max_image_size']:
            raise ValueError('pixels must be nonempty RGB CHW within configured bounds')
        if not pixels.is_floating_point() or not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
            raise ValueError('pixels must contain finite floating values in [0, 1]')
        return pixels

    def prepare(self, value):
        from PIL import Image
        pixels = self.validate(value)
        rgb = (pixels.detach().cpu().float().clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).contiguous()
        image = Image.frombytes('RGB', (pixels.shape[2], pixels.shape[1]), rgb.numpy().tobytes())
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': value['question']}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        batch = self.processor(text=prompt, images=[image], return_tensors='pt')
        if batch['input_ids'].shape[-1] > self.config['max_input_tokens']:
            raise ValueError('processed image/question exceeds configured token limit')
        device, dtype = self.gate.device, self.down.module.weight.dtype
        batch = {key: tensor.to(device=device, dtype=dtype if tensor.is_floating_point() else tensor.dtype) for key, tensor in batch.items()}
        # Foundation visual tokens retain their pretrained spatial organization.
        visual = self.model.model.get_image_features(batch.pop('pixel_values'), batch.pop('pixel_attention_mask', None), return_dict=True).pooler_output
        original_shape = visual.shape
        visual_sequence = visual.reshape(1, -1, original_shape[-1])
        text = self.model.get_input_embeddings()(batch['input_ids'])
        encoded = self.down(torch.cat([visual_sequence, text], dim=1))
        mask = torch.cat([torch.ones((1, visual_sequence.shape[1]), dtype=torch.bool, device=device), batch['attention_mask'].bool()], dim=1)
        workspace = self.workspace(encoded, mask)
        read, _ = self.read(encoded[:, :visual_sequence.shape[1]], workspace['conditioning'], workspace['conditioning'], need_weights=False)
        revised = visual_sequence + torch.tanh(self.gate) * self.up(read)
        batch['image_hidden_states'] = revised.reshape(original_shape)
        return batch, workspace, visual_sequence.shape[1]

    def loss(self, value, target):
        if not isinstance(target, str) or not target.strip():
            raise ValueError('language target must be nonempty reviewer-supplied text')
        if len(target) > self.config['max_target_chars']:
            raise ValueError('language target exceeds configured limit')
        batch, _, _ = self.prepare(value)
        # The target is appended only after workspace interpretation is complete.
        ids = self.processor.tokenizer.encode(target, add_special_tokens=False)
        eos = self.model.generation_config.eos_token_id
        if isinstance(eos, list):
            eos = eos[0]
        if eos is not None:
            ids.append(eos)
        if len(ids) > self.config['max_new_tokens']:
            raise ValueError('language target exceeds configured token limit')
        if batch['input_ids'].shape[1] + len(ids) > self.model.config.text_config.max_position_embeddings:
            raise ValueError('target and input exceed model context capacity')
        target_ids = torch.tensor([ids], dtype=torch.long, device=self.gate.device)
        prefix = batch['input_ids']
        batch['input_ids'] = torch.cat([prefix, target_ids], dim=1)
        batch['attention_mask'] = torch.cat([batch['attention_mask'], torch.ones_like(target_ids)], dim=1)
        labels = torch.cat([torch.full_like(prefix, -100), target_ids], dim=1)
        return self.model(**batch, labels=labels, use_cache=False, return_dict=True).loss

    @torch.no_grad()
    def interpret(self, value, *, max_new_tokens=None):
        import hashlib
        limit = self.config['max_new_tokens'] if max_new_tokens is None else max_new_tokens
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= self.config['max_new_tokens']:
            raise ValueError('max_new_tokens exceeds configured bounds')
        batch, workspace, visual_tokens = self.prepare(value)
        if batch['input_ids'].shape[1] + limit > self.model.config.text_config.max_position_embeddings:
            raise ValueError('generation and input exceed model context capacity')
        generated = self.model.generate(**batch, max_new_tokens=limit, do_sample=False, return_dict_in_generate=False)
        answer_ids = generated[0, batch['input_ids'].shape[1]:]
        description = self.processor.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        complete = bool(len(answer_ids) and int(answer_ids[-1]) in eos)
        pixels = value['pixels'].detach().cpu().contiguous()
        fingerprint = hashlib.sha256(pixels.view(torch.uint8).numpy().tobytes() + str((tuple(pixels.shape), pixels.dtype)).encode()).hexdigest()
        return {
            'interpretation': description,
            'verification': 'unverified',
            'uncertainty': {'status': 'uncalibrated', 'confidence': None},
            'source': {'source_id': value['source_id'], 'kind': 'full-image', 'shape': list(pixels.shape), 'sha256': fingerprint},
            'question': value['question'],
            'completion_status': 'complete' if complete else 'token_limit',
            'workspace': {'active': bool(self.gate.detach().abs() > 0), 'visual_tokens': visual_tokens, 'attention': workspace['attention'][0].detach().cpu().tolist(), 'relations': workspace['relations'][0].detach().cpu().tolist()},
            'foundation_source': copy.deepcopy(self.config.get('foundation_source')),
        }


class Scene(PretrainedTool):
    """Rank explicit descriptions using a learned image/text workspace.

    This interface supplies no object vocabulary or spatial truth rules. Training
    determines behavior; selected candidates remain fallible interpretations.
    """

    def __init__(self, config):
        config = dict(config)
        tokenizer_json = config.pop('_tokenizer_json', None)
        language_assets = config.pop('_language_assets', None)
        if config.get('mode') == 'language':
            if config.get('architecture_version', 1) != 1:
                raise ValueError('unsupported scene language architecture_version')
            config['architecture_version'] = 1
            for key, default in [('workspace_dimensions', 64), ('workspace_slots', 8), ('workspace_steps', 2), ('max_image_size', 4096), ('max_question_chars', 4096), ('max_input_tokens', 4096), ('max_target_chars', 4096), ('max_new_tokens', 256)]:
                config.setdefault(key, default)
                if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
                    raise ValueError(f'{key} must be a positive integer')
            config.setdefault('freeze_foundation', True)
            if type(config['freeze_foundation']) is not bool:
                raise ValueError('freeze_foundation must be boolean')
            super().__init__(config)
            self.language = SceneLanguage(config, language_assets)
            self.objective = SceneLanguageObjective(self)
            return
        vocabulary = config.get('vocabulary')
        if not isinstance(vocabulary, list) or not vocabulary or any(not isinstance(word, str) or not word for word in vocabulary) or len(set(vocabulary)) != len(vocabulary):
            raise ValueError('vocabulary must contain unique nonempty strings')
        for name, default in [('dimensions', 32), ('slots', 4), ('steps', 2), ('max_tokens', 256)]:
            config.setdefault(name, default)
            if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 1:
                raise ValueError(f'{name} must be a positive integer')
        if config.get('architecture_version', 1) != 1:
            raise ValueError('unsupported scene architecture_version')
        config['architecture_version'] = 1
        for name, default in [('patch_size', 8), ('in_channels', 3), ('max_image_size', 256), ('max_candidates', 64)]:
            config.setdefault(name, default)
            if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 1:
                raise ValueError(f'{name} must be a positive integer')
        if config['max_image_size'] < config['patch_size']:
            raise ValueError('max_image_size must accommodate a patch')
        super().__init__(config)
        if 'foundation_config' in config:
            import hashlib
            if not isinstance(tokenizer_json, str) or hashlib.sha256(tokenizer_json.encode()).hexdigest() != config.get('tokenizer_sha256'):
                raise ValueError('foundation construction requires matching tokenizer assets')
            self.rank = FoundationSceneRank(config, tokenizer_json)
        else:
            self.rank = SceneRank(config)
        self.objective = RankingObjective(self)

    def forward(self, inputs, *, context=None):
        if context:
            raise ValueError('Scene does not accept context')
        if hasattr(self, 'language'):
            return self.interpret(inputs)
        logits, workspace, coordinates = self.rank.compute(inputs)
        probabilities = logits.softmax(-1).detach().cpu().tolist()
        return {
            'selected_id': inputs['candidates'][int(logits.argmax())]['id'],
            'candidates': [dict(copy.deepcopy(item), predicted_score=score, probability=p) for item, score, p in zip(inputs['candidates'], logits.detach().cpu().tolist(), probabilities)],
            'source_id': inputs['source_id'],
            'patch_coordinates': coordinates.detach().cpu().tolist(),
            'attention': workspace['attention'][0].detach().cpu().tolist(),
            'attention_source_ids': [inputs['source_id']] * len(coordinates) + [None] * len(self.rank.tokens(inputs['question'])),
            'relations': workspace['relations'][0].detach().cpu().tolist(),
        }

    predict = forward

    def loss(self, inputs, targets):
        if hasattr(self, 'language'):
            return self.language.loss(inputs, targets)
        logits = self.rank(inputs)
        if isinstance(targets, str):
            ids = [item['id'] for item in inputs['candidates']]
            if targets not in ids:
                raise ValueError('target must identify a supplied candidate')
            targets = ids.index(targets)
        if isinstance(targets, bool) or not isinstance(targets, int) or not 0 <= targets < logits.numel():
            raise ValueError('target must be a valid candidate index or ID')
        return F.cross_entropy(logits.unsqueeze(0), torch.tensor([targets], device=logits.device))

    @property
    def training_operation(self):
        return self.objective

    training_inputs_include_targets = True

    def operation_bindings(self):
        return bindings(self)

    @classmethod
    def from_foundation(cls, repo_id='openai/clip-vit-base-patch32', *, revision, local_files_only=False, dimensions=32, slots=4, steps=2):
        """Explicitly import pinned pretrained perception; ranking starts random."""
        import hashlib
        import json
        from transformers import CLIPModel, CLIPTokenizerFast, CLIPImageProcessor
        foundation = CLIPModel.from_pretrained(repo_id, revision=revision, local_files_only=local_files_only)
        tokenizer = CLIPTokenizerFast.from_pretrained(repo_id, revision=revision, local_files_only=local_files_only)
        processor = CLIPImageProcessor.from_pretrained(repo_id, revision=revision, local_files_only=local_files_only)
        tokenizer_json = tokenizer.backend_tokenizer.to_str()
        config = {'vocabulary': ['<foundation>'], 'dimensions': dimensions, 'slots': slots, 'steps': steps,
                  'foundation_config': json.loads(json.dumps(foundation.config.to_dict())), 'foundation_source': {'repo_id': repo_id, 'revision': revision},
                  '_tokenizer_json': tokenizer_json, 'tokenizer_sha256': hashlib.sha256(tokenizer_json.encode()).hexdigest(),
                  'image_mean': processor.image_mean, 'image_std': processor.image_std,
                  'preprocessing': 'bicubic-antialiased-square-resize', 'patch_size': foundation.config.vision_config.patch_size}
        model = cls(json.loads(json.dumps(config)))
        model.rank.foundation.load_state_dict(foundation.state_dict())
        return model

    def _save_pretrained_assets(self, directory):
        if hasattr(self, 'language'):
            import shutil
            folder = directory / 'processor'
            if folder.is_symlink():
                folder.unlink()
            elif folder.exists():
                shutil.rmtree(folder)
            folder.mkdir()
            for name, value in self.language.assets.items():
                (folder / name).parent.mkdir(parents=True, exist_ok=True)
                (folder / name).write_text(value, encoding='utf-8')
        elif isinstance(self.rank, FoundationSceneRank):
            path = directory / 'tokenizer.json'
            if path.is_symlink():
                path.unlink()
            path.write_text(self.rank.tokenizer.to_str(), encoding='utf-8')

    @classmethod
    def _load_pretrained_config(cls, config, directory):
        if config.get('mode') == 'language':
            from pathlib import Path
            names = config.get('processor_hashes')
            if not isinstance(names, dict) or any(not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts or Path(name).suffix not in {'.json', '.jinja', '.txt'} for name in names):
                raise ValueError('invalid processor asset names')
            config = dict(config, _language_assets={name: (directory / 'processor' / name).read_text(encoding='utf-8') for name in config['processor_hashes']})
        elif 'foundation_config' in config:
            config = dict(config, _tokenizer_json=(directory / 'tokenizer.json').read_text(encoding='utf-8'))
        return config

    def interpret(self, inputs, *, max_new_tokens=None):
        """Produce an unverified full-image interpretation, without fact extraction."""
        if not hasattr(self, 'language'):
            raise ValueError('interpret requires a Scene language model checkpoint')
        return self.language.interpret(inputs, max_new_tokens=max_new_tokens)

    @classmethod
    def from_language_foundation(cls, repo_id='HuggingFaceTB/SmolVLM-500M-Instruct', *, revision, local_files_only=False, freeze_foundation=True):
        """Explicitly import an owned Idefics3 VLM and its local processor assets.

        Initial competence belongs to the supplied pretrained foundation. The
        trainable workspace residual starts inactive and requires supervision.
        """
        import hashlib
        import json
        import tempfile
        from pathlib import Path
        from transformers import Idefics3ForConditionalGeneration, Idefics3Processor
        model = Idefics3ForConditionalGeneration.from_pretrained(repo_id, revision=revision, local_files_only=local_files_only, dtype=torch.float32)
        processor = Idefics3Processor.from_pretrained(repo_id, revision=revision, local_files_only=local_files_only)
        with tempfile.TemporaryDirectory() as directory:
            processor.save_pretrained(directory)
            assets = {path.relative_to(directory).as_posix(): path.read_text(encoding='utf-8') for path in Path(directory).rglob('*') if path.is_file()}
        config = {'mode': 'language', 'language_config': model.config.to_dict(), 'generation_config': model.generation_config.to_dict(),
                  'foundation_source': {'repo_id': repo_id, 'revision': revision}, 'freeze_foundation': freeze_foundation,
                  'processor_hashes': {name: hashlib.sha256(value.encode()).hexdigest() for name, value in assets.items()}, '_language_assets': assets}
        result = cls(json.loads(json.dumps(config)))
        # Nested foundation configs may request mixed construction dtypes. Own a
        # consistent float32 graph; callers may explicitly cast the whole tool.
        result.language.float()
        result.language.model.load_state_dict(model.state_dict())
        result.eval()
        return result
