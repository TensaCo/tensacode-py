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
from .._internal.ranking import RankingObjective, bindings, normalize_config
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


class Scene(PretrainedTool):
    """Rank explicit descriptions using a learned image/text workspace.

    This interface supplies no object vocabulary or spatial truth rules. Training
    determines behavior; selected candidates remain fallible interpretations.
    """

    def __init__(self, config):
        config = normalize_config(config)
        for name, default in [('patch_size', 8), ('in_channels', 3), ('max_image_size', 256), ('max_candidates', 64)]:
            config.setdefault(name, default)
            if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 1:
                raise ValueError(f'{name} must be a positive integer')
        if config['max_image_size'] < config['patch_size']:
            raise ValueError('max_image_size must accommodate a patch')
        super().__init__(config)
        self.rank = SceneRank(config)
        self.objective = RankingObjective(self)

    def forward(self, inputs, *, context=None):
        if context:
            raise ValueError('Scene does not accept context')
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
