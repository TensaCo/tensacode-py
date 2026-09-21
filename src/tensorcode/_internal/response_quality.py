"""Owned, supervised response-quality judgments over complete supplied inputs.

Three independent binary heads assess support, completeness and constraints.
Foundation relevance training does not train these newly initialized semantics.
Scores are model judgments; neither source truth nor answer correctness is assured.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import threading
import weakref

import torch
from torch import nn
from torch.nn import functional as F

from .latent_ops import LatentOperation
from .vec.text import _native_config, _tokenizer, _tokenizer_config
from ..tracing import invoke
from ..training.calibration import TemperatureCalibration

AXES = ('support', 'completeness', 'constraints')


class _QualityObjective(nn.Module):
    replayable = True

    def __init__(self, owner):
        super().__init__()
        self._owner = weakref.ref(owner)

    def parameters(self, recurse=True):
        return self._owner().parameters(recurse=recurse)

    def configuration(self):
        owner = self._owner()
        return {'operation': type(self).__module__ + '.' + type(self).__qualname__,
                'owner': owner._tool_identity(), 'config': owner.configuration()}

    def __call__(self, value, *, context=None):
        return invoke(self, value, context, super().__call__)

    def forward(self, value, *, context=None):
        if context:
            raise ValueError('response quality objective does not consume context')
        if not isinstance(value, dict) or set(value) != {'inputs', 'targets'}:
            raise ValueError('objective requires inputs and targets')
        return self._owner().loss(value['inputs'], value['targets'])


class ResponseQualityAssessor(LatentOperation):
    """JSON-configured native encoder and three owned binary classifier heads.

    ``forward`` returns raw logits in ``AXES`` order, with shape [3] for one
    input or [batch, 3] for a nonempty list. Only ``loss`` consumes targets.
    ``receipt`` reports one input's scores and full-input truncation status.
    All tokenizer assets are embedded in the data-only configuration.
    ``input_format`` defaults to ``json`` for the single serialized input;
    ``paired`` encodes question/candidate against the complete evidence list
    using the tokenizer's native pair separators and segment IDs.
    """
    training_inputs_include_targets = True

    def __init__(self, config):
        config = self._validated_config(config)
        allowed = {'foundation_config', 'tokenizer_json', 'tokenizer_special_tokens',
                   'tokenizer_options', 'max_tokens', 'foundation', 'calibration', 'input_format'}
        if set(config) - allowed:
            raise ValueError('unsupported response quality configuration fields')
        if config.get('input_format', 'json') not in ('json', 'paired'):
            raise ValueError('input_format must be json or paired')
        if not isinstance(config.get('foundation_config'), dict) or not isinstance(config.get('tokenizer_json'), str):
            raise ValueError('complete native encoder and tokenizer configuration required')
        native = config['foundation_config']
        if native.get('model_type') not in ('bert', 'electra') or native.get('is_decoder') or native.get('is_encoder_decoder'):
            raise ValueError('response quality supports native BERT or Electra encoders')
        config.setdefault('max_tokens', 512)
        config.setdefault('tokenizer_special_tokens', {})
        config.setdefault('tokenizer_options', {})
        config.setdefault('calibration', {})
        if type(config['max_tokens']) is not int or not 2 <= config['max_tokens'] <= native.get('max_position_embeddings', 512):
            raise ValueError('max_tokens must fit the native position capacity and be at least two')
        if any(not isinstance(config[key], dict) for key in ('tokenizer_special_tokens', 'tokenizer_options', 'calibration')):
            raise ValueError('tokenizer and calibration options must be objects')
        options = config['tokenizer_options']
        if set(options) - {'options', 'padding_side', 'truncation_side'} or any(
                options.get(side, 'right') != 'right' for side in ('padding_side', 'truncation_side')):
            raise ValueError('tokenizer options require right padding/truncation and known asset fields')
        if 'foundation' in config and not isinstance(config['foundation'], dict):
            raise ValueError('foundation provenance must be an object')
        super().__init__(config)
        from transformers import AutoModel
        self.encoder = AutoModel.from_config(_native_config(native), trust_remote_code=False,
                                             **({'add_pooling_layer': False} if native['model_type'] == 'bert' else {}))
        self.head = nn.Linear(self.encoder.config.hidden_size, len(AXES))
        self.tokenizer = _tokenizer({'json': config['tokenizer_json'],
                                    'special_tokens': config['tokenizer_special_tokens'],
                                    **config['tokenizer_options']})
        if self.tokenizer.pad_token_id is None:
            raise ValueError('response quality tokenizer requires a padding token')
        if config['max_tokens'] < self.tokenizer.num_special_tokens_to_add(pair=config.get('input_format', 'json') == 'paired'):
            raise ValueError('max_tokens must accommodate the native tokenizer special tokens')
        if len(self.tokenizer) > self.encoder.config.vocab_size:
            raise ValueError('tokenizer vocabulary exceeds native embedding capacity')
        self.calibrations = nn.ModuleDict({axis: TemperatureCalibration(**config['calibration']) for axis in AXES})
        self.register_buffer('calibration_weight_digest', torch.zeros(32, dtype=torch.uint8))
        self._calibration_versions = None
        self._lock = threading.RLock()
        self.objective = _QualityObjective(self)

    @property
    def training_operation(self):
        return self.objective

    @staticmethod
    def validate(inputs):
        if not isinstance(inputs, dict) or set(inputs) != {'question', 'evidence', 'candidate'}:
            raise ValueError('inputs require only question, evidence, candidate; labels belong in targets')
        if any(not isinstance(inputs[key], str) or not inputs[key].strip() for key in ('question', 'candidate')):
            raise ValueError('question and candidate must be nonempty strings')
        evidence = inputs['evidence']
        if not isinstance(evidence, list) or any(not isinstance(item, dict) or set(item) != {'source_id', 'text'} or
                any(not isinstance(item[key], str) or not item[key].strip() for key in ('source_id', 'text')) for item in evidence):
            raise ValueError('evidence must contain only nonempty source_id/text objects')
        if len({item['source_id'] for item in evidence}) != len(evidence):
            raise ValueError('source IDs must be unique')

    @classmethod
    def _text(cls, inputs):
        cls.validate(inputs)
        # JSON escaping preserves field boundaries and all supplied text; no gold,
        # rationale, or authored semantic policy enters the encoded representation.
        return json.dumps({key: inputs[key] for key in ('question', 'evidence', 'candidate')}, ensure_ascii=False)

    def _encode(self, batch, **options):
        if self.config.get('input_format', 'json') == 'json':
            return self.tokenizer([self._text(item) for item in batch], **options)
        for item in batch:
            self.validate(item)
        first = [f"Question: {item['question']}\nCandidate: {item['candidate']}" for item in batch]
        evidence = [json.dumps(item['evidence'], ensure_ascii=False) for item in batch]
        return self.tokenizer(first, text_pair=evidence, return_token_type_ids=True, **options)

    def forward(self, inputs, *, context=None):
        if context:
            raise ValueError('response quality does not consume context')
        single = isinstance(inputs, dict)
        batch = [inputs] if single else inputs
        if not isinstance(batch, list) or not batch:
            raise ValueError('inputs must be one input object or a nonempty list')
        tokens = self._encode(batch, padding=True, truncation=True,
                              max_length=self.config['max_tokens'], return_tensors='pt')
        device = next(self.encoder.parameters()).device
        hidden = self.encoder(**{key: value.to(device) for key, value in tokens.items()}).last_hidden_state
        logits = self.head(hidden[:, 0])
        if not torch.isfinite(logits).all():
            raise ValueError('response quality produced nonfinite logits')
        return logits[0] if single else logits

    @staticmethod
    def _targets(targets):
        if not isinstance(targets, dict) or set(targets) != set(AXES) or any(
                value is not None and type(value) is not bool for value in targets.values()):
            raise ValueError('targets must map each response quality axis to bool or None')
        return [float(targets[axis] or False) for axis in AXES], [targets[axis] is not None for axis in AXES]

    def loss(self, inputs, targets):
        single = isinstance(inputs, dict)
        rows = [targets] if single else targets
        if not isinstance(rows, list) or not rows or (not single and len(rows) != len(inputs)):
            raise ValueError('one target object is required per input')
        parsed = [self._targets(row) for row in rows]
        if not all(any(mask) for _, mask in parsed):
            raise ValueError('at least one reviewed axis is required per input')
        for item in ([inputs] if single else inputs):
            if self.input_metadata(item)['input_truncated']:
                raise ValueError('supervised response quality inputs cannot be truncated')
        self._invalidate_calibration()
        logits = self(inputs).reshape(-1, len(AXES))
        values = torch.tensor([value for value, _ in parsed], device=logits.device, dtype=logits.dtype)
        mask = torch.tensor([mask for _, mask in parsed], device=logits.device, dtype=torch.bool)
        elementwise = F.binary_cross_entropy_with_logits(logits, values, reduction='none')
        return ((elementwise * mask).sum(-1) / mask.sum(-1)).mean()

    def _weight_digest(self):
        digest = hashlib.sha256()
        for prefix, module in (('encoder', self.encoder), ('head', self.head)):
            for name, value in module.state_dict().items():
                digest.update((prefix + '.' + name + str(value.dtype) + str(tuple(value.shape))).encode())
                digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return torch.tensor(list(digest.digest()), dtype=torch.uint8, device=self.calibration_weight_digest.device)

    def _versions(self):
        return tuple((id(value), value._version, str(value.dtype), str(value.device), tuple(value.shape), value.data_ptr())
                     for module in (self.encoder, self.head) for value in [*module.parameters(), *module.buffers()])

    def _invalidate_calibration(self):
        for calibration in self.calibrations.values():
            calibration.calibrated.fill_(False)
            calibration.sample_count.zero_()
            calibration.temperature.fill_(1.)
        self._calibration_versions = None

    def _validate_calibration_weights(self):
        versions = self._versions()
        if any(bool(cal.calibrated) for cal in self.calibrations.values()) and self._calibration_versions != versions:
            if not torch.equal(self.calibration_weight_digest, self._weight_digest()):
                self._invalidate_calibration()
            self._calibration_versions = versions

    @torch.no_grad()
    def fit_calibration(self, logits, targets):
        """Fit separate temperatures on explicit held-out logits and masked labels.

        Callers must supply held-out logits from this exact model's current weights.
        The API cannot establish dataset independence or the origin of supplied scores.
        """
        if not isinstance(logits, torch.Tensor) or not logits.is_floating_point() or logits.ndim != 2 or logits.shape[1] != 3 or not logits.shape[0] or not torch.isfinite(logits).all():
            raise ValueError('calibration logits must be finite [samples, 3]')
        if not isinstance(targets, list) or len(targets) != len(logits):
            raise ValueError('one calibration target object is required per sample')
        parsed = [self._targets(row) for row in targets]
        with self._lock:
            self._invalidate_calibration()
            results = {}
            for index, axis in enumerate(AXES):
                selected = [row for row, (_, mask) in enumerate(parsed) if mask[index]]
                if not selected:
                    results[axis] = None
                    continue
                values = logits[selected, index]
                binary = torch.stack((torch.zeros_like(values), values), dim=-1)
                labels = torch.tensor([int(parsed[row][0][index]) for row in selected], dtype=torch.long)
                results[axis] = self.calibrations[axis].fit(binary, labels)
            self.calibration_weight_digest.copy_(self._weight_digest())
            self._calibration_versions = self._versions()
            return results

    def input_metadata(self, inputs):
        """Inspect full-input token coverage without inference or truncation."""
        with self._lock:
            count = len(self._encode([inputs], truncation=False)['input_ids'][0])
        return {'source_ids': [item['source_id'] for item in inputs['evidence']],
                'input_truncated': count > self.config['max_tokens'],
                'input_token_count': count, 'max_tokens': self.config['max_tokens']}

    def receipt(self, inputs):
        metadata = self.input_metadata(inputs)
        with self._lock:
            self._validate_calibration_weights()
            modes = [(module, module.training) for module in self.modules()]
            try:
                self.eval()
                with torch.no_grad():
                    logits = self(inputs)
                    scores = {axis: float(self.calibrations[axis](torch.stack((logits[index] * 0, logits[index]))).softmax(-1)[1])
                              for index, axis in enumerate(AXES)}
                    if any(not math.isfinite(value) for value in scores.values()):
                        raise ValueError('response quality produced nonfinite scores')
            finally:
                for module, mode in modes:
                    module.training = mode
            return {'scores': scores, 'logits': dict(zip(AXES, logits.cpu().tolist())),
                    'calibrated': {axis: bool(cal.calibrated) for axis, cal in self.calibrations.items()},
                    'calibration_sample_count': {axis: int(cal.sample_count) for axis, cal in self.calibrations.items()},
                    **metadata, 'origin': 'model_inference',
                    'model': copy.deepcopy(self.config.get('foundation', {'initialization': 'random_native_encoder'})),
                    'head_initialization': 'new_random_binary_heads; foundation task does not train these semantics',
                    'semantics': 'supervised model judgments, not truth guarantees'}

    @staticmethod
    def accepts(receipt, threshold=.5):
        """Apply a caller-owned threshold; incomplete input coverage cannot pass."""
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('threshold must be finite in [0, 1]')
        if not isinstance(receipt, dict) or type(receipt.get('input_truncated')) is not bool:
            raise ValueError('receipt requires explicit input truncation status')
        count, limit = receipt.get('input_token_count'), receipt.get('max_tokens')
        if type(count) is not int or count < 1 or type(limit) is not int or limit < 2 or receipt['input_truncated'] != (count > limit):
            raise ValueError('receipt requires consistent token coverage metadata')
        scores = receipt.get('scores')
        if not isinstance(scores, dict) or set(scores) != set(AXES) or any(
                type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values()):
            raise ValueError('receipt requires finite scores for all axes')
        return not receipt['input_truncated'] and all(value >= threshold for value in scores.values())

    @classmethod
    def from_foundation(cls, repo, *, revision=None, max_tokens=512, local_files_only=False, input_format='json'):
        """Explicit native safetensors bootstrap; response-quality heads start random."""
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        native = AutoConfig.from_pretrained(repo, revision=revision, local_files_only=local_files_only, trust_remote_code=False)
        if native.model_type not in ('bert', 'electra') or native.is_decoder or native.is_encoder_decoder:
            raise ValueError('response quality supports native BERT or Electra encoders')
        encoder, info = AutoModel.from_pretrained(repo, revision=revision, local_files_only=local_files_only,
                           trust_remote_code=False, use_safetensors=True, output_loading_info=True,
                           **({'add_pooling_layer': False} if native.model_type == 'bert' else {}))
        if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
            raise ValueError(f'foundation encoder has incomplete weights: {info}')
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, local_files_only=local_files_only,
                                                  trust_remote_code=False, use_fast=True)
        assets = _tokenizer_config(tokenizer)
        resolved = getattr(encoder.config, '_commit_hash', None) or revision
        if not Path(repo).is_dir() and not resolved:
            raise ValueError('foundation provenance requires a resolved revision')
        result = cls({'foundation_config': json.loads(encoder.config.to_json_string()), 'tokenizer_json': assets.pop('json'),
                      'tokenizer_special_tokens': assets.pop('special_tokens'), 'tokenizer_options': assets,
                      'max_tokens': max_tokens, 'input_format': input_format,
                      'foundation': {'repository': str(repo), 'revision': resolved,
                      'initialization': 'pretrained_encoder_only', 'response_quality_heads_pretrained': False}})
        result.encoder.load_state_dict(encoder.state_dict(), strict=True)
        result.eval()
        return result
