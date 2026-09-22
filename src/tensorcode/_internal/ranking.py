"""Owned text candidate models; predictions are not observations or actions."""
from __future__ import annotations

import copy
import re

import torch
from torch import nn

from .vec.adapter import TensorAdapter as Transform
from ..tracing import invoke
from .workspace import Workspace
from .proposals import conversation_context, conversation_block


def normalize_config(config):
    result = dict(config)
    if 'foundation_config' in result:
        import json
        result['foundation_config'] = json.loads(json.dumps(result['foundation_config']))
        if not isinstance(result.get('tokenizer_json'), str):
            raise ValueError('foundation requires tokenizer_json')
        result.setdefault('freeze_foundation', True)
        if not isinstance(result['freeze_foundation'], bool):
            raise ValueError('freeze_foundation must be boolean')
        result.setdefault('vocabulary', ['<foundation>'])
    vocabulary = result.get('vocabulary')
    if not isinstance(vocabulary, list) or not vocabulary or any(not isinstance(x, str) or not x for x in vocabulary):
        raise ValueError('vocabulary must be a nonempty list of unique nonempty strings')
    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError('vocabulary must contain unique strings')
    for key, default in [('dimensions', 32), ('slots', 4), ('steps', 2), ('max_tokens', 256)]:
        result.setdefault(key, default)
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if result.get('architecture_version', 1) != 1:
        raise ValueError('unsupported ranking architecture_version')
    result.setdefault('cache_records', 0)
    if type(result['cache_records']) is not int or result['cache_records'] < 0:
        raise ValueError('cache_records must be a nonnegative integer')
    result['architecture_version'] = 1
    return result


class FoundationEncoding(nn.Module):
    """Native contextual encoder; configuration and tokenizer are checkpoint-owned."""

    def __init__(self, config):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        self.config = copy.deepcopy(config)
        native = dict(config['foundation_config'])
        model_type = native.pop('model_type')
        self.model = AutoModel.from_config(AutoConfig.for_model(model_type, **native))
        self.frozen = config['freeze_foundation']
        self.model.requires_grad_(not self.frozen)
        if self.frozen:
            self.model.eval()

    def configuration(self):
        return {'foundation_config': self.config['foundation_config'], 'freeze_foundation': self.frozen}

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def forward(self, inputs):
        return self.model(**inputs).last_hidden_state


def from_foundation(cls, repo, *, revision=None, local_files_only=False, **options):
    import json
    from pathlib import Path
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, local_files_only=local_files_only, use_fast=True)
    model = AutoModel.from_pretrained(repo, revision=revision, local_files_only=local_files_only)
    if not hasattr(tokenizer, 'backend_tokenizer'):
        raise ValueError('foundation requires a serializable fast tokenizer')
    resolved = getattr(model.config, '_commit_hash', None) or revision
    if not Path(repo).is_dir() and not resolved:
        raise ValueError('foundation provenance requires a resolved revision')
    config = dict(options, foundation_config=json.loads(json.dumps(model.config.to_dict())), tokenizer_json=tokenizer.backend_tokenizer.to_str(), tokenizer_special_tokens={k: str(v) for k, v in tokenizer.special_tokens_map.items() if isinstance(v, str)}, foundation={'repository': str(repo), 'revision': resolved, 'workspace_initialization': 'random'})
    result = cls(config)
    result.rank.encode.module.model.load_state_dict(model.state_dict())
    return result


class RankOperation(nn.Module):
    replayable = True

    def __init__(self, config, *, task_key, candidates_key):
        super().__init__()
        self.config = copy.deepcopy(config)
        from collections import OrderedDict
        self._encoding_cache = OrderedDict()
        self.task_key, self.candidates_key = task_key, candidates_key
        self.vocabulary = {token: index + 1 for index, token in enumerate(config['vocabulary'])}
        dimensions = config['dimensions']
        if 'foundation_config' in config:
            from tokenizers import Tokenizer
            from transformers import PreTrainedTokenizerFast
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(config['tokenizer_json']), **config.get('tokenizer_special_tokens', {}))
            self.encode = Transform(FoundationEncoding(config))
            hidden_size = self.encode.module.model.config.hidden_size
            self.projection = Transform(nn.Linear(hidden_size, dimensions))
        else:
            self.tokenizer = None
            self.encode = Transform(nn.Embedding(len(self.vocabulary) + 1, dimensions))
        self.workspace = Workspace(dimensions, config['slots'], config['steps'])
        self.query = Transform(nn.Linear(dimensions, dimensions))
        self.candidate = Transform(nn.Linear(dimensions, dimensions))
        self.score = Transform(nn.Sequential(nn.Linear(dimensions * 4, dimensions), nn.GELU(), nn.Linear(dimensions, 1)))

    def configuration(self):
        return {'operation': type(self).__module__ + '.' + type(self).__qualname__, 'config': copy.deepcopy(self.config), 'task_key': self.task_key, 'candidates_key': self.candidates_key}

    def __call__(self, value, *, context=None):
        return invoke(self, value, context, super().__call__)

    def validate(self, value):
        if not isinstance(value, dict) or not isinstance(value.get(self.task_key), str) or not value[self.task_key].strip():
            raise ValueError(f'{self.task_key} must be nonempty text')
        evidence = value.get('evidence', [])
        candidates = value.get(self.candidates_key)
        if not isinstance(evidence, list) or not isinstance(candidates, list) or not candidates:
            raise ValueError('evidence must be a list and candidates a nonempty list')
        for records, id_key in [(evidence, 'source_id'), (candidates, 'id')]:
            ids = []
            for record in records:
                if not isinstance(record, dict) or not isinstance(record.get(id_key), str) or not record[id_key] or not isinstance(record.get('text'), str) or not record['text'].strip():
                    raise ValueError(f'each record requires nonempty {id_key} and text')
                ids.append(record[id_key])
            if len(set(ids)) != len(ids):
                raise ValueError(f'{id_key} values must be unique')
        return evidence, candidates

    def clear_encoding_cache(self):
        self._encoding_cache.clear()

    def _apply(self, fn, recurse=True):
        self.clear_encoding_cache()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self.clear_encoding_cache()
        return super()._load_from_state_dict(*args, **kwargs)

    def tokens(self, text):
        words = re.findall(r"\w+|[^\w\s]", text.casefold())
        ids = [self.vocabulary.get(word, 0) for word in words[:self.config['max_tokens']]] or [0]
        return torch.tensor(ids, device=self.encode.module.weight.device, dtype=torch.long)

    def compute(self, value, *, workspace_ablation=None):
        evidence, candidates = self.validate(value)
        # Encode each source separately so attention retains exact source identity.
        dialogue = conversation_context(value)
        dialogue_text = conversation_block(dialogue)
        if dialogue:
            length = (len(self.tokenizer(dialogue_text)['input_ids']) if self.tokenizer is not None
                      else len(re.findall(r'\w+|[^\w\s]', dialogue_text.casefold())))
            if length > self.config['max_tokens']:
                raise ValueError('Conversation context exceeds ranking token budget')
        segments = [(None, value[self.task_key])] + ([(None, dialogue_text)] if dialogue else []) + [(item['source_id'], item['text']) for item in evidence]
        states, sources = [], []
        texts = [text for _, text in segments] + [item['text'] for item in candidates]
        if self.tokenizer is not None:
            encoded_inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=self.config['max_tokens'], return_tensors='pt')
            device = next(self.encode.parameters()).device
            encoded_inputs = {key: value.to(device) for key, value in encoded_inputs.items()}
            cache_enabled = self.config['cache_records'] > 0 and not any(p.requires_grad for p in self.encode.parameters()) and not self.encode.module.model.training
            cache_key = tuple(texts)
            cached = self._encoding_cache.get(cache_key) if cache_enabled else None
            if cached is None:
                hidden = self.encode(encoded_inputs)
                if cache_enabled:
                    # Inference-warmed caches must remain usable by later autograd.
                    with torch.inference_mode(False):
                        self._encoding_cache[cache_key] = hidden.detach().cpu().clone()
                    if len(self._encoding_cache) > self.config['cache_records']:
                        self._encoding_cache.popitem(last=False)
            else:
                self._encoding_cache.move_to_end(cache_key)
                hidden = cached.to(device=device, dtype=next(self.encode.parameters()).dtype)
            batch = self.projection(hidden)
            encoded_texts = [row[mask.bool()] for row, mask in zip(batch, encoded_inputs['attention_mask'])]
        else:
            encoded_texts = [self.encode(self.tokens(text)) for text in texts]
        for (source_id, text), encoded in zip(segments, encoded_texts):
            states.append(encoded)
            sources.extend([source_id] * encoded.shape[0])
        workspace = self.workspace(torch.cat(states).unsqueeze(0))
        if workspace_ablation not in (None, 'zero', 'bypass'):
            raise ValueError('workspace_ablation must be None, zero, or bypass')
        conditioning = workspace['conditioning'].mean(1)
        if workspace_ablation == 'zero':
            conditioning = torch.zeros_like(conditioning)
        elif workspace_ablation == 'bypass':
            conditioning = torch.cat(states).mean(0, keepdim=True)
        query = self.query(conditioning).expand(len(candidates), -1)
        candidate = self.candidate(torch.stack([encoded.mean(0) for encoded in encoded_texts[len(segments):]]))
        scores = self.score(torch.cat([query, candidate, query * candidate, torch.abs(query - candidate)], -1)).squeeze(-1)
        return scores, workspace, sources

    def forward(self, value, *, context=None):
        if context:
            raise ValueError('RankOperation does not accept context')
        return self.compute(value)[0]

    def receipt(self, value, *, probabilities=False):
        scores, workspace, sources = self.compute(value)
        selected = int(scores.argmax().item())
        candidates = value[self.candidates_key]
        result = {
            'selected_id': candidates[selected]['id'],
            'candidates': [dict(copy.deepcopy(item), predicted_score=float(score)) for item, score in zip(candidates, scores.detach().cpu().tolist())],
            'evidence': copy.deepcopy(value.get('evidence', [])),
            'attention': workspace['attention'][0].detach().cpu().tolist(),
            'attention_source_ids': sources,
            'relations': workspace['relations'][0].detach().cpu().tolist(),
        }
        if probabilities:
            for item, probability in zip(result['candidates'], scores.softmax(-1).detach().cpu().tolist()):
                item['probability'] = probability
        return result


class RankingObjective(nn.Module):
    replayable = True

    def __init__(self, tool):
        super().__init__()
        import weakref
        object.__setattr__(self, '_tool_ref', weakref.ref(tool))

    def parameters(self, recurse=True):
        return self._tool_ref().parameters(recurse=recurse)

    def configuration(self):
        tool = self._tool_ref()
        return {'operation': type(self).__module__ + '.' + type(self).__qualname__, 'tool': type(tool).__name__, 'config': tool.configuration()}

    def __call__(self, value, *, context=None):
        return invoke(self, value, context, super().__call__)

    def forward(self, value, *, context=None):
        if context:
            raise ValueError('RankingObjective does not accept context')
        tool = self._tool_ref()
        inputs = value['inputs']
        mode = value.get('mode', 'rank')
        if isinstance(inputs, dict) and set(inputs) == {'mode', 'inputs'}:
            mode, inputs = inputs['mode'], inputs['inputs']
        if mode == 'rank':
            return tool.loss(inputs, value['targets'])
        if mode == 'proposal':
            objective = getattr(tool, 'proposal_loss', None) or getattr(tool, 'generation_loss', None)
        elif mode == 'verification':
            objective = getattr(tool, 'verification_loss', None)
        elif mode == 'retrieval':
            objective = getattr(tool, 'retrieval_loss', None)
        else:
            raise ValueError('training mode must be rank, proposal, verification, or retrieval')
        if objective is None:
            raise ValueError(f'{mode} training capability is not configured')
        return objective(inputs, value['targets']).clone()


def bindings(tool):
    return {name: module for name, module in tool.named_modules() if name and getattr(module, 'replayable', False)}


class RankingSession:
    """Independent receipt history; each call supplies its complete evidence."""

    def __init__(self, tool):
        self.tool = tool
        self._history = []

    @property
    def history(self):
        return copy.deepcopy(self._history)

    def __call__(self, inputs, *, context=None):
        if context:
            raise ValueError('RankingSession does not accept context')
        snapshot = copy.deepcopy(inputs)
        receipt = self.tool(snapshot)
        key = self.tool.rank.candidates_key
        if key not in snapshot:
            # Persist the actual generated alternatives, never regenerate on load.
            snapshot[key] = [{k: copy.deepcopy(v) for k, v in candidate.items()
                              if k not in {'predicted_score', 'probability', 'verifications', 'joint_verification'}}
                             for candidate in receipt['candidates']]
        previous = self._history[-1]['receipt']['selected_id'] if self._history else None
        receipt['previous_selected_id'] = previous
        receipt['revised'] = previous is not None and previous != receipt['selected_id']
        self._history.append({'inputs': snapshot, 'receipt': copy.deepcopy(receipt)})
        return receipt

    def _identity(self):
        import hashlib
        import json
        config = {'tool': type(self.tool).__module__ + '.' + type(self.tool).__qualname__, 'config': self.tool.configuration()}
        return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def save(self, path):
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps({'version': 1, 'identity': self._identity(), 'history': self._history}, allow_nan=False), encoding='utf-8')

    @classmethod
    def load(cls, path, tool):
        import json
        from pathlib import Path
        result = cls(tool)
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('version') != 1 or data.get('identity') != result._identity() or not isinstance(data.get('history'), list):
            raise ValueError('session architecture identity or format mismatch')
        previous = None
        for item in data['history']:
            if not isinstance(item, dict) or set(item) != {'inputs', 'receipt'}:
                raise ValueError('invalid session record')
            receipt = item['receipt']
            if (isinstance(receipt, dict) and receipt.get('abstained') is True
                    and receipt.get('selected_id') is None and receipt.get('candidates') == []
                    and item['inputs'].get(tool.rank.candidates_key) == []):
                from .proposals import proposal_prompt
                proposal_prompt(item['inputs'], tool.rank.task_key)
                if (receipt.get('evidence') != item['inputs'].get('evidence', [])
                        or receipt.get('previous_selected_id') != previous
                        or receipt.get('revised') is not (previous is not None)):
                    raise ValueError('invalid abstention receipt')
                previous = None
                continue
            evidence, candidates = tool.rank.validate(item['inputs'])
            if not isinstance(receipt, dict) or receipt.get('evidence') != evidence or receipt.get('selected_id') not in {x['id'] for x in candidates}:
                raise ValueError('session receipt does not match its evidence/candidates')
            source_ids = receipt.get('attention_source_ids')
            if not isinstance(source_ids, list) or any(source is not None and source not in {entry['source_id'] for entry in evidence} for source in source_ids):
                raise ValueError('session attention source mismatch')
            recorded = receipt.get('candidates')
            if not isinstance(recorded, list) or len(recorded) != len(candidates) or any(not isinstance(a, dict) or {k: v for k, v in a.items() if k not in {'predicted_score', 'probability', 'verifications', 'joint_verification'}} != {k: v for k, v in b.items() if k not in {'predicted_score', 'probability', 'verifications', 'joint_verification'}} for a, b in zip(recorded, candidates)):
                raise ValueError('session candidate receipt mismatch')
            if receipt.get('previous_selected_id') != previous or receipt.get('revised') is not (previous is not None and previous != receipt['selected_id']):
                raise ValueError('session revision history mismatch')
            import math
            if any(isinstance(candidate.get('predicted_score'), bool) or not isinstance(candidate.get('predicted_score'), (int, float)) or not math.isfinite(candidate['predicted_score']) for candidate in recorded):
                raise ValueError('session contains invalid predicted scores')
            for candidate in recorded:
                verified = ('verification_semantics' in receipt or
                            candidate.get('origin') == 'generated' and getattr(tool, 'verifier', None) is not None)
                if verified or 'verifications' in candidate:
                    checks = candidate.get('verifications')
                    if (not isinstance(checks, list) or len(checks) != len(evidence)
                            or any(not isinstance(check, dict) or check.get('source_id') != source['source_id']
                                   for check, source in zip(checks, evidence))):
                        raise ValueError('session verification source mismatch')
                    extra_checks = []
                    if tool.config.get('verification_scope') == 'joint':
                        joint = candidate.get('joint_verification')
                        if evidence:
                            if (not isinstance(joint, dict)
                                    or joint.get('source_ids') != [source['source_id'] for source in evidence]
                                    or joint.get('scope') != 'joint'
                                    or type(joint.get('token_count')) is not int or joint['token_count'] < 1
                                    or type(joint.get('max_tokens')) is not int or joint['max_tokens'] < 1
                                    or type(joint.get('input_truncated')) is not bool
                                    or joint['input_truncated'] != (joint['token_count'] > joint['max_tokens'])):
                                raise ValueError('invalid session joint verification coverage')
                            extra_checks.append(joint)
                        elif joint is not None:
                            raise ValueError('invalid session joint verification without evidence')
                    elif 'joint_verification' in candidate:
                        raise ValueError('session joint verification conflicts with configured scope')
                    for check in [*checks, *extra_checks]:
                        distribution = check.get('distribution')
                        if (not isinstance(distribution, dict) or set(distribution) != {'support', 'contradiction', 'unknown'}
                                or any(type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
                                       for score in distribution.values())
                                or not math.isclose(sum(distribution.values()), 1., abs_tol=1e-5)):
                            raise ValueError('invalid session verification distribution')
                        calibrated, count = check.get('calibrated'), check.get('calibration_sample_count')
                        if (type(calibrated) is not bool or type(count) is not int or count < 0
                                or calibrated != (count > 0) or check.get('origin') != 'model_inference'
                                or type(check.get('input_truncated')) is not bool
                                or not isinstance(check.get('model'), dict)):
                            raise ValueError('invalid session verification provenance or calibration')
                        verifier = getattr(tool, 'verifier', None)
                        if verifier is not None and check['model'] != verifier.config.get('verifier_foundation', {'initialization': 'configured_weights'}):
                            raise ValueError('session verifier model provenance mismatch')
            previous = receipt['selected_id']
        result._history = copy.deepcopy(data['history'])
        return result
