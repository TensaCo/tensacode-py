"""Trainable evidence-conditioned hypothesis ranking."""
import copy
import json

import torch
from torch.nn import functional as F

from .._internal.pretrained import PretrainedTool
from .._internal.proposals import generate_proposals, proposal_loss
from .chatbot import Chatbot
from ..training.calibration import TemperatureCalibration
from .._internal.ranking import RankOperation, RankingObjective, RankingSession, bindings, from_foundation, normalize_config


class Investigator(PretrainedTool):
    """Own an encoder, shared workspace, and learned hypothesis scoring head.

    Candidate hypotheses and evidence are supplied inputs, not discovered facts.
    Probabilities are uncalibrated model scores. Construction uses random weights;
    use ``from_pretrained`` to load learned weights.
    """

    def __init__(self, config):
        config = json.loads(json.dumps(config))
        generator = Chatbot(config['generator']) if 'generator' in config else None
        if generator is not None:
            config['generator'] = generator.configuration()
        config.setdefault('max_proposals', 16)
        config.setdefault('proposal_template_version', 1)
        if type(config['max_proposals']) is not int or config['max_proposals'] < 1:
            raise ValueError('max_proposals must be a positive integer')
        super().__init__(normalize_config(config))
        self.rank = RankOperation(self.config, task_key='question', candidates_key='hypotheses')
        self.objective = RankingObjective(self)
        self.generator = generator
        self.verifier = EvidenceVerifier(self.config) if 'verifier_config' in self.config else None

    from_foundation = classmethod(from_foundation)

    def forward(self, inputs, *, context=None):
        if context:
            raise ValueError('This tool does not accept context')
        return self.rank.receipt(inputs, probabilities=True) if 'hypotheses' in inputs else self.investigate(inputs)

    def propose(self, inputs, *, count=3):
        return generate_proposals(self.generator, inputs, task_key='question', count=count,
                                  max_count=self.config['max_proposals'])

    def proposal_loss(self, inputs, targets):
        return proposal_loss(self.generator, inputs, targets, task_key='question')

    def investigate(self, inputs, *, count=3):
        if self.verifier is None:
            raise ValueError('evidence verification capability is not configured')
        value = copy.deepcopy(inputs)
        if 'hypotheses' not in value:
            value['hypotheses'] = self.propose(value, count=count)
        if not value['hypotheses']:
            raise ValueError('generator produced no nonempty distinct hypotheses')
        self.rank.validate(value)
        result = self.rank.receipt(value, probabilities=True)
        for candidate in result['candidates']:
            candidate['verifications'] = self.verifier.verify(candidate['text'], value.get('evidence', []))
        result['verification_semantics'] = 'source-wise NLI model distributions; calibration fit status is explicit and does not establish facts'
        return result

    def verification_loss(self, inputs, targets):
        if self.verifier is None:
            raise ValueError('evidence verification capability is not configured')
        return self.verifier.loss(inputs, targets)

    def configuration(self):
        config = super().configuration()
        if self.generator is not None:
            config['generator'] = self.generator.configuration()
        return config

    @classmethod
    def from_foundations(cls, encoder_repo, generator_repo, verifier_repo, *,
                         encoder_revision=None, generator_revision=None, verifier_revision=None,
                         verifier_labels, local_files_only=False, generator_options=None, **options):
        from pathlib import Path
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        generator = Chatbot.from_foundation(generator_repo, revision=generator_revision,
                    local_files_only=local_files_only, **(generator_options or {}))
        tokenizer = AutoTokenizer.from_pretrained(verifier_repo, revision=verifier_revision,
                    local_files_only=local_files_only, use_fast=True)
        verifier = AutoModelForSequenceClassification.from_pretrained(verifier_repo,
                    revision=verifier_revision, local_files_only=local_files_only)
        resolved = getattr(verifier.config, '_commit_hash', None) or verifier_revision
        if not Path(verifier_repo).is_dir() and not resolved:
            raise ValueError('verifier provenance requires a resolved Hub revision')
        if not hasattr(tokenizer, 'backend_tokenizer'):
            raise ValueError('verifier requires a serializable fast tokenizer')
        result = cls.from_foundation(encoder_repo, revision=encoder_revision,
                    local_files_only=local_files_only, generator=generator.configuration(),
                    verifier_config=verifier.config.to_dict(),
                    verifier_tokenizer_json=tokenizer.backend_tokenizer.to_str(),
                    verifier_tokenizer_special_tokens={k: str(v) for k, v in tokenizer.special_tokens_map.items() if isinstance(v, str)},
                    verifier_labels=verifier_labels,
                    verifier_foundation={'repository': str(verifier_repo), 'revision': resolved}, **options)
        result.generator.load_state_dict(generator.state_dict())
        result.verifier.model.load_state_dict(verifier.state_dict())
        return result

    predict = forward

    def new_session(self):
        return RankingSession(self)

    def loss(self, inputs, targets):
        logits = self.rank(inputs)
        if isinstance(targets, (list, tuple)) or isinstance(targets, torch.Tensor) and targets.ndim == 1:
            target = torch.as_tensor(targets, dtype=logits.dtype, device=logits.device)
            if target.shape != logits.shape or not torch.isfinite(target).all() or (target < 0).any() or not torch.isclose(target.sum(), target.new_tensor(1.0), atol=1e-5):
                raise ValueError('target distribution must be finite, nonnegative and sum to one')
            return -(target * logits.log_softmax(-1)).sum()
        if isinstance(targets, str):
            ids = [item['id'] for item in inputs['hypotheses']]
            if targets not in ids:
                raise ValueError('target must identify a supplied hypothesis')
            targets = ids.index(targets)
        if isinstance(targets, bool) or not isinstance(targets, int) or not 0 <= targets < logits.numel():
            raise ValueError('target must be a valid hypothesis index or ID')
        return F.cross_entropy(logits.unsqueeze(0), torch.tensor([targets], device=logits.device))

    @property
    def training_operation(self):
        return self.objective

    training_inputs_include_targets = True

    def operation_bindings(self):
        result = bindings(self)
        if self.generator is not None:
            result.update({f'generator.{key}': value for key, value in self.generator.operation_bindings().items()})
        return result


class _VerifierCalibration(TemperatureCalibration):
    def __init__(self, owner, **options):
        super().__init__(**options)
        import weakref
        self._owner = weakref.ref(owner)

    def fit(self, logits, labels):
        result = super().fit(logits, labels)
        self._owner()._record_calibration_weights()
        return result


class EvidenceVerifier(torch.nn.Module):
    """Owned classifier with explicit semantic label mapping and source identity."""
    replayable = True

    def __init__(self, config):
        super().__init__()
        from tokenizers import Tokenizer
        from transformers import AutoConfig, AutoModelForSequenceClassification, PreTrainedTokenizerFast
        self.config = {key: copy.deepcopy(value) for key, value in config.items() if key.startswith('verifier_')}
        native = dict(config['verifier_config'])
        self.model = AutoModelForSequenceClassification.from_config(AutoConfig.for_model(native.pop('model_type'), **native))
        self.calibration = _VerifierCalibration(self, **config.get('verifier_calibration', {}))
        self.register_buffer('calibration_weight_digest', torch.zeros(32, dtype=torch.uint8))
        self._calibration_versions = None
        import threading
        self._lock = threading.RLock()
        self.labels = config.get('verifier_labels')
        if (not isinstance(self.labels, dict) or set(self.labels) != {'support', 'contradiction', 'unknown'}
                or any(type(index) is not int for index in self.labels.values())
                or set(self.labels.values()) != set(range(self.model.config.num_labels))):
            raise ValueError('verifier_labels must explicitly map support, contradiction, unknown to the three classifier indices')
        self.max_tokens = config.get('verifier_max_tokens', 512)
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError('verifier_max_tokens must be a positive integer')
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(config['verifier_tokenizer_json']),
                    **config.get('verifier_tokenizer_special_tokens', {}))
        if self.tokenizer.pad_token_id is None:
            raise ValueError('verifier tokenizer requires padding token')

    def configuration(self):
        return copy.deepcopy(self.config)

    def forward(self, inputs, *, context=None):
        if context:
            raise ValueError('verifier does not consume context')
        if not isinstance(inputs, list) or not inputs or any(not isinstance(x, dict) or any(not isinstance(x.get(k), str) or not x[k].strip() for k in ('premise', 'hypothesis')) for x in inputs):
            raise ValueError('verifier inputs must be nonempty premise/hypothesis pairs')
        tokens = self.tokenizer([x['premise'] for x in inputs], [x['hypothesis'] for x in inputs],
                    padding=True, truncation=True, max_length=self.max_tokens, return_tensors='pt')
        device = next(self.model.parameters()).device
        return self.model(**{k: v.to(device) for k, v in tokens.items()}).logits

    def __call__(self, value, *, context=None):
        from ..tracing import invoke
        return invoke(self, value, context, super().__call__)

    def loss(self, inputs, targets):
        if not isinstance(targets, (list, tuple)) or len(targets) != len(inputs) or any(not isinstance(t, str) or t not in self.labels for t in targets):
            raise ValueError('targets must provide one named NLI label per pair')
        self.calibration.calibrated.fill_(False)
        self.calibration.sample_count.zero_()
        self.calibration.temperature.fill_(1.)
        logits = self(inputs)
        return F.cross_entropy(logits, torch.tensor([self.labels[t] for t in targets], device=logits.device))

    def _weight_digest(self):
        import hashlib
        digest = hashlib.sha256()
        for name, tensor in self.model.state_dict().items():
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return torch.tensor(list(digest.digest()), dtype=torch.uint8, device=self.calibration_weight_digest.device)

    def _versions(self):
        return tuple((name, id(tensor), tensor._version, str(tensor.dtype), str(tensor.device),
                      tuple(tensor.shape), tensor.data_ptr())
                     for name, tensor in [*self.model.named_parameters(), *self.model.named_buffers()])

    def _record_calibration_weights(self):
        self.calibration_weight_digest.copy_(self._weight_digest())
        self._calibration_versions = self._versions()

    def _validate_calibration_weights(self):
        if bool(self.calibration.calibrated) and self._calibration_versions != self._versions():
            if not torch.equal(self.calibration_weight_digest, self._weight_digest()):
                self.calibration.calibrated.fill_(False)
                self.calibration.sample_count.zero_()
                self.calibration.temperature.fill_(1.)
            self._calibration_versions = self._versions()

    def verify(self, hypothesis, evidence):
        with self._lock:
            return self._verify(hypothesis, evidence)

    def _verify(self, hypothesis, evidence):
        self._validate_calibration_weights()
        if not evidence:
            return []
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            with torch.no_grad():
                logits = self([{'premise': x['text'], 'hypothesis': hypothesis} for x in evidence])
                probabilities = self.calibration(logits).softmax(-1).cpu().tolist()
        finally:
            for module, training in modes:
                module.training = training
        return [{'source_id': source['source_id'], 'distribution': {label: row[index] for label, index in self.labels.items()},
                 'origin': 'model_inference', 'calibrated': bool(self.calibration.calibrated.item()),
                 'calibration_sample_count': int(self.calibration.sample_count.item()),
                 'model': copy.deepcopy(self.config.get('verifier_foundation', {'initialization': 'configured_weights'})),
                 'input_truncated': len(self.tokenizer(source['text'], hypothesis)['input_ids']) > self.max_tokens}
                for source, row in zip(evidence, probabilities)]
