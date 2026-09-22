"""Explicit source-grounded session composition; generated claims remain hypotheses."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import threading
from functools import wraps
from pathlib import Path

import torch

from .cognitive_state import Assessment, CognitiveState, Evidence, Hypothesis, _text
from .episodic import EpisodicMemory


_LOCK_CREATION = threading.Lock()


def _ensure_lock(investigator):
    with _LOCK_CREATION:
        if not hasattr(investigator, '_cognition_lock'):
            investigator._cognition_lock = threading.RLock()
        if not hasattr(investigator, '_cognition_fingerprint'):
            investigator._cognition_fingerprint = _ModelFingerprint()


def _model_locked(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self.investigator._cognition_lock:
            return method(self, *args, **kwargs)
    return locked


class _ModelFingerprint:
    """Cache content hashes by tensor identity/version and configuration.

    Standard optimizer/no_grad mutations invalidate the hash. Mutating via .data
    bypasses PyTorch version counters and is unsupported; call invalidate after
    any such external mutation. Concurrent weight updates are unsupported.
    """
    def __init__(self):
        self._key = None
        self._digest = None

    def invalidate(self):
        self._key = None

    def __call__(self, modules, config):
        serialized = json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False)
        tensors = [(f'{prefix}.{name}', tensor) for prefix, module in modules
                   for name, tensor in list(module.named_parameters()) + list(module.named_buffers())]
        key = (serialized, tuple((name, id(t), t._version, str(t.dtype), str(t.device), tuple(t.shape)) for name, t in tensors))
        if key != self._key:
            digest = hashlib.sha256(serialized.encode())
            for name, tensor in tensors:
                digest.update(name.encode())
                digest.update(str(tensor.dtype).encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            self._digest = digest.hexdigest()
            self._key = key
        return self._digest


@dataclass(frozen=True)
class SelectionPolicy:
    """Authored thresholds on model scores, not a learned truth criterion.

    Require support from at least one source, inspect unknown on that strongest
    supporting source, and veto contradiction from ANY current source. Explicit
    joint verification instead requires support/unknown on the combined premise,
    retaining every source contradiction veto. Source trust is caller supplied.
    """
    min_support: float = .7
    max_contradiction: float = .2
    max_unknown: float = .3

    def __post_init__(self):
        for value in asdict(self).values():
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('policy thresholds must be finite values in [0, 1]')

    def accepts(self, distributions, *, joint_distribution=None):
        if not distributions:
            return False
        checked = list(distributions) + ([joint_distribution] if joint_distribution is not None else [])
        for row in checked:
            if set(row) != {'support', 'contradiction', 'unknown'} or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in row.values()):
                raise ValueError('expected finite support/contradiction/unknown distributions')
        strongest = joint_distribution if joint_distribution is not None else max(distributions, key=lambda row: row['support'])
        return (strongest['support'] >= self.min_support
                and strongest['unknown'] <= self.max_unknown
                and max(row['contradiction'] for row in checked) <= self.max_contradiction)

    def accepts_verification(self, verification, source_ids, *, scope):
        """Require complete identified evidence coverage before screening scores."""
        checks = verification['verifications']
        if (len(checks) != len(source_ids) or len(set(source_ids)) != len(source_ids)
                or {row['source_id'] for row in checks} != set(source_ids)):
            raise ValueError('verification sources must match active evidence exactly')
        joint = verification.get('joint_verification')
        if scope not in ('source', 'joint'):
            raise ValueError('verification_scope must be source or joint')
        if scope == 'joint':
            if not source_ids:
                return False
            if (not isinstance(joint, dict) or joint.get('source_ids') != source_ids
                    or joint.get('scope') != 'joint'):
                raise ValueError('joint verification sources must match active evidence exactly')
            if (type(joint.get('input_truncated')) is not bool
                    or type(joint.get('token_count')) is not int or joint['token_count'] < 1
                    or type(joint.get('max_tokens')) is not int or joint['max_tokens'] < 1
                    or joint['input_truncated'] != (joint['token_count'] > joint['max_tokens'])
                    or any(type(row.get('input_truncated')) is not bool for row in checks)):
                raise ValueError('joint verification requires explicit input coverage metadata')
        if any(row.get('input_truncated', False) for row in checks):
            return False
        if scope == 'joint' and joint.get('input_truncated', False):
            return False
        return self.accepts([row['distribution'] for row in checks],
                            joint_distribution=joint['distribution'] if scope == 'joint' else None)

    def receipt(self, *, scope='source'):
        return dict(asdict(self), origin='authored_policy',
                    aggregation=('joint-premise support/unknown; maximum contradiction across joint and individual sources'
                                 if scope == 'joint' else
                                 'strongest-source support/unknown; maximum contradiction across current sources'),
                    semantics='model-score screening, not established truth; source trust supplied by caller')


class LearnedEpisodicMemory:
    """Owned retrieval embeddings, or explicit rank-pooling artifact fallback.

    Dedicated sentence encoders preserve their trained embedding geometry. Rank
    encoder pooling is retained for existing artifacts, without claiming that
    ranking supervision learned a useful cosine retrieval space.
    """
    def __init__(self, investigator, *, capacity=256):
        self.investigator = investigator
        _ensure_lock(investigator)
        self._fingerprint = _ModelFingerprint()
        self.memory = EpisodicMemory(capacity=capacity, model_fingerprint=self.fingerprint)
        self._evidence = {}

    @property
    @_model_locked
    def fingerprint(self):
        dedicated = getattr(self.investigator, 'episodic_encoder', None)
        if dedicated is not None:
            return self._fingerprint([('episodic_encoder', dedicated)], dedicated.configuration())
        rank = self.investigator.rank
        modules = [('encode', rank.encode)]
        if rank.tokenizer is not None:
            modules.append(('projection', rank.projection))
        return self._fingerprint(modules, rank.configuration())

    @property
    def metadata(self):
        dedicated = getattr(self.investigator, 'episodic_encoder', None)
        if dedicated is not None:
            return dedicated.metadata
        rank = self.investigator.rank
        return {'encoder': 'rank_encoder_pooling', 'pooling': 'masked_mean',
                'normalized': True, 'max_tokens': rank.config['max_tokens'],
                'dimensions': rank.config['dimensions'],
                'foundation': copy.deepcopy(rank.config.get('foundation', {'initialization': 'configured_weights'})),
                'semantics': 'legacy rank-space cosine proximity; retrieval quality is not established by rank training'}

    def invalidate_fingerprint(self):
        """Required after unsupported .data writes to model tensors."""
        self._fingerprint.invalidate()

    @_model_locked
    def _embed(self, text):
        _text(text, 'text')
        dedicated = getattr(self.investigator, 'episodic_encoder', None)
        if dedicated is not None:
            return dedicated.receipt([text])['embeddings'][0]
        rank = self.investigator.rank
        modes = [(module, module.training) for module in rank.modules()]
        try:
            rank.eval()
            with torch.no_grad():
                if rank.tokenizer is None:
                    encoded = rank.encode(rank.tokens(text))
                else:
                    tokens = rank.tokenizer([text], padding=True, truncation=True,
                                            max_length=rank.config['max_tokens'], return_tensors='pt')
                    device = next(rank.encode.parameters()).device
                    tokens = {key: value.to(device) for key, value in tokens.items()}
                    encoded = rank.projection(rank.encode(tokens))[0][tokens['attention_mask'][0].bool()]
                return encoded.mean(0).detach().cpu().tolist()
        finally:
            for module, training in modes:
                module.training = training

    @_model_locked
    def remember(self, evidence, *, episode_id, question='', outcome=''):
        fingerprint = self.fingerprint
        if fingerprint != self.memory.model_fingerprint:
            raise ValueError('stale embedding index; rebuild required')
        self.memory.insert(evidence, self._embed(evidence.text), episode_id=episode_id,
                           question=question, outcome=outcome, model_fingerprint=fingerprint)
        ids = set(self.memory._entries)
        self._evidence = {key: value for key, value in self._evidence.items() if key in ids}
        self._evidence[evidence.id] = evidence

    @_model_locked
    def retrieve(self, question, *, k=5, exclude_episode_id=None):
        fingerprint = self.fingerprint
        if fingerprint != self.memory.model_fingerprint:
            raise ValueError('stale embedding index; rebuild required')
        return self.memory.query(self._embed(question), model_fingerprint=fingerprint,
                                 k=k, exclude_episode_id=exclude_episode_id)

    @_model_locked
    def rebuild_index(self):
        fingerprint = self.fingerprint
        vectors = {key: self._embed(value.text) for key, value in self._evidence.items()}
        self.memory.rebuild_index(vectors, model_fingerprint=fingerprint)

    @_model_locked
    def fork(self):
        result = LearnedEpisodicMemory.__new__(LearnedEpisodicMemory)
        result.investigator = self.investigator
        result._fingerprint = self._fingerprint
        result.memory = EpisodicMemory(capacity=self.memory.capacity, model_fingerprint=self.memory.model_fingerprint)
        result.memory._entries = dict(self.memory._entries)
        result.memory._vectors = dict(self.memory._vectors)
        result._evidence = dict(self._evidence)
        return result

    @_model_locked
    def remove(self, evidence_id):
        self.memory.remove(evidence_id)
        del self._evidence[evidence_id]

    @_model_locked
    def snapshot(self):
        return {'schema_version': 1, 'capacity': self.memory.capacity,
                'records': [{'evidence': asdict(entry.evidence), 'episode_id': entry.episode_id,
                             'question': entry.question, 'outcome': entry.outcome}
                            for entry in self.memory._entries.values()]}

    @classmethod
    def from_snapshot(cls, snapshot, *, investigator):
        if not isinstance(snapshot, dict) or set(snapshot) != {'schema_version','capacity','records'} or type(snapshot['schema_version']) is not int or snapshot['schema_version'] != 1:
            raise ValueError('invalid episodic snapshot schema')
        if not isinstance(snapshot['records'], list):
            raise ValueError('episodic records must be an array')
        result = cls(investigator, capacity=snapshot['capacity'])
        if len(snapshot['records']) > result.memory.capacity:
            raise ValueError('episodic snapshot exceeds capacity')
        seen = set()
        for row in snapshot['records']:
            if not isinstance(row, dict) or set(row) != {'evidence','episode_id','question','outcome'} or not isinstance(row['evidence'], dict) or set(row['evidence']) != {'id','text','source_id'}:
                raise ValueError('invalid episodic record schema')
            evidence = Evidence(**row['evidence'])
            if evidence.id in seen:
                raise ValueError('duplicate episodic evidence id')
            seen.add(evidence.id)
            result.remember(evidence, episode_id=row['episode_id'], question=row['question'], outcome=row['outcome'])
        return result


class CognitiveSession:
    """Persistent source evidence, revisable hypotheses and explicit abstention.

    Session snapshots contain data and policy only. Pass the Investigator model
    separately on restore. No generated hypothesis is inserted as evidence or an
    actual observation. Methods commit state only after successful validation.
    """
    def __init__(self, investigator, *, state=None, policy=None, memory=None):
        self.investigator = investigator
        _ensure_lock(investigator)
        self._state = state if state is not None else CognitiveState()
        if not isinstance(self._state, CognitiveState):
            raise ValueError('state must be CognitiveState')
        if isinstance(policy, dict):
            if set(policy) - {'min_support', 'max_contradiction', 'max_unknown'}:
                raise ValueError('invalid policy configuration')
            policy = SelectionPolicy(**policy)
        self.policy = policy if policy is not None else SelectionPolicy()
        if not isinstance(self.policy, SelectionPolicy):
            raise ValueError('policy must be SelectionPolicy')
        self.retrieval_k = 5
        if isinstance(memory, dict):
            if set(memory) - {'capacity', 'top_k'}:
                raise ValueError('invalid episodic memory configuration')
            self.retrieval_k = memory.get('top_k', 5)
            if type(self.retrieval_k) is not int or self.retrieval_k < 0:
                raise ValueError('top_k must be nonnegative')
            memory = LearnedEpisodicMemory(investigator, capacity=memory.get('capacity', 256))
        if memory is not None and (not isinstance(memory, LearnedEpisodicMemory) or memory.investigator is not investigator):
            raise ValueError('memory must use this session Investigator')
        self.memory = memory
        self._inactive_evidence = set()
        self._episode = 0
        self._active_evidence = {e.id: e.id for e in self._state.evidence}
        self._evidence_lineage = {e.id: (e.id,) for e in self._state.evidence}
        # Every session owns separate evidence but shares this immutable-model
        # content cache. Tensor versions/configuration still invalidate it.
        self._fingerprint = investigator._cognition_fingerprint
        self._observed_model = self._state.assessments[-1].model_provenance if self._state.assessments else None

    def _model_identity(self):
        return self._fingerprint([('investigator', self.investigator)], self.investigator.configuration())

    def invalidate_fingerprint(self):
        self._fingerprint.invalidate()
        if self.memory is not None:
            self.memory.invalidate_fingerprint()

    @property
    @_model_locked
    def state(self):
        if self._observed_model is not None:
            current = self._model_identity()
            if current != self._observed_model:
                self._state = replace(self._state, revision=self._state.revision + 1)
                self._observed_model = current
        return self._state

    @state.setter
    def state(self, value):
        if not isinstance(value, CognitiveState):
            raise ValueError('state must be CognitiveState')
        self._state = value

    @_model_locked
    def fork(self, *, copy_memory=True):
        memory = self.memory.fork() if copy_memory and self.memory is not None else self.memory
        result = CognitiveSession(self.investigator, state=self.state, policy=self.policy, memory=memory)
        result._active_evidence = dict(self._active_evidence)
        result._evidence_lineage = dict(self._evidence_lineage)
        result._inactive_evidence = set(self._inactive_evidence)
        result.retrieval_k = self.retrieval_k
        result._fingerprint = self._fingerprint
        result._observed_model = self._observed_model
        result._episode = self._episode
        return result

    @property
    def episode_id(self):
        return f'episode-{self._episode}'

    @_model_locked
    def new_episode(self):
        self.state = replace(self.state, revision=self.state.revision + 1,
                             hypotheses=(), assessments=(), selection=(), selection_revision=None,
                             goals=(), plans=())
        self._active_evidence = {}
        self._episode += 1
        return self.state

    @_model_locked
    def ingest(self, evidence):
        records = tuple(evidence)
        updated = self.state.add_evidence(records)
        active = dict(self._active_evidence)
        lineage = dict(self._evidence_lineage)
        versions = {version for history in lineage.values() for version in history}
        for record in records:
            if record.id in self._inactive_evidence:
                raise ValueError('archived evidence cannot be reactivated implicitly')
            if record.id not in active:
                if record.id in versions and record.id not in lineage:
                    raise ValueError('revision record cannot become a separate logical source')
                active[record.id] = record.id
                lineage.setdefault(record.id, (record.id,))
        self.state, self._active_evidence = updated, active
        self._evidence_lineage = lineage
        return self.state

    def _resolve_logical_evidence(self, evidence_id):
        """Resolve explicit lineage; never infer relationships from user ID syntax."""
        _text(evidence_id, 'evidence_id')
        lineage = self._evidence_lineage.get(evidence_id)
        if lineage is not None:
            latest = lineage[-1]
            if latest in self._inactive_evidence:
                raise ValueError('logical evidence has been removed')
            source = next(e for e in self.state.evidence if e.id == latest)
            return source, lineage
        if any(evidence_id in history for history in self._evidence_lineage.values()):
            raise ValueError('revision requires the logical evidence id, not a version')
        if evidence_id in self._inactive_evidence:
            raise ValueError('archived evidence cannot be revised')
        source = next((e for e in self.state.evidence if e.id == evidence_id), None)
        entry = self.memory.memory._entries.get(evidence_id) if self.memory is not None else None
        if source is not None and entry is not None and source != entry.evidence:
            raise ValueError('evidence id conflicts with remembered content')
        if source is None and entry is not None:
            source = entry.evidence
        if source is None:
            raise ValueError('unknown logical evidence id')
        return source, (source.id,)

    @_model_locked
    def revise_evidence(self, evidence_id, text, source_id=None):
        old, history = self._resolve_logical_evidence(evidence_id)
        current = self.state
        new_id = f'{evidence_id}@{current.revision + 1}'
        if any(e.id == new_id for e in current.evidence) or (self.memory is not None and new_id in self.memory.memory._entries):
            raise ValueError('revision id conflict')
        replacement = Evidence(new_id, text, old.source_id if source_id is None else source_id)
        updated = current.add_evidence([old, replacement])
        memory = self.memory
        if memory is not None and old.id in memory.memory._entries:
            entry = memory.memory._entries[old.id]
            if entry.evidence != old:
                raise ValueError('evidence id conflicts with remembered content')
            memory = memory.fork()
            memory.remove(old.id)
            # Prior outcome feedback described old content, not this correction.
            memory.remember(replacement, episode_id=entry.episode_id,
                            question=entry.question, outcome='')
        active = dict(self._active_evidence); active[evidence_id] = replacement.id
        lineage = dict(self._evidence_lineage); lineage[evidence_id] = (*history, replacement.id)
        self.state, self._active_evidence = updated, active
        self._evidence_lineage = lineage
        self._inactive_evidence = self._inactive_evidence | {old.id}
        self.memory = memory
        return self.state

    @_model_locked
    def remove_evidence(self, evidence_id):
        old, history = self._resolve_logical_evidence(evidence_id)
        updated = self.state.add_evidence([old])
        updated = replace(updated, revision=updated.revision + 1)
        active = dict(self._active_evidence); active.pop(evidence_id, None)
        lineage = dict(self._evidence_lineage); lineage[evidence_id] = history
        self.state, self._active_evidence = updated, active
        self._evidence_lineage = lineage
        self._inactive_evidence = self._inactive_evidence | {old.id}
        return self.state

    @property
    def active_evidence(self):
        by_id = {e.id: e for e in self.state.evidence}
        return tuple(by_id[key] for key in self._active_evidence.values())

    @_model_locked
    def investigate(self, question, *, hypotheses=None, count=3, episode_id=None):
        _text(question, 'question')
        evidence = self.active_evidence
        hits = ()
        working = self.state
        if self.memory is not None:
            # Request the bounded index before filtering archived/active sources,
            # so obsolete nearest neighbors cannot crowd out eligible sources.
            retrieved = self.memory.retrieve(question, k=self.memory.memory.capacity, exclude_episode_id=self.episode_id if episode_id is None else episode_id)
            active_ids = {e.id for e in evidence}
            hits = tuple(hit for hit in retrieved if hit.evidence.id not in self._inactive_evidence and hit.evidence.id not in active_ids)[:self.retrieval_k]
            working = working.add_evidence([hit.evidence for hit in hits])
            evidence = evidence + tuple(hit.evidence for hit in hits)
        # Investigator requires unique source IDs. Immutable evidence IDs provide
        # that identity even when several passages cite the same document.
        inputs = {'question': question, 'evidence': [{'source_id': e.id, 'text': e.text} for e in evidence]}
        if hypotheses is not None:
            inputs['hypotheses'] = copy.deepcopy(hypotheses)
        modes = [(module, module.training) for module in self.investigator.rank.modules()]
        try:
            self.investigator.rank.eval()
            self.investigator.rank.clear_encoding_cache()
            with torch.no_grad():
                result = self.investigator.investigate(inputs, count=count)
        finally:
            for module, training in modes:
                module.training = training
        # Verification may invalidate stale calibration buffers during inference.
        provenance = self._model_identity()
        proposals = [Hypothesis(candidate['id'], candidate['text'],
                                origin='generated' if hypotheses is None else 'supplied',
                                model_provenance=provenance if hypotheses is None else 'caller-supplied')
                     for candidate in result['candidates']]
        updated = working.add_hypotheses(proposals)
        assessments, accepted = [], []
        by_id = {e.id: e for e in evidence}
        candidates = copy.deepcopy(result['candidates'])
        for candidate in candidates:
            checks = candidate['verifications']
            if len(checks) != len(evidence) or {check['source_id'] for check in checks} != set(by_id):
                raise ValueError('verification sources must match active evidence exactly')
            candidate['accepted_by_policy'] = self.policy.accepts_verification(
                candidate, list(by_id), scope=self.investigator.config['verification_scope'])
            for check in checks:
                evidence_id = check['source_id']
                distribution = check['distribution']
                assessments.append(Assessment(evidence_id, candidate['id'], distribution, provenance))
                check['evidence_id'] = evidence_id
                check['source_id'] = by_id[evidence_id].source_id
            truncated = any(check.get('input_truncated', False) for check in checks)
            joint = candidate.get('joint_verification')
            if joint is not None:
                truncated = truncated or joint.get('input_truncated', False)
                joint['evidence_ids'] = list(joint['source_ids'])
                joint['source_ids'] = [by_id[evidence_id].source_id for evidence_id in joint['evidence_ids']]
            candidate['verification_coverage'] = ('truncated; abstention required' if truncated else
                'complete joint premise and supplied pairs' if joint is not None else 'complete supplied pairs')
            candidate['epistemic_status'] = 'hypothesis'
            score = candidate['predicted_score']
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError('candidate rank scores must be finite')
            if candidate['accepted_by_policy']:
                accepted.append(candidate)
        updated = updated.assess(assessments)
        selected = max(accepted, key=lambda candidate: candidate['predicted_score'])['id'] if accepted else None
        updated = updated.select([selected] if selected is not None else [])
        receipt = {'question': question, 'selected_id': selected, 'abstained': selected is None,
                   'candidates': candidates, 'evidence': [asdict(e) for e in evidence],
                   'retrieval': [asdict(hit) for hit in hits],
                   'retrieval_encoder': self.memory.metadata if self.memory is not None else None,
                   'state_revision': updated.revision, 'policy': self.policy.receipt(scope=self.investigator.config['verification_scope']),
                   'verification_scope': self.investigator.config['verification_scope'],
                   'model_provenance': provenance,
                   'semantics': 'Generated and supplied candidates remain hypotheses; selection is authored screening of model scores, not truth.'}
        lineage = dict(self._evidence_lineage)
        known_versions = {version for history in lineage.values() for version in history}
        for hit in hits:
            if hit.evidence.id not in known_versions:
                lineage[hit.evidence.id] = (hit.evidence.id,)
        self.state = updated
        self._evidence_lineage = lineage
        self._observed_model = provenance
        return receipt

    @_model_locked
    def remember(self, evidence_id, *, episode_id=None, question='', outcome=''):
        if self.memory is None:
            raise ValueError('episodic memory is not configured')
        resolved = self._evidence_lineage[evidence_id][-1] if evidence_id in self._evidence_lineage else evidence_id
        if resolved in self._inactive_evidence:
            raise ValueError('archived evidence cannot be remembered')
        source = next((e for e in self.state.evidence if e.id == resolved), None)
        if source is None:
            raise ValueError('unknown evidence id')
        return self.memory.remember(source, episode_id=self.episode_id if episode_id is None else episode_id, question=question, outcome=outcome)

    @_model_locked
    def retrieve(self, question, *, k=5, exclude_episode_id=None):
        if self.memory is None:
            raise ValueError('episodic memory is not configured')
        if type(k) is not int or k < 0:
            raise ValueError('k must be nonnegative')
        hits = self.memory.retrieve(question, k=self.memory.memory.capacity,
                                    exclude_episode_id=exclude_episode_id)
        return tuple(hit for hit in hits if hit.evidence.id not in self._inactive_evidence)[:k]

    @_model_locked
    def snapshot(self):
        return {'schema_version': 2, 'state': self.state.to_dict(),
                'evidence_lineage': {key: list(history) for key, history in self._evidence_lineage.items()},
                'active_evidence': dict(self._active_evidence), 'policy': asdict(self.policy),
                'inactive_evidence': sorted(self._inactive_evidence), 'retrieval_k': self.retrieval_k,
                'memory': self.memory.snapshot() if self.memory is not None else None,
                'model_provenance': self._observed_model, 'episode': self._episode}

    @classmethod
    def from_snapshot(cls, snapshot, *, investigator, memory=None):
        if not isinstance(snapshot, dict) or set(snapshot) != {'schema_version','state','active_evidence','evidence_lineage','policy','inactive_evidence','retrieval_k','memory','model_provenance','episode'} or type(snapshot['schema_version']) is not int or snapshot['schema_version'] != 2:
            raise ValueError('unsupported cognitive session schema')
        policy = snapshot['policy']
        if not isinstance(policy, dict) or set(policy) != {'min_support','max_contradiction','max_unknown'}:
            raise ValueError('invalid policy schema')
        state = CognitiveState.from_dict(snapshot['state'])
        active = snapshot['active_evidence']
        ids = {e.id for e in state.evidence}
        if not isinstance(active, dict) or any(not isinstance(k,str) or not k or k not in ids or not isinstance(v,str) or v not in ids for k,v in active.items()) or len(set(active.values())) != len(active):
            raise ValueError('invalid active evidence references')
        inactive = snapshot['inactive_evidence']
        if not isinstance(inactive, list) or any(not isinstance(i,str) or i not in ids for i in inactive) or len(set(inactive)) != len(inactive) or set(inactive) & set(active.values()):
            raise ValueError('invalid archived evidence references')
        lineage = snapshot['evidence_lineage']
        if not isinstance(lineage, dict):
            raise ValueError('invalid evidence lineage')
        seen = set()
        for root, history in lineage.items():
            if (not isinstance(root, str) or root not in ids or not isinstance(history, list)
                    or not history or history[0] != root
                    or any(not isinstance(version, str) or version not in ids for version in history)
                    or len(set(history)) != len(history) or seen.intersection(history)
                    or any(version not in inactive for version in history[:-1])):
                raise ValueError('invalid evidence lineage')
            seen.update(history)
        if seen != ids:
            raise ValueError('evidence lineage must cover every historical record')
        if any(root not in lineage or lineage[root][-1] != latest for root, latest in active.items()):
            raise ValueError('active evidence conflicts with lineage')
        retrieval_k = snapshot['retrieval_k']
        if type(retrieval_k) is not int or retrieval_k < 0:
            raise ValueError('invalid retrieval limit')
        if snapshot['memory'] is not None:
            if memory is not None:
                raise ValueError('snapshot already supplies episodic memory')
            memory = LearnedEpisodicMemory.from_snapshot(snapshot['memory'], investigator=investigator)
        result = cls(investigator, state=state, policy=SelectionPolicy(**policy), memory=memory)
        if result.memory is not None:
            historical = {e.id: e for e in state.evidence}
            for entry in result.memory.memory._entries.values():
                if entry.evidence.id in historical and entry.evidence != historical[entry.evidence.id]:
                    raise ValueError('remembered evidence conflicts with immutable session history')
        result._active_evidence = dict(active)
        result._evidence_lineage = {root: tuple(history) for root, history in lineage.items()}
        result._inactive_evidence = set(inactive)
        result.retrieval_k = retrieval_k
        provenance = snapshot['model_provenance']
        if provenance is not None and (not isinstance(provenance,str) or not provenance):
            raise ValueError('invalid model provenance')
        if state.assessments and provenance is None:
            raise ValueError('assessed session requires model provenance')
        if type(snapshot['episode']) is not int or snapshot['episode'] < 0:
            raise ValueError('invalid episode counter')
        result._episode = snapshot['episode']
        result._observed_model = provenance
        # Trigger weight compatibility invalidation before exposing restored state.
        result.state
        return result

    def save(self, path):
        Path(path).write_text(json.dumps(self.snapshot(), allow_nan=False, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path, *, investigator, memory=None):
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError('duplicate JSON key')
                result[key] = value
            return result
        return cls.from_snapshot(json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs),
                                 investigator=investigator, memory=memory)
