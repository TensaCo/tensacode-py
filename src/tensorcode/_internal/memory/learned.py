"""Private model-owned episodic retrieval and persistence."""
from dataclasses import asdict
import copy
import torch
from .episodic import EpisodicMemory
from ..cognition.locking import _ensure_lock, _model_locked, _ModelFingerprint
from ...tools.cognition import Evidence, _text


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
