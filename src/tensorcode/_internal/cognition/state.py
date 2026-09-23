"""Bounded, immutable session data. Records do not infer facts or execute policies."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import json
from pathlib import Path


from ...tools.cognition import Evidence, Hypothesis, Assessment, Goal, Plan, Observation


_RECORD_TYPES = {'evidence': Evidence, 'hypotheses': Hypothesis,
                 'assessments': Assessment, 'goals': Goal, 'plans': Plan,
                 'observations': Observation}


@dataclass(frozen=True)
class CognitiveState:
    max_records: int = 256
    revision: int = 0
    evidence: tuple[Evidence, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    assessments: tuple[Assessment, ...] = ()
    goals: tuple[Goal, ...] = ()
    plans: tuple[Plan, ...] = ()
    observations: tuple[Observation, ...] = ()
    selection: tuple[str, ...] = ()
    selection_revision: int | None = None

    def __post_init__(self):
        if type(self.max_records) is not int or self.max_records < 1:
            raise ValueError('max_records must be positive')
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError('invalid state revision')
        for name, kind in _RECORD_TYPES.items():
            records = getattr(self, name)
            if not isinstance(records, (tuple, list)) or any(type(r) is not kind for r in records):
                raise ValueError(f'invalid {name} records')
            object.__setattr__(self, name, tuple(records))
            if name != 'assessments' and len({r.id for r in records}) != len(records):
                raise ValueError(f'duplicate {name} ids')
        if sum(len(getattr(self, name)) for name in _RECORD_TYPES) > self.max_records:
            raise ValueError('state capacity exceeded')
        evidence_ids = {r.id for r in self.evidence}
        hypothesis_ids = {r.id for r in self.hypotheses}
        for a in self.assessments:
            if a.evidence_id not in evidence_ids or a.hypothesis_id not in hypothesis_ids:
                raise ValueError('assessment refers to absent record')
            if a.revision > self.revision:
                raise ValueError('assessment revision is in the future')
        if not isinstance(self.selection, (tuple, list)) or any(not isinstance(i, str) or i not in hypothesis_ids for i in self.selection):
            raise ValueError('selection refers to absent hypothesis')
        object.__setattr__(self, 'selection', tuple(self.selection))
        if self.selection_revision is not None and (type(self.selection_revision) is not int or not 0 <= self.selection_revision <= self.revision):
            raise ValueError('invalid selection revision')
        if self.selection and self.selection_revision is None:
            raise ValueError('selection needs revision')

    def _add(self, name, records, *, revisable=False):
        items = dict((r.id, r) for r in getattr(self, name))
        batch = tuple(records)
        if any(type(r) is not _RECORD_TYPES[name] for r in batch):
            raise ValueError(f'invalid {name} record')
        if len({r.id for r in batch}) != len(batch):
            raise ValueError('duplicate ids in batch')
        for record in batch:
            if record.id in items and items[record.id] != record and not revisable:
                raise ValueError('immutable record id conflict')
            items[record.id] = record
        updated = tuple(items.values())
        if updated == getattr(self, name):
            return self
        return replace(self, **{name: updated, 'revision': self.revision + 1})

    def add_evidence(self, records):
        return self._add('evidence', records)

    def add_hypotheses(self, records):
        return self._add('hypotheses', records, revisable=True)

    def add_goals(self, records):
        return self._add('goals', records, revisable=True)

    def add_plans(self, records):
        return self._add('plans', records, revisable=True)

    def observe(self, records):
        return self._add('observations', records)

    def assess(self, records):
        batch = tuple(records)
        if any(type(r) is not Assessment for r in batch):
            raise ValueError('invalid assessment')
        stamped = tuple(replace(r, revision=self.revision + 1) for r in batch)
        return replace(self, revision=self.revision + 1, assessments=self.assessments + stamped,
                       selection_revision=None, selection=())

    def select(self, hypothesis_ids):
        return replace(self, selection=tuple(hypothesis_ids), selection_revision=self.revision)

    def is_stale(self, assessment):
        return assessment.revision != self.revision

    @property
    def selection_stale(self):
        return self.selection_revision != self.revision

    def to_dict(self):
        result = {'schema_version': 1, 'max_records': self.max_records,
                  'revision': self.revision, 'selection': list(self.selection),
                  'selection_revision': self.selection_revision}
        for name in _RECORD_TYPES:
            result[name] = []
            for record in getattr(self, name):
                data = {f.name: getattr(record, f.name) for f in fields(record)}
                if isinstance(record, Assessment):
                    data['scores'] = dict(record.scores)
                if isinstance(record, Plan):
                    data['steps'] = list(record.steps)
                    data['predicted_outcomes'] = list(record.predicted_outcomes)
                result[name].append(data)
        return result

    @classmethod
    def from_dict(cls, data):
        expected = {f.name for f in fields(cls)} | {'schema_version'}
        if not isinstance(data, dict) or set(data) != expected or type(data['schema_version']) is not int or data['schema_version'] != 1:
            raise ValueError('unsupported session schema')
        values = {k: v for k, v in data.items() if k != 'schema_version'}
        for name, kind in _RECORD_TYPES.items():
            if not isinstance(values[name], list):
                raise ValueError(f'{name} must be an array')
            records = []
            for row in values[name]:
                if not isinstance(row, dict) or set(row) != {f.name for f in fields(kind)}:
                    raise ValueError(f'invalid {name} schema')
                records.append(kind(**row))
            values[name] = tuple(records)
        return cls(**values)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), allow_nan=False, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path):
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError('duplicate JSON key')
                result[key] = value
            return result
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs))
