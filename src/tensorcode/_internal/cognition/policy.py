"""Explicit authored selection thresholds."""
from dataclasses import dataclass, asdict
import math


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
