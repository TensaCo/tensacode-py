"""Private independent ranking receipt history."""
import copy


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
                from ..proposals import proposal_prompt
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
