"""Reproduce one empirical applicability probe in an authored hidden-wiring world.

Training contexts, feature projection, candidate bindings, and available actions
are supplied. Actual executed transitions determine the fitted outcome forecasts.
Run after source freeze: python -m eval.learning.evaluate_model_investigation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields, is_dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

from eval.learning.model_investigation_fixture import bindings, make_setup, make_noisy_setup


def encode(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {field.name: encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((encode(v) for v in value), key=lambda item: json.dumps(item, sort_keys=True))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'unsupported audit value {type(value).__name__}')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_case(*, noisy=False):
    started = time.perf_counter()
    setup = make_noisy_setup(mode=2) if noisy else make_setup(mode=2, tie=False)
    training_ms = (time.perf_counter() - started) * 1000
    agent, plugin = setup.agent, setup.plugin
    group_before = agent.interpretations.get(setup.group_id)
    snapshots = tuple(model.snapshot() for model in setup.models)
    initial_executions = len(plugin.executions)
    proposal = agent.propose_experience_investigation(
        setup.group_id, bindings(setup), setup.before_source_id, setup.calls)
    executions_after_proposal = len(plugin.executions)
    retained_forecast = agent.interpretations.get_source(proposal.record_source_id)
    source_ids_before_probe = tuple(source.id for source in agent.interpretations.sources())
    before = time.perf_counter()
    result = agent.execute_experience_investigation(proposal.id)
    execution_ms = (time.perf_counter() - before) * 1000
    executions_after_probe = len(plugin.executions)
    retry = agent.execute_experience_investigation(proposal.id)
    source_ids_after_probe = tuple(source.id for source in agent.interpretations.sources())
    models = []
    for model in setup.models:
        examples = model.examples
        split_counts = Counter(example.split for example in examples)
        per_button = Counter((example.split, example.action.arg('button')) for example in examples)
        training_ids = {example.attempt_id for example in examples if example.split == 'training'}
        validation_ids = {example.attempt_id for example in examples if example.split == 'evaluation'}
        models.append({'snapshot': model.snapshot(), 'artifact': model.artifact, 'evaluation': model.evaluation,
                       'validation_policy': model.policy, 'split_counts': dict(split_counts),
                       'per_button_counts': [{'split': split, 'button': button, 'count': count}
                                             for (split, button), count in sorted(per_button.items())],
                       'split_ids_disjoint': training_ids.isdisjoint(validation_ids),
                       'fit_examples': examples, 'rule_evidence': model.evidence})
    checks = {
        'proposal_did_not_execute': initial_executions == executions_after_proposal,
        'forecast_record_existed_before_probe': proposal.record_source_id in source_ids_before_probe,
        'forecast_precedes_fresh_probe_observations': bool(result.source_ids) and all(
            source_ids_after_probe.index(proposal.record_source_id) < source_ids_after_probe.index(source_id)
            for source_id in result.source_ids),
        'one_probe_executed': executions_after_probe == initial_executions + 1,
        'replay_did_not_execute': len(plugin.executions) == executions_after_probe,
        'workspace_group_unchanged': agent.interpretations.get(setup.group_id) == group_before,
        'workspace_unselected': agent.interpretations.get(setup.group_id).selected_id is None,
        'models_unsuspended': tuple(model.snapshot() for model in setup.models) == snapshots,
    }
    report = {'evaluation': 'authored_hidden_wiring_empirical_applicability_probe',
              'design': {'contexts': [1, 2], 'fresh_probe_context': 2,
                         'hidden_context_excluded_from_features': True,
                         'supplied': ['environment', 'context partition', 'projection',
                                      'candidate-model bindings', 'available probe calls'],
                         'learned': 'decision-list transition outcome associations from applied receipts',
                         'selection_policy': 'predefined context-2 probe; deterministic and authored noisy schedule cases',
                         'noisy_case': noisy},
              'models': models, 'candidate_ids': setup.candidate_ids,
              'before_probe_observation': agent.interpretations.get_source(setup.before_source_id),
              'proposal': proposal, 'retained_forecast_record': retained_forecast,
              'result': result, 'replay_result': retry,
              'probe_observation_sources': [agent.interpretations.get_source(sid) for sid in result.source_ids],
              'checks': checks, 'initial_applied_actions_including_resets': initial_executions,
              'setup_and_training_ms': training_ms, 'fresh_probe_ms': execution_ms,
              'limitations': ['Small deterministic authored simulator; no external generalization estimate.',
                              'Separate contexts and interpretation bindings supplied, not inferred.',
                              'World-response model applicability does not identify user intent.',
                              'Validation outcomes contribute empirical forecast support; only fresh probe is separate.',
                              'Validation partitions exclude training attempts but repeat a narrow action distribution.']}
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=Path('eval/results/model_investigation.json'))
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__).resolve(), root / 'eval/learning/model_investigation_fixture.py',
             *[root / 'src/tensorcode' / name for name in (
                 'agent/core.py', 'agent/experience_investigation.py', 'agent/experience_planning.py',
                 'agent/interpretation.py', 'agent/plugin.py', 'learning/experience.py',
                 'learning/induce.py', 'learning/literals.py')]]
    hashes = {str(path): sha(path) for path in paths}
    cases = {'deterministic': audit_case(), 'noisy_overlap': audit_case(noisy=True)}
    report = {'evaluation': 'empirical_outcome_support_investigation',
              'source_hashes': hashes,
              'source_hashes_verified_after_run': {path: sha(path) == value for path, value in hashes.items()},
              'cases': cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(encode(report), indent=2, allow_nan=False) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
