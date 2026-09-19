"""Reused real-input audit of pending expansion before supplied hypothesis testing.

Run ``python -m eval.parsing.evaluate_pending_investigation`` after source freeze.
Empty predictions are an authored diagnostic, not inferred grounding hypotheses.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from eval.parsing.evaluate_continuation import candidate_record, forbid_decoding, sha
from eval.parsing.span_evaluation import load_records

BASIS = 'No inferred grounding model; candidate remains unmodeled'
POLICY = {'max_expansions': 64, 'max_candidates': 4, 'max_probes': 2}


def audit_input(record, reader):
    from tensorcode.agent import Agent
    from tensorcode.agent.investigation import CandidateHypothesis
    agent = Agent([], reader=reader)
    interpreted = agent.interpret(record.text)
    source = agent.interpretations.get_source(interpreted.source_id)
    groups = []
    for group_id in interpreted.group_ids:
        initial = agent.interpretations.get(group_id)
        old = deepcopy(initial.candidates)
        initial_status = agent.interpretations.continuation_status(group_id)
        factory_calls = []
        def factory(group):
            factory_calls.append({'candidate_ids': [c.id for c in group.candidates],
                                  'revision': group.revision})
            return tuple(CandidateHypothesis(c.id, (), (BASIS,)) for c in group.candidates)
        before = time.perf_counter()
        with forbid_decoding(reader) as calls:
            resolved = agent.resolve_interpretation(group_id, factory, **POLICY)
        elapsed = (time.perf_counter() - before) * 1000
        current = agent.interpretations.get(group_id)
        expansion = resolved.expansion
        result = resolved.investigation.result
        candidate_ids = [c.id for c in current.candidates]
        assessment_ids = [a.candidate_id for a in result.assessments]
        row = {'initial_candidates': [candidate_record(c, source.text) for c in initial.candidates],
               'initial_pending': initial_status.pending,
               'factory_calls': factory_calls,
               'expansion': None if expansion is None else {
                   'candidate_ids': expansion.candidate_ids, 'explored': expansion.explored,
                   'pending': expansion.pending},
               'final_candidates': [candidate_record(c, source.text) for c in current.candidates],
               'assessment_ids': assessment_ids,
               'investigation_reason': result.reason,
               'decision_reason': resolved.investigation.decision.reason,
               'selected_id': current.selected_id, 'decision_id': resolved.investigation.decision.candidate_id,
               'observations': len(result.observations), 'latency_ms': elapsed,
               'decoder_calls_during_resolution': dict(calls),
               'checks': {
                   'expanded_before_factory': (expansion is not None and bool(expansion.candidate_ids)
                       and factory_calls == [{'candidate_ids': [c.id for c in expansion.group.candidates],
                                              'revision': expansion.group.revision}]),
                   'factory_exact_coverage': len(factory_calls) == 1 and factory_calls[0]['candidate_ids'] == candidate_ids,
                   'assessment_exact_coverage': assessment_ids == candidate_ids,
                   'old_candidates_unchanged': current.candidates[:len(old)] == old,
                   'within_budgets': expansion is not None and expansion.explored <= POLICY['max_expansions']
                       and len(expansion.candidate_ids) <= POLICY['max_candidates'],
                   'no_selection': current.selected_id is None and resolved.investigation.decision.candidate_id is None,
                   'no_observation_without_predictions': not result.observations,
                   'no_decoder_replay': not any(calls.values())}}
        row['checks']['all_source_anchors_valid'] = not any(c['anchor_errors'] for c in row['final_candidates'])
        groups.append(row)
    return {'sent_id': record.sent_id, 'text': record.text, 'groups': groups,
            'source_text_unchanged': source.text == record.text,
            'tasks_created': len(agent.tasks), 'plugins': len(agent.plugins),
            'reader_unavailable': str(interpreted.unavailable) if interpreted.unavailable else None}


def main():
    from tensorcode.agent.understand import LearnedReader
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cohort', type=Path, default=Path('eval/results/continuation_cohort.json'))
    ap.add_argument('--treebank', type=Path, default=Path.home() / '.cache/tensorcode/seeds/UD_English-EWT/en_ewt-ud-test.conllu')
    ap.add_argument('--model', type=Path, default=Path.home() / '.cache/tensorcode/models/ud_ewt_parser.pickle')
    ap.add_argument('--segmentation-model', type=Path, default=Path.home() / '.cache/tensorcode/models/ud_ewt_segmenter.json')
    ap.add_argument('--output', type=Path, default=Path('eval/results/pending_investigation.json'))
    args = ap.parse_args()
    cohort = json.loads(args.cohort.read_text())
    records = {r.sent_id: r for r in load_records(args.treebank)}
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__).resolve(), root / 'eval/parsing/evaluate_continuation.py',
             root / 'eval/parsing/span_evaluation.py', *[
                 root / 'src/tensorcode' / name for name in (
                     'agent/core.py', 'agent/interpretation.py', 'agent/investigation.py', 'agent/understand.py',
                     'language/deps_semantics.py', 'language/learned_parser.py', 'language/segmentation.py',
                     'language/treebank.py', 'language/chart.py')]]
    hashes = {str(p): sha(p) for p in paths}
    inputs = {str(p): sha(p) for p in (args.cohort, args.treebank, args.model, args.segmentation_model)}
    reader = LearnedReader(args.model, segmentation_model_path=args.segmentation_model)
    rows = []
    for sent_id in cohort['sent_ids']:
        try:
            rows.append(audit_input(records[sent_id], reader))
        except Exception as exc:
            rows.append({'sent_id': sent_id, 'error': f'{type(exc).__name__}: {exc}'})
    report = {'evaluation': 'pending_interpretation_expansion_before_supplied_hypotheses',
              'cohort': cohort, 'selection': 'Exact reuse of doc53 three-input cohort; no new selection',
              'policy': POLICY, 'hypothesis_basis': BASIS,
              'source_hashes': hashes, 'input_hashes': inputs,
              'source_hashes_verified_after_run': {p: sha(p) == h for p, h in hashes.items()},
              'input_hashes_verified_after_run': {p: sha(p) == h for p, h in inputs.items()},
              'sentences': rows,
              'limitations': ['Authored empty predictions test scheduling and coverage, not hypothesis formation.',
                              'No grounding, semantic accuracy, execution, or generality claim.',
                              'Three previously exposed diagnostic inputs; no holdout claim.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
