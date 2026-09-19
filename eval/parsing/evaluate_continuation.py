"""Audit in-memory interpretation continuation on predeclared real source text.

No training, downloads, gold scoring, or meaning selection. Run with
``python -m eval.parsing.evaluate_continuation`` after runtime source freeze.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

from eval.parsing.span_evaluation import load_records, reader_groups


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def forbid_decoding(reader):
    """Guard classes, so copied model objects cannot evade the audit."""
    calls = Counter()
    methods = ((reader.segmenter, ('segment',)),
               (reader.tagger, ('tag', 'tag_candidates', 'greedy_candidate')),
               (reader.parser, ('parse_candidates', 'greedy_search', 'greedy_candidate')))
    with ExitStack() as stack:
        for model, names in methods:
            if model is None:
                continue
            for name in names:
                if not hasattr(type(model), name):
                    continue
                key = f'{type(model).__name__}.{name}'
                calls[key] = 0
                def reject(*args, _key=key, **kwargs):
                    calls[_key] += 1
                    raise AssertionError(f'continuation invoked decoder: {_key}')
                stack.enter_context(patch.object(type(model), name, reject))
        yield calls


def candidate_record(candidate, text):
    payload = candidate.payload
    metadata = payload.metadata
    span = metadata.get('sentence_span')
    errors = []
    if metadata.get('syntax_complete'):
        try:
            raw = text[slice(*span)]
            _, errors = reader_groups(text, [SimpleNamespace(text=raw, alternatives=(payload,))])
        except (TypeError, ValueError):
            errors = ['invalid sentence span']
    else:
        errors = ['candidate has no complete anchored syntax']
    return {'id': candidate.id, 'provenance': candidate.provenance,
            'acts': [act.kind for act in payload.acts],
            'anchor_errors': list(errors),
            'evidence': {key: metadata.get(key) for key in (
                'sentence_span', 'tokens', 'token_anchors', 'heads', 'labels', 'tags',
                'model_artifact', 'segmentation_artifact', 'semantic_choices',
                'semantic_unresolved', 'semantic_frontier', 'continuation_proposal')}}


def audit_input(record, reader, policy):
    from tensorcode.agent import Agent
    agent = Agent([], reader=reader)
    started = time.perf_counter()
    interpreted = agent.interpret(record.text)
    initial_ms = (time.perf_counter() - started) * 1000
    source = agent.interpretations.get_source(interpreted.source_id)
    groups = []
    for group_id in interpreted.group_ids:
        initial = agent.interpretations.get(group_id)
        cursor = agent.interpretations.get_continuation(group_id)
        row = {'initial_candidates': [candidate_record(c, source.text) for c in initial.candidates],
               'initial_pending': cursor.pending if cursor is not None else None,
               'continuation_available': cursor is not None, 'calls': []}
        with forbid_decoding(reader) as decoder_calls:
            for _ in range(policy['expansion_calls_per_group']):
                previous = agent.interpretations.get(group_id)
                old = deepcopy(previous.candidates)
                before = time.perf_counter()
                batch = agent.expand_interpretation(group_id,
                    max_expansions=policy['max_expansions_per_call'],
                    max_candidates=policy['max_candidates_per_call'])
                elapsed = (time.perf_counter() - before) * 1000
                current = batch.group
                appended = [c for c in current.candidates if c.id in batch.candidate_ids]
                row['calls'].append({'latency_ms': elapsed, 'explored': batch.explored,
                    'pending': batch.pending, 'candidate_ids': batch.candidate_ids,
                    'new_candidates': [candidate_record(c, source.text) for c in appended],
                    'old_candidates_unchanged': current.candidates[:len(old)] == old,
                    'selected_id': current.selected_id, 'revision': current.revision,
                    'within_budgets': (batch.explored <= policy['max_expansions_per_call']
                                       and len(appended) <= policy['max_candidates_per_call'])})
            row['decoder_calls_during_expansion'] = dict(decoder_calls)
        groups.append(row)
    return {'sent_id': record.sent_id, 'text': record.text, 'initial_ms': initial_ms,
            'source_text_unchanged': source.text == record.text,
            'groups': groups, 'tasks_created': len(agent.tasks),
            'reader_unavailable': str(interpreted.unavailable) if interpreted.unavailable else None}


def main():
    from tensorcode.agent.understand import LearnedReader
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cohort', type=Path, default=Path('eval/results/continuation_cohort.json'))
    ap.add_argument('--treebank', type=Path, default=Path.home() / '.cache/tensorcode/seeds/UD_English-EWT/en_ewt-ud-test.conllu')
    ap.add_argument('--model', type=Path, default=Path.home() / '.cache/tensorcode/models/ud_ewt_parser.pickle')
    ap.add_argument('--segmentation-model', type=Path, default=Path.home() / '.cache/tensorcode/models/ud_ewt_segmenter.json')
    ap.add_argument('--output', type=Path, default=Path('eval/results/interpretation_continuation.json'))
    args = ap.parse_args()
    policy = json.loads(args.cohort.read_text())
    records = {r.sent_id: r for r in load_records(args.treebank)}
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__).resolve(), root / 'eval/parsing/span_evaluation.py', *[
        root / 'src/tensorcode' / name for name in (
            'agent/core.py', 'agent/interpretation.py', 'agent/understand.py',
            'language/deps_semantics.py', 'language/learned_parser.py',
            'language/segmentation.py', 'language/treebank.py', 'language/chart.py')]]
    hashes = {str(p): sha(p) for p in files}
    inputs = {str(p): sha(p) for p in (args.cohort, args.treebank, args.model, args.segmentation_model)}
    reader = LearnedReader(args.model, segmentation_model_path=args.segmentation_model)
    rows = []
    for sent_id in policy['sent_ids']:
        try:
            rows.append(audit_input(records[sent_id], reader, policy))
        except Exception as exc:
            rows.append({'sent_id': sent_id, 'error': f'{type(exc).__name__}: {exc}'})
    result = {'evaluation': 'in_memory_workspace_continuation_mechanism',
              'policy': policy, 'source_hashes': hashes, 'input_hashes': inputs,
              'initial_reader_budgets': {key: getattr(reader, key) for key in (
                  'tag_beam_width', 'tag_max_candidates', 'parse_beam_width', 'parse_max_candidates',
                  'parse_ranking', 'max_expansions', 'max_sentence_expansions', 'max_alternatives',
                  'semantic_max_candidates', 'semantic_max_expansions', 'max_sentence_semantic_expansions',
                  'segmentation_beam_width', 'segmentation_max_candidates', 'segmentation_max_expansions')},
              'source_hashes_verified_after_run': {p: sha(p) == h for p, h in hashes.items()},
              'input_hashes_verified_after_run': {p: sha(p) == h for p, h in inputs.items()},
              'sentences': rows,
              'limitations': ['Three reused short diagnostic inputs, no quality scoring.',
                              'No selected meaning, execution, restart persistence, or generality claim.',
                              'Authored fair scheduler; only already generated syntax can continue.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
