"""Prepare training-only evidence interventions for explicit assistant review.

This script reads only train.jsonl. It creates no labels, trains no model, and
never treats the original full-context judgment as a shuffled-context target.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

AXES = ('support', 'completeness', 'constraints')
ORIGIN = 'authored_source_intervention_on_natural_candidate'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_keys(evidence):
    keys = set()
    for item in evidence:
        if any(not isinstance(item.get(key), str) or not item[key].strip()
               for key in ('id', 'source_id', 'text')):
            raise ValueError('evidence requires original nonempty id/source_id/text')
        keys.add(('source_id', item['source_id']))
        keys.add(('text_sha256', hashlib.sha256(item['text'].encode()).hexdigest()))
    return keys


def interventions(rows, *, count=64):
    """Select first all-known-true variant per question; never transfer labels."""
    if type(count) is not int or count < 2:
        raise ValueError('at least two questions are required for cyclic donors')
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate input candidate IDs')
    selected, seen = [], set()
    for row in rows:
        if row['question_id'] in seen or not all(row.get('targets', {}).get(axis) is True for axis in AXES):
            continue
        if not row['evidence']:
            raise ValueError('known-good original candidate requires source evidence')
        source_keys(row['evidence'])
        selected.append(row)
        seen.add(row['question_id'])
        if len(selected) == count:
            break
    if len(selected) < 2:
        raise ValueError('at least two distinct known-good training questions required')
    used_sources = set()
    for row in selected:
        keys = source_keys(row['evidence'])
        if used_sources & keys:
            raise ValueError('selected donor questions must be source-ID/text disjoint')
        used_sources.update(keys)
    result = []
    for index, original in enumerate(selected):
        donor = selected[(index + 1) % len(selected)]
        for kind in ('source-shuffled', 'evidence-free'):
            shuffled = kind == 'source-shuffled'
            evidence = donor['evidence'] if shuffled else []
            candidate = {
                'id': original['id'] + ':' + kind,
                'question_id': original['question_id'], 'question': original['question'],
                'candidate': original['candidate'], 'evidence': copy.deepcopy(evidence),
                'candidate_origin': ORIGIN, 'original_id': original['id'],
                'donor_question_id': donor['question_id'] if shuffled else None,
                'review_status': 'pending_explicit_review',
                'provenance': {
                    'partition': 'train', 'intervention': kind,
                    'selection': 'first all-three-known-true candidate per question in training input order',
                    'donor_policy': 'next selected training question cyclically' if shuffled else 'remove all evidence',
                    'original_source_keys': sorted(source_keys(original['evidence'])),
                    'intervention_source_keys': sorted(source_keys(evidence)),
                    'original_labels_role': 'selection only; not transferred or inferred for intervened context',
                    'supervision': 'no targets supplied; requires explicit review and label merge before training'}}
            if 'reference_answer' in original:
                candidate['original_reference_answer'] = original['reference_answer']
            result.append(candidate)
    output_ids = [row['id'] for row in result]
    if len(output_ids) != len(set(output_ids)) or set(output_ids) & set(ids):
        raise ValueError('duplicate or colliding intervention candidate IDs')
    return result


def write_jsonl(path, rows):
    with Path(path).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def prepare(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    training = Path(args.data) / 'train.jsonl'
    rows = [json.loads(line) for line in training.read_text().splitlines() if line.strip()]
    candidates = interventions(rows, count=args.count)
    original_lookup = {row['id']: row for row in rows}
    review = []
    for offset in range(0, len(candidates), 2):
        pair = candidates[offset:offset + 2]
        original = original_lookup[pair[0]['original_id']]
        review.append({'question_id': original['question_id'], 'original_id': original['id'],
                       'question': original['question'], 'candidate': original['candidate'],
                       'original_evidence': original['evidence'],
                       'original_reference_answer': original.get('reference_answer'),
                       'reference_role': 'original full-source reference; review metadata, never inference input',
                       'review_status': 'pending_explicit_review', 'interventions': pair})
    manifest = {'format': 'tensorcode.response_quality_training_interventions.v1',
                'input': str(training), 'input_sha256': sha256(training),
                'input_partition': 'train', 'input_candidates': len(rows), 'requested_questions': args.count,
                'selected_questions': len(review), 'candidates': len(candidates),
                'question_ids': [row['question_id'] for row in review],
                'original_ids': [row['original_id'] for row in review],
                'method': 'first eligible distinct training questions; fixed question/candidate; next selected question evidence cyclically and empty evidence',
                'candidate_origin': ORIGIN,
                'source_provenance': [{'id': row['id'], 'original_id': row['original_id'],
                                       'donor_question_id': row['donor_question_id'],
                                       'source_keys': row['provenance']['intervention_source_keys']} for row in candidates],
                'supervision': 'none; every intervened candidate requires explicit review and label merge',
                'excluded_inference_metadata': ['original_reference_answer', 'provenance', 'review_status'],
                'script_sha256': sha256(__file__)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + output.name + '-', dir=output.parent))
    try:
        write_jsonl(temporary / 'candidates.jsonl', candidates)
        write_jsonl(temporary / 'review.jsonl', review)
        manifest['files'] = {name: sha256(temporary / name) for name in ('candidates.jsonl', 'review.jsonl')}
        (temporary / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        if output.exists():
            raise FileExistsError(output)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(json.dumps({'output': str(output), 'selected_questions': len(review),
                      'candidates': len(candidates), 'review_status': 'pending_explicit_review'}))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    preparation = commands.add_parser('prepare')
    preparation.add_argument('--data', required=True)
    preparation.add_argument('--output', required=True)
    preparation.add_argument('--count', type=int, default=64)
    return prepare(parser.parse_args())


if __name__ == '__main__':
    main()
