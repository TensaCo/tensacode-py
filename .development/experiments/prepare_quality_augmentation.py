"""Append explicitly reviewed TRAIN-only controls; retain held-out bytes exactly.

This builder never creates labels, loads models, or changes inference fields.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

_spec = importlib.util.spec_from_file_location('quality_augmentation_data', Path(__file__).resolve().parents[2] / 'examples/train_response_quality.py')
helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helper)


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def read(path):
    return json.loads(Path(path).read_text())


def verify_files(directory, files):
    for name, digest in files.items():
        if Path(name).name != name or helper.sha256(directory / name) != digest:
            raise ValueError(f'pack checksum or local filename mismatch: {name}')


def reviewed(records, labels):
    rows = helper.merge_labels(records, labels)
    for row in rows:
        if all(value is None for value in row['targets'].values()):
            raise ValueError('pending/unreviewed candidate has no supervised axis')
        row['review_status'] = 'explicit_labels_joined_for_training'
    return rows


def evidence_additions(directory, train_hash, training, base_manifest):
    manifest = read(directory / 'manifest.json')
    receipt = read(directory / 'training-manifest.json')
    if receipt.get('format') != 'tensorcode.response_quality_supervision.v1':
        raise ValueError('unsupported evidence review receipt')
    if manifest.get('format') != 'tensorcode.response_quality_training_interventions.v1' or manifest.get('input_partition') != 'train' or manifest.get('input_sha256') != train_hash:
        raise ValueError('evidence pack training provenance differs')
    # The removed pending review template is historical metadata, not supervision.
    verify_files(directory, {'candidates.jsonl': manifest['files']['candidates.jsonl']})
    if receipt.get('selection_manifest') != base_manifest.get('selection_manifest'):
        raise ValueError('evidence review selection provenance differs')
    for name in ('calibration.jsonl', 'development.jsonl'):
        if receipt['files'][name] != base_manifest['files'][name]:
            raise ValueError('evidence review held-out provenance differs')
    label_files = {Path(name).name: digest for name, digest in receipt['inputs']['labels'].items()
                   if Path(name).parent.name == directory.name and Path(name).name.startswith('labels-')
                   and Path(name).suffix == '.jsonl'}
    if not label_files:
        raise ValueError('evidence pack has no recorded reviewed labels')
    verify_files(directory, label_files)
    rows = helper.read_jsonl(directory / 'candidates.jsonl')
    if len(rows) != manifest['candidates']:
        raise ValueError('evidence candidate count differs')
    for row in rows:
        anchor = training.get(row['original_id'])
        if anchor is None or any(row[key] != anchor[key] for key in ('question_id', 'question', 'candidate')):
            raise ValueError('evidence anchor/question/candidate differs from original training row')
        kind = row['provenance']['intervention']
        if kind == 'evidence-free':
            if row['evidence'] or row.get('donor_question_id') is not None:
                raise ValueError('evidence-free variant must have no source or donor')
        elif kind == 'source-shuffled':
            donor = row.get('donor_question_id')
            if donor == row['question_id'] or not any(other['question_id'] == donor and other['evidence'] == row['evidence'] for other in training.values()):
                raise ValueError('swapped evidence must exactly match declared original training donor')
        else:
            raise ValueError('undeclared evidence intervention')
    labels = [label for name in sorted(label_files) for label in helper.read_jsonl(directory / name)]
    result = reviewed(rows, labels)
    for row in result:
        row['augmentation_origin'] = 'reviewed authored training evidence intervention'
    return result, {'manifest_sha256': helper.sha256(directory / 'manifest.json'),
                    'review_receipt_sha256': helper.sha256(directory / 'training-manifest.json'),
                    'candidate_sha256': manifest['files']['candidates.jsonl'], 'label_files_sha256': label_files,
                    'historical_pending_review_template_used': False}


def near_additions(directory, train_hash, training):
    manifest = read(directory / 'manifest.json')
    receipt = read(directory / 'adjudication.json')
    if receipt.get('format') != 'tensorcode.train_near_correct_adjudication.v1':
        raise ValueError('unsupported near-correct adjudication receipt')
    if manifest.get('format') != 'tensorcode.train_only_near_correct_review_pack.v1' or manifest.get('input_partition') != 'train' or manifest.get('source_sha256') != train_hash:
        raise ValueError('near-correct pack training provenance differs')
    verify_files(directory, manifest['files'])
    if receipt.get('frozen_manifest_sha256') != helper.sha256(directory / 'manifest.json'):
        raise ValueError('adjudication frozen manifest mismatch')
    files = receipt.get('review_files_sha256', {})
    if not {'adjudicated.jsonl', 'reviews-first.jsonl', 'reviews-second.jsonl'} <= set(files):
        raise ValueError('adjudication must bind labels and both independent reviews')
    verify_files(directory, files)
    rows = helper.read_jsonl(directory / 'candidates.jsonl')
    authors = helper.read_jsonl(directory / 'authorship.jsonl')
    author_by_id = {row['id']: row for row in authors}
    if len(author_by_id) != len(authors) or set(author_by_id) != {row['id'] for row in rows} or len(rows) != manifest['variants']:
        raise ValueError('authorship IDs/count differ')
    anchors = helper.read_jsonl(directory / 'anchors.jsonl')
    if any(row['id'] not in training or canonical(row) != canonical(training[row['id']]) for row in anchors):
        raise ValueError('near-correct anchors differ from original training rows')
    if set(manifest['question_edit_flags']) != set(author_by_id):
        raise ValueError('question edit flags do not cover exact variants')
    for row in rows:
        author = author_by_id[row['id']]
        anchor = training.get(author['original_id'])
        if anchor is None or author['partition'] != 'train' or row['question_id'] != anchor['question_id'] or author['question_id'] != anchor['question_id']:
            raise ValueError('near-correct anchor must belong to original training')
        if author['original_question'] != anchor['question'] or author['original_candidate'] != anchor['candidate'] or row['evidence'] != anchor['evidence']:
            raise ValueError('near-correct original inputs/evidence changed')
        edited = row['question'] != anchor['question']
        if type(author['question_edited']) is not bool or author['question_edited'] != edited or manifest['question_edit_flags'][row['id']] is not edited:
            raise ValueError('question edit flag differs from literal inputs')
        if edited and author['category'] != 'unsupported_inherited_qualifier':
            raise ValueError('question edits restricted to explicit qualifier category')
        row['original_id'] = anchor['id']
        row['authorship'] = author
    if dict(Counter(row['category'] for row in authors)) != manifest['category_counts']:
        raise ValueError('authored category counts differ')
    labels = helper.read_jsonl(directory / 'adjudicated.jsonl')
    result = reviewed(rows, labels)
    for row in result:
        row['augmentation_origin'] = 'reviewed authored near-correct training contrast'
    return result, {'manifest_sha256': helper.sha256(directory / 'manifest.json'),
                    'adjudication_sha256': helper.sha256(directory / 'adjudication.json'),
                    'frozen_files_sha256': manifest['files'], 'reviewed_files_sha256': files}


def build(base, evidence_pack, near_pack, output):
    base, evidence_pack, near_pack, output = map(Path, (base, evidence_pack, near_pack, output))
    if output.exists():
        raise FileExistsError(output)
    manifest, splits = helper.load_data(base)
    training = {row['id']: row for row in splits['train']}
    train_hash = manifest['files']['train.jsonl']
    evidence, evidence_provenance = evidence_additions(evidence_pack, train_hash, training, manifest)
    near, near_provenance = near_additions(near_pack, train_hash, training)
    additions = evidence + near
    original_sources = {canonical(item) for row in training.values() for item in row['evidence']}
    if any(canonical(item) not in original_sources for row in additions for item in row['evidence']):
        raise ValueError('augmentation evidence is not an exact original training source object')
    combined = {**splits, 'train': splits['train'] + additions}
    helper.validate_splits(combined)
    for rows in combined.values():
        helper.merge_labels(rows, [{'id': row['id'], 'targets': row['targets']} for row in rows])
    result = copy.deepcopy(manifest)
    result['origin'] = ('Assistant-reviewed natural proposals plus reviewed authored TRAIN-only '
                        'evidence interventions and near-correct contrasts; not final evaluation')
    result['excluded_inference_metadata'] = sorted(set(manifest.get('excluded_inference_metadata', [])) |
        {'authorship', 'provenance', 'augmentation_origin', 'candidate_origin', 'review', 'targets', 'rationale'})
    result['splits'] = {name: helper.describe(rows) for name, rows in combined.items()}
    result['augmentation'] = {'role': 'TRAIN-only reviewed authored controls; not natural failures or held-out qualification',
        'parent_manifest_sha256': helper.sha256(base / 'manifest.json'), 'parent_files_sha256': manifest['files'],
        'inherited_inputs_scope': 'Top-level inputs describe only the original parent corpus; added candidate and label inputs are recorded in evidence_pack and near_pack below.',
        'original_training_prefix_rows': len(splits['train']), 'added_evidence_rows': len(evidence),
        'added_near_correct_rows': len(near), 'evidence_pack': evidence_provenance, 'near_pack': near_provenance,
        'heldout_bytes_unchanged': True, 'inference_allowlist': ['question', 'candidate', 'evidence.id', 'evidence.text'],
        'label_authorship': 'supplied assistant reviews/adjudication; not human gold',
        'builder_sha256': helper.sha256(__file__)}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}.stage-', dir=output.parent))
    try:
        prefix = (base / 'train.jsonl').read_bytes()
        with (staging / 'train.jsonl').open('wb') as stream:
            stream.write(prefix)
            if prefix and not prefix.endswith(b'\n'):
                stream.write(b'\n')
            for row in additions:
                stream.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n').encode())
        for name in ('calibration.jsonl', 'development.jsonl'):
            shutil.copyfile(base / name, staging / name)
        result['files'] = {f'{name}.jsonl': helper.sha256(staging / f'{name}.jsonl') for name in helper.SPLITS}
        helper.write_json(staging / 'manifest.json', result)
        helper.load_data(staging)
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'evidence-pack', 'near-pack', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    build(args.base, args.evidence_pack, args.near_pack, args.output)
