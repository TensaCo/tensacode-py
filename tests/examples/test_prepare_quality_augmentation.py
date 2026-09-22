import copy
import importlib.util
import json
from pathlib import Path

import pytest


def runner():
    path = Path(__file__).parents[2] / '.development/experiments/prepare_quality_augmentation.py'
    spec = importlib.util.spec_from_file_location('augmentation_builder', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.write_text(json.dumps(value))


def lines(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def fixture(tmp_path):
    mod = runner()
    base, evidence, near = [tmp_path / name for name in ('base', 'evidence', 'near')]
    for path in (base, evidence, near):
        path.mkdir()
    def row(key):
        return {'id': key, 'question_id': 'q' + key, 'question': 'Who founded ' + key + '?',
                'candidate': 'Alice founded ' + key + '.',
                'evidence': [{'id': 'doc', 'source_id': key, 'text': 'Alice founded ' + key + '.'}],
                'targets': {'support': True, 'completeness': True, 'constraints': True}}
    a, b = row('a'), row('b')
    for name, rows in [('train', [a, b]), ('calibration', [row('c')]), ('development', [row('d')])]:
        lines(base / f'{name}.jsonl', rows)
    base_manifest = {'selection_manifest': {'fixture': True}, 'files': {
        f'{name}.jsonl': mod.helper.sha256(base / f'{name}.jsonl') for name in mod.helper.SPLITS}}
    write(base / 'manifest.json', base_manifest)
    extra = {**copy.deepcopy(a), 'id': 'e', 'original_id': 'a', 'evidence': [],
             'provenance': {'intervention': 'evidence-free'}, 'donor_question_id': None}
    extra.pop('targets')
    lines(evidence / 'candidates.jsonl', [extra])
    labels = [{'id': 'e', 'targets': {'support': False, 'completeness': None, 'constraints': None}, 'reviewer': 'fixture'}]
    lines(evidence / 'labels-000.jsonl', labels)
    write(evidence / 'manifest.json', {'format': 'tensorcode.response_quality_training_interventions.v1',
          'input_partition': 'train', 'input_sha256': base_manifest['files']['train.jsonl'], 'candidates': 1,
          'files': {'candidates.jsonl': mod.helper.sha256(evidence / 'candidates.jsonl'), 'review.jsonl': 'removed-template'}})
    write(evidence / 'training-manifest.json', {'format': 'tensorcode.response_quality_supervision.v1', 'selection_manifest': base_manifest['selection_manifest'],
          'files': base_manifest['files'], 'inputs': {'labels': {
          'evidence/labels-000.jsonl': mod.helper.sha256(evidence / 'labels-000.jsonl')}}})
    near_row = {**copy.deepcopy(a), 'id': 'n', 'candidate': 'Bob founded a.',
                'targets': dict.fromkeys(mod.helper.AXES)}
    author = {'id': 'n', 'original_id': 'a', 'partition': 'train', 'question_id': 'qa',
              'original_question': a['question'], 'original_candidate': a['candidate'],
              'question_edited': False, 'category': 'predicate_attachment'}
    lines(near / 'candidates.jsonl', [near_row])
    lines(near / 'anchors.jsonl', [a])
    lines(near / 'authorship.jsonl', [author])
    write(near / 'manifest.json', {'format': 'tensorcode.train_only_near_correct_review_pack.v1',
          'input_partition': 'train', 'source_sha256': base_manifest['files']['train.jsonl'], 'variants': 1,
          'question_edit_flags': {'n': False}, 'category_counts': {'predicate_attachment': 1},
          'files': {name: mod.helper.sha256(near / name) for name in ('candidates.jsonl', 'anchors.jsonl', 'authorship.jsonl')}})
    near_labels = [{'id': 'n', 'targets': {'support': False, 'completeness': True, 'constraints': None}, 'review_authorship': 'fixture'}]
    for name in ('adjudicated.jsonl', 'reviews-first.jsonl', 'reviews-second.jsonl'):
        lines(near / name, near_labels)
    refresh_receipt(mod, near)
    return mod, base, evidence, near


def refresh_receipt(mod, near):
    write(near / 'adjudication.json', {'format': 'tensorcode.train_near_correct_adjudication.v1',
          'frozen_manifest_sha256': mod.helper.sha256(near / 'manifest.json'), 'review_files_sha256': {
          name: mod.helper.sha256(near / name) for name in ('adjudicated.jsonl', 'reviews-first.jsonl', 'reviews-second.jsonl')}})


def test_append_only_training_preserves_bytes_unknowns_and_inference_allowlist(tmp_path):
    mod, base, evidence, near = fixture(tmp_path)
    output = tmp_path / 'prepared'
    result = mod.build(base, evidence, near, output)
    _, splits = mod.helper.load_data(output)
    assert len(splits['train']) == 4
    assert (output / 'train.jsonl').read_bytes().startswith((base / 'train.jsonl').read_bytes())
    for name in ('calibration.jsonl', 'development.jsonl'):
        assert (output / name).read_bytes() == (base / name).read_bytes()
    assert splits['train'][-1]['targets']['constraints'] is None
    assert splits['train'][-2]['targets']['completeness'] is None
    assert set(mod.helper.model_inputs(splits['train'][-1])) == {'question', 'candidate', 'evidence'}
    assert result['augmentation']['parent_manifest_sha256'] == mod.helper.sha256(base / 'manifest.json')
    assert 'natural proposals plus reviewed authored TRAIN-only' in result['origin']
    assert {'authorship', 'provenance', 'augmentation_origin', 'candidate_origin'} <= set(result['excluded_inference_metadata'])
    assert 'original parent corpus' in result['augmentation']['inherited_inputs_scope']
    with pytest.raises(FileExistsError):
        mod.build(base, evidence, near, output)


@pytest.mark.parametrize('mutation', ['pending', 'ids', 'question', 'evidence', 'anchor', 'label_hash', 'adjudication'])
def test_rejects_unreviewed_or_nontraining_changes_without_output(tmp_path, mutation):
    mod, base, evidence, near = fixture(tmp_path)
    if mutation == 'adjudication':
        receipt = mod.read(near / 'adjudication.json')
        receipt['frozen_manifest_sha256'] = '0' * 64
        write(near / 'adjudication.json', receipt)
    elif mutation in ('pending', 'ids', 'label_hash'):
        labels = mod.helper.read_jsonl(near / 'adjudicated.jsonl')
        if mutation == 'pending':
            labels[0]['targets'] = dict.fromkeys(mod.helper.AXES)
        elif mutation == 'ids':
            labels[0]['id'] = 'absent'
        else:
            labels[0]['targets']['support'] = True
        lines(near / 'adjudicated.jsonl', labels)
        if mutation != 'label_hash':
            refresh_receipt(mod, near)
    else:
        rows = mod.helper.read_jsonl(near / 'candidates.jsonl')
        authors = mod.helper.read_jsonl(near / 'authorship.jsonl')
        if mutation == 'question':
            rows[0]['question'] = 'Changed question?'
        elif mutation == 'evidence':
            rows[0]['evidence'] = mod.helper.read_jsonl(base / 'calibration.jsonl')[0]['evidence']
        else:
            authors[0]['original_id'] = 'd'
        lines(near / 'candidates.jsonl', rows)
        lines(near / 'authorship.jsonl', authors)
        manifest = mod.read(near / 'manifest.json')
        manifest['files'] = {name: mod.helper.sha256(near / name) for name in manifest['files']}
        write(near / 'manifest.json', manifest)
        refresh_receipt(mod, near)
    with pytest.raises(ValueError):
        mod.build(base, evidence, near, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()
