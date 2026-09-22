import copy
import importlib.util
import json
from pathlib import Path
import runpy

import pytest
import torch


def runner():
    path = Path(__file__).parents[2] / '.development/experiments/evaluate_quality_evidence_controls.py'
    spec = importlib.util.spec_from_file_location('evidence_controls_eval', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def controls_path():
    return Path(__file__).parents[2] / '.development/datasets/quality-evidence-controls'


def test_frozen_pack_pending_labels_and_exact_review_join(tmp_path):
    mod = runner()
    pack = mod.load_controls(controls_path())
    assert len(pack['anchors']) == 12 and len(pack['variants']) == 24
    assert all(row['targets'] == dict.fromkeys(mod.AXES) for row in pack['variants'])
    labels = [{'id': row['id'], 'targets': {'support': False, 'completeness': None,
                                          'constraints': None}} for row in pack['variants']]
    path = tmp_path / 'labels.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in labels))
    reviewed = mod.load_controls(controls_path(), path)
    assert reviewed['review_status'] == 'adjudicated_support_labels_supplied'
    assert all(row['targets']['support'] is False for row in reviewed['variants'])
    labels[0]['id'] = 'unrecognized'
    path.write_text(''.join(json.dumps(row) + '\n' for row in labels))
    with pytest.raises(ValueError, match='IDs'):
        mod.load_controls(controls_path(), path)


def test_native_assessment_both_paths_preserve_inputs_and_exclude_overflow():
    from tensorcode.tools.chatbot import Chatbot
    mod = runner()
    config = runpy.run_path(str(Path(__file__).parents[1] / 'models/test_chatbot_model.py'))['tiny_config']()
    config['max_input_tokens'] = 256
    model = Chatbot(config).eval()
    anchor = {'id': 'a', 'question': 'hello', 'candidate': 'world',
              'evidence': [{'id': 'e', 'text': 'hello world'}],
              'targets': {axis: True for axis in mod.AXES}}
    variant = {**copy.deepcopy(anchor), 'id': 'v', 'original_id': 'a',
               'intervention': 'evidence-free', 'evidence': [],
               'targets': {'support': False, 'completeness': None, 'constraints': None}}
    overflow = {**copy.deepcopy(variant), 'id': 'overflow', 'intervention': 'source-swapped',
                'question': 'hello ' * 300}
    records = mod.evaluate(model, [anchor], [variant, overflow], yes_id=5, no_id=6)
    assert len(records) == 3
    for mode, ablation in [('active', None), ('bypass', 'bypass')]:
        expected = mod.probe.assess(model, mod.inputs_helper.model_inputs(anchor), yes_id=5, no_id=6, workspace_ablation=ablation)
        assert records[0][mode] == expected
        assert records[2][mode]['input_truncated']
        assert records[2][mode]['scores'] is None
    summary = mod.summarize(records)
    assert summary['active']['eligible'] == 2
    assert summary['active']['excluded_ids'] == ['overflow']
    assert summary['pairs'][1]['active']['support_delta'] is None
    assert all('targets' not in value for value in mod.probe.prompts(mod.inputs_helper.model_inputs(anchor)).values())


def test_summary_keeps_unknown_support_separate_and_uses_fixed_threshold():
    mod = runner()
    def row(key, support, score, **kw):
        receipt = {'input_truncated': False, 'input_token_counts': dict.fromkeys(mod.AXES, 4),
                   'scores': dict.fromkeys(mod.AXES, score)}
        return {'id': key, 'targets': {'support': support, 'completeness': None, 'constraints': None},
                'active': receipt, 'bypass': receipt, **kw}
    records = [row('a', True, .7, role='anchor'),
               row('v', False, .5, role='variant', original_id='a', intervention='evidence-free'),
               row('u', None, .2, role='variant', original_id='a', intervention='source-swapped')]
    summary = mod.summarize(records)
    assert summary['active']['support']['false_accepts'] == 1
    assert summary['active']['support']['unknown_labels'] == 1
    assert summary['active']['support']['rejected'] == 1
    grouped = summary['active']['by_role_and_intervention']
    assert grouped['anchors']['support']['known_positive'] == 1
    assert grouped['evidence-free']['support']['false_accepts'] == 1
    assert grouped['source-swapped']['support']['unknown_labels'] == 1
    assert summary['pairs'][0]['active']['support_delta'] == pytest.approx(-.2)


def test_saved_tiny_run_evaluates_all_frozen_controls_and_keeps_review_caveats(tmp_path):
    from types import SimpleNamespace
    from tensorcode.tools.chatbot import Chatbot
    mod = runner()
    config = runpy.run_path(str(Path(__file__).parents[1] / 'models/test_chatbot_model.py'))['tiny_config']()
    config['max_input_tokens'] = 128
    tokenizer = json.loads(config['tokenizer_json'])
    tokenizer['model']['vocab'].update(yes=8, no=9)
    config['tokenizer_json'] = json.dumps(tokenizer)
    config['foundation_config']['vocab_size'] = 10
    model = Chatbot(config).eval()
    model.save_pretrained(tmp_path / 'model')
    source = {'data_manifest_sha256': mod.load_controls(controls_path())['manifest']['prepared_manifest_sha256'],
              'instructions': mod.probe.INSTRUCTIONS, 'dtype': 'float32', 'autocast_dtype': None,
              'max_tokens': 128, 'label_ids': {'yes': [8], 'no': [9]},
              **{key: model.configuration()[key] for key in ('memory_update', 'memory_mode', 'workspace')}}
    (tmp_path / 'report.json').write_text(json.dumps(source))
    labels = controls_path() / 'adjudicated.jsonl'
    result = mod.run(SimpleNamespace(run=tmp_path, controls=controls_path(), labels=labels,
                                    output=tmp_path / 'result.json', device='cpu'))
    assert len(result['records']) == 36
    assert len(result['summary']['pairs']) == 24
    assert result['review_status'] == 'adjudicated_support_labels_supplied'
    assert any(row.get('review', {}).get('anchor_label_concerns') for row in result['records'])
    assert result['summary']['active']['excluded_ids']
    assert result['labels_sha256'] == mod.reload_helper.sha256(labels)
    assert json.loads((tmp_path / 'result.json').read_text()) == result


def test_run_rejects_unrelated_training_manifest_before_model_load(tmp_path):
    from types import SimpleNamespace
    mod = runner()
    (tmp_path / 'report.json').write_text(json.dumps({'data_manifest_sha256': 'wrong'}))
    with pytest.raises(ValueError, match='data manifest'):
        mod.run(SimpleNamespace(run=tmp_path, controls=controls_path(), labels=None,
                                output=tmp_path / 'result.json', device='cpu'))
    assert not (tmp_path / 'result.json').exists()


def supervised_fixture(tmp_path):
    mod = runner()
    anchor = {'id': 'dev', 'question_id': 'qd', 'question': 'which?', 'candidate': 'answer',
              'evidence': [{'id': 'd', 'source_id': 'dev-source', 'text': 'dev evidence'}],
              'targets': dict.fromkeys(mod.AXES, True)}
    train = {**copy.deepcopy(anchor), 'id': 'train', 'question_id': 'qt',
             'evidence': [{'id': 't', 'source_id': 'train-source', 'text': 'train evidence'}]}
    calibration = {**copy.deepcopy(anchor), 'id': 'cal', 'question_id': 'qc',
                   'evidence': [{'id': 'c', 'source_id': 'cal-source', 'text': 'cal evidence'}]}
    for name, rows in [('train', [train]), ('calibration', [calibration]), ('development', [anchor])]:
        (tmp_path / f'{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    manifest = {'files': {f'{name}.jsonl': mod.reload_helper.sha256(tmp_path / f'{name}.jsonl')
                          for name in ('train', 'calibration', 'development')}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    pack = {'manifest': {'prepared_manifest_sha256': mod.reload_helper.sha256(tmp_path / 'manifest.json'),
                         'prepared_files_sha256': copy.deepcopy(manifest['files'])},
            'anchors': [anchor], 'variants': []}
    return mod, pack, manifest, train


@pytest.mark.parametrize('change', ['train_only', 'heldout', 'leak', 'source_leak', 'text_leak'])
def test_actual_corpus_guard_checks_heldout_bytes_and_train_disjointness(tmp_path, change):
    mod, pack, manifest, train = supervised_fixture(tmp_path)
    if change == 'train_only':
        (tmp_path / 'train.jsonl').write_text(json.dumps({**train, 'id': 'extra'}) + '\n')
    elif change == 'heldout':
        (tmp_path / 'development.jsonl').write_text(json.dumps({**pack['anchors'][0], 'candidate': 'changed'}) + '\n')
    elif change == 'leak':
        (tmp_path / 'train.jsonl').write_text(json.dumps({**train, 'question_id': 'qd'}) + '\n')
    else:
        source = train['evidence'][0]
        source['source_id' if change == 'source_leak' else 'text'] = ('dev-source' if change == 'source_leak' else 'dev evidence')
        (tmp_path / 'train.jsonl').write_text(json.dumps(train) + '\n')
    manifest['files'] = {name: mod.reload_helper.sha256(tmp_path / name) for name in manifest['files']}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    report = {'data_manifest_sha256': mod.reload_helper.sha256(tmp_path / 'manifest.json')}
    if change == 'train_only':
        receipt = mod.validate_data_provenance(report, pack, tmp_path)
        assert receipt['files_sha256'] == manifest['files']
    else:
        with pytest.raises(ValueError):
            mod.validate_data_provenance(report, pack, tmp_path)
