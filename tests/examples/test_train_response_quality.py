import importlib.util
from pathlib import Path
import pytest


def example():
    spec = importlib.util.spec_from_file_location('train_response_quality_example', Path(__file__).parents[2] / 'examples/train_response_quality.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(i, source):
    return {'id':str(i),'question_id':str(i),'question':'Which?', 'candidate':'A.',
            'evidence':[{'id':'e','source_id':source,'text':'source '+source}], 'reference_answer':'secret target'}


def test_model_inputs_exclude_reference_and_review():
    mod = example()
    r = record(0,'doc'); r['rationale']='secret review'
    inputs = mod.model_inputs(r)
    assert set(inputs)=={'question','candidate','evidence'}
    assert inputs['evidence']==[{'source_id':'e','text':'source doc'}]
    assert 'secret' not in str(inputs)


def test_split_keeps_connected_sources_and_question_variants_together():
    mod = example()
    records=[record(i,str(i)) for i in range(12)]
    records[1]['evidence']=records[0]['evidence']
    repeated=dict(records[2],id='variant')
    records.append(repeated)
    splits=mod.split_records(records,seed=7,train_questions=6,calibration_questions=3)
    locations={r['id']:name for name,rows in splits.items() for r in rows}
    assert locations['0']==locations['1']
    assert locations['2']==locations['variant']
    assert len(locations)==13
    mod.validate_splits(splits)
    with pytest.raises(ValueError,match='overlap'):
        mod.validate_splits({'train':[records[0]],'calibration':[records[1]],'development':[]})


def test_labels_must_cover_exact_candidates_and_keep_unknowns():
    mod=example();records=[record(0,'doc')]
    label={'id':'0','targets':{'support':True,'completeness':False,'constraints':None},'reviewer':'assistant','rationale':'incomplete','source_ids':['doc']}
    merged=mod.merge_labels(records,[label])
    assert merged[0]['targets']['constraints'] is None
    for labels in ([],[label,label],[dict(label,id='different')],[dict(label,targets={'support':1,'completeness':False,'constraints':None})]):
        with pytest.raises(ValueError):mod.merge_labels(records,labels)


def test_split_connects_identical_text_under_different_source_names():
    mod = example()
    records = [record(i, str(i)) for i in range(12)]
    records[1]['evidence'][0]['text'] = records[0]['evidence'][0]['text']
    splits = mod.split_records(records, seed=4, train_questions=6, calibration_questions=3)
    locations = {row['id']: split for split, rows in splits.items() for row in rows}
    assert locations['0'] == locations['1']
    assert splits == mod.split_records(list(reversed(records)), seed=4, train_questions=6, calibration_questions=3)


def test_metrics_do_not_treat_unknown_as_negative_or_certify_it():
    mod = example()
    rows = [{'targets': dict(support=True, completeness=True, constraints=None)},
            {'targets': dict(support=True, completeness=False, constraints=True)}]
    scores = [dict.fromkeys(mod.AXES, .8)] * 2
    result = mod.metrics(rows, scores)
    assert result['constraints']['labelled'] == 1
    assert result['constraints']['accuracy'] == 1
    assert result['all_axes'] == {'accepted': 2, 'accepted_all_known_true': 0,
                                  'accepted_known_failure': 1, 'accepted_unresolved': 1,
                                  'all_known_true_total': 0}
    with pytest.raises(ValueError):
        mod.metrics(rows, scores[:1])
    with pytest.raises(ValueError):
        mod.metrics(rows, [dict.fromkeys(mod.AXES, float('nan'))] * 2)


def test_majority_baseline_is_fitted_only_from_training():
    mod = example()
    training = [{'targets': dict.fromkeys(mod.AXES, False)}]
    development = [{'targets': dict.fromkeys(mod.AXES, True)}]
    result = mod.constant_baselines(training, development)
    assert result['training_majority']['scores'] == dict.fromkeys(mod.AXES, 0.)
    assert result['training_majority']['metrics']['support']['accuracy'] == 0.
    assert result['all_positive']['metrics']['support']['accuracy'] == 1.


def test_load_data_rejects_tampered_partition(tmp_path):
    import json
    mod = example()
    for split in mod.SPLITS:
        (tmp_path / f'{split}.jsonl').write_text('')
    manifest = {'files': {f'{split}.jsonl': mod.sha256(tmp_path / f'{split}.jsonl') for split in mod.SPLITS}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'development.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='checksum'):
        mod.load_data(tmp_path)


def test_foundation_verification_checks_revision_and_actual_bytes(tmp_path):
    import hashlib
    mod = example()
    cache = tmp_path / '.cache/huggingface/download'
    cache.mkdir(parents=True)
    for name in ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'):
        content = b'fixture bytes'
        (tmp_path / name).write_bytes(content)
        etag = (hashlib.sha256(content).hexdigest() if name.endswith('safetensors') else
                hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest())
        (cache / (name + '.metadata')).write_text(mod.REVISION + '\n' + etag + '\n0\n')
    result = mod.verify_foundation(tmp_path)
    assert result['assets']['model.safetensors']['etag_algorithm'] == 'sha256'
    (tmp_path / 'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        mod.verify_foundation(tmp_path)
    (cache / 'config.json.metadata').write_text('wrong revision\n' + '0' * 40)
    with pytest.raises(ValueError, match='revision'):
        mod.verify_foundation(tmp_path)


def test_prescribed_partitions_reject_missing_and_overlapping_questions():
    mod = example()
    records = [record(i, str(i)) for i in range(3)]
    groups = {'train': ['0'], 'calibration': ['1'], 'development': ['2']}
    assert mod.prescribed_splits(records, groups)['development'] == [records[2]]
    with pytest.raises(ValueError, match='exact'):
        mod.prescribed_splits(records, dict(groups, development=[]))
    with pytest.raises(ValueError, match='exact'):
        mod.prescribed_splits(records, dict(groups, development=['0', '2']))


def test_ablation_preserves_targets_and_replaces_whole_other_question_evidence():
    import copy
    mod = example()
    rows = [record(i, str(i)) for i in range(3)]
    for row in rows:
        row['targets'] = dict.fromkeys(mod.AXES, True)
    rows.append(dict(rows[0], id='variant'))
    before = copy.deepcopy(rows)
    shuffled = mod.ablation_rows(rows, 'source_shuffled', seed=3)
    assert rows == before
    assert shuffled[0]['evidence'] == shuffled[3]['evidence']
    for original, changed in zip(rows, shuffled):
        assert original['evidence'] != changed['evidence']
        assert original['targets'] == changed['targets']
    assert all(not r['evidence'] for r in mod.ablation_rows(rows, 'evidence_free', seed=3))
    with pytest.raises(ValueError, match='two'):
        mod.ablation_rows(rows[:1], 'source_shuffled', seed=3)


def test_prepare_preserves_manifest_groups_and_explicit_protocol(tmp_path):
    import argparse
    import json
    mod = example()
    rows = [record(i, str(i)) for i in range(3)]
    candidates, labels, selection = [tmp_path / name for name in ('candidates.jsonl','labels.jsonl','selection.json')]
    candidates.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    labels.write_text(''.join(json.dumps({'id': r['id'], 'targets': dict.fromkeys(mod.AXES, True)})+'\n' for r in rows))
    groups = {'train':['2'], 'calibration':['0'], 'development':['1']}
    selection.write_text(json.dumps({'question_groups': groups, 'seed': 123}))
    output = tmp_path / 'prepared'
    result = mod.prepare(argparse.Namespace(candidates=candidates, labels=[labels], output=output,
                                           seed=123, train_questions=1, calibration_questions=1,
                                           selection_manifest=selection))
    assert result['seed'] == 123
    assert mod.read_jsonl(output / 'train.jsonl')[0]['id'] == '2'
    assert result['requested_questions'] == dict.fromkeys(mod.SPLITS, 1)


def test_ablation_report_compares_scores_but_names_original_label_limitation():
    mod = example()
    class EvidenceSensitive:
        def receipt(self, inputs):
            probability = .9 if inputs['evidence'] else .1
            return {'scores':dict.fromkeys(mod.AXES, probability)}
    rows = [dict(record(i, str(i)), targets=dict.fromkeys(mod.AXES, True)) for i in range(2)]
    model = EvidenceSensitive()
    result = mod.evaluate_ablations(model, rows, mod.evaluate(model, rows), seed=9)
    empty = result['evidence_free']
    assert empty['mean_score_delta_from_full']['support'] == pytest.approx(-.8)
    assert empty['acceptance_delta_from_full'] == -2
    assert 'metrics' not in empty
    assert empty['metrics_against_original_full_source_labels']['support']['false_rejects'] == 2


def test_ablation_predictions_record_actual_context_provenance():
    mod = example()
    class Scorer:
        def receipt(self, inputs): return {'scores':dict.fromkeys(mod.AXES, .8)}
    rows = [dict(record(i, str(i)), targets=dict.fromkeys(mod.AXES, True)) for i in range(2)]
    result = mod.evaluate_ablations(Scorer(), rows, mod.evaluate(Scorer(), rows), seed=4)
    contexts = result['source_shuffled']['contexts']
    assert contexts[0]['id'] == '0'
    assert contexts[0]['evidence'][0]['source_id'] == '1'
    assert result['evidence_free']['contexts'][0]['evidence'] == []
