import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def example():
    path = Path(__file__).parents[2] / '.development/experiments/prepare_quality_interventions.py'
    spec = importlib.util.spec_from_file_location('quality_interventions', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(identity, question, title, *, targets=None):
    return {'id': identity, 'question_id': question, 'question': 'Question '+question,
            'candidate': 'Answer '+question, 'reference_answer': 'SECRET',
            'targets': targets or dict(support=True, completeness=True, constraints=True),
            'review': {'rationale': 'original full-context review'},
            'evidence': [{'id': 'doc-9', 'source_id': title, 'text': 'Source '+title}]}


def test_pairing_selects_first_known_good_variant_and_changes_only_context():
    mod = example()
    rows = [record('bad', 'bad', 'bad', targets=dict(support=False, completeness=True, constraints=True)),
            record('a', 'q1', 'one'), record('a2', 'q1', 'one'), record('b', 'q2', 'two'), record('c', 'q3', 'three')]
    original = copy.deepcopy(rows)
    result = mod.interventions(rows, count=2)
    assert rows == original
    assert [row['id'] for row in result] == ['a:source-shuffled', 'a:evidence-free', 'b:source-shuffled', 'b:evidence-free']
    for row in result:
        base = rows[1] if row['question_id'] == 'q1' else rows[3]
        assert (row['question'], row['candidate']) == (base['question'], base['candidate'])
        assert 'targets' not in row and 'review' not in row and 'reference_answer' not in row
        assert row['original_reference_answer'] == 'SECRET'
        assert row['candidate_origin'] == 'authored_source_intervention_on_natural_candidate'
        assert row['review_status'] == 'pending_explicit_review'
    assert result[0]['evidence'] == rows[3]['evidence']
    assert result[0]['donor_question_id'] == 'q2'
    assert result[1]['evidence'] == []
    assert result[1]['donor_question_id'] is None
    assert result[2]['evidence'] == rows[1]['evidence']


def test_pairing_rejects_shared_sources_texts_duplicate_ids_and_insufficient_donors():
    mod = example()
    a, b = record('a', 'q1', 'one'), record('b', 'q2', 'two')
    for rows in ([a], [a, dict(b, id='a')], [a, dict(b, evidence=a['evidence'])],
                 [a, dict(b, evidence=[dict(b['evidence'][0], text=a['evidence'][0]['text'])])]):
        with pytest.raises(ValueError):
            mod.interventions(rows)
    with pytest.raises(ValueError):
        mod.interventions([a,b], count=0)


def test_prepare_reads_only_train_and_outputs_pending_review_atomically(tmp_path, monkeypatch):
    mod = example()
    data = tmp_path / 'prepared'
    data.mkdir()
    training = data / 'train.jsonl'
    rows = [record('a', 'q1', 'one'), record('b', 'q2', 'two')]
    training.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (data / 'development.jsonl').write_text('DO NOT READ')
    (data / 'calibration.jsonl').write_text('DO NOT READ')
    old_open = Path.open
    def guarded_open(path, *args, **kwargs):
        assert path.name not in ('development.jsonl', 'calibration.jsonl')
        return old_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    output = tmp_path / 'interventions'
    manifest = mod.prepare(SimpleNamespace(data=data, output=output, count=64))
    assert manifest['selected_questions'] == 2
    assert manifest['candidates'] == 4
    assert manifest['input_sha256'] == mod.sha256(training)
    reviews = [json.loads(line) for line in (output / 'review.jsonl').read_text().splitlines()]
    assert len(reviews) == 2
    assert reviews[0]['question_id'] == 'q1'
    assert len(reviews[0]['interventions']) == 2
    assert all('targets' not in item for row in reviews for item in row['interventions'])
    with pytest.raises(FileExistsError):
        mod.prepare(SimpleNamespace(data=data, output=output, count=64))
    assert not list(tmp_path.glob('.interventions-*'))


def test_interventions_do_not_pass_original_gold_or_metadata_to_inference():
    mod = example()
    path = Path(__file__).parents[2] / 'examples/train_response_quality.py'
    spec = importlib.util.spec_from_file_location('quality_training_input', path)
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    candidates = mod.interventions([record('a','q1','one'), record('b','q2','two')])
    for row in candidates:
        inputs = training.model_inputs(row)
        assert set(inputs) == {'question', 'candidate', 'evidence'}
        assert 'SECRET' not in json.dumps(inputs)
        assert 'authored_source_intervention' not in json.dumps(inputs)


def test_failed_write_never_publishes_partial_directory(tmp_path, monkeypatch):
    mod = example()
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'train.jsonl').write_text('\n'.join(json.dumps(r) for r in [record('a','q1','one'),record('b','q2','two')]))
    def failed_write(path, rows):
        Path(path).write_text('partial')
        raise OSError('simulated disk failure')
    monkeypatch.setattr(mod, 'write_jsonl', failed_write)
    output = tmp_path / 'result'
    with pytest.raises(OSError, match='disk failure'):
        mod.prepare(SimpleNamespace(data=data, output=output, count=64))
    assert not output.exists()
    assert not list(tmp_path.glob('.result-*'))
