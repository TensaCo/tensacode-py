import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).parents[2] / 'examples/train_chatbot.py'
spec = importlib.util.spec_from_file_location('train_chatbot_example', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_records_preserve_explicit_targets_and_ids(tmp_path):
    path = tmp_path / 'data.jsonl'
    record = {'id': 'source-1', 'input': 'observed evidence', 'target': 'reviewed answer'}
    path.write_text(json.dumps(record) + '\n')
    assert module.read_records(path) == [record]
    path.write_text(json.dumps(record) + '\n' + json.dumps(record))
    with pytest.raises(ValueError, match='Duplicate'):
        module.read_records(path)


def test_qa_metrics_handle_empty_and_partial_answers():
    assert module.scores('The Paris.', 'paris') == {'exact_match': 1.0, 'token_f1': 1.0}
    assert module.scores('', 'Paris')['token_f1'] == 0.0
    assert module.scores('red green', 'red blue')['token_f1'] == 0.5


def test_records_reject_empty_target(tmp_path):
    path = tmp_path / 'data.jsonl'
    path.write_text(json.dumps({'id': 'one', 'input': 'hello', 'target': ''}))
    with pytest.raises(ValueError, match='nonempty'):
        module.read_records(path)
