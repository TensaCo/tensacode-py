import importlib.util
from pathlib import Path
import json
import pytest


def runner():
    path=Path(__file__).parents[2]/'.development/experiments/probe_claim_verbalization.py'
    spec=importlib.util.spec_from_file_location('claim_probe',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_verbalization_only_receives_question_and_proposal():
    row={'question':'When was the pilot born?', 'candidate':'February',
         'evidence':[{'source_id':'a','text':'SECRET_SOURCE'}], 'reference_answer':'SECRET_TARGET'}
    prompt=runner().verbalization_prompt(row)
    assert 'SECRET' not in prompt
    assert json.loads(prompt.split('\n',1)[1])=={'question':row['question'],'proposed_answer':'February'}
    for value in ['', None, 7]:
        with pytest.raises(ValueError):runner().verbalization_prompt(dict(row,candidate=value))


def test_preserved_answer_is_a_format_guard_not_semantic_equivalence():
    mod=runner()
    assert mod.answer_preserved('February','The pilot was born in February.')
    assert not mod.answer_preserved('February','The pilot was born in October.')
    assert not mod.answer_preserved('7','The distance is 17 miles.')
    # Deliberately demonstrates the limit: the guard cannot certify the predicate.
    assert mod.answer_preserved('February','The pilot crashed in February.')
