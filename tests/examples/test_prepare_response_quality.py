import importlib.util
from pathlib import Path
import pytest


def example():
    path = Path(__file__).parents[2] / 'examples/prepare_response_quality.py'
    spec = importlib.util.spec_from_file_location('prepare_quality', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case(i, title):
    return {'id': str(i), 'question': 'Which?', 'answer': 'SECRET',
            'supporting_facts': {'title': [title]},
            'context': {'title': [title], 'sentences': [['The source '+title+'.']]}}


def test_selection_reserves_validation_and_existing_sources_and_exact_counts():
    mod = example()
    rows = [case(0,'final'), case(1,'old'), case(2,'a'), case(3,'a'), case(4,'b'), case(5,'c')]
    selected, groups = mod.select_cases(rows, {'final'}, [{'question_id':'old-id', 'evidence':[{'source_id':'old','text':'old text'}]}], counts=(1,1,1))
    assert [r['id'] for r in selected] == ['2','4','5']
    assert groups == {'train':['2'], 'calibration':['4'], 'development':['5']}
    with pytest.raises(ValueError, match='requested'):
        mod.select_cases(rows, {'final'}, [], counts=(5,1,1))


def test_candidates_keep_reference_only_metadata_and_allow_natural_deduplication():
    mod = example()
    prepared = {'id':'q', 'question':'Which?', 'target':'SECRET', 'evidence':[{'id':'doc-9','source_id':'original','text':'Evidence.'}]}
    class Generator:
        def propose(self, inputs, *, count):
            assert inputs == {'question':'Which?', 'evidence':[{'source_id':'doc-9','text':'Evidence.'}]}
            assert count == 3
            return [{'id':'natural', 'text':'Answer.', 'input_truncated':False}]
    rows = mod.generate_case(Generator(), prepared, 0)
    assert rows[0]['id'] == 'q:natural'
    assert rows[0]['evidence'] == prepared['evidence']
    assert rows[0]['reference_answer'] == 'SECRET'
    assert rows[0]['input_truncated'] is False
    class Empty:
        def propose(self, inputs, *, count): return []
    with pytest.raises(ValueError, match='empty'):
        mod.generate_case(Empty(), prepared, 0)


def test_validation_projection_reads_only_title_leaf(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    mod = example()
    path = tmp_path / 'validation.parquet'
    pq.write_table(pa.Table.from_pylist([case(0, 'reserved')]), path)
    original = pq.read_table
    def projection(path, *, columns):
        assert columns == ['context.title']
        table = original(path, columns=columns)
        assert 'SECRET' not in str(table.to_pylist())
        return table
    monkeypatch.setattr(pq, 'read_table', projection)
    assert mod.validation_titles(path) == {'reserved'}
