"""Retired benchmarks disclose removal without scoring or replacing historical data."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('script,artifacts', [
    ('eval/structures/gsm8k.py', ('eval/results/structures_gsm8k.json',)),
    ('eval/relations/eval_gsm8k_transfer.py', ('eval/results/relation_gsm8k.json',)),
    ('eval/relations/eval_relations.py', ('eval/results/relation_hotpot.json',
                                       'eval/results/relation_hotpot_after_defect_fixes.json')),
])
def test_retired_entrypoint_runs_without_dependencies_or_overwriting_history(tmp_path, script, artifacts):
    before = {name: (ROOT / name).read_bytes() for name in artifacts}
    # Isolated Python without site packages exercises the actual CLI: retired
    # evaluators must not import the removed readers or dataset/model packages.
    result = subprocess.run([sys.executable, '-I', '-S', str(ROOT / script)],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert not result.stderr
    assert report['status'] == 'retired' and report['measured'] is False
    assert tuple(report['historical_artifacts']) == artifacts
    assert 'no language benchmark replacement' in report['replacement']
    assert not {'accuracy', 'accuracy_overall', 'coverage', 'correct', 'n', 'scores'} & report.keys()
    assert not list(tmp_path.iterdir())
    assert {name: (ROOT / name).read_bytes() for name in artifacts} == before


def test_automatic_numeric_and_relation_readers_cannot_be_imported_as_fallbacks():
    from tensorcode import semantics_bridge
    assert importlib.util.find_spec('tensorcode.relation') is None
    for name in ('Mention', 'quantities_in', 'quantities_in_text', 'tell_mentions', 'WORD_NUMBERS', 'MONEY'):
        assert not hasattr(semantics_bridge, name)
    from eval.structures import gsm8k
    for name in ('solve', 'run', '_search', 'TOTAL_CUES', 'LEFT_CUES', 'EACH_CUES', 'QUESTION'):
        assert not hasattr(gsm8k, name)
