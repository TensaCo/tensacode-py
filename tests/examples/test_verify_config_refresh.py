import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tensorcode.tools.investigator import Investigator


def runner():
    path = Path(__file__).parents[2] / '.development/experiments/verify_config_refresh.py'
    spec = importlib.util.spec_from_file_location('verify_config_refresh', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def records(tmp_path):
    mod = runner()
    torch.manual_seed(17)
    model = Investigator({'vocabulary': ['red', 'blue', 'city', 'Rome', 'Paris'],
                          'dimensions': 8, 'slots': 2, 'steps': 1}).eval()
    original, refreshed = tmp_path / 'original', tmp_path / 'refreshed'
    model.save_pretrained(original)
    model.save_pretrained(refreshed)
    manifest = original / 'tensorcode_config.json'
    value = json.loads(manifest.read_text())
    for key in mod.DEFAULT_ADDITIONS:
        del value['config'][key]
    manifest.write_text(json.dumps(value))
    # The retained in-memory model supplies the historical fixture behavior;
    # current from_pretrained correctly rejects that obsolete manifest.
    baseline = mod.record(model, original, tool='investigator', source_revision='historical-fixture')
    restored = Investigator.from_pretrained(refreshed)
    current = mod.record(restored, refreshed, tool='investigator', source_revision='current-fixture')
    return mod, baseline, current


def test_refresh_preserves_all_weights_and_three_authored_probe_receipts(tmp_path):
    mod, baseline, current = records(tmp_path)
    result = mod.compare(baseline, current)
    assert result['verified'] is True
    assert result['added_defaults'] == {'verification_scope': 'source', 'max_proposals': 16,
                                         'proposal_template_version': 1}
    assert len(current['predictions']) == 3
    assert current['state']['tensor_count'] > 0
    assert current['state']['parameter_count'] > 0


@pytest.mark.parametrize('mutation', ['weights', 'state', 'dtype', 'prediction', 'config', 'default_type', 'extra_default'])
def test_refresh_rejects_unapproved_changes(tmp_path, mutation):
    mod, baseline, current = records(tmp_path)
    current = copy.deepcopy(current)
    if mutation == 'weights':
        current['weights_sha256'] = '0' * 64
    elif mutation == 'state':
        current['state']['sha256'] = '0' * 64
    elif mutation == 'dtype':
        current['state']['tensors'][0]['dtype'] = 'torch.float64'
    elif mutation == 'prediction':
        current['predictions'][0]['selected_id'] = 'unapproved'
    elif mutation == 'config':
        current['manifest']['config']['dimensions'] += 1
    elif mutation == 'default_type':
        current['manifest']['config']['proposal_template_version'] = True
    else:
        current['manifest']['config']['unexpected'] = 1
    with pytest.raises(ValueError):
        mod.compare(baseline, current)


def test_streaming_digest_detects_parameter_values_and_dtype():
    mod = runner()
    model = torch.nn.Linear(2, 2)
    initial = mod.state_digest(model, chunk_bytes=3)
    with torch.no_grad():
        model.weight[0, 0].add_(1)
    assert mod.state_digest(model, chunk_bytes=3)['sha256'] != initial['sha256']
    before_dtype = mod.state_digest(model)
    model.double()
    assert mod.state_digest(model)['sha256'] != before_dtype['sha256']
    assert all(row['dtype'] == 'torch.float64' for row in mod.state_digest(model)['tensors'])


def test_compare_cli_loads_refreshed_model_in_fresh_process(tmp_path):
    import os
    import subprocess
    import sys
    mod, baseline, expected = records(tmp_path)
    # Exercise the baseline CLI independently with a loadable tiny artifact.
    # Historical source loading itself requires the separate real run.
    baseline_cli = tmp_path / 'baseline-cli.json'
    subprocess.run([sys.executable, mod.__file__, 'baseline', '--tool', 'investigator',
                    '--artifact', str(tmp_path / 'refreshed'), '--source-revision', 'current-fixture',
                    '--output', str(baseline_cli)], check=True, capture_output=True, text=True,
                   env={**os.environ, 'OMP_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''})
    recorded = json.loads(baseline_cli.read_text())
    assert recorded['state'] == expected['state']
    assert recorded['predictions'] == expected['predictions']
    baseline_path = tmp_path / 'baseline.json'
    baseline_path.write_text(json.dumps(baseline))
    output = tmp_path / 'comparison.json'
    subprocess.run([sys.executable, mod.__file__, 'compare', '--tool', 'investigator',
                    '--artifact', str(tmp_path / 'refreshed'), '--source-revision', 'current-fixture',
                    '--baseline', str(baseline_path), '--output', str(output)],
                   check=True, capture_output=True, text=True,
                   env={**os.environ, 'OMP_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''})
    result = json.loads(output.read_text())
    assert result['comparison']['verified'] is True
    assert result['state'] == expected['state']
    assert result['predictions'] == expected['predictions']
