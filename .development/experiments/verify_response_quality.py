"""Fresh-process comparison against an existing response-quality pilot report."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from tensorcode._internal.response_quality import ResponseQualityAssessor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--data', type=Path, required=True)
args = parser.parse_args()
if not torch.cuda.is_available():
    raise RuntimeError('run real model verification on the authorized CUDA host')
torch.set_num_threads(8)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
script = Path(__file__).resolve().parents[2] / 'examples/train_response_quality.py'
spec = importlib.util.spec_from_file_location('quality_pilot_helpers', script)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
report = json.loads((args.run / 'report.json').read_text())
manifest, splits = helpers.load_data(args.data)
if helpers.sha256(args.data / 'manifest.json') != report['data_manifest_sha256']:
    raise ValueError('verification data differ from the training report')
model = ResponseQualityAssessor.from_pretrained(args.run / 'model', device='cuda')
model.eval()
compared = 0
for split, rows in splits.items():
    lookup = {item['id']: item for item in rows}
    for expected in report['final_calibrated'][split]['predictions']:
        actual = {'id': expected['id'], **model.receipt(helpers.model_inputs(lookup[expected['id']]))}
        if actual != expected:
            raise AssertionError(f'fresh-process receipt differs: {expected["id"]}')
        compared += 1
result = {'fresh_process': True, 'all_receipts_exact': True, 'compared_candidates': compared,
          'report_sha256': helpers.sha256(args.run / 'report.json'),
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'scope': 'complete calibrated artifact, all eligible partitions; no new final questions'}
helpers.write_json(args.run / 'fresh-process-verification.json', result)
print(json.dumps(result, indent=2))
