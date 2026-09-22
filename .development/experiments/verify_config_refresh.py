"""Compare a historical ranking artifact with a config-only refresh.

Run baseline and compare in separate Python processes, selecting the historical
or current source tree with PYTHONPATH. Only the two named tool classes load.
Three authored supplied-candidate probes establish finite behavior equivalence,
not training quality, general accuracy, or a qualified cognitive capability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import torch

DEFAULT_ADDITIONS = {'verification_scope': 'source', 'max_proposals': 16,
                     'proposal_template_version': 1}
PROBES = [
    {'question': 'Which city is named in the source?',
     'evidence': [{'source_id': 'note', 'text': 'The meeting took place in Rome.'}],
     'hypotheses': [{'id': 'rome', 'text': 'The meeting took place in Rome.'},
                    {'id': 'paris', 'text': 'The meeting took place in Paris.'}]},
    {'question': 'Which color is reported for the door?',
     'evidence': [{'source_id': 'first', 'text': 'The door is red.'},
                  {'source_id': 'second', 'text': 'The door is blue.'}],
     'hypotheses': [{'id': 'blue', 'text': 'The door is blue.'},
                    {'id': 'red', 'text': 'The door is red.'},
                    {'id': 'conflict', 'text': 'The two reports disagree about the color.'}]},
    {'question': 'Which year was the bridge built?', 'evidence': [],
     'hypotheses': [{'id': 'old', 'text': 'The bridge was built in 1900.'},
                    {'id': 'new', 'text': 'The bridge was built in 2000.'}]},
]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def state_digest(model, *, chunk_bytes=1024 * 1024):
    """Hash every named state tensor, including dtype/shape and alias entries.

    Transfer at most one chunk per tensor to CPU. These owned models have dense
    contiguous parameters; reject other layouts rather than copying whole tensors.
    """
    if type(chunk_bytes) is not int or chunk_bytes < 1:
        raise ValueError('chunk_bytes must be a positive integer')
    tensors = []
    total = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise ValueError('state digest requires dense contiguous tensors')
        flat = tensor.detach().reshape(-1)
        width = max(1, chunk_bytes // tensor.element_size())
        digest = hashlib.sha256()
        for start in range(0, flat.numel(), width):
            chunk = flat[start:start + width].cpu().view(torch.uint8)
            digest.update(chunk.numpy().tobytes())
        row = {'name': name, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
               'sha256': digest.hexdigest()}
        total.update(canonical(row).encode())
        total.update(b'\n')
        tensors.append(row)
    return {'sha256': total.hexdigest(), 'tensors': tensors, 'tensor_count': len(tensors),
            'parameter_count': sum(parameter.numel() for parameter in model.parameters())}


def record(model, directory, *, tool, source_revision):
    directory = Path(directory)
    manifest = json.loads((directory / 'tensorcode_config.json').read_text())
    if tool not in ('investigator', 'decision'):
        raise ValueError('unsupported tool')
    if any(key in manifest['config'] for key in ('generator', 'verifier_config')):
        raise ValueError('this protocol only covers supplied-candidate ranking artifacts')
    model.eval()
    with torch.inference_mode():
        predictions = [model(json.loads(canonical(probe))) for probe in PROBES]
    # Validate that receipts are finite data-only records before any comparison.
    canonical(predictions)
    return {'format': 'tensorcode.config_refresh_probe', 'version': 1, 'tool': tool,
            'source_revision': source_revision, 'manifest': manifest,
            'weights_sha256': file_digest(directory / 'model.safetensors'),
            'state': state_digest(model), 'probes': PROBES, 'predictions': predictions,
            'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                            'device': str(next(model.parameters()).device),
                            'threads': torch.get_num_threads(),
                            'tool_module': sys.modules[type(model).__module__].__file__}}


def compare(baseline, refreshed):
    for key in ('format', 'version', 'tool', 'weights_sha256', 'state', 'probes', 'predictions'):
        if canonical(baseline[key]) != canonical(refreshed[key]):
            raise ValueError(f'configuration refresh changed {key}')
    if baseline['format'] != 'tensorcode.config_refresh_probe' or baseline['version'] != 1:
        raise ValueError('unsupported baseline record')
    if canonical(baseline['probes']) != canonical(PROBES):
        raise ValueError('baseline does not use the fixed authored probes')
    before, after = baseline['manifest'], refreshed['manifest']
    if set(before) != set(after) or canonical({k: v for k, v in before.items() if k != 'config'}) != canonical({k: v for k, v in after.items() if k != 'config'}):
        raise ValueError('artifact envelope changed')
    old = before['config']
    if any(key in old for key in DEFAULT_ADDITIONS):
        raise ValueError('baseline must omit exactly the three historical defaults')
    if canonical({**old, **DEFAULT_ADDITIONS}) != canonical(after['config']):
        raise ValueError('configuration changes exceed the three allowed defaults')
    return {'verified': True, 'added_defaults': DEFAULT_ADDITIONS,
            'limitation': 'Identical weights, full state and three authored ranking probe receipts; '
                          'not new training or task-performance qualification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['baseline', 'compare'])
    parser.add_argument('--tool', required=True, choices=['investigator', 'decision'])
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if not args.artifact.is_dir():
        parser.error('--artifact must be an existing local directory')
    if args.output.exists():
        parser.error('--output must be a new file')
    if (args.phase == 'compare') != (args.baseline is not None):
        parser.error('--baseline is required only for compare')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from tensorcode.tools.investigator import Investigator
    from tensorcode.tools.decision import Decision
    cls = {'investigator': Investigator, 'decision': Decision}[args.tool]
    model = cls.from_pretrained(args.artifact, local_files_only=True, device=args.device)
    result = record(model, args.artifact, tool=args.tool, source_revision=args.source_revision)
    if args.phase == 'compare':
        baseline = json.loads(args.baseline.read_text())
        result['comparison'] = compare(baseline, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
