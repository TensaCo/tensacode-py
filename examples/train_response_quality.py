"""Bounded response-quality pilot using explicit assistant-reviewed supervision.

Run ``prepare`` on reviewed JSONL, then ``train`` on a CUDA training host. No
model download, final-set access, automatic labels or tool-policy promotion occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

AXES = ('support', 'completeness', 'constraints')
SEED = 20260921
FOUNDATION = 'cross-encoder/qnli-electra-base'
REVISION = 'c7dea87c98b2269a935686c31336e97e837cbbeb'
SPLITS = ('train', 'calibration', 'development')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def model_inputs(record):
    """Keep reference answers, targets and reviewer rationales outside inference."""
    return {'question': record['question'], 'candidate': record['candidate'],
            'evidence': [{'source_id': item['id'], 'text': item['text']} for item in record['evidence']]}


def merge_labels(records, labels):
    ids = [record['id'] for record in records]
    reviewed = [label['id'] for label in labels]
    if len(set(ids)) != len(ids) or len(set(reviewed)) != len(reviewed) or set(ids) != set(reviewed):
        raise ValueError('labels must cover exact candidate IDs without duplicates')
    lookup = {label['id']: label for label in labels}
    result = []
    for record in records:
        label = lookup[record['id']]
        targets = label.get('targets')
        if not isinstance(targets, dict) or set(targets) != set(AXES) or any(
                value is not None and type(value) is not bool for value in targets.values()):
            raise ValueError('targets require every axis with bool or None values')
        result.append({**record, 'targets': dict(targets),
                       'review': {key: value for key, value in label.items() if key not in ('id', 'targets')}})
    return result


def source_keys(record):
    keys = set()
    for item in record['evidence']:
        source = item.get('source_id')
        if not isinstance(source, str) or not source.strip() or not isinstance(item.get('text'), str):
            raise ValueError('split provenance needs source_id and source text')
        keys.add(('source', source))
        keys.add(('text', hashlib.sha256(item['text'].encode()).hexdigest()))
    return keys


def validate_splits(splits):
    owners = {}
    for split, rows in splits.items():
        seen = set()
        for row in rows:
            if row['id'] in seen:
                raise ValueError('duplicate candidate ID in split')
            seen.add(row['id'])
            keys = {('candidate', row['id']), ('question', row['question_id']), *source_keys(row)}
            for key in keys:
                previous = owners.setdefault(key, split)
                if previous != split:
                    raise ValueError(f'split overlap for {key[0]} between {previous} and {split}')


def split_records(records, *, seed=SEED, train_questions=20, calibration_questions=6):
    if type(train_questions) is not int or type(calibration_questions) is not int or min(train_questions, calibration_questions) < 1:
        raise ValueError('training and calibration question counts must be positive integers')
    grouped = {}
    for row in records:
        grouped.setdefault(row['question_id'], []).append(row)
    parent = {key: key for key in grouped}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    sources = {}
    for question in sorted(grouped):
        for row in grouped[question]:
            for key in sorted(source_keys(row)):
                previous = sources.setdefault(key, question)
                parent[root(question)] = root(previous)
    components = {}
    for question in sorted(grouped):
        components.setdefault(root(question), []).append(question)
    groups = sorted(components.values())
    random.Random(seed).shuffle(groups)
    result = {split: [] for split in SPLITS}
    counts = {split: 0 for split in SPLITS}
    # Sequential whole-component allocation: never split a document to hit a
    # requested count. Actual counts and component sizes are recorded explicitly.
    for group in groups:
        split = ('train' if counts['train'] < train_questions else
                 'calibration' if counts['calibration'] < calibration_questions else 'development')
        counts[split] += len(group)
        for question in group:
            result[split].extend(sorted(grouped[question], key=lambda row: row['id']))
    validate_splits(result)
    return result


def describe(rows):
    return {'candidates': len(rows), 'questions': len({row['question_id'] for row in rows}),
            'documents': len({item['source_id'] for row in rows for item in row['evidence']}),
            'labels': {axis: {name: sum(row['targets'][axis] is value for row in rows)
                             for name, value in (('true', True), ('false', False), ('unknown', None))} for axis in AXES}}


def prepare(args):
    labels = [label for path in args.labels for label in read_jsonl(path)]
    records = merge_labels(read_jsonl(args.candidates), labels)
    splits = split_records(records)
    if any(not rows for rows in splits.values()):
        raise ValueError('source-connected split leaves an empty partition; revise protocol explicitly')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for split, rows in splits.items():
        (output / f'{split}.jsonl').write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
    manifest = {'format': 'tensorcode.response_quality_supervision.v1', 'seed': SEED,
                'label_authorship': 'assistant-authored judgments; not human ground truth',
                'origin': '88 natural proposals from previously inspected 32-question development run; not held-out final evidence',
                'inputs': {'candidates': sha256(args.candidates), 'labels': {str(path): sha256(path) for path in args.labels}},
                'requested_questions': {'train': 20, 'calibration': 6, 'development': 6},
                'split_policy': 'shuffle connected source-ID/text-hash components; allocate whole components in sequence',
                'splits': {split: describe(rows) for split, rows in splits.items()},
                'files': {f'{split}.jsonl': sha256(output / f'{split}.jsonl') for split in SPLITS},
                'inference_fields': ['question', 'candidate', 'evidence.source_id', 'evidence.text'],
                'excluded_inference_metadata': ['reference_answer', 'targets', 'review', 'rationale']}
    write_json(output / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


def metrics(rows, scores):
    """Unknown labels neither count as negatives nor certify accepted answers."""
    if len(rows) != len(scores):
        raise ValueError('one score row is required per record')
    if any(set(score) != set(AXES) or any(type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1
                                       for p in score.values()) for score in scores):
        raise ValueError('finite probabilities are required for every axis')
    result = {}
    for axis in AXES:
        pairs = [(row['targets'][axis], score[axis]) for row, score in zip(rows, scores) if row['targets'][axis] is not None]
        n = len(pairs)
        result[axis] = {'labelled': n, 'unknown': len(rows) - n,
                        'accuracy': sum((p >= .5) == y for y, p in pairs) / n if n else None,
                        'bce': -sum(math.log(max(1e-12, p if y else 1 - p)) for y, p in pairs) / n if n else None,
                        'false_accepts': sum(p >= .5 and not y for y, p in pairs),
                        'false_rejects': sum(p < .5 and y for y, p in pairs)}
    accepted = [row for row, score in zip(rows, scores) if all(score[axis] >= .5 for axis in AXES)]
    result['all_axes'] = {'accepted': len(accepted),
                          'accepted_all_known_true': sum(all(row['targets'][axis] is True for axis in AXES) for row in accepted),
                          'accepted_known_failure': sum(any(row['targets'][axis] is False for axis in AXES) for row in accepted),
                          'accepted_unresolved': sum(not any(row['targets'][axis] is False for axis in AXES) and
                                                     any(row['targets'][axis] is None for axis in AXES) for row in accepted),
                          'all_known_true_total': sum(all(row['targets'][axis] is True for axis in AXES) for row in rows)}
    return result


def constant_baselines(training_rows, rows):
    majority = {}
    for axis in AXES:
        labels = [row['targets'][axis] for row in training_rows if row['targets'][axis] is not None]
        if not labels:
            raise ValueError('majority baseline needs a known training label for every axis')
        majority[axis] = float(sum(labels) >= len(labels) / 2)
    positive = {axis: 1. for axis in AXES}
    return {'training_majority': {'scores': majority, 'metrics': metrics(rows, [majority] * len(rows))},
            'all_positive': {'scores': positive, 'metrics': metrics(rows, [positive] * len(rows))}}


def evaluate(model, rows):
    predictions = [{'id': row['id'], **model.receipt(model_inputs(row))} for row in rows]
    return {'metrics': metrics(rows, [row['scores'] for row in predictions]), 'predictions': predictions}


def tensor_digest(module):
    import torch
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update((name + str(value.dtype) + str(tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_equal(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(state_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(state_equal(a, b) for a, b in zip(left, right))
    return left == right


def load_data(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if set(manifest['files']) != {f'{split}.jsonl' for split in SPLITS}:
        raise ValueError('manifest requires exactly train/calibration/development files')
    for name, digest in manifest['files'].items():
        if sha256(directory / name) != digest:
            raise ValueError('data manifest checksum mismatch')
    splits = {split: read_jsonl(directory / f'{split}.jsonl') for split in SPLITS}
    validate_splits(splits)
    return manifest, splits


def verify_foundation(directory):
    """Verify local HF download metadata and bytes without network access.

    Download metadata is a local provenance assertion, not a signed remote proof.
    SHA256 (LFS) and Git blob SHA1 ETags detect subsequent asset changes.
    """
    directory = Path(directory)
    required = {'config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'}
    optional = {'special_tokens_map.json', 'vocab.txt', 'added_tokens.json'}
    if any(not (directory / name).is_file() for name in required):
        raise ValueError('pinned foundation requires native safetensors and tokenizer assets')
    result = {}
    for name in sorted(required | {name for name in optional if (directory / name).is_file()}):
        metadata = directory / '.cache' / 'huggingface' / 'download' / (name + '.metadata')
        if not metadata.is_file():
            raise ValueError(f'missing Hugging Face download metadata for {name}')
        lines = metadata.read_text().splitlines()
        if len(lines) < 2 or lines[0] != REVISION:
            raise ValueError(f'foundation revision metadata mismatch for {name}')
        etag = lines[1].strip('"')
        content = (directory / name).read_bytes()
        if len(etag) == 64:
            actual = hashlib.sha256(content).hexdigest()
            algorithm = 'sha256'
        elif len(etag) == 40:
            actual = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
            algorithm = 'git_blob_sha1'
        else:
            raise ValueError(f'unsupported foundation ETag for {name}')
        if actual != etag:
            raise ValueError(f'foundation asset checksum mismatch for {name}')
        result[name] = {'revision': lines[0], 'etag': etag, 'etag_algorithm': algorithm,
                        'sha256': hashlib.sha256(content).hexdigest()}
    return {'assets': result, 'provenance': 'locally asserted HF download revision; asset bytes match recorded ETags'}


def train(args):
    # Must precede CUDA initialization for deterministic matrix multiplication.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import platform
    import torch
    from tensorcode._internal.response_quality import ResponseQualityAssessor
    from tensorcode.training import ToolTrainer
    if not torch.cuda.is_available():
        raise RuntimeError('real-model training requires the authorized CUDA training host')
    if not Path(args.foundation).is_dir():
        raise ValueError('foundation must be an existing local pinned snapshot; this script does not download')
    foundation_verification = verify_foundation(args.foundation)
    manifest, splits = load_data(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)
    model = ResponseQualityAssessor.from_foundation(args.foundation, revision=REVISION,
                                                    max_tokens=512, local_files_only=True).to('cuda')
    model.config['foundation']['repository'] = FOUNDATION
    model.config['foundation']['revision'] = REVISION
    report = {'foundation': {'repository': FOUNDATION, 'revision': REVISION, 'local_path': args.foundation,
                            'files': {str(p.relative_to(args.foundation)): sha256(p) for p in sorted(Path(args.foundation).rglob('*'))
                                      if p.is_file() and p.suffix in ('.json', '.safetensors', '.txt')}},
              'foundation_verification': foundation_verification,
              'data': manifest, 'data_manifest_sha256': sha256(Path(args.data) / 'manifest.json'),
              'script_sha256': sha256(__file__), 'host': platform.node(), 'gpu': torch.cuda.get_device_name(),
              'torch': torch.__version__, 'protocol': {'seed': SEED, 'epochs': 5, 'batch': 4, 'adamw_lr': 2e-5,
                                                     'max_tokens': 512, 'threshold': .5, 'attention': 'math SDPA only',
                                                     'deterministic_algorithms': True},
              'limitations': ['small assistant-reviewed development sample', 'shared generator/foundation exposure',
                              'no fresh final evaluation; no production policy promotion',
                              'temperature calibration preserves decisions at threshold 0.5'],
              'excluded': {}, 'eligible': {}}
    for split, rows in splits.items():
        eligible, excluded = [], []
        for row in rows:
            metadata = model.input_metadata(model_inputs(row))
            reasons = ([] if not metadata['input_truncated'] else ['full_input_truncated'])
            if all(row['targets'][axis] is None for axis in AXES):
                reasons.append('all_targets_unknown')
            if reasons:
                excluded.append({'id': row['id'], 'reasons': reasons, **metadata})
            else:
                eligible.append(row)
        splits[split] = eligible
        report['excluded'][split] = excluded
        report['eligible'][split] = describe(eligible)
    write_json(output / 'eligibility.json', report)
    if any(not rows for rows in splits.values()):
        raise ValueError('an entire split is excluded; eligibility report saved, no training performed')
    if any(not any(row['targets'][axis] is not None for row in splits[split]) for split in ('train', 'calibration') for axis in AXES):
        raise ValueError('training/calibration need at least one known label for each axis')
    report['constant_baselines'] = {split: constant_baselines(splits['train'], rows) for split, rows in splits.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    trainer = ToolTrainer(model, optimizer=optimizer)
    report['initial_parameter_digests'] = {name: tensor_digest(getattr(model, name)) for name in ('encoder', 'head')}
    report['initial'] = {split: evaluate(model, rows) for split, rows in splits.items()}
    write_json(output / 'initial.json', report)
    experiences = output / 'experiences'
    experiences.mkdir()
    report['epoch_losses'] = []
    source = 'assistant-reviewed:' + report['data_manifest_sha256']
    for epoch in range(5):
        model.train()
        order = list(splits['train'])
        random.shuffle(order)
        total = 0.
        for index in range(0, len(order), 4):
            rows = order[index:index + 4]
            with torch.no_grad():
                session = trainer.capture([model_inputs(row) for row in rows], [row['targets'] for row in rows], source=source)
            session.save(experiences / f'epoch-{epoch + 1}-batch-{index // 4}.json', operations=trainer.operations, release=True)
            loss = trainer.step(session)
            total += loss * len(rows)
            del session
        report['epoch_losses'].append(total / len(order))
        print(json.dumps({'epoch': epoch + 1, 'loss': report['epoch_losses'][-1]}), flush=True)
    model.eval()
    report['training_steps'] = trainer.steps
    report['final_parameter_digests'] = {name: tensor_digest(getattr(model, name)) for name in ('encoder', 'head')}
    report['parameters_changed'] = {name: report['initial_parameter_digests'][name] != report['final_parameter_digests'][name]
                                    for name in ('encoder', 'head')}
    report['final_uncalibrated'] = {split: evaluate(model, rows) for split, rows in splits.items()}
    model.save_pretrained(output / 'model-uncalibrated')
    trainer.save_checkpoint(output / 'training', progress={'epochs': 5, 'data_manifest_sha256': report['data_manifest_sha256']})
    # Restore optimizer/RNG and compare one identical continuation step. Probe
    # updates are discarded by restoring the checkpoint before calibration.
    restored = ResponseQualityAssessor.from_pretrained(output / 'model-uncalibrated', device='cuda')
    restored_trainer = ToolTrainer(restored, optimizer=torch.optim.AdamW(restored.parameters(), lr=2e-5))
    restored_trainer.load_checkpoint(output / 'training')
    report['reload'] = {'tensors_equal': state_equal(model.state_dict(), restored.state_dict()),
                        'optimizer_equal': state_equal(optimizer.state_dict(), restored_trainer.optimizer.state_dict())}
    probe_rows = splits['train'][:4]
    probe_inputs, probe_targets = [model_inputs(row) for row in probe_rows], [row['targets'] for row in probe_rows]
    with torch.no_grad():
        probe = trainer.capture(probe_inputs, probe_targets, source=source)
        restored_probe = restored_trainer.capture(probe_inputs, probe_targets, source=source)
    probe.release()
    restored_probe.release()
    trainer.load_checkpoint(output / 'training')
    model.train()
    first_loss = trainer.step(probe)
    restored_trainer.load_checkpoint(output / 'training')
    restored.train()
    second_loss = restored_trainer.step(restored_probe)
    report['restart_probe'] = {'loss_equal': first_loss == second_loss,
                               'tensors_equal': state_equal(model.state_dict(), restored.state_dict()),
                               'optimizer_equal': state_equal(optimizer.state_dict(), restored_trainer.optimizer.state_dict()),
                               'same_process_fresh_instances': True}
    trainer.load_checkpoint(output / 'training')
    if trainer.steps != report['training_steps']:
        raise RuntimeError('continuation probe was not rolled back')
    model.eval()
    del restored_trainer, restored, probe, restored_probe
    torch.cuda.empty_cache()
    calibration = splits['calibration']
    with torch.no_grad():
        logits = torch.stack([model(model_inputs(row)).cpu() for row in calibration])
    report['temperature_fit'] = model.fit_calibration(logits, [row['targets'] for row in calibration])
    report['final_calibrated'] = {split: evaluate(model, rows) for split, rows in splits.items()}
    model.save_pretrained(output / 'model')
    restored = ResponseQualityAssessor.from_pretrained(output / 'model', device='cuda')
    report['reload']['calibrated_tensors_equal'] = state_equal(model.state_dict(), restored.state_dict())
    report['reload']['receipt_equal'] = all(model.receipt(model_inputs(row)) == restored.receipt(model_inputs(row))
                                           for row in splits['development'])
    report['elapsed_seconds'] = time.time() - started
    write_json(output / 'report.json', report)
    if not all(report['reload'].values()) or not all(report['restart_probe'].values()):
        raise RuntimeError('artifact or exact continuation parity failed; see report')
    print(json.dumps({'completed': str(output), 'development': report['final_calibrated']['development']['metrics']}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--candidates', required=True)
    prep.add_argument('--labels', nargs='+', required=True)
    prep.add_argument('--output', required=True)
    training = commands.add_parser('train')
    training.add_argument('--data', required=True)
    training.add_argument('--foundation', required=True)
    training.add_argument('--output', required=True)
    args = parser.parse_args()
    return prepare(args) if args.command == 'prepare' else train(args)


if __name__ == '__main__':
    main()
