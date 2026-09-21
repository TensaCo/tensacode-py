"""Fine-tune an owned NLI verifier and calibrate only on a separate SNLI split.

Run substantial workloads on the authorized remote GPU. The foundation already
trained on SNLI/MultiNLI; held-out means isolated from this run, not demonstrably
unseen during foundation development. NLI scores are not factual truth guarantees.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

MODEL = 'cross-encoder/nli-deberta-v3-small'
MODEL_REVISION = 'fa2804872c3b4bd748f38c0185cc85775361e735'
DATASET = 'stanfordnlp/snli'
DATASET_REVISION = 'cdb5c3d5eed6ead6e5a341c8e56e669bb666725b'
LABELS = {0: 'support', 1: 'unknown', 2: 'contradiction'}


def prepare_record(row, split, index):
    if row['label'] == -1:
        return None
    if row['label'] not in LABELS:
        raise ValueError('unknown SNLI label')
    pair = {key: row[key] for key in ('premise', 'hypothesis')}
    if any(not isinstance(value, str) or not value.strip() for value in pair.values()):
        raise ValueError('empty NLI input')
    fingerprint = hashlib.sha256(json.dumps(pair, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {'id': f'{split}:{index}', **pair, 'target': LABELS[row['label']], 'pair_sha256': fingerprint}


def model_inputs(records):
    return [{key: record[key] for key in ('premise', 'hypothesis')} for record in records]


def select_records(rows, split, count):
    selected, seen = [], set()
    for index, row in enumerate(rows):
        record = prepare_record(row, split, index)
        if record is not None and record['pair_sha256'] not in seen:
            selected.append(record)
            seen.add(record['pair_sha256'])
        if len(selected) == count:
            return selected
    raise ValueError('requested count exceeds available distinct labeled records')


def check_splits(splits):
    ids, pairs = set(), set()
    for records in splits.values():
        for record in records:
            if record['id'] in ids or record['pair_sha256'] in pairs:
                raise ValueError('split ID or text-pair overlap')
            ids.add(record['id'])
            pairs.add(record['pair_sha256'])


def label_mapping(id2label):
    semantic = {'entailment': 'support', 'neutral': 'unknown', 'contradiction': 'contradiction'}
    if len(id2label) != 3 or set(id2label.values()) != set(semantic):
        raise ValueError('classifier label mapping must explicitly name the three NLI classes')
    result = {semantic[name]: int(index) for index, name in id2label.items()}
    if set(result.values()) != {0, 1, 2}:
        raise ValueError('classifier label indices must be 0, 1, 2')
    return result


def load_records(split, count):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    filename = f'plain_text/{split}-00000-of-00001.parquet'
    path = Path(hf_hub_download(DATASET, filename, repo_type='dataset', revision=DATASET_REVISION, token=False))
    def rows():
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            yield from batch.to_pylist()
    return select_records(rows(), split, count), {'file': filename, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def make_verifier(max_tokens):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from tensorcode.tools.investigator import EvidenceVerifier
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, token=False, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=MODEL_REVISION, token=False)
    config = {'verifier_config': model.config.to_dict(),
              'verifier_tokenizer_json': tokenizer.backend_tokenizer.to_str(),
              'verifier_tokenizer_special_tokens': {k: str(v) for k, v in tokenizer.special_tokens_map.items() if isinstance(v, str)},
              'verifier_labels': label_mapping(model.config.id2label),
              'verifier_max_tokens': max_tokens,
              'verifier_foundation': {'repository': MODEL, 'revision': MODEL_REVISION}}
    verifier = EvidenceVerifier(config)
    verifier.model.load_state_dict(model.state_dict())
    return verifier


def predict(verifier, records, batch_size):
    import torch
    verifier.eval()
    with torch.no_grad():
        return torch.cat([verifier(model_inputs(records[start:start + batch_size])).detach().float().cpu()
                          for start in range(0, len(records), batch_size)])


def labels_tensor(verifier, records):
    import torch
    return torch.tensor([verifier.labels[row['target']] for row in records])


def authored_diagnostics(verifier):
    """Fixed authored checks: report outcomes, never train or select on them."""
    pairs = [
        ('specific_to_general', 'A dog is running through a field.', 'An animal is running.', 'support'),
        ('general_to_specific', 'An animal is running.', 'A dog is running through a field.', 'unknown'),
        ('contradiction', 'The door is open.', 'The door is closed.', 'contradiction'),
        ('contradiction_reversed', 'The door is closed.', 'The door is open.', 'contradiction'),
        ('unrelated', 'A person is reading a book.', 'It is raining outside.', 'unknown'),
    ]
    records = [{'premise': p, 'hypothesis': h} for _, p, h, _ in pairs]
    scores = verifier.calibration(predict(verifier, records, 8)).softmax(-1).tolist()
    inverse = {index: label for label, index in verifier.labels.items()}
    return [{'id': name, 'premise': p, 'hypothesis': h, 'authored_expected': expected,
             'predicted': inverse[max(range(3), key=lambda i: row[i])],
             'distribution': {label: row[index] for label, index in verifier.labels.items()}}
            for (name, p, h, expected), row in zip(pairs, scores)]


def run(output, *, train_count=1024, calibration_count=256, test_count=256,
        epochs=1, batch_size=8, learning_rate=1e-5, seed=20260921, max_tokens=192, device='cuda'):
    import platform
    import importlib.metadata as metadata
    import torch
    from safetensors.torch import save_file, load_file
    from tensorcode.training.calibration import evaluate_calibration
    from tensorcode.tools.investigator import EvidenceVerifier

    if not torch.cuda.is_available() or device != 'cuda':
        raise ValueError('this substantial training example requires the authorized CUDA host')
    if any(type(x) is not int or x < 1 for x in (train_count, calibration_count, test_count, epochs, batch_size, max_tokens)):
        raise ValueError('counts and sizes must be positive integers')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(8)
    torch.manual_seed(seed)
    random.seed(seed)
    # All source revisions and row-selection settings are fixed before downloads.
    manifest = {'model': {'repository': MODEL, 'revision': MODEL_REVISION},
                'dataset': {'repository': DATASET, 'revision': DATASET_REVISION},
                'settings': dict(train_count=train_count, calibration_count=calibration_count, test_count=test_count,
                                 epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, seed=seed, max_tokens=max_tokens),
                'selection': 'first distinct labeled text pairs in pinned parquet row order; split:row IDs',
                'isolation': 'train fits model weights; validation fits temperature only; test used only for final reporting, never checkpoint selection',
                'limitations': ['Foundation already trained on SNLI and MultiNLI; foundation split exposure cannot be independently certified.',
                                'Caption-domain NLI does not establish real-world truth or general evidence reliability.',
                                'Fixed one-run hyperparameters; no test-driven tuning; no bitwise GPU determinism guarantee.']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    splits, sources = {}, {}
    for split, count in [('train', train_count), ('validation', calibration_count), ('test', test_count)]:
        splits[split], sources[split] = load_records(split, count)
    check_splits(splits)
    manifest['sources'] = sources
    manifest['partitions'] = {split: [{'id': row['id'], 'pair_sha256': row['pair_sha256']} for row in records] for split, records in splits.items()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    verifier = make_verifier(max_tokens).to(device)
    # Baseline predictions are retained, but test metrics are computed only at final reporting.
    before = {split: predict(verifier, splits[split], batch_size) for split in ('validation', 'test')}
    torch.cuda.reset_peak_memory_stats()
    verifier.train()
    pilot_loss = verifier.loss(model_inputs(splits['train'][:batch_size]), [row['target'] for row in splits['train'][:batch_size]])
    pilot_loss.backward()
    torch.cuda.synchronize()
    pilot = {'batch_size': batch_size, 'loss': float(pilot_loss.detach()), 'peak_cuda_bytes': torch.cuda.max_memory_allocated()}
    print(json.dumps({'pilot': pilot}), flush=True)
    verifier.zero_grad(set_to_none=True)
    del pilot_loss
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(verifier.model.parameters(), lr=learning_rate, weight_decay=0.01)
    losses = []
    step = 0
    for epoch in range(epochs):
        order = list(range(len(splits['train'])))
        random.Random(seed + epoch).shuffle(order)
        verifier.train()
        total = 0.
        for start in range(0, len(order), batch_size):
            rows = [splits['train'][index] for index in order[start:start + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss = verifier.loss(model_inputs(rows), [row['target'] for row in rows])
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(verifier.model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(rows)
            step += 1
            if step % 16 == 0:
                print(json.dumps({'epoch': epoch + 1, 'step': step, 'loss': float(loss.detach()), 'elapsed_seconds': time.time() - started}), flush=True)
        losses.append(total / len(order))
    verifier.eval()
    development = predict(verifier, splits['validation'], batch_size)
    fit = verifier.calibration.fit(development, labels_tensor(verifier, splits['validation']))
    test = predict(verifier, splits['test'], batch_size)
    labels = labels_tensor(verifier, splits['test'])
    metrics = {'before_test': evaluate_calibration(before['test'], labels),
               'after_test_uncalibrated': evaluate_calibration(test, labels),
               'after_test_calibrated': evaluate_calibration(verifier.calibration(test), labels),
               'development_temperature_fit': fit,
               'before_development': evaluate_calibration(before['validation'], labels_tensor(verifier, splits['validation']))}
    # The ordinary HF model and tokenizer can initialize another owned verifier.
    verifier.model.save_pretrained(output / 'model')
    verifier.tokenizer.save_pretrained(output / 'model')
    config = verifier.configuration()
    (output / 'verifier_config.json').write_text(json.dumps(config, indent=2))
    save_file({key: value.detach().cpu().contiguous() for key, value in verifier.state_dict().items()}, str(output / 'verifier.safetensors'))
    save_file({'before_test_logits': before['test'], 'after_test_logits': test, 'test_labels': labels,
               'development_logits': development, 'development_labels': labels_tensor(verifier, splits['validation'])}, str(output / 'evaluation.safetensors'))
    diagnostics = authored_diagnostics(verifier)
    # Reload from constructor and ordinary state, with no model/network reads.
    restored = EvidenceVerifier(json.loads((output / 'verifier_config.json').read_text())).to(device)
    restored.load_state_dict(load_file(str(output / 'verifier.safetensors')))
    restored.eval()
    reloaded = predict(restored, splits['test'][:batch_size], batch_size)
    reload_error = float((reloaded - test[:batch_size]).abs().max())
    if not torch.allclose(reloaded, test[:batch_size], atol=1e-5, rtol=1e-5):
        raise RuntimeError('saved verifier reload prediction mismatch')
    manifest.update({'status': 'completed', 'steps': step, 'epoch_train_loss': losses,
                     'pilot': pilot, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
                     'elapsed_seconds': time.time() - started, 'metrics': metrics, 'authored_diagnostics': diagnostics,
                     'reload_max_abs_logit_error': reload_error,
                     'calibration': {'temperature': float(verifier.calibration.temperature), 'sample_count': int(verifier.calibration.sample_count)},
                     'host': platform.node(), 'gpu': torch.cuda.get_device_name(),
                     'versions': {name: metadata.version(name) for name in ('torch', 'transformers', 'huggingface-hub', 'pyarrow', 'tensorcode')}})
    (output / 'report.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'completed': str(output), 'metrics': metrics, 'elapsed_seconds': manifest['elapsed_seconds']}), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-count', type=int, default=1024)
    parser.add_argument('--calibration-count', type=int, default=256)
    parser.add_argument('--test-count', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=20260921)
    parser.add_argument('--max-tokens', type=int, default=192)
    run(**vars(parser.parse_args()))
