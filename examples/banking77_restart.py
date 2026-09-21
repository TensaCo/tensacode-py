"""Persist supervised Banking77 traces, train after process exit, then evaluate.

Run with official train/test CSV files. Artifacts stay in --artifacts; this script
never downloads data. Each stage is a separate, sequentially terminated process.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import torch
from torch import nn
import tensorcode as tc
from tensorcode.ops.vec import Classify
from tensorcode.ops.vec.encode import VocabularyEncoder


def tokenize(text):
    """Authored vocabulary policy matching the mechanical encoder tokenizer."""
    return re.findall(r'\w+|[^\w\s]', text.lower())


def read(path):
    with path.open(newline='') as stream:
        return [(row['text'], row['category']) for row in csv.DictReader(stream)]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def bindings(manifest):
    return {
        'encode': VocabularyEncoder(vocabulary=manifest['vocabulary'], dimensions=manifest['dimensions']),
        'classify': Classify(nn.Linear(manifest['dimensions'], len(manifest['labels'])), labels=manifest['labels']),
    }


def predict(operations, texts):
    return operations['classify'](operations['encode'](texts))


def stage(args):
    from tensorcode.training import Trainer, load, load_checkpoint, save_checkpoint

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    root = args.artifacts
    started = time.perf_counter()
    if args.stage == 'capture':
        training, testing = read(args.train), read(args.test)
        heldout = {text.strip().lower() for text, _ in testing}
        original_count = len(training)
        training = [(text, label) for text, label in training if text.strip().lower() not in heldout]
        excluded = original_count - len(training)
        labels = sorted({label for _, label in training})
        # Vocabulary uses only official training text; held-out labels never guide fitting.
        vocabulary = sorted({token for text, _ in training for token in tokenize(text)})
        manifest = {
            'vocabulary': vocabulary, 'labels': labels, 'dimensions': args.dimensions,
            'seed': args.seed, 'batch_size': args.batch_size, 'train_rows': len(training),
            'test_rows': len(testing), 'excluded_overlap_rows': excluded,
            'train_sha256': hashlib.sha256(args.train.read_bytes()).hexdigest(),
            'test_sha256': hashlib.sha256(args.test.read_bytes()).hexdigest(),
        }
        operations = bindings(manifest)
        save_checkpoint(root / 'initial.json', operations=operations)
        random.Random(args.seed).shuffle(training)
        files = []
        for start in range(0, len(training), args.batch_size):
            batch = training[start:start + args.batch_size]
            with torch.no_grad(), tc.trace() as session:
                output = predict(operations, tuple(text for text, _ in batch))
            session.supervise(output, tuple(label for _, label in batch), source='Banking77 official training labels')
            filename = f'experience-{len(files):04d}.json'
            session.save(root / filename, operations=operations, release=True)
            files.append(filename)
        manifest['experiences'] = files
        write(root / 'manifest.json', manifest)
        result = {'experiences': len(files), 'corrections': len(training)}
    else:
        manifest = json.loads((root / 'manifest.json').read_text())
        operations = bindings(manifest)
        if args.stage == 'train':
            optimizer = torch.optim.Adam([p for op in operations.values() for p in op.parameters()], lr=args.lr)
            load_checkpoint(root / 'initial.json', operations=operations)
            experiences = [load(root / name, operations=operations) for name in manifest['experiences']]
            trainer = Trainer(operations, optimizer=optimizer)
            rng = random.Random(args.seed)
            losses = []
            for _ in range(args.epochs):
                rng.shuffle(experiences)
                losses.append(sum(trainer.step(session) for session in experiences) / len(experiences))
            save_checkpoint(root / 'trained.json', operations=operations, optimizer=optimizer)
            result = {'epochs': args.epochs, 'mean_batch_loss_by_epoch': losses, 'loaded_experiences': len(experiences)}
        else:
            checkpoint = 'initial.json' if args.stage == 'before' else 'trained.json'
            load_checkpoint(root / checkpoint, operations=operations)
            for op in operations.values():
                op.eval()
            testing = read(args.test)
            if hashlib.sha256(args.test.read_bytes()).hexdigest() != manifest['test_sha256']:
                raise ValueError('Held-out data changed between stages')
            lookup = {label: index for index, label in enumerate(manifest['labels'])}
            correct, total_loss = 0, 0.0
            with torch.no_grad():
                for start in range(0, len(testing), 256):
                    batch = testing[start:start + 256]
                    output = predict(operations, tuple(text for text, _ in batch))
                    target = torch.tensor([lookup[label] for _, label in batch])
                    correct += int((output.logits.argmax(-1) == target).sum())
                    total_loss += nn.functional.cross_entropy(output.logits, target, reduction='sum').item()
            result = {'accuracy': correct / len(testing), 'cross_entropy': total_loss / len(testing), 'rows': len(testing)}
    result.update(stage=args.stage, pid=os.getpid(), seconds=time.perf_counter() - started)
    write(root / f'{args.stage}-receipt.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dimensions', type=int, default=96)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--stage', choices=('capture', 'before', 'train', 'after'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.epochs, args.dimensions, args.batch_size) < 1 or args.lr <= 0:
        parser.error('epochs, dimensions, batch size, and learning rate must be positive')
    if args.stage:
        stage(args)
        return
    args.artifacts.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), '--train', str(args.train.resolve()),
               '--test', str(args.test.resolve()), '--artifacts', str(args.artifacts.resolve()),
               '--epochs', str(args.epochs), '--dimensions', str(args.dimensions),
               '--batch-size', str(args.batch_size), '--seed', str(args.seed), '--lr', str(args.lr)]
    receipts = []
    for name in ('capture', 'before', 'train', 'after'):
        completed = subprocess.run(command + ['--stage', name], check=True)
        receipt = json.loads((args.artifacts / f'{name}-receipt.json').read_text())
        receipt['returncode'] = completed.returncode
        receipt['terminated_before_next_stage'] = True
        receipts.append(receipt)
        print(json.dumps(receipt), flush=True)
    manifest = json.loads((args.artifacts / 'manifest.json').read_text())
    report = {
        'dataset': 'Banking77 official CSV splits',
        'source': 'https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data',
        **{key: value for key, value in manifest.items() if key not in ('vocabulary', 'labels', 'experiences')},
        'vocabulary_size': len(manifest['vocabulary']), 'label_count': len(manifest['labels']),
        'epochs': args.epochs, 'optimizer': 'Adam', 'learning_rate': args.lr,
        'experience_files': len(manifest['experiences']), 'processes': receipts,
        'before': {key: receipts[1][key] for key in ('accuracy', 'cross_entropy')},
        'after': {key: receipts[3][key] for key in ('accuracy', 'cross_entropy')},
        'artifacts_bytes': sum(path.stat().st_size for path in args.artifacts.iterdir() if path.is_file()),
        'command': command, 'tensorcode': tc.__version__, 'torch': torch.__version__,
        'limitations': 'Supplied ground-truth labels and authored tokenizer/mean pooling. No pretrained model, autonomous discovery, or calibration claim. Fixed settings; held-out split used only for before/after evaluation. Session replay differentiates these local torch operations, not arbitrary Python or remote services.',
    }
    write(args.artifacts / 'results.json', report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
