"""Train a real text decision pipeline and evaluate a fixed held-out split.

Download the official CSVs from https://github.com/PolyAI-LDN/task-specific-datasets
or pass existing files. Nothing is downloaded or sent to a model by this script.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
import tensorcode as tc
from tensorcode.ops.vec import TextEncoder, Classify
from tensorcode.ops.vec.encode import tokenize
from tensorcode.tools.decision import Decision


def read(path):
    with path.open(newline='') as stream:
        return [(r['text'], r['category']) for r in csv.DictReader(stream)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True, type=Path)
    parser.add_argument('--test', required=True, type=Path)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    torch.set_num_threads(2)
    torch.manual_seed(7)
    rng = random.Random(7)
    training, testing = read(args.train), read(args.test)
    # Exclude any literal overlap without examining held-out labels.
    heldout_text = {text.strip().lower() for text, _ in testing}
    original_count = len(training)
    training = [(x, y) for x, y in training if x.strip().lower() not in heldout_text]
    labels = tuple(sorted({y for _, y in training}))
    lookup = {label: i for i, label in enumerate(labels)}
    vocabulary = tuple(sorted({word for text, _ in training for word in tokenize(text)}))
    encoder = TextEncoder(vocabulary=vocabulary, dimensions=96)
    classifier = Classify(nn.Linear(96, len(labels)), labels=labels)
    tool = Decision(encode=encoder, decide=classifier)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=.01)

    def evaluate():
        total_loss, correct = 0., 0
        encoder.eval()
        classifier.eval()
        with torch.no_grad():
            for start in range(0, len(testing), 256):
                batch = testing[start:start+256]
                target = torch.tensor([lookup[y] for _, y in batch])
                prediction = tool(tuple(x for x, _ in batch))
                total_loss += nn.functional.cross_entropy(prediction.logits, target, reduction='sum').item()
                correct += int((prediction.logits.argmax(-1) == target).sum())
        return {'accuracy': correct / len(testing), 'cross_entropy': total_loss / len(testing)}

    before = evaluate()
    started = time.perf_counter()
    for epoch in range(args.epochs):
        encoder.train()
        classifier.train()
        rng.shuffle(training)
        for start in range(0, len(training), 128):
            batch = training[start:start+128]
            optimizer.zero_grad()
            with tc.trace() as episode:
                prediction = tool(tuple(x for x, _ in batch))
            target = torch.tensor([lookup[y] for _, y in batch])
            loss = nn.functional.cross_entropy(prediction.logits, target)
            loss.backward()
            optimizer.step()
        print(f'epoch {epoch+1}/{args.epochs}: final training batch loss {loss.item():.4f}', flush=True)
    seconds = time.perf_counter() - started
    after = evaluate()
    example = episode.example(episode.ref(prediction))
    # Replay uses the current parameters and roots; it does not re-use saved logits.
    replayed = episode.replay(example.target)
    assert replayed.logits.requires_grad
    report = {
        'dataset': 'Banking77 official CSV splits',
        'source': 'https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data',
        'train_sha256': hashlib.sha256(args.train.read_bytes()).hexdigest(),
        'test_sha256': hashlib.sha256(args.test.read_bytes()).hexdigest(),
        'train_rows': len(training), 'test_rows': len(testing),
        'excluded_overlap_rows': original_count-len(training), 'labels': len(labels),
        'vocabulary': len(vocabulary), 'seed': 7, 'epochs': args.epochs,
        'dimensions': 96, 'optimizer': 'Adam', 'learning_rate': .01,
        'before': before, 'after': after, 'training_seconds': seconds,
        'last_trace_calls': len(example.calls), 'last_trace_external_leaves': len(example.inputs),
        'tensorcode': tc.__version__, 'torch': torch.__version__,
        'limitations': 'Authored tokenizer and pooling; supervised labels; no calibration guarantee, pretrained semantics, or autonomous agent claim. Test split used only for fixed before/after evaluation.',
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered+'\n')


if __name__ == '__main__':
    main()
