"""Learn revisable interpretations from caller-reviewed evidence sequences.

JSONL: {"case_id": "...", "evidence": [{"source_id": "...", "text": "...",
"target": "reviewed hypothesis", "reviewer": "..."}, ...]}. Prediction inputs
omit target/reviewer. Hypotheses are explicitly supplied at collection time.

The learned component is a small bag-of-words classifier over each growing
prefix, not general reasoning. Evidence order/provenance is retained in receipts,
but mean pooling cannot model order, negation, or source reliability reliably.
Probabilities are uncalibrated. Realization is an authored display template.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

import tensorcode as tc
from tensorcode.ops import vec
from tensorcode.ops.vec.encode import tokenize
from tensorcode.training import Trainer, load, load_checkpoint, save_checkpoint


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def read_cases(path, *, labels=None):
    """Validate bounded input, preserving supplied source text and step order."""
    cases, seen = [], set()
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
                if not isinstance(case, dict) or set(case) != {'case_id', 'evidence'}:
                    raise ValueError('expected case_id and evidence')
                case_id = case['case_id']
                if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
                    raise ValueError('case_id must be a unique nonempty string')
                evidence = case['evidence']
                if not isinstance(evidence, list) or not 1 <= len(evidence) <= 100:
                    raise ValueError('evidence must contain 1..100 steps')
                sources = set()
                for step in evidence:
                    required = {'source_id', 'text'} | ({'target', 'reviewer'} if labels is not None else set())
                    if not isinstance(step, dict) or set(step) != required:
                        raise ValueError(f'each step requires exactly {sorted(required)}')
                    if any(not isinstance(step[k], str) or not step[k].strip() for k in required):
                        raise ValueError('step fields must be nonempty strings')
                    if step['source_id'] in sources:
                        raise ValueError('source_id must be unique within a case')
                    if len(step['text']) > 10000:
                        raise ValueError('evidence text exceeds 10000 characters')
                    sources.add(step['source_id'])
                    if labels is not None and step['target'] not in labels:
                        raise ValueError('reviewed target is outside the hypothesis vocabulary')
                seen.add(case_id)
                cases.append(case)
                if len(cases) > 10000:
                    raise ValueError('input exceeds 10000 cases')
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f'{path}:{number}: {error}') from error
    if not cases:
        raise ValueError('input contains no cases')
    return cases


def bindings(manifest):
    """Construct all public operations before collecting or replaying any input."""
    return {
        'evidence': vec.VocabularyEncoder(vocabulary=manifest['vocabulary'], dimensions=manifest['dimensions']),
        'interpretation': vec.Classify(
            nn.Linear(manifest['dimensions'], len(manifest['labels'])), labels=manifest['labels']),
    }


def infer(operations, text):
    return operations['interpretation'](operations['evidence'](text))


def prefixes(case):
    texts = []
    for index, step in enumerate(case['evidence']):
        texts.append(step['text'])
        yield index, step, '\n'.join(texts)


def collect(input_path, artifacts, labels, *, dimensions=24, seed=7):
    labels = list(labels)
    if len(labels) < 2 or len(set(labels)) != len(labels) or any(not isinstance(x, str) or not x.strip() for x in labels):
        raise ValueError('supply at least two unique nonempty hypotheses')
    if dimensions < 1:
        raise ValueError('dimensions must be positive')
    cases = read_cases(input_path, labels=labels)
    # Only training evidence builds the vocabulary. Reviewed targets never enter text.
    vocabulary = sorted({token for case in cases for step in case['evidence'] for token in tokenize(step['text'])})
    manifest = {'labels': labels, 'vocabulary': vocabulary, 'dimensions': dimensions,
                'seed': seed, 'experiences': []}
    torch.manual_seed(seed)
    operations = bindings(manifest)
    artifacts = Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=False)
    save_checkpoint(artifacts / 'initial.json', operations=operations)
    for case in cases:
        for index, step, text in prefixes(case):
            with torch.no_grad(), tc.trace() as session:
                output = infer(operations, text)
            session.supervise(output, step['target'], source=step['reviewer'])
            filename = f'experience-{len(manifest["experiences"]):06d}.json'
            session.save(artifacts / filename, operations=operations, release=True)
            manifest['experiences'].append({'file': filename, 'case_id': case['case_id'], 'step': index})
    # Source IDs and exact original evidence remain available beside portable traces.
    write_json(artifacts / 'evidence.json', cases)
    write_json(artifacts / 'manifest.json', manifest)
    return {'cases': len(cases), 'supervised_steps': len(manifest['experiences'])}


def restore(artifacts, checkpoint='trained.json'):
    artifacts = Path(artifacts)
    manifest = json.loads((artifacts / 'manifest.json').read_text(encoding='utf-8'))
    operations = bindings(manifest)
    load_checkpoint(artifacts / checkpoint, operations=operations)
    return manifest, operations


def train(artifacts, *, epochs=30, lr=.03):
    if epochs < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError('epochs and finite learning rate must be positive')
    artifacts = Path(artifacts)
    manifest, operations = restore(artifacts, 'initial.json')
    experiences = [load(artifacts / row['file'], operations=operations) for row in manifest['experiences']]
    optimizer = torch.optim.Adam([p for operation in operations.values() for p in operation.parameters()], lr=lr)
    trainer = Trainer(operations, optimizer=optimizer)
    losses = []
    for _ in range(epochs):
        losses.append(sum(trainer.step(session) for session in experiences) / len(experiences))
    save_checkpoint(artifacts / 'trained.json', operations=operations, optimizer=optimizer)
    receipt = {'epochs': epochs, 'mean_step_loss_by_epoch': losses,
               'supervision': 'caller-reviewed targets', 'probabilities_calibrated': False}
    write_json(artifacts / 'training.json', receipt)
    return receipt


def predict(input_path, artifacts):
    cases = read_cases(input_path)
    _, operations = restore(artifacts)
    for operation in operations.values():
        operation.eval()
    receipts = []
    with torch.no_grad():
        for case in cases:
            previous = None
            for index, _, text in prefixes(case):
                prediction = infer(operations, text)
                # Interpretation is chosen before the authored language rendering.
                interpretation = prediction.value
                receipts.append({
                    'case_id': case['case_id'], 'step': index,
                    'evidence': case['evidence'][:index + 1],
                    'distribution': dict(zip(prediction.labels, prediction.probabilities.tolist())),
                    'interpretation': interpretation,
                    'revised': previous is not None and previous != interpretation,
                    'realization': f'Current interpretation: {interpretation}.',
                    'probabilities_calibrated': False,
                })
                previous = interpretation
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    capture = stages.add_parser('collect', help='initialize operations and persist reviewed traces')
    capture.add_argument('--input', type=Path, required=True)
    capture.add_argument('--artifacts', type=Path, required=True)
    capture.add_argument('--hypothesis', action='append', required=True)
    capture.add_argument('--dimensions', type=int, default=24)
    capture.add_argument('--seed', type=int, default=7)
    fit = stages.add_parser('train', help='rebuild operations, replay traces, save learned weights')
    fit.add_argument('--artifacts', type=Path, required=True)
    fit.add_argument('--epochs', type=int, default=30)
    fit.add_argument('--lr', type=float, default=.03)
    apply = stages.add_parser('predict', help='load weights into fresh operations and revise interpretations')
    apply.add_argument('--input', type=Path, required=True)
    apply.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        if args.stage == 'collect':
            result = collect(args.input, args.artifacts, args.hypothesis, dimensions=args.dimensions, seed=args.seed)
        elif args.stage == 'train':
            result = train(args.artifacts, epochs=args.epochs, lr=args.lr)
        else:
            result = predict(args.input, args.artifacts)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
