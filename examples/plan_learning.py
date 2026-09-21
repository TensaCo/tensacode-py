"""Learn to rank supplied plans from observed outcomes; never execute a plan.

JSONL rows: id, task, evidence=[{id, text}], plans=[{id, text, outcome,
source}]. Training outcomes must be finite numbers on a shared scale, where
higher is better; source identifies the actual observation. Prediction rows
omit outcome/source, or include both for held-out MSE. Candidate generation,
text pooling, and maximizing predicted outcome are authored policies. This is
not autonomous planning, causal inference, or calibrated confidence.
"""
from __future__ import annotations

import argparse
import json
import re
import math
from pathlib import Path

import torch
from tensorcode import trace, training
from tensorcode.ops.vec.decode import Decode
from tensorcode.ops.vec import latent_codecs
from tensorcode.ops.vec.encode import VocabularyEncoder


def tokenize(text):
    """Authored vocabulary policy matching the mechanical encoder tokenizer."""
    return re.findall(r'\w+|[^\w\s]', text.lower())


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def read_rows(path, *, supervised):
    rows = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not all(nonempty(row.get(k)) for k in ('id', 'task')):
            raise ValueError('Each task needs nonempty id and task strings')
        evidence, plans = row.get('evidence'), row.get('plans')
        if not isinstance(evidence, list) or not isinstance(plans, list) or not plans:
            raise ValueError('Each task needs evidence list and nonempty plans list')
        for items in (evidence, plans):
            if any(not isinstance(item, dict) or not all(nonempty(item.get(k)) for k in ('id', 'text')) for item in items):
                raise ValueError('Evidence and plans need nonempty id/text strings')
            if len({item['id'] for item in items}) != len(items):
                raise ValueError('Evidence and plan IDs must be unique within each list')
        for plan in plans:
            if supervised or 'outcome' in plan or 'source' in plan:
                outcome = plan.get('outcome')
                if isinstance(outcome, bool) or not isinstance(outcome, (int, float)) or not math.isfinite(outcome):
                    raise ValueError('Observed outcomes must be finite numbers')
                if not nonempty(plan.get('source')):
                    raise ValueError('Observed outcomes require a source ID')
        if not supervised and any('outcome' in p for p in plans) and not all('outcome' in p for p in plans):
            raise ValueError('Held-out outcomes must cover every candidate or none')
        rows.append(row)
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Input must contain tasks with unique IDs')
    return rows


def texts(row):
    # Role prefixes distinguish evidence words from candidate-plan words.
    task = ' '.join('task_' + word for word in tokenize(row['task']))
    evidence = ' '.join('evidence_' + word for item in row['evidence'] for word in tokenize(item['text']))
    return [task + ' ' + evidence + ' ' + ' '.join('plan_' + word for word in tokenize(plan['text'])) for plan in row['plans']]


def bindings(manifest):
    """Construct every operation before capture, replay, or inference."""
    width = manifest['dimensions']
    space = {'name': 'application.plan-text', 'dimensions': width}
    return {
        'interpret': VocabularyEncoder({
            'vocabulary': manifest['vocabulary'], 'dimensions': width, 'output_space': space,
        }),
        'anticipate': Decode({
            'architecture': 'mlp', 'input_space': space, 'hidden_dimensions': [width],
            'output_dimensions': 1, 'output': 'outcome',
        }),
    }


def predict(operations, row):
    return operations['anticipate'](operations['interpret'](texts(row)))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def collect(input_path, artifacts, *, dimensions=24, seed=7):
    if dimensions < 1:
        raise ValueError('dimensions must be positive')
    rows = read_rows(input_path, supervised=True)
    manifest = {
        'dimensions': dimensions,
        'vocabulary': sorted({token for row in rows for text in texts(row) for token in tokenize(text)}),
        'training_ids': [row['id'] for row in rows],
        'training_texts': [text for row in rows for text in texts(row)],
        'seed': seed, 'experiences': [],
        'observations': [{ 'task_id': row['id'], 'evidence_ids': [e['id'] for e in row['evidence']],
                           'plans': [{'id': p['id'], 'source': p['source']} for p in row['plans']]} for row in rows],
    }
    root = Path(artifacts)
    root.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    operations = bindings(manifest)
    training.save_checkpoint(root / 'initial.json', operations=operations)
    for i, row in enumerate(rows):
        with torch.no_grad(), trace() as session:
            scores = predict(operations, row)
        session.supervise(scores, torch.tensor([[p['outcome']] for p in row['plans']], dtype=torch.float32),
                          loss='mse', source=json.dumps({'task': row['id'], 'outcomes': [p['source'] for p in row['plans']]}))
        name = f'experience-{i:04d}.json'
        session.save(root / name, operations=operations, codecs=latent_codecs(), release=True)
        manifest['experiences'].append(name)
    # Preserve original evidence and feedback alongside normalized replay inputs.
    write(root / 'observations.json', rows)
    write(root / 'manifest.json', manifest)
    return {'tasks': len(rows), 'observed_outcomes': sum(len(r['plans']) for r in rows)}


def restore(artifacts, checkpoint='trained.json'):
    root = Path(artifacts)
    manifest = json.loads((root / 'manifest.json').read_text())
    operations = bindings(manifest)
    training.load_checkpoint(root / checkpoint, operations=operations)
    return manifest, operations


def train(artifacts, *, epochs=100, lr=0.01):
    if epochs < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError('epochs and learning rate must be positive and finite')
    root = Path(artifacts)
    manifest, operations = restore(root, 'initial.json')
    sessions = [training.load(root / name, operations=operations, codecs=latent_codecs()) for name in manifest['experiences']]
    optimizer = torch.optim.Adam([p for op in operations.values() for p in op.parameters()], lr=lr)
    trainer = training.Trainer(operations, optimizer=optimizer)
    losses = [sum(trainer.step(session) for session in sessions) / len(sessions)
              for _ in range(epochs)]
    training.save_checkpoint(root / 'trained.json', operations=operations, optimizer=optimizer)
    report = {'first_loss': losses[0], 'last_loss': losses[-1], 'updates': epochs * len(sessions),
              'mean_loss_by_epoch': losses}
    write(root / 'training.json', report)
    return report


def evaluate(input_path, artifacts, *, checkpoint='trained.json'):
    rows = read_rows(input_path, supervised=False)
    manifest, operations = restore(artifacts, checkpoint)
    evaluated = [row for row in rows if any('outcome' in plan for plan in row['plans'])]
    if set(manifest['training_ids']) & {row['id'] for row in evaluated} or set(manifest['training_texts']) & {text for row in evaluated for text in texts(row)}:
        raise ValueError('Prediction inputs must be held out from collection (IDs and exact inputs)')
    for op in operations.values():
        op.eval()
    result, errors = [], []
    with torch.no_grad():
        for row in rows:
            # Anticipate the outcome of EVERY supplied plan before choosing any.
            values = predict(operations, row).flatten().tolist()
            if not all(math.isfinite(v) for v in values):
                raise ValueError('Model produced nonfinite predicted outcomes')
            candidates = [{'id': p['id'], 'text': p['text'], 'predicted_outcome': v} for p, v in zip(row['plans'], values)]
            result.append({'task_id': row['id'], 'task': row['task'], 'evidence': row['evidence'], 'candidates': candidates,
                           'selected_plan': max(candidates, key=lambda p: p['predicted_outcome'])['id'],
                           'executed': False, 'uncertainty': 'Uncalibrated regression; scores are not confidence or causal effects.'})
            errors.extend((v - p['outcome']) ** 2 for p, v in zip(row['plans'], values) if 'outcome' in p)
    return {'tasks': result, 'heldout_mse': sum(errors) / len(errors) if errors else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('collect', 'train', 'predict'))
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--dimensions', type=int, default=24)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--checkpoint', choices=('initial.json', 'trained.json'), default='trained.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage in ('collect', 'predict') and args.input is None:
        parser.error('--input is required for collect/predict')
    if args.stage == 'collect':
        report = collect(args.input, args.artifacts, dimensions=args.dimensions, seed=args.seed)
    elif args.stage == 'train':
        report = train(args.artifacts, epochs=args.epochs, lr=args.lr)
    else:
        report = evaluate(args.input, args.artifacts, checkpoint=args.checkpoint)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
