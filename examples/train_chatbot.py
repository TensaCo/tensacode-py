"""Fine-tune an owned Chatbot on explicit JSONL input/target/source records.

The prescribed train schedule never selects checkpoints using held-out scores.
A new workspace is initialized over an explicitly selected seq2seq foundation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import string
import time


def read_records(path):
    records = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not records:
        raise ValueError('Dataset must contain records')
    for row in records:
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip()
                                           for key in ('id', 'input', 'target')):
            raise ValueError('Every record requires nonempty string id, input, target')
    if len({row['id'] for row in records}) != len(records):
        raise ValueError('Duplicate source IDs')
    return records


def normalize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def scores(prediction, target):
    predicted, expected = normalize(prediction), normalize(target)
    a, b = predicted.split(), expected.split()
    common = sum((Counter(a) & Counter(b)).values())
    f1 = 2 * common / (len(a) + len(b)) if a and b else float(a == b)
    return {'exact_match': float(predicted == expected), 'token_f1': f1}


def evaluate(model, records, batch_size, *, ablation=None):
    import torch
    predictions, losses = [], []
    model.eval()
    for offset in range(0, len(records), batch_size):
        batch = records[offset:offset + batch_size]
        inputs, targets = [r['input'] for r in batch], [r['target'] for r in batch]
        with torch.no_grad():
            loss = model.loss_batch(inputs, targets, workspace_ablation=ablation)
        losses.append((float(loss), len(batch)))
        outputs = model.generate_batch(inputs, workspace_ablation=ablation)
        predictions.extend({'id': row['id'], 'target': row['target'], 'prediction': output,
                            **scores(output, row['target'])} for row, output in zip(batch, outputs))
    return {'examples': len(records),
            'mean_batch_cross_entropy': sum(loss * n for loss, n in losses) / len(records),
            'exact_match': sum(row['exact_match'] for row in predictions) / len(records),
            'token_f1': sum(row['token_f1'] for row in predictions) / len(records),
            'predictions': predictions}


def run(args):
    import torch
    from tensorcode.tools.chatbot import Chatbot
    from tensorcode.training import Trainer
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    train, test = read_records(args.train), read_records(args.test)
    if {r['id'] for r in train} & {r['id'] for r in test}:
        raise ValueError('Train/test source IDs overlap')
    if {r['input'] for r in train} & {r['input'] for r in test}:
        raise ValueError('Train/test inputs overlap')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be new or empty')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model = Chatbot.from_foundation(args.foundation, revision=args.revision,
                                   local_files_only=args.local_files_only,
                                   max_input_tokens=args.max_input_tokens,
                                   max_target_tokens=args.max_target_tokens,
                                   max_new_tokens=args.max_target_tokens,
                                   workspace={'slots': args.slots, 'steps': 2}).to(args.device)
    initial_gate = float(model.memory_gate.detach())
    report = {'seed': args.seed, 'foundation': args.foundation, 'revision': args.revision,
              'schedule': {'epochs': args.epochs, 'batch_size': args.batch_size,
                           'foundation_lr': args.lr, 'workspace_lr': args.workspace_lr},
              'data': {name: {'path': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                               'source_ids': [r['id'] for r in records]}
                       for name, path, records in [('train', args.train, train), ('test', args.test, test)]}}
    print('Evaluating fixed held-out split before training', flush=True)
    report['before'] = evaluate(model, test, args.batch_size)
    manifest = Path(args.train).parent / 'chat-data-manifest.json'
    if manifest.is_file():
        report['data_provenance'] = json.loads(manifest.read_text(encoding='utf-8'))
    model.save_pretrained(output / 'initial')
    other = [p for name, p in model.named_parameters() if not name.startswith('foundation.')]
    optimizer = torch.optim.AdamW([{'params': model.foundation.parameters(), 'lr': args.lr},
                                   {'params': other, 'lr': args.workspace_lr}])
    trainer = Trainer.from_tool(model, optimizer=optimizer)
    # Capture one explicit, portable feedback batch to demonstrate durable replay.
    sample = train[:args.batch_size]
    experience = trainer.capture([r['input'] for r in sample], [r['target'] for r in sample],
                                 source='train JSONL ids: ' + ','.join(r['id'] for r in sample))
    experience.save(output / 'experience.json', operations=trainer.operations)
    report['epoch_losses'] = []
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(train)))
        random.shuffle(order)
        total = 0.0
        for offset in range(0, len(order), args.batch_size):
            batch = [train[i] for i in order[offset:offset + args.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss_batch([r['input'] for r in batch], [r['target'] for r in batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            trainer.steps += 1
            total += float(loss.detach()) * len(batch)
        report['epoch_losses'].append(total / len(train))
        print(json.dumps({'epoch': epoch + 1, 'training_loss': report['epoch_losses'][-1]}), flush=True)
    report['after'] = evaluate(model, test, args.batch_size)
    report['bypass'] = evaluate(model, test, args.batch_size, ablation='bypass')
    report['zero'] = evaluate(model, test, args.batch_size, ablation='zero')
    report['memory_gate'] = {'before': initial_gate, 'after': float(model.memory_gate.detach())}
    report['elapsed_seconds'] = time.monotonic() - started
    model.save_pretrained(output / 'model')
    # Persist optimizer/RNG separately. The JSON representation is intentionally
    # explicit and can be large; do not include it in the Hub weight package.
    if args.training_checkpoint:
        trainer.save_checkpoint(output / 'training', progress={'epochs': args.epochs})
    restored = Chatbot.from_pretrained(output / 'model', device=args.device)
    probe = [test[0]['input']]
    report['reload_generation_equal'] = model.generate_batch(probe) == restored.generate_batch(probe)
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output / 'model' / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    card = f'''---
library_name: tensorcode
pipeline_tag: text2text-generation
base_model: {args.foundation}
license: apache-2.0
---
# TensorCode evidence-conditioned chatbot

Owned sequence encoder, relational slot workspace, and language decoder. Initialized
from `{args.foundation}` at `{args.revision}` and fine-tuned using the explicit data
hashes and source IDs in evaluation.json. Schedule: {args.epochs} epochs; seed {args.seed}.

This checkpoint is a narrow evidence-conditioned QA demonstration, not evidence of
general autonomous cognition. Foundation instruction behavior is inherited; training
updates both foundation and workspace. Data preparation may supply oracle supporting
passages: retrieval competence is not established by this experiment. The model does
not independently verify its generated claims. Workspace bypass and zero-evidence
ablations are reported separately; zero ablation alone does not show workspace utility.

Held-out exact match: {report['after']['exact_match']:.4f}; token F1:
{report['after']['token_f1']:.4f}. Workspace-bypassed exact match:
{report['bypass']['exact_match']:.4f}. See evaluation.json for every prediction,
before/after metrics, source provenance and limitations. Published weights contain no
runtime conversation; training examples must be licensed and reviewed by their supplier.

```python
from tensorcode.tools.chatbot import Chatbot
model = Chatbot.from_pretrained("PATH_OR_HUB_ID")
answer = model.generate_batch(["Question: ...\\nEvidence: ..."])[0]
```
'''
    (output / 'model' / 'README.md').write_text(card, encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items()
                      if key in ('epoch_losses', 'elapsed_seconds', 'memory_gate', 'reload_generation_equal')}), flush=True)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--train', required=True)
    result.add_argument('--test', required=True)
    result.add_argument('--output', required=True)
    result.add_argument('--foundation', default='google/flan-t5-small')
    result.add_argument('--revision', default='0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab')
    result.add_argument('--local-files-only', action='store_true')
    result.add_argument('--epochs', type=int, default=3)
    result.add_argument('--batch-size', type=int, default=4)
    result.add_argument('--lr', type=float, default=3e-5)
    result.add_argument('--workspace-lr', type=float, default=1e-3)
    result.add_argument('--seed', type=int, default=7)
    result.add_argument('--device', default='cpu')
    result.add_argument('--threads', type=int, default=4)
    result.add_argument('--max-input-tokens', type=int, default=512)
    result.add_argument('--max-target-tokens', type=int, default=64)
    result.add_argument('--slots', type=int, default=8)
    result.add_argument('--training-checkpoint', action='store_true')
    return result


if __name__ == '__main__':
    args = parser().parse_args()
    if min(args.epochs, args.batch_size, args.threads, args.max_input_tokens,
           args.max_target_tokens, args.slots) <= 0 or min(args.lr, args.workspace_lr) <= 0:
        raise SystemExit('Counts and learning rates must be positive')
    run(args)
