"""Teach faithful realization of an explicitly supplied selected statement.

This is deliberately a selected-statement copying/realization task, NOT answer
inference. A human QA2D declaration appears in the legitimate selected-hypothesis
input field and is also the decoder target. Original question and source context
are retained. Only existing QA2D train/dev document partitions are read; test is
never opened. Full model training/evaluation require the authorized CUDA host.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import re
import time

FOUNDATION_REVISION = '7bcac572ce56db69c1ea7c8af255c5d7c9672fc2'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interpretation(row):
    if row.get('target_origin') != 'QA2D.turker_answer' or not isinstance(row.get('target'), str) or not row['target'].strip():
        raise ValueError('explicit human selected statement required')
    return {'selected_id': 'selected-statement',
            'candidates': [{'id': 'selected-statement', 'text': row['target']}],
            'evidence': [{'id': item['source_id'], 'source_id': item['source_id'], 'text': item['text']}
                         for item in row['evidence']]}


def realization_input(model, row):
    from tensorcode.tools.chatbot import Chatbot
    return Chatbot._realization_input(model, row['question'], interpretation(row))


def validate_tokens(model, records, *, target_limit):
    if type(target_limit) is not int or target_limit < 1:
        raise ValueError('positive target token limit required')
    source_truncated = 0
    for row in records:
        if len(model.tokenizer(row['target'], truncation=False)['input_ids']) > target_limit:
            raise ValueError(f'target would be truncated for {row["id"]}')
        prompt, _, truncation = realization_input(model, row)
        if (len(model.tokenizer(prompt, truncation=False)['input_ids']) > model.config['max_input_tokens']
                or row['target'] not in prompt):
            raise ValueError(f'input would lose selected statement for {row["id"]}')
        source_truncated += bool(truncation)
    return {'target_truncated_count': 0, 'input_truncated_count': 0,
            'source_context_truncated_count': source_truncated}


def select_records(records, count, *, seed):
    if type(count) is not int or count < 1 or len(records) < count:
        raise ValueError('requested count exceeds available records or is invalid')
    if len({row['id'] for row in records}) != len(records):
        raise ValueError('duplicate source IDs')
    groups = defaultdict(list)
    for row in records:
        groups[row['document_id']].append(row)
    rng = random.Random(seed)
    titles = sorted(groups)
    rng.shuffle(titles)
    pools = []
    for title in titles:
        pool = sorted(groups[title], key=lambda row: row['id'])
        rng.shuffle(pool)
        pools.append(pool)
    selected = []
    while len(selected) < count:
        for pool in pools:
            if pool and len(selected) < count:
                selected.append(pool.pop())
    rng.shuffle(selected)
    return selected


def check_splits(train, dev):
    for key in ('id', 'document_id'):
        if {row[key] for row in train} & {row[key] for row in dev}:
            raise ValueError(f'train/dev {key} overlap')
    if {item['text'] for row in train for item in row['evidence']} & {item['text'] for row in dev for item in row['evidence']}:
        raise ValueError('train/dev source context overlap')


def preservation_scores(prediction, target):
    predicted, expected = re.findall(r'\w+', prediction.casefold()), re.findall(r'\w+', target.casefold())
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    return {'verbatim_exact': float(prediction.strip() == target.strip()),
            'statement_exact': float(predicted == expected),
            'token_f1': 2 * overlap / (len(predicted) + len(expected)) if predicted or expected else 1.}


def evaluate(model, records, batch_size):
    model.eval()
    predictions = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        outputs = model.generate_batch([realization_input(model, row)[0] for row in batch])
        predictions.extend({'id': row['id'], 'selected_statement': row['target'], 'prediction': output,
                            **preservation_scores(output, row['target'])} for row, output in zip(batch, outputs))
    return {'sample_count': len(records), 'nonempty_count': sum(bool(row['prediction'].strip()) for row in predictions),
            **{name: sum(row[name] for row in predictions) / len(records)
               for name in ('verbatim_exact', 'statement_exact', 'token_f1')}, 'predictions': predictions}


def model_card(*, foundation, revision, epochs):
    """Describe the configured run without assigning an unknown source license."""
    return f'''---
library_name: tensorcode
pipeline_tag: text2text-generation
---
# Faithful selected-statement realization

Native TensorCode Chatbot initialized from `{foundation}` at revision `{revision}` with an owned
workspace. Prescribed schedule: {epochs} epochs. The selected human QA2D declaration is deliberately supplied in the
production realization prompt and is the decoder target. This learns statement
preservation/copying; it is not target-blind answer inference or fact verification.
Question and original source paragraph are also supplied. Training updates the
foundation and workspace. Use as the language realization stage of an owned
cognitive Chatbot; hypothesis generation and verification remain separate roles.

See evaluation.json for train/dev article separation, source IDs and hashes,
model/source revisions, token truncation checks, all development predictions and
native full model/optimizer restart checks. The prescribed {epochs} epochs are used. The test partition is never opened. Development scores measure preservation
of known statements, not factual correctness or final end-to-end accuracy. The
original native baseline is separately preserved in the initial artifact.

Foundation licensing follows the selected source. Original SQuAD sources are CC-BY-SA-4.0,
and the QA2D mirror declares MIT. Dataset/paper attribution and pinned revisions
are in evaluation.json. Human target errors can be preserved by this model.
'''


def make_optimizer(model, learning_rate, workspace_lr):
    import torch
    other = [parameter for name, parameter in model.named_parameters() if not name.startswith('foundation.')]
    return torch.optim.AdamW([{'params': model.foundation.parameters(), 'lr': learning_rate},
                              {'params': other, 'lr': workspace_lr}])


def run(args):
    import torch
    from tensorcode.tools.chatbot import Chatbot
    from tensorcode.training import Trainer
    if args.device != 'cuda' or not torch.cuda.is_available():
        raise ValueError('full-model realization training requires authorized CUDA host')
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    data = Path(args.data)
    manifest = json.loads((data / 'manifest.json').read_text())
    splits = {}
    # Deliberately never open test.jsonl, including for preprocessing or diagnosis.
    for split, count in [('train', args.train_count), ('dev', args.dev_count)]:
        path = data / f'{split}.jsonl'
        if sha256(path) != manifest['splits'][split]['sha256']:
            raise ValueError('source split checksum mismatch')
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        splits[split] = select_records(rows, count, seed=args.seed)
    check_splits(splits['train'], splits['dev'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    model = Chatbot.from_foundation(args.foundation, revision=args.revision, local_files_only=args.local_files_only,
                                    max_input_tokens=args.max_input_tokens, max_target_tokens=args.max_target_tokens,
                                    max_new_tokens=args.max_target_tokens, workspace={'slots': 8, 'steps': 2}).to(args.device)
    token_checks = {split: validate_tokens(model, records, target_limit=args.max_target_tokens)
                    for split, records in splits.items()}
    optimizer = make_optimizer(model, args.learning_rate, args.workspace_lr)
    trainer = Trainer.from_tool(model, optimizer=optimizer)
    report = {'task': 'faithful realization of supplied selected human statement; not target-blind QA inference',
              'input_contract': 'production Chatbot._realization_input(question, interpretation); selected statement is intentionally the decoder target',
              'foundation': {'source': args.foundation, 'revision': args.revision},
              'schedule': {'epochs': args.epochs, 'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
                           'workspace_lr': args.workspace_lr, 'seed': args.seed},
              'source_provenance': {key: manifest[key] for key in ('qa2d', 'squad', 'files')},
              'data_manifest_sha256': sha256(data / 'manifest.json'),
              'selection_policy': 'seeded article-round-robin subsets of existing disjoint train/dev partitions',
              'splits': {}, 'token_checks': token_checks,
              'token_limits': {'input': args.max_input_tokens, 'target': args.max_target_tokens},
              'script_sha256': sha256(__file__),
              'chatbot_source_sha256': sha256(Path(__file__).parents[1] / 'src/tensorcode/tools/chatbot.py'),
              'host': platform.node(), 'gpu': torch.cuda.get_device_name(),
              'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'tensorcode')},
              'epoch_losses': [],
              'limitations': ['Selected statement is explicitly supplied; this does not measure finding a correct answer.',
                              'Human declarations can contain errors; this task learns to preserve them, not verify them.',
                              'No test partition or final end-to-end holdout is opened or evaluated by this script.',
                              f'Development is diagnostic for the prescribed {args.epochs} epochs, not an unseen final test.',
                              'Prior foundation exposure to underlying SQuAD sources cannot be excluded.']}
    for split, rows in splits.items():
        path = output / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        report['splits'][split] = {'count': len(rows), 'sha256': sha256(path), 'ids': [row['id'] for row in rows],
                                   'document_count': len({row['document_id'] for row in rows}),
                                   'documents': sorted({row['document_id'] for row in rows})}
    model.save_pretrained(output / 'initial')
    print('Evaluating faithful realization baseline on development only', flush=True)
    report['before_development'] = evaluate(model, splits['dev'], args.batch_size)
    (output / 'before.json').write_text(json.dumps(report, indent=2))
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(splits['train'])))
        random.shuffle(order)
        total = 0.
        for start in range(0, len(order), args.batch_size):
            batch = [splits['train'][index] for index in order[start:start + args.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss_batch([realization_input(model, row)[0] for row in batch], [row['target'] for row in batch])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite realization training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            trainer.steps += 1
            total += float(loss.detach()) * len(batch)
            if trainer.steps % 16 == 0:
                print(json.dumps({'epoch': epoch + 1, 'step': trainer.steps, 'loss': float(loss.detach())}), flush=True)
        report['epoch_losses'].append(total / len(order))
    report['after_development'] = evaluate(model, splits['dev'], args.batch_size)
    model.save_pretrained(output / 'model')
    trainer.save_checkpoint(output / 'training', progress={'epochs': args.epochs,
                            'data_manifest_sha256': report['data_manifest_sha256'],
                            'train_subset_sha256': report['splits']['train']['sha256']})
    probe = [realization_input(model, splits['dev'][0])[0]]
    expected = model.generate_batch(probe)
    restored = Chatbot.from_pretrained(output / 'model', device=args.device)
    restored_trainer = Trainer.from_tool(restored, optimizer=make_optimizer(restored, args.learning_rate, args.workspace_lr))
    progress = restored_trainer.load_checkpoint(output / 'training')
    report['reload_generation_equal'] = expected == restored.generate_batch(probe)
    report['full_checkpoint'] = {'steps': restored_trainer.steps, 'optimizer_state_entries': len(restored_trainer.optimizer.state),
                                  'progress': progress, 'steps_equal': restored_trainer.steps == trainer.steps}
    if not report['reload_generation_equal'] or not report['full_checkpoint']['steps_equal']:
        raise RuntimeError('native realization checkpoint verification failed')
    report['elapsed_seconds'] = time.time() - started
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    (output / 'model' / 'evaluation.json').write_text(json.dumps(report, indent=2))
    (output / 'model' / 'README.md').write_text(model_card(
        foundation=args.foundation, revision=args.revision, epochs=args.epochs))
    print(json.dumps({'completed': str(output), 'before': {k: v for k, v in report['before_development'].items() if k != 'predictions'},
                      'after': {k: v for k, v in report['after_development'].items() if k != 'predictions'},
                      'elapsed_seconds': report['elapsed_seconds']}), flush=True)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--data', required=True)
    result.add_argument('--output', required=True)
    result.add_argument('--foundation', required=True)
    result.add_argument('--revision', default=FOUNDATION_REVISION)
    result.add_argument('--local-files-only', action='store_true')
    result.add_argument('--train-count', type=int, default=256)
    result.add_argument('--dev-count', type=int, default=64)
    result.add_argument('--epochs', type=int, default=3)
    result.add_argument('--batch-size', type=int, default=8)
    result.add_argument('--learning-rate', type=float, default=3e-5)
    result.add_argument('--workspace-lr', type=float, default=1e-3)
    result.add_argument('--seed', type=int, default=20260921)
    result.add_argument('--max-input-tokens', type=int, default=1024)
    result.add_argument('--max-target-tokens', type=int, default=96)
    result.add_argument('--device', default='cuda')
    return result


if __name__ == '__main__':
    args = parser().parse_args()
    if min(args.epochs, args.batch_size, args.max_input_tokens, args.max_target_tokens, args.learning_rate, args.workspace_lr) <= 0:
        raise SystemExit('counts and learning rates must be positive')
    run(args)
