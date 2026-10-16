"""Train owned tools on pinned HotpotQA support annotations, not invented outcomes.

Requires pyarrow, huggingface_hub and torch. Downloads only the first official
training shard and official distractor validation shard. Validation examples are
fixed before fitting and never used for vocabulary, gradients or model selection.
Planner supervision is document relevance, NOT observed utility of executed plans.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re

REVISION = '1908d6afbbead072334abe2965f91bd2709910ab'
DATASET = 'hotpotqa/hotpot_qa'


def prepare_record(row):
    titles, sentences = row['context']['title'], row['context']['sentences']
    if len(titles) != len(sentences) or not titles:
        raise ValueError('context titles and passages must align')
    support = set(row['supporting_facts']['title'])
    targets = [float(title in support) for title in titles]
    if not any(targets):
        raise ValueError('record has no supporting passage')
    candidates = [{'id': f'doc-{i}', 'text': title + '\n' + ' '.join(parts)}
                  for i, (title, parts) in enumerate(zip(titles, sentences))]
    return {'id': row['id'], 'question': row['question'], 'candidates': candidates,
            'targets': targets, 'target': targets.index(1.0)}


def vocabulary(records, limit=20000):
    counts = Counter()
    for record in records:
        for text in [record['question']] + [c['text'] for c in record['candidates']]:
            counts.update(re.findall(r'\w+|[^\w\s]', text.casefold()))
    return [word for word, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def check_splits(train, validation):
    a, b = [r['id'] for r in train], [r['id'] for r in validation]
    if len(set(a)) != len(a) or len(set(b)) != len(b) or set(a) & set(b):
        raise ValueError('training and validation must have unique disjoint question IDs')


def inputs(record, kind):
    # The question is evidence available before reading any candidate document.
    # Candidate passages are supplied options, never added as verified evidence.
    return {('question' if kind == 'investigator' else 'goal'): record['question'],
            'evidence': [],
            ('hypotheses' if kind == 'investigator' else 'plans'): record['candidates']}


def load_records(split, count, *, offset=0):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    filename = f'distractor/{split}-00000-of-0000{2 if split == "train" else 1}.parquet'
    path = Path(hf_hub_download(DATASET, filename, repo_type='dataset', revision=REVISION))
    records = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=min(count, 256)):
        records.extend(prepare_record(row) for row in batch.to_pylist())
        if len(records) >= count + offset:
            break
    if len(records) < count + offset:
        raise ValueError('requested count exceeds available shard')
    return records[offset:count + offset], {'file': filename, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def lexical_baseline(records):
    """Authored token-overlap comparison, not a learned cognitive model."""
    hit = recall = 0.0
    for record in records:
        question = set(re.findall(r'\w+', record['question'].casefold()))
        order = sorted(range(len(record['candidates'])), key=lambda index:
                       -len(question & set(re.findall(r'\w+', record['candidates'][index]['text'].casefold()))))
        hit += record['targets'][order[0]]
        recall += sum(record['targets'][index] for index in order[:2]) / sum(record['targets'])
    return {'method': 'unique casefold alphanumeric token overlap; candidate-order tie break',
            'support_hit_at_1': hit / len(records), 'support_recall_at_2': recall / len(records)}


def evaluate(tool, records, kind, *, zero_workspace=False):
    import torch
    from torch.nn import functional as F
    tool.eval()
    loss = hit1 = recall2 = 0.0
    with torch.no_grad():
        for record in records:
            scores = tool.rank.compute(inputs(record, kind), workspace_ablation='zero' if zero_workspace else None)[0]
            truth = scores.new_tensor(record['targets'])
            loss += float(F.cross_entropy(scores[None], (truth / truth.sum())[None] if tool.config.get('foundation_config') or tool.config.get('encoder_type') == 'foundation' else torch.tensor([record['target']], device=scores.device)) if kind == 'investigator' else F.mse_loss(scores, truth))
            hit1 += record['targets'][int(scores.argmax())]
            recall2 += float(truth[scores.topk(min(2, len(scores))).indices].sum() / truth.sum())
    return {'loss': loss / len(records), 'support_hit_at_1': hit1 / len(records), 'support_recall_at_2': recall2 / len(records)}


def run(output, *, train_count=512, validation_count=128, epochs=8, dimensions=32, seed=17, learning_rate=0.003, foundation=None, foundation_revision=None, validation_offset=0, device="cpu", dev_count=0):
    import torch
    from tensorcode.tools.investigator import Investigator
    from tensorcode.tools.planner import Planner
    torch.set_num_threads(1)
    random.seed(seed)
    torch.manual_seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train, train_source = load_records('train', train_count)
    validation, validation_source = load_records('validation', validation_count, offset=validation_offset) if validation_offset else load_records('validation', validation_count)
    check_splits(train, validation)
    development = load_records('train', dev_count, offset=train_count)[0] if dev_count else []
    if development:
        check_splits(train, development)
        check_splits(development, validation)
    config = {'vocabulary': vocabulary(train), 'dimensions': dimensions, 'slots': 4, 'steps': 2, 'max_tokens': 128}
    manifest = {'dataset': DATASET, 'revision': REVISION, 'train_source': train_source,
                'validation_source': validation_source, 'train_ids': [r['id'] for r in train],
                'validation_ids': [r['id'] for r in validation], 'development_ids': [r['id'] for r in development], 'seed': seed, 'epochs': epochs,
                'learning_rate': learning_rate, 'batch_size': 8, 'config': {k: v for k, v in config.items() if k != 'vocabulary'},
                'foundation': foundation, 'foundation_revision': foundation_revision, 'validation_offset': validation_offset, 'device': device,
                'target_policy': 'uniform over all annotated supporting titles' if foundation else 'first supporting title',
                'vocabulary_size': None if foundation else len(config['vocabulary']), 'vocabulary_source': 'inherited pinned foundation tokenizer' if foundation else 'training subset only',
                'selection': 'first N official rows, fixed before fitting; final epoch, no validation selection',
                'limitations': ['Investigator pilot used first supporting title; foundation runs use uniform support distribution. Hit/recall accept any supporting title.',
                                'Planner targets are human document support annotations, a relevance proxy, not executed action outcomes.',
                                'Candidates are supplied passages; no hypothesis generation, general planning, or general cognition is established.'],
                'lexical_baseline': lexical_baseline(validation), 'results': {}}
    for kind, cls in [('investigator', Investigator), ('planner', Planner)]:
        torch.manual_seed(seed)
        tool = (cls.from_foundation(foundation, revision=foundation_revision, freeze_foundation=True, cache_records=2048, dimensions=dimensions, slots=4, steps=2, max_tokens=128) if foundation else cls(config)).to(device)
        optimizer = torch.optim.AdamW(tool.parameters(), lr=learning_rate)
        before = evaluate(tool, validation, kind)
        history = []
        development_history = []
        for epoch in range(epochs):
            tool.train()
            order = list(range(len(train)))
            random.Random(seed + epoch).shuffle(order)
            total = 0.0
            for start in range(0, len(order), 8):
                batch = [train[i] for i in order[start:start + 8]]
                optimizer.zero_grad()
                loss = torch.stack([tool.loss(inputs(r, kind), ([v / sum(r['targets']) for v in r['targets']] if foundation else r['target']) if kind == 'investigator' else r['targets']) for r in batch]).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tool.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(batch)
            history.append(total / len(train))
            if development:
                development_history.append(evaluate(tool, development, kind))
            print(json.dumps({'tool': kind, 'epoch': epoch + 1, 'training_loss': history[-1]}), flush=True)
        after = evaluate(tool, validation, kind)
        ablation = evaluate(tool, validation, kind, zero_workspace=True)
        folder = output / kind
        tool.save_pretrained(folder)
        restored = cls.from_pretrained(folder).to(device)
        reloaded = evaluate(restored, validation, kind)
        if reloaded != after:
            raise RuntimeError('saved model predictions differ after reload')
        result = {'before': before, 'after': after, 'zero_workspace': ablation, 'reloaded': reloaded, 'training_loss': history, 'development_metrics': development_history}
        manifest['results'][kind] = result
        (folder / 'evaluation.json').write_text(json.dumps(result, indent=2) + '\n')
        (folder / 'README.md').write_text('---\nlibrary_name: tensorcode\ndatasets:\n- hotpotqa/hotpot_qa\ntags:\n- tensorcode\n- experimental\n---\n\n# TensorCode ' + cls.__name__ + ' support relevance prototype\n\nOwned encoder, shared differentiable workspace and candidate scoring head; the manifest identifies inherited foundation weights when used. '
            'This checkpoint ranks supplied HotpotQA documents using human supporting-fact annotations. It is not a general cognitive agent. '
            'Planner scores are relevance proxies, not measured plan utility. Investigator target policy is in the manifest; foundation runs supervise all supporting titles. '
            'Newly built vocabulary and gradients use training questions only; foundation tokenizer assets are inherited; held-out official validation IDs are fixed before fitting. '
            'The final epoch is saved without validation-based selection. Inherited foundation models, if used, are pinned in the manifest.\n\n'
            'Load with `from tensorcode.tools.' + kind + ' import ' + cls.__name__ + '` then `' + cls.__name__ + '.from_pretrained(path_or_repo)`. '
            'Inputs contain question/hypotheses for Investigator or goal/plans for Planner; each candidate has id/text. See the TensorCode example for the complete schema.\n\n'
            '## Measured results\n\n```json\n' + json.dumps(result, indent=2) + '\n```\n\n'
            'Data, split IDs, hashes, hyperparameters and limitations are recorded in training-manifest.json. '
            'Attention is learned routing, not proof of factual support. Scores are uncalibrated. '
            'Compare the zero-workspace ablation and authored lexical baseline before attributing quality to the workspace.\n\n'
            '## Authored lexical comparison\n\n```json\n' + json.dumps(manifest['lexical_baseline'], indent=2) + '\n```\n')
    (output / 'training-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for kind in manifest['results']:
        (output / kind / 'training-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-count', type=int, default=512)
    parser.add_argument('--validation-count', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--dimensions', type=int, default=32)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--learning-rate', type=float, default=0.003)
    parser.add_argument('--foundation')
    parser.add_argument('--foundation-revision')
    parser.add_argument('--validation-offset', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dev-count', type=int, default=0)
    args = parser.parse_args()
    if args.validation_offset < 0 or args.dev_count < 0:
        parser.error('offset and development count must be nonnegative')
    if args.foundation and not args.foundation_revision:
        parser.error('foundation loading requires an explicit pinned revision')
    if min(args.train_count, args.validation_count, args.epochs, args.dimensions) < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('counts, epochs, dimensions and learning rate must be positive')
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
