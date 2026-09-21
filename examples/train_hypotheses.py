"""Train declarative hypotheses from original SQuAD context and human QA2D labels.

Only the joined original question and paragraph enter the production proposal
prompt. Short answers and QA2D rule-based outputs never enter model inputs.
Document-disjoint evaluation isolates this run, not FLAN's prior benchmark exposure.
Human declarations can be noisy; separate NLI scores are model judgments, not facts.
Substantial training and generation require the authorized CUDA training host.
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
import urllib.request

QA2D = 'domenicrosati/QA2D'
QA2D_REVISION = 'd38d3f42978e72c8c3ccc5dca0d3a2ac745f1fcf'
SQUAD_REVISION = 'eee5fdbf62f8613a7812b03419e6b29617b74fd1'
FOUNDATION_REVISION = '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def question_key(text):
    return ''.join(re.findall(r'\w+', text.casefold()))


def squad_index(sources):
    result = {}
    for source in sources:
        for article in source['data']:
            for paragraph in article['paragraphs']:
                for question in paragraph['qas']:
                    identifier = question['id']
                    if identifier in result:
                        raise ValueError('duplicate original source ID')
                    result[identifier] = {'question': question['question'], 'context': paragraph['context'],
                                          'document_id': article['title']}
    return result


def join_record(row, originals, source_split):
    if row['dataset'] != 'SQuAD':
        return None
    identifier = row['example_uid']
    if identifier not in originals:
        raise ValueError(f'original source missing for {identifier}')
    original = originals[identifier]
    if question_key(original['question']) != question_key(row['question']):
        raise ValueError(f'question mismatch for {identifier}')
    target = row.get('turker_answer')
    if not isinstance(target, str) or not target.strip():
        raise ValueError('nonempty human declaration required')
    if not original['context'].strip() or not original['question'].strip():
        raise ValueError('original source context and question must be nonempty')
    context_id = hashlib.sha256(original['context'].encode()).hexdigest()
    return {'id': identifier, 'document_id': original['document_id'], 'question': original['question'],
            'evidence': [{'source_id': 'squad:' + context_id[:16], 'text': original['context']}],
            'target': target.strip(), 'target_origin': 'QA2D.turker_answer', 'qa2d_split': source_split}


def model_input(record):
    from tensorcode._internal.proposals import proposal_prompt
    return proposal_prompt({'question': record['question'], 'evidence': record['evidence']}, 'question')


def check_splits(splits):
    previous_ids, previous_documents, previous_contexts = set(), set(), set()
    for records in splits.values():
        ids = {r['id'] for r in records}
        documents = {r['document_id'] for r in records}
        contexts = {e['text'] for r in records for e in r['evidence']}
        if len(ids) != len(records) or ids & previous_ids or documents & previous_documents or contexts & previous_contexts:
            raise ValueError('source ID, document or context overlap between splits')
        previous_ids.update(ids)
        previous_documents.update(documents)
        previous_contexts.update(contexts)


def document_splits(records, *, train_count=1024, dev_count=128, test_count=128, seed=20260921):
    counts = {'train': train_count, 'dev': dev_count, 'test': test_count}
    if any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError('split counts must be positive integers')
    groups = defaultdict(list)
    seen = set()
    for record in records:
        if record['id'] in seen:
            continue
        seen.add(record['id'])
        groups[record['document_id']].append(record)
    titles = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(titles)
    if len(titles) < 3:
        raise ValueError('at least three original documents are required')
    heldout_titles = max(1, len(titles) // 10)
    partitions = {'train': titles[2 * heldout_titles:],
                  'dev': titles[:heldout_titles], 'test': titles[heldout_titles:2 * heldout_titles]}
    result = {}
    for split, count in counts.items():
        pools = []
        for title in partitions[split]:
            pool = sorted(groups[title], key=lambda row: row['id'])
            rng.shuffle(pool)
            pools.append(pool)
        selected = []
        # Round-robin across all article groups before taking a second row from
        # any article. This prevents a few large articles dominating the subset.
        while len(selected) < count and any(pools):
            for pool in pools:
                if pool and len(selected) < count:
                    selected.append(pool.pop())
        if len(selected) != count:
            raise ValueError('not enough records in the fixed document partition')
        rng.shuffle(selected)
        result[split] = selected
    check_splits(result)
    return result


def declaration_scores(prediction, target):
    # Normalize case and token spacing while retaining all declaration words.
    predicted, expected = re.findall(r'\w+', prediction.casefold()), re.findall(r'\w+', target.casefold())
    common = sum((Counter(predicted) & Counter(expected)).values())
    return {'exact_declaration': float(predicted == expected),
            'token_f1': 2 * common / (len(predicted) + len(expected)) if predicted or expected else 1.}


def prepare_data(directory, *, train_count=1024, dev_count=128, test_count=128, seed=20260921):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files, sources, rows = [], [], []
    for split in ('train', 'dev'):
        url = f'https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/{SQUAD_REVISION}/dataset/{split}-v1.1.json'
        path = directory / f'squad-{split}.json'
        urllib.request.urlretrieve(url, path)
        files.append({'url': url, 'sha256': sha256(path)})
        sources.append(json.loads(path.read_text()))
        filename = f'data/{split}-00000-of-00001.parquet'
        path = hf_hub_download(QA2D, filename, repo_type='dataset', revision=QA2D_REVISION, token=False)
        files.append({'repository': QA2D, 'revision': QA2D_REVISION, 'file': filename, 'sha256': sha256(path)})
        rows.extend((row, split) for row in pq.read_table(path).to_pylist())
    originals = squad_index(sources)
    records = []
    excluded = Counter()
    for row, split in rows:
        try:
            record = join_record(row, originals, split)
        except ValueError as error:
            excluded[str(error).split(' for ')[0]] += 1
            continue
        if record is None:
            excluded['non-SQuAD source'] += 1
        else:
            records.append(record)
    splits = document_splits(records, train_count=train_count, dev_count=dev_count, test_count=test_count, seed=seed)
    manifest = {'qa2d': {'repository': QA2D, 'revision': QA2D_REVISION, 'label_field': 'turker_answer',
                         'paper': 'https://arxiv.org/abs/1809.02922'},
                'squad': {'repository': 'rajpurkar/SQuAD-explorer', 'revision': SQUAD_REVISION,
                          'license': 'CC-BY-SA-4.0'}, 'files': files, 'seed': seed,
                'eligible_records': len(records), 'excluded_records': dict(excluded),
                'split_policy': 'seeded 80/10/10 original-article groups; round-robin samples; repartitions QA2D train and dev',
                'input_policy': 'original question and unaltered paragraph only; production proposal_prompt',
                'limitations': ['human declaration labels can contain grammatical errors',
                                'source paragraph is supplied; retrieval competence is not evaluated',
                                'foundation may previously have seen SQuAD or related benchmarks'], 'splits': {}}
    for split, selected in splits.items():
        path = directory / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in selected))
        manifest['splits'][split] = {'sha256': sha256(path), 'count': len(selected),
                                     'ids': [r['id'] for r in selected],
                                     'document_count': len({r['document_id'] for r in selected}),
                                     'documents': sorted({r['document_id'] for r in selected})}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return splits, manifest


def token_limits(model, splits, *, input_limit, target_limit):
    """Audit the prefixes actually used by the bounded encoder and decoder."""
    return {'input': input_limit, 'target': target_limit,
            'input_truncated_counts': {split: sum(
                len(model.tokenizer(model_input(row), truncation=False)['input_ids']) > input_limit
                for row in records) for split, records in splits.items()},
            'target_truncated_counts': {split: sum(
                len(model.tokenizer(row['target'], truncation=False)['input_ids']) > target_limit
                for row in records) for split, records in splits.items()},
            'target_truncation_policy': 'loss_batch truncates decoder targets exceeding the configured limit'}


def nli_summary(predictions):
    """Keep whole-split, nonempty and untruncated NLI denominators explicit."""
    evaluated = [row for row in predictions if row['nli_model_label'] is not None]
    untruncated = [row for row in evaluated if not row['nli_input_truncated']]
    supported = sum(row['nli_model_label'] == 'support' for row in evaluated)
    return {'nli_evaluated_count': len(evaluated),
            'nli_truncated_count': len(evaluated) - len(untruncated),
            'nli_untruncated_count': len(untruncated),
            'nli_model_support_rate': supported / len(evaluated) if evaluated else None,
            'nli_support_rate_all_samples': supported / len(predictions) if predictions else None,
            'nli_model_support_rate_untruncated': (
                sum(row['nli_model_label'] == 'support' for row in untruncated) / len(untruncated)
                if untruncated else None)}


def evaluate(model, records, batch_size, verifier=None):
    import torch
    model.eval()
    predictions = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        outputs = model.generate_batch([model_input(row) for row in batch])
        predictions.extend({'id': row['id'], 'target': row['target'], 'prediction': output,
                            **declaration_scores(output, row['target'])} for row, output in zip(batch, outputs))
    result = {'count': len(records), 'predictions': predictions,
              **{key: sum(row[key] for row in predictions) / len(records) for key in ('exact_declaration', 'token_f1')}}
    if verifier is not None:
        verifier.eval()
        valid = [(row, prediction) for row, prediction in zip(records, predictions) if prediction['prediction'].strip()]
        for prediction in predictions:
            prediction['nli_model_label'] = None
            prediction['nli_input_truncated'] = None
        inverse = {index: name for name, index in verifier.labels.items()}
        for start in range(0, len(valid), batch_size):
            batch = valid[start:start + batch_size]
            pairs = [{'premise': row['evidence'][0]['text'], 'hypothesis': prediction['prediction']}
                     for row, prediction in batch]
            for (_, prediction), pair in zip(batch, pairs):
                prediction['nli_input_truncated'] = len(verifier.tokenizer(
                    pair['premise'], pair['hypothesis'], truncation=False)['input_ids']) > verifier.max_tokens
            with torch.no_grad():
                labels = verifier(pairs).argmax(-1).cpu().tolist()
            for (_, prediction), label in zip(batch, labels):
                prediction['nli_model_label'] = inverse[label]
        result.update(nli_summary(predictions))
        result['nli_max_tokens'] = verifier.max_tokens
        result['nli_semantics'] = ('separate model judgment, not human correctness or truth; '
                                   'the overall rate includes truncated pairs, with untruncated rate reported separately')
    return result


def model_card(*, foundation, revision, epochs):
    """Describe the configured run without assigning an unknown source license."""
    return f'''---
library_name: tensorcode
pipeline_tag: text2text-generation
---
# Evidence-conditioned declarative hypothesis generator

Native TensorCode Chatbot initialized from `{foundation}` at revision `{revision}`, with an owned
relational workspace. Prescribed schedule: {epochs} epochs. Trained using original SQuAD questions and paragraphs,
with human QA2D turker_answer declarations as decoder targets. No short-answer
annotations or rule-based pseudo-labels enter the encoder. Training updates
foundation and workspace weights. Use as Investigator.generator with its exact
production proposal_prompt template; generated text remains an uncertain proposal.

See evaluation.json for source hashes, document-disjoint splits, all predictions,
baseline/final declaration metrics, optional independent NLI model judgments, and
input/target and NLI pair truncation counts. Human labels can be noisy. Supplied passages do not test
retrieval; prior foundation benchmark exposure cannot be excluded. NLI scores do
not establish factual truth or answer relevance. SQuAD sources are CC-BY-SA-4.0;
QA2D mirror declares MIT. Foundation licensing follows the selected source. Source revisions and attribution are in evaluation.json.

Load with tensorcode.tools.chatbot.Chatbot.from_pretrained(path). The separate
training artifact stores full model, optimizer, RNG and epoch progress for resume.
'''


def make_optimizer(model, learning_rate, workspace_lr):
    import torch
    other = [parameter for name, parameter in model.named_parameters() if not name.startswith('foundation.')]
    return torch.optim.AdamW([{'params': model.foundation.parameters(), 'lr': learning_rate},
                              {'params': other, 'lr': workspace_lr}])


def run(args):
    import torch
    from tensorcode.tools.chatbot import Chatbot
    from tensorcode.training import ToolTrainer
    if args.prepare_only:
        _, manifest = prepare_data(args.data, train_count=args.train_count, dev_count=args.dev_count,
                                   test_count=args.test_count, seed=args.seed)
        print(json.dumps({k: v for k, v in manifest.items() if k not in ('files', 'splits')}, indent=2))
        return manifest
    if args.device != 'cuda' or not torch.cuda.is_available():
        raise ValueError('training and model evaluation require the authorized CUDA host')
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    data = Path(args.data)
    manifest = json.loads((data / 'manifest.json').read_text())
    splits = {name: [json.loads(line) for line in (data / f'{name}.jsonl').read_text().splitlines() if line.strip()]
              for name in ('train', 'dev', 'test')}
    check_splits(splits)
    for split in splits:
        if sha256(data / f'{split}.jsonl') != manifest['splits'][split]['sha256']:
            raise ValueError('prepared data checksum mismatch')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    model = Chatbot.from_foundation(args.foundation, revision=args.revision, local_files_only=args.local_files_only,
                                    max_input_tokens=args.max_input_tokens, max_target_tokens=args.max_target_tokens,
                                    max_new_tokens=args.max_target_tokens,
                                    workspace={'slots': 8, 'steps': 2}).to(args.device)
    verifier = None
    if args.verifier_path:
        from safetensors.torch import load_file
        from tensorcode.tools.investigator import EvidenceVerifier
        path = Path(args.verifier_path)
        verifier = EvidenceVerifier(json.loads((path / 'verifier_config.json').read_text())).to(args.device)
        verifier.load_state_dict(load_file(str(path / 'verifier.safetensors')))
    optimizer = make_optimizer(model, args.learning_rate, args.workspace_lr)
    trainer = ToolTrainer(model, optimizer=optimizer)
    report = {'data': manifest, 'foundation': args.foundation, 'revision': args.revision,
              'schedule': {'epochs': args.epochs, 'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
                           'workspace_lr': args.workspace_lr, 'seed': args.seed},
              'host': platform.node(), 'gpu': torch.cuda.get_device_name(),
              'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'tensorcode')},
              'proposal_template_sha256': sha256(Path(__file__).parents[1] / 'src/tensorcode/_internal/proposals.py'),
              'script_sha256': sha256(__file__), 'verifier_path': args.verifier_path,
              'evaluation_policy': 'fixed final checkpoint; no selection or optimization on dev/test', 'epoch_losses': []}
    report['token_limits'] = token_limits(model, splits, input_limit=args.max_input_tokens,
                                         target_limit=args.max_target_tokens)
    first_epoch = 0
    if args.resume:
        progress = trainer.load_checkpoint(args.resume)
        if progress.get('data_manifest_sha256') != sha256(data / 'manifest.json'):
            raise ValueError('resume data provenance mismatch')
        first_epoch = progress['epochs']
        report['resumed_from'] = args.resume
    print('Evaluating declaration baseline', flush=True)
    report['before'] = {split: evaluate(model, splits[split], args.batch_size, verifier) for split in ('dev', 'test')}
    (output / 'before.json').write_text(json.dumps(report, indent=2))
    for epoch in range(first_epoch, args.epochs):
        model.train()
        order = list(range(len(splits['train'])))
        random.shuffle(order)
        total = 0.
        for start in range(0, len(order), args.batch_size):
            batch = [splits['train'][index] for index in order[start:start + args.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss_batch([model_input(row) for row in batch], [row['target'] for row in batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            trainer.steps += 1
            total += float(loss.detach()) * len(batch)
            if trainer.steps % 32 == 0:
                print(json.dumps({'epoch': epoch + 1, 'step': trainer.steps, 'loss': float(loss.detach())}), flush=True)
        report['epoch_losses'].append(total / len(order))
        print(json.dumps({'epoch': epoch + 1, 'mean_training_loss': report['epoch_losses'][-1]}), flush=True)
    report['after'] = {split: evaluate(model, splits[split], args.batch_size, verifier) for split in ('dev', 'test')}
    model.save_pretrained(output / 'model')
    trainer.save_checkpoint(output / 'training', progress={'epochs': args.epochs,
                            'data_manifest_sha256': sha256(data / 'manifest.json')})
    expected = model.generate_batch([model_input(splits['test'][0])])
    # Reconstruct native model and full optimizer, then restore resumable state.
    restored = Chatbot.from_pretrained(output / 'model', device=args.device)
    restored_trainer = ToolTrainer(restored, optimizer=make_optimizer(restored, args.learning_rate, args.workspace_lr))
    restored_progress = restored_trainer.load_checkpoint(output / 'training')
    restored.eval()
    report['reload_generation_equal'] = expected == restored.generate_batch([model_input(splits['test'][0])])
    report['optimizer_resume'] = {'steps_equal': restored_trainer.steps == trainer.steps,
                                  'state_entries': len(restored_trainer.optimizer.state),
                                  'progress': restored_progress}
    if not report['reload_generation_equal'] or not report['optimizer_resume']['steps_equal']:
        raise RuntimeError('native checkpoint verification failed')
    report['elapsed_seconds'] = time.time() - started
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    (output / 'model' / 'evaluation.json').write_text(json.dumps(report, indent=2))
    (output / 'model' / 'README.md').write_text(model_card(
        foundation=args.foundation, revision=args.revision, epochs=args.epochs))
    print(json.dumps({'completed': str(output), 'test_before': {k: v for k, v in report['before']['test'].items() if k != 'predictions'},
                      'test_after': {k: v for k, v in report['after']['test'].items() if k != 'predictions'},
                      'elapsed_seconds': report['elapsed_seconds']}), flush=True)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--data', required=True)
    result.add_argument('--output')
    result.add_argument('--prepare-only', action='store_true')
    result.add_argument('--foundation', default='google/flan-t5-small')
    result.add_argument('--revision', default=FOUNDATION_REVISION)
    result.add_argument('--local-files-only', action='store_true')
    result.add_argument('--verifier-path')
    result.add_argument('--resume')
    result.add_argument('--train-count', type=int, default=1024)
    result.add_argument('--dev-count', type=int, default=128)
    result.add_argument('--test-count', type=int, default=128)
    result.add_argument('--epochs', type=int, default=3)
    result.add_argument('--batch-size', type=int, default=8)
    result.add_argument('--learning-rate', type=float, default=3e-5)
    result.add_argument('--workspace-lr', type=float, default=1e-3)
    result.add_argument('--seed', type=int, default=20260921)
    result.add_argument('--max-input-tokens', type=int, default=512)
    result.add_argument('--max-target-tokens', type=int, default=64)
    result.add_argument('--device', default='cuda')
    return result


if __name__ == '__main__':
    args = parser().parse_args()
    if not args.prepare_only and not args.output:
        raise SystemExit('--output required for training')
    if min(args.epochs, args.batch_size, args.max_input_tokens, args.max_target_tokens, args.learning_rate, args.workspace_lr) <= 0:
        raise SystemExit('counts and learning rates must be positive')
    run(args)
