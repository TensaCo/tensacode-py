"""Select source-disjoint Hotpot training questions and generate natural proposals.

Local pinned parquet and owned model artifacts only. Validation reads project
only context titles; gold answers are review metadata, never generator inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

SEED = 20260922
DATASET = 'hotpotqa/hotpot_qa'
REVISION = '1908d6afbbead072334abe2965f91bd2709910ab'
PINNED = {'train': '76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68',
          'validation': 'c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6'}
SPLITS = ('train', 'calibration', 'development')


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def select_cases(rows, reserved_titles, excluded, *, counts=(128, 32, 32)):
    if len(counts) != 3 or any(type(n) is not int or n < 1 for n in counts):
        raise ValueError('three positive question counts required')
    prepare_hotpot = sibling('evaluate_cognition').prepare_hotpot
    used_ids = {row.get('question_id', row.get('id')) for row in excluded}
    used_titles = set(reserved_titles) | {e['source_id'] for row in excluded for e in row['evidence']}
    used_texts = {hashlib.sha256(e['text'].encode()).hexdigest() for row in excluded for e in row['evidence']}
    selected = []
    for row in rows:
        if row['id'] in used_ids:
            continue
        prepared = prepare_hotpot(row)
        titles = {e['source_id'] for e in prepared['evidence']}
        texts = {hashlib.sha256(e['text'].encode()).hexdigest() for e in prepared['evidence']}
        if titles & used_titles or texts & used_texts:
            continue
        selected.append(prepared)
        used_ids.add(row['id'])
        used_titles.update(titles)
        used_texts.update(texts)
        if len(selected) == sum(counts):
            break
    if len(selected) != sum(counts):
        raise ValueError(f'only {len(selected)} eligible questions; requested {sum(counts)}')
    groups, offset = {}, 0
    for split, count in zip(SPLITS, counts):
        groups[split] = [row['id'] for row in selected[offset:offset + count]]
        offset += count
    return selected, groups


def validation_titles(path):
    import pyarrow.parquet as pq
    # A leaf-column projection avoids decoding any validation question, answer,
    # supporting fact, or context sentence column.
    table = pq.read_table(path, columns=['context.title'])
    titles = set()
    for row in table.to_pylist():
        value = row.get('context', row)
        titles.update(value['title'])
    if not titles:
        raise ValueError('validation title projection was empty')
    return titles


def select(args):
    import pyarrow.parquet as pq
    if args.dataset_revision != REVISION:
        raise ValueError('dataset revision differs from the declared pinned protocol')
    paths = {'train': args.train_parquet, 'validation': args.validation_parquet}
    hashes = {key: sha256(path) for key, path in paths.items()}
    if hashes != PINNED:
        raise ValueError('parquet bytes differ from the pinned protocol')
    titles = validation_titles(args.validation_parquet)
    excluded = [row for path in args.exclude_candidates for row in read_jsonl(path)]
    rows = (row for batch in pq.ParquetFile(args.train_parquet).iter_batches(batch_size=256) for row in batch.to_pylist())
    cases, groups = select_cases(rows, titles, excluded,
                                counts=(args.train_questions, args.calibration_questions, args.development_questions))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'cases.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in cases))
    manifest = {'format': 'tensorcode.response_quality_selection.v2', 'seed': args.seed,
                'dataset': {'repository': DATASET, 'revision': REVISION, 'configuration': 'distractor',
                            'train_shard': 'train-00000-of-00002.parquet', 'sha256': hashes},
                'exclusions': {str(path): sha256(path) for path in args.exclude_candidates},
                'reserved_validation_titles': len(titles),
                'selection_policy': 'first eligible training-shard rows; reserve all validation context titles and excluded source IDs/text hashes; selected questions mutually source disjoint',
                'question_groups': groups, 'cases_sha256': sha256(output / 'cases.jsonl'),
                'reference_answer_role': 'review metadata only; oracle supporting passages are supplied evidence'}
    atomic_json(output / 'manifest.json', manifest)
    print(json.dumps({'selected': len(cases), 'output': str(output)}), flush=True)
    return manifest


def generate_case(investigator, case, index):
    inputs = {'question': case['question'], 'evidence': [
        {'source_id': item['id'], 'text': item['text']} for item in case['evidence']]}
    proposals = investigator.propose(inputs, count=3)
    if not proposals:
        raise ValueError('empty natural proposal set')
    if len(proposals) > 3 or len({p['id'] for p in proposals}) != len(proposals):
        raise ValueError('invalid natural proposal count or duplicate IDs')
    return [{'id': case['id'] + ':' + proposal['id'], 'question_index': index,
             'question_id': case['id'], 'question': case['question'], 'candidate': proposal['text'],
             'evidence': case['evidence'], 'reference_answer': case['target'],
             'candidate_origin': 'natural owned Investigator.propose; up to three deduplicated beam proposals',
             'review_role': 'assistant supervision pending; not human labels',
             'input_truncated': proposal['input_truncated'], 'generation': proposal} for proposal in proposals]


def generate(args):
    import torch
    from tensorcode.tools.chatbot import Chatbot
    from tensorcode.tools.investigator import Investigator
    if not torch.cuda.is_available():
        raise RuntimeError('natural proposal generation requires CUDA')
    model_path, cases_path, output = Path(args.model), Path(args.cases), Path(args.output)
    if not model_path.is_dir():
        raise ValueError('model must be a complete existing local owned artifact')
    manifest = json.loads((cases_path / 'manifest.json').read_text())
    if sha256(cases_path / 'cases.jsonl') != manifest['cases_sha256']:
        raise ValueError('selection cases checksum mismatch')
    cases = read_jsonl(cases_path / 'cases.jsonl')
    if [row['id'] for row in cases] != [key for split in SPLITS for key in manifest['question_groups'][split]]:
        raise ValueError('selection manifest question order mismatch')
    output.mkdir(parents=True, exist_ok=False)
    progress = output / 'progress'
    progress.mkdir()
    torch.set_num_threads(8)
    torch.manual_seed(manifest['seed'])
    torch.cuda.manual_seed_all(manifest['seed'])
    model = (Chatbot if args.model_kind == 'chatbot' else Investigator).from_pretrained(model_path)
    model.to('cuda').eval()
    investigator = model.investigator if args.model_kind == 'chatbot' else model
    provenance = {'selection_manifest': manifest, 'requested_proposals_per_question': 3,
                  'generator_artifact': {str(p.relative_to(model_path)): sha256(p) for p in sorted(model_path.rglob('*')) if p.is_file()},
                  'seed': manifest['seed'], 'model_kind': args.model_kind, 'gpu': torch.cuda.get_device_name(),
                  'script_sha256': sha256(__file__), 'status': 'incomplete'}
    atomic_json(output / 'manifest.json', provenance)
    records, failures = [], []
    for index, case in enumerate(cases):
        try:
            with torch.no_grad():
                candidates = generate_case(investigator, case, index)
            records.extend(candidates)
            result = {'question_id': case['id'], 'candidates': candidates, 'count': len(candidates)}
        except (ValueError, RuntimeError) as error:
            result = {'question_id': case['id'], 'error': {'type': type(error).__name__, 'message': str(error)}}
            failures.append(result)
        atomic_json(progress / f'{index:04d}.json', result)
        print(json.dumps({'question_id': case['id'], 'count': result.get('count', 0), 'error': result.get('error')}), flush=True)
    provenance.update(actual_candidates=len(records), completed_questions=len(cases) - len(failures), failures=failures,
                      actual_counts={row['id']: sum(r['question_id'] == row['id'] for r in records) for row in cases})
    if failures:
        atomic_json(output / 'manifest.json', provenance)
        raise RuntimeError('candidate generation incomplete; explicit failures saved, no complete candidates artifact')
    temporary = output / 'candidates.jsonl.tmp'
    temporary.write_text(''.join(json.dumps(row) + '\n' for row in records))
    temporary.replace(output / 'candidates.jsonl')
    provenance.update(status='complete', candidates_sha256=sha256(output / 'candidates.jsonl'))
    atomic_json(output / 'manifest.json', provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    selection = commands.add_parser('select')
    selection.add_argument('--train-parquet', required=True)
    selection.add_argument('--validation-parquet', required=True)
    selection.add_argument('--exclude-candidates', nargs='+', required=True)
    selection.add_argument('--dataset-revision', default=REVISION)
    selection.add_argument('--seed', type=int, default=SEED)
    selection.add_argument('--train-questions', type=int, default=128)
    selection.add_argument('--calibration-questions', type=int, default=32)
    selection.add_argument('--development-questions', type=int, default=32)
    selection.add_argument('--output', required=True)
    generation = commands.add_parser('generate')
    generation.add_argument('--cases', required=True)
    generation.add_argument('--model', required=True)
    generation.add_argument('--model-kind', choices=['chatbot', 'investigator'], default='chatbot')
    generation.add_argument('--output', required=True)
    args = parser.parse_args()
    return select(args) if args.command == 'select' else generate(args)


if __name__ == '__main__':
    main()
