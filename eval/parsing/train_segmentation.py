"""Train source segmentation on local UD train annotations and measure span proposals.

No normalization, downloads, syntax parsing, or automatic semantic selection.
Pilot: --train-limit 1000 --epochs 2 --dev-sample 250 --skip-test --model /tmp/segmenter-pilot.json
Full: --epochs 5 --dev-sample 0 --test-sample 0
Zero sample/limit means all records. Only training-split records enter training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

from eval.parsing.span_evaluation import Gold, load_records


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(records: tuple[Gold, ...], count: int, seed: int) -> tuple[Gold, ...]:
    if count < 0:
        raise ValueError('sample count cannot be negative')
    return tuple(random.Random(seed).sample(list(records), min(count, len(records)))) if count else records


def validate_spans(text: str, spans) -> tuple[tuple[int, int], ...]:
    """Require exact original-character coverage, not repaired/normalized tokens."""
    checked = []
    covered = set()
    end = 0
    for span in spans:
        if len(span) != 2 or any(type(v) is not int for v in span):
            raise ValueError('spans must contain two exact integers')
        lo, hi = span
        if not (end <= lo < hi <= len(text)) or any(c.isspace() for c in text[lo:hi]):
            raise ValueError('spans overlap, exceed source, or contain whitespace')
        covered.update(range(lo, hi))
        checked.append((lo, hi))
        end = hi
    if covered != {i for i, c in enumerate(text) if not c.isspace()}:
        raise ValueError('spans do not cover every non-whitespace source character exactly')
    return tuple(checked)


def span_counts(gold, predicted) -> dict:
    gold_set, predicted_set = set(gold), set(predicted)
    matched = len(gold_set & predicted_set)
    return {'matched': matched, 'gold': len(gold_set), 'predicted': len(predicted_set),
            'exact': tuple(gold) == tuple(predicted)}


def old_tokenizer_spans(text: str):
    """Historical chart tokenizer reference, only used by this evaluation."""
    from tensorcode.language.chart import tokenize
    cursor = 0
    spans = []
    for token in tokenize(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if not text.startswith(token, cursor):
            raise ValueError('historical token cannot be source-aligned')
        spans.append((cursor, cursor + len(token)))
        cursor += len(token)
    if text[cursor:].strip():
        raise ValueError('historical tokenizer left source characters uncovered')
    # Historical quoted spans may contain whitespace; this is a measured convention,
    # not permission to normalize them into the learned model's word boundaries.
    return tuple(spans)


def evaluate(model, records: tuple[Gold, ...], *, beam_width: int, max_candidates: int,
             max_expansions: int, compare_old: bool = True) -> dict:
    rows = []
    for record in records:
        truth = tuple(record.spans[i] for i in sorted(record.spans))
        candidates = []
        invalid = []
        search = None
        error = record.error
        start = time.perf_counter()
        if not error:
            try:
                search = model.segment(record.text, beam_width=beam_width, max_candidates=max_candidates,
                                       max_expansions=max_expansions)
                for candidate in search.candidates:
                    try:
                        spans = validate_spans(record.text, candidate.spans)
                        if spans not in candidates:
                            candidates.append(spans)
                    except (TypeError, ValueError) as exc:
                        invalid.append(str(exc))
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
        ms = (time.perf_counter() - start) * 1000
        top = span_counts(truth, candidates[0] if candidates else ())
        options = [span_counts(truth, candidate) for candidate in candidates]
        # One complete candidate maximizes matched gold spans (oracle recall).
        # No union of mutually incompatible boundaries is credited.
        oracle = max(options, key=lambda row: (row['matched'], -row['predicted'])) if options else span_counts(truth, ())
        if record.error:
            top['exact'] = oracle['exact'] = False
        old = None
        if compare_old and not record.error:
            try:
                old = span_counts(truth, old_tokenizer_spans(record.text))
            except ValueError as exc:
                old = {**span_counts(truth, ()), 'error': str(exc)}
        rows.append({'sent_id': record.sent_id, 'characters': len(record.text),
                     'gold_words': len(record.words), 'top': top, 'oracle': oracle,
                     'exact_in_candidates': any(o['exact'] for o in options),
                     'candidate_count': len(candidates), 'error': error,
                     'invalid_candidates': invalid, 'old_tokenizer': old,
                     'expansions': search.expansions if search is not None else 0,
                     'truncated': search.truncated if search is not None else False,
                     'complete': search.complete if search is not None else False,
                     'reason': search.reason if search is not None else None, 'latency_ms': ms})
    metrics = {}
    for name in ('top', 'oracle', 'old_tokenizer'):
        entries = [r[name] for r in rows if r[name] is not None]
        matched = sum(r['matched'] for r in entries)
        gold = sum(r['gold'] for r in entries)
        predicted = sum(r['predicted'] for r in entries)
        metrics[name] = {'matched_spans': matched, 'gold_spans': gold, 'predicted_spans': predicted,
                         'precision': matched / predicted if predicted else None,
                         'recall': matched / gold if gold else None,
                         'exact_sentences': sum(r['exact'] for r in entries),
                         'exact_rate': sum(r['exact'] for r in entries) / len(entries) if entries else None}
    exact_oracle = sum(r['exact_in_candidates'] for r in rows)
    latencies = sorted(r['latency_ms'] for r in rows)
    return {'metrics': metrics, 'sample_size': len(rows),
            'exact_candidate_recall': exact_oracle / len(rows) if rows else None,
            'search': {'mean_candidates': statistics.mean(r['candidate_count'] for r in rows) if rows else 0,
                       'empty_sets': sum(not r['candidate_count'] for r in rows),
                       'error_inputs': sum(bool(r['error']) for r in rows),
                       'invalid_candidate_inputs': sum(bool(r['invalid_candidates']) for r in rows),
                       'truncated_inputs': sum(r['truncated'] for r in rows),
                       'budget_exhaustions': sum(r['reason'] == 'budget_exhausted' for r in rows),
                       'reasons': dict(Counter(r['reason'] for r in rows if r['reason']))},
            'latency_ms': {'median': statistics.median(latencies) if rows else None,
                           'p95': latencies[min(len(rows) - 1, int(.95 * len(rows)))] if rows else None,
                           'total': sum(latencies)}, 'sentences': rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--treebank', type=Path, default=Path.home() / '.cache/tensorcode/seeds/UD_English-EWT')
    ap.add_argument('--model', type=Path, default=Path.home() / '.cache/tensorcode/models/ud_ewt_segmenter.json')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--seed', type=int, default=20260922)
    ap.add_argument('--train-limit', type=int, default=0)
    ap.add_argument('--dev-sample', type=int, default=250)
    ap.add_argument('--test-sample', type=int, default=250)
    ap.add_argument('--skip-test', action='store_true')
    ap.add_argument('--beam-width', type=int, default=4)
    ap.add_argument('--max-candidates', type=int, default=3)
    ap.add_argument('--max-expansions', type=int, default=100000)
    ap.add_argument('--output', type=Path, default=Path('eval/results/segmentation.json'))
    args = ap.parse_args()
    if (min(args.epochs, args.beam_width, args.max_candidates, args.max_expansions) < 1
            or min(args.train_limit, args.dev_sample, args.test_sample) < 0):
        ap.error('epochs positive; sample counts nonnegative (zero means all)')
    from tensorcode.language import segmentation
    import tensorcode.language.learned_parser as learned_parser
    import tensorcode.language.chart as chart
    paths = (Path(__file__), Path(__file__).with_name('span_evaluation.py'),
             Path(segmentation.__file__), Path(learned_parser.__file__), Path(chart.__file__))
    sources = {str(p): sha(p) for p in paths}
    split_paths = {split: args.treebank / f'en_ewt-ud-{split}.conllu' for split in ('train', 'dev', 'test')}
    split_hashes = {split: sha(path) for split, path in split_paths.items()}
    records = {split: load_records(path) for split, path in split_paths.items()}
    training = select(records['train'], args.train_limit, args.seed)
    bad_training = [r for r in training if r.error]
    if bad_training:
        ap.error(f'Training alignment errors: {[(r.sent_id, r.error) for r in bad_training]}')
    examples = [(r.text, validate_spans(r.text, tuple(r.spans[i] for i in sorted(r.spans)))) for r in training]
    start = time.perf_counter()
    model = segmentation.Segmenter.train(examples, epochs=args.epochs, seed=args.seed)
    training_seconds = time.perf_counter() - start
    metadata = {'training_source': 'UD English EWT train only; original # text and exact basic-word character spans',
                'training_split_sha256': split_hashes['train'], 'train_sentences': len(training),
                'train_sent_ids_sha256': hashlib.sha256('\n'.join(r.sent_id for r in training).encode()).hexdigest(),
                'train_words': sum(len(r.words) for r in training), 'train_characters': sum(len(r.text) for r in training),
                'epochs': args.epochs, 'seed': args.seed, 'training_seconds': training_seconds,
                'sources': sources, 'confidence': 'uncalibrated margin; no user-intent probability'}
    args.model.parent.mkdir(parents=True, exist_ok=True)
    segmentation.save_model(args.model, model, metadata)
    loaded = segmentation.load_model(args.model)
    outputs = {}
    for split, sample in (('dev', args.dev_sample), ('test', args.test_sample)):
        if split == 'test' and args.skip_test:
            continue
        selected = select(records[split], sample, args.seed + (1 if split == 'dev' else 2))
        outputs[split] = evaluate(loaded, selected, beam_width=args.beam_width,
                                  max_candidates=args.max_candidates, max_expansions=args.max_expansions)
    result = {'evaluation': 'learned_source_segmentation', 'training': metadata,
              'artifact': {'path': str(args.model), 'sha256': sha(args.model), 'bytes': args.model.stat().st_size},
              'datasets': {split: {'path': str(split_paths[split]), 'sha256': split_hashes[split],
                                  'records': len(records[split]), 'alignment_errors': [{'sent_id': r.sent_id, 'error': r.error} for r in records[split] if r.error]}
                           for split in records},
              'selection': {'train_limit': args.train_limit, 'dev_sample': args.dev_sample,
                            'test_sample': None if args.skip_test else args.test_sample,
                            'seed': args.seed, 'dev_seed': args.seed + 1, 'test_seed': args.seed + 2,
                            'method': 'sample without replacement in source-file order; zero count uses full split'},
              'budgets': {'beam_width': args.beam_width, 'max_candidates': args.max_candidates,
                          'max_expansions': args.max_expansions},
              'source_hashes_verified_after_run': {str(p): sha(p) == h for p, h in ((Path(p), h) for p, h in sources.items())},
              'dataset_hashes_verified_after_run': {split: sha(path) == split_hashes[split] for split, path in split_paths.items()},
              'limitations': ['Training annotations supply word boundaries; no semantic meaning or grounding is learned here.',
                              'Whitespace is a declared segmentation boundary; all current corpus words satisfy this assumption.',
                              'Oracle recall chooses a whole candidate using gold boundaries, not an actual agent selection policy.',
                              'Test split is previously exposed during this project; no pristine held-out generalization claim.',
                              'Historical tokenizer comparison is evaluation-only and supplies no fallback to production.'],
              'results': outputs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({**result, 'results': {k: {a: b for a, b in v.items() if a != 'sentences'} for k, v in outputs.items()}}, indent=2))


if __name__ == '__main__':
    main()
