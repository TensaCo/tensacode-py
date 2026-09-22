"""Compare generated-JSON and likelihood decoding for owned typed decisions.

Zero-shot diagnostic of the *decoding mechanism* on a supplied foundation: no
training, threshold selection or prompt search occurs. Two inputs are used:

* Banking77 test rows (first ``--per-class`` per intent), raw intent labels.
* Response-quality development candidates with assistant-reviewed labels, asked
  as three true/false questions (support, completeness, constraints). These rows are development
  data, not a benchmark, and the labels are not human ground truth.

Example::

    python examples/evaluate_typed_decisions.py --foundation PATH_TO_FLAN_T5 \
        --banking77 ~/.cache/tensorcode/data/banking77_test.csv \
        --candidates .development/datasets/response-quality-candidates.jsonl \
        --labels .development/datasets/response-quality-labels-{a,b,c}.jsonl \
        --output docs/results/typed-decisions.json
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from tensorcode.ops import text

AXES = {
    'support': 'Do the evidence passages support every factual assertion in the candidate answer?',
    'completeness': 'Does the candidate answer supply the kind of value the question asks for?',
    'constraints': 'Does the candidate answer satisfy every restriction stated in the question?',
}
MODES = {
    'generate': {},
    'likelihood_sum': {'decoding': 'likelihood'},
    'likelihood_mean': {'decoding': 'likelihood', 'likelihood_normalization': 'mean'},
}


def expected_calibration_error(pairs, bins=10):
    if not pairs:
        return None
    grouped = defaultdict(list)
    for confidence, correct in pairs:
        grouped[min(int(confidence * bins), bins - 1)].append((confidence, correct))
    return sum(
        len(rows) / len(pairs) * abs(sum(c for c, _ in rows) / len(rows) - sum(k for _, k in rows) / len(rows))
        for rows in grouped.values()
    )


def auroc(scored):
    positives = [score for score, label in scored if label]
    negatives = [score for score, label in scored if not label]
    if not positives or not negatives:
        return None
    wins = sum((p > n) + .5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def build(cls, foundation, device, **config):
    return cls.from_foundation(foundation, config=config).to(device)


def banking77(args):
    rows, counts = [], defaultdict(int)
    with open(Path(args.banking77).expanduser(), newline='') as handle:
        for row in csv.DictReader(handle):
            if counts[row['category']] < args.per_class:
                counts[row['category']] += 1
                rows.append(row)
    labels = sorted(counts)
    report = {'rows': len(rows), 'labels': len(labels), 'majority_accuracy': max(counts.values()) / len(rows)}
    for mode, decoding in MODES.items():
        op = build(text.Classify, args.foundation, args.device, labels=labels,
                   instructions='Which banking customer intent does this message express?',
                   generation={'max_new_tokens': 64}, **decoding)
        correct = valid = 0
        calibration = []
        started = time.perf_counter()
        for row in rows:
            try:
                result = op((text.Message('user', row['text']),))
            except (text.InvalidModelOutput, ValueError):
                continue
            valid += 1
            hit = result.label == row['category']
            correct += hit
            if result.confidence is not None:
                calibration.append((result.confidence, hit))
        report[mode] = {
            'valid_rate': valid / len(rows),
            'accuracy': correct / len(rows),
            'ece': expected_calibration_error(calibration),
            'seconds_per_row': (time.perf_counter() - started) / len(rows),
        }
        print('banking77', mode, report[mode], flush=True)
    return report


def response_quality(args):
    labels = {}
    for path in args.labels:
        for line in open(path):
            row = json.loads(line)
            labels[row['id']] = row['targets']
    candidates = [json.loads(line) for line in open(args.candidates)]
    report = {'candidates': len(candidates), 'label_authorship': 'assistant-reviewed development labels'}
    for mode, decoding in MODES.items():
        questions = {
            axis: build(text.Classify, args.foundation, args.device, labels=['true', 'false'],
                        instructions=instruction, generation={'max_new_tokens': 64}, **decoding)
            for axis, instruction in AXES.items()
        }
        outcomes = {axis: {'valid': 0, 'correct': 0, 'labelled': 0, 'scored': []} for axis in AXES}
        started = time.perf_counter()
        for row in candidates:
            evidence = '\n'.join(f"[{item['source_id']}] {item['text']}" for item in row['evidence'])
            message = text.Message('user', f"Question: {row['question']}\nCandidate answer: {row['candidate']}\n"
                                           f"Evidence:\n{evidence}")
            answers = {}
            for axis, operation in questions.items():
                # Each axis is scored separately so one invalid response does not
                # discard the others; text.ask would raise for the whole set.
                try:
                    answers[axis] = operation((message,))
                except (text.InvalidModelOutput, ValueError):
                    answers[axis] = None
            for axis, answer in answers.items():
                target = labels.get(row['id'], {}).get(axis)
                if target is None:
                    continue
                stats = outcomes[axis]
                stats['labelled'] += 1
                if answer is None:
                    continue
                stats['valid'] += 1
                stats['correct'] += (answer.label == 'true') == target
                if answer.distribution is not None:
                    stats['scored'].append((answer.distribution['true'], target))
        summary = {}
        for axis, stats in outcomes.items():
            positives = sum(label for _, label in stats['scored'])
            summary[axis] = {
                'labelled': stats['labelled'],
                'valid_rate': stats['valid'] / stats['labelled'],
                'accuracy': stats['correct'] / stats['labelled'],
                'auroc_true': auroc(stats['scored']),
                'scored_rows': len(stats['scored']),
                'scored_positive_rate': positives / len(stats['scored']) if stats['scored'] else None,
            }
        summary['seconds_per_candidate'] = (time.perf_counter() - started) / len(candidates)
        report[mode] = summary
        print('response_quality', mode, json.dumps(summary), flush=True)
    majority = {}
    for axis in AXES:
        values = [targets[axis] for targets in labels.values() if targets.get(axis) is not None]
        majority[axis] = max(values.count(True), values.count(False)) / len(values)
    report['majority_accuracy'] = majority
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--foundation', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--banking77')
    parser.add_argument('--per-class', type=int, default=5)
    parser.add_argument('--candidates')
    parser.add_argument('--labels', nargs='*', default=[])
    parser.add_argument('--output')
    args = parser.parse_args()
    report = {'foundation': Path(args.foundation).name if Path(args.foundation).exists() else args.foundation,
              'training': 'none (zero-shot)', 'modes': MODES}
    if args.banking77:
        report['banking77'] = banking77(args)
    if args.candidates:
        report['response_quality'] = response_quality(args)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
