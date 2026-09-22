"""Frozen development evidence controls, using existing native yes/no scoring.

No training or promotion decision. Use the authorized GPU host for real models.
Unknown labels stay unknown; changed evidence does not imply a negative label.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load_module('evidence_controls_native_probe', Path(__file__).with_name('probe_generative_quality.py'))
reload_helper = load_module('evidence_controls_reload', Path(__file__).with_name('verify_generative_quality_reload.py'))
inputs_helper = load_module('evidence_controls_inputs', Path(__file__).resolve().parents[2] / 'examples/train_response_quality.py')
AXES = tuple(probe.INSTRUCTIONS)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_targets(targets):
    if not isinstance(targets, dict) or set(targets) != set(AXES) or any(value is not None and type(value) is not bool for value in targets.values()):
        raise ValueError('targets require boolean or unknown values for exactly three axes')


def load_controls(directory, labels=None):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('format') != 'tensorcode.response_quality_development_evidence_controls.v1':
        raise ValueError('unsupported control pack')
    for name in ('anchors.jsonl', 'candidates.jsonl', 'review.jsonl'):
        if reload_helper.sha256(directory / name) != manifest['files'][name]:
            raise ValueError(f'frozen control checksum differs: {name}')
    anchors, variants = jsonl(directory / 'anchors.jsonl'), jsonl(directory / 'candidates.jsonl')
    if len(anchors) != 12 or len(variants) != 24:
        raise ValueError('frozen protocol requires twelve anchors and twenty-four variants')
    by_id = {row['id']: row for row in anchors + variants}
    if len(by_id) != 36:
        raise ValueError('control IDs must be unique')
    anchor_ids = {row['id'] for row in anchors}
    for row in anchors:
        validate_targets(row['targets'])
    seen = set()
    for row in variants:
        if row['original_id'] not in anchor_ids:
            raise ValueError('variant anchor is missing')
        anchor = by_id[row['original_id']]
        if any(row[key] != anchor[key] for key in ('question', 'candidate', 'question_id')):
            raise ValueError('variant must preserve anchor question and candidate')
        pair = row['original_id'], row['intervention']
        if pair in seen or row['intervention'] not in ('evidence-free', 'source-swapped'):
            raise ValueError('variant intervention must occur once per anchor')
        seen.add(pair)
        row['targets'] = dict.fromkeys(AXES)
    status = 'pending_support_review'
    if labels is not None:
        reviewed = jsonl(labels)
        mapped = {row['id']: row for row in reviewed}
        if len(mapped) != len(reviewed) or set(mapped) != {row['id'] for row in variants}:
            raise ValueError('review IDs must match every variant exactly once')
        for row in variants:
            review = mapped[row['id']]
            validate_targets(review['targets'])
            if any(review['targets'][axis] is not None for axis in ('completeness', 'constraints')):
                raise ValueError('altered-context completeness and constraints must remain unknown')
            for key in ('question', 'candidate', 'evidence', 'original_id', 'question_id', 'intervention'):
                if key in review and review[key] != row[key]:
                    raise ValueError(f'review input differs: {row["id"]}/{key}')
            row['targets'] = copy.deepcopy(review['targets'])
            row['review'] = {key: copy.deepcopy(value) for key, value in review.items()
                             if key not in ('id', 'targets', 'question', 'candidate', 'evidence')}
        status = 'adjudicated_support_labels_supplied'
    return {'manifest': manifest, 'anchors': anchors, 'variants': variants, 'review_status': status}


def evaluate(model, anchors, variants, *, yes_id, no_id, autocast_dtype=None):
    records = []
    for role, rows in [('anchor', anchors), ('variant', variants)]:
        for row in rows:
            inputs = inputs_helper.model_inputs(row)
            record = {'id': row['id'], 'role': role, 'targets': copy.deepcopy(row['targets'])}
            for key in ('original_id', 'intervention', 'review'):
                if key in row:
                    record[key] = row[key]
            for mode, ablation in [('active', None), ('bypass', 'bypass')]:
                receipt = probe.assess(model, inputs, yes_id=yes_id, no_id=no_id,
                                       workspace_ablation=ablation, autocast_dtype=autocast_dtype)
                scores = receipt['scores']
                if scores is not None and (set(scores) != set(AXES) or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in scores.values())):
                    raise ValueError('native quality scores must be finite three-axis probabilities')
                if receipt['input_truncated'] != (scores is None):
                    raise ValueError('inconsistent input coverage receipt')
                record[mode] = receipt
            if any(record['active'][key] != record['bypass'][key] for key in ('input_token_counts', 'input_truncated')):
                raise ValueError('active and bypass prompt coverage differs')
            records.append(record)
    return records


def summarize_mode(records, mode):
    eligible = [row for row in records if not row[mode]['input_truncated']]
    support = {'accepted': 0, 'rejected': 0, 'known_positive': 0, 'known_negative': 0,
               'unknown_labels': 0, 'false_accepts': 0, 'false_rejects': 0}
    for row in eligible:
        accepted = row[mode]['scores']['support'] >= .5
        target = row['targets']['support']
        support['accepted' if accepted else 'rejected'] += 1
        support['known_positive' if target is True else 'known_negative' if target is False else 'unknown_labels'] += 1
        support['false_accepts'] += target is False and accepted
        support['false_rejects'] += target is True and not accepted
    return {'eligible': len(eligible), 'total': len(records),
                    'coverage': len(eligible) / len(records) if records else None,
                    'excluded_ids': [row['id'] for row in records if row[mode]['input_truncated']],
                    'support': support}


def summarize(records):
    result = {'threshold': .5, 'pairs': []}
    by_id = {row['id']: row for row in records}
    groups = {'anchors': [row for row in records if row['role'] == 'anchor'],
              **{kind: [row for row in records if row.get('intervention') == kind]
                 for kind in ('evidence-free', 'source-swapped')}}
    for mode in ('active', 'bypass'):
        result[mode] = summarize_mode(records, mode)
        result[mode]['by_role_and_intervention'] = {
            name: summarize_mode(rows, mode) for name, rows in groups.items()}
    for row in records:
        if row['role'] != 'variant':
            continue
        anchor = by_id[row['original_id']]
        pair = {'anchor_id': anchor['id'], 'variant_id': row['id'], 'intervention': row['intervention'],
                'variant_support_target': row['targets']['support']}
        for mode in ('active', 'bypass'):
            a, v = anchor[mode]['scores'], row[mode]['scores']
            pair[mode] = {'anchor_support': None if a is None else a['support'],
                          'variant_support': None if v is None else v['support'],
                          'support_delta': None if a is None or v is None else v['support'] - a['support'],
                          'variant_rejected': None if v is None else v['support'] < .5}
        result['pairs'].append(pair)
    return result


def run(args):
    output, run_dir = Path(args.output), Path(args.run)
    if output.exists():
        raise FileExistsError(output)
    pack = load_controls(args.controls, getattr(args, 'labels', None))
    source_report = json.loads((run_dir / 'report.json').read_text())
    if source_report.get('data_manifest_sha256') != pack['manifest']['prepared_manifest_sha256']:
        raise ValueError('checkpoint data manifest differs from frozen controls provenance')
    if source_report.get('instructions') != probe.INSTRUCTIONS:
        raise ValueError('checkpoint report uses different quality instructions')
    if source_report.get('dtype') not in ('float32', 'bfloat16') or source_report.get('autocast_dtype') not in (None, 'bfloat16'):
        raise ValueError('unsupported computation dtype')
    reload_helper.configure_runtime(args.device)
    import torch
    from tensorcode.tools.chatbot import Chatbot
    # One model instance; ablate the workspace on that same owned foundation.
    model = Chatbot.from_pretrained(run_dir / 'model', device=args.device).eval()
    reload_helper.validate_conditioning(source_report, model)
    if source_report.get('foundation') != model.configuration().get('foundation'):
        raise ValueError('artifact foundation differs from source report')
    if {str(p.dtype).removeprefix('torch.') for p in model.parameters()} != {source_report['dtype']}:
        raise ValueError('artifact dtype differs from source report')
    if model.config['max_input_tokens'] != source_report.get('max_tokens'):
        raise ValueError('artifact input budget differs from source report')
    ids = {word: model.tokenizer(word, add_special_tokens=False)['input_ids'] for word in ('yes', 'no')}
    if any(len(value) != 1 for value in ids.values()) or ids['yes'] == ids['no'] or ids != source_report.get('label_ids'):
        raise ValueError('native yes/no label IDs differ')
    autocast = torch.bfloat16 if source_report.get('autocast_dtype') == 'bfloat16' else None
    records = evaluate(model, pack['anchors'], pack['variants'], yes_id=ids['yes'][0], no_id=ids['no'][0], autocast_dtype=autocast)
    result = {'role': 'frozen development evidence sensitivity; no promotion or performance qualification',
              'instructions': probe.INSTRUCTIONS, 'review_status': pack['review_status'],
              'records': records, 'summary': summarize(records), 'device': args.device,
              'dtype': source_report['dtype'], 'autocast_dtype': source_report.get('autocast_dtype'),
              'label_ids': ids, 'max_tokens': model.config['max_input_tokens'],
              'controls_manifest_sha256': reload_helper.sha256(Path(args.controls) / 'manifest.json'),
              'controls_files_sha256': pack['manifest']['files'],
              'labels_sha256': reload_helper.sha256(args.labels) if getattr(args, 'labels', None) else None,
              'source_report_sha256': reload_helper.sha256(run_dir / 'report.json'),
              'artifact_sha256': {name: reload_helper.sha256(run_dir / 'model' / name) for name in ('tensorcode_config.json', 'model.safetensors')},
              'script_sha256': reload_helper.sha256(__file__), 'probe_sha256': reload_helper.sha256(probe.__file__),
              'limitations': ['Authored development interventions, not reserved final data.',
                             'Conditional native yes/no scores are not calibrated factual truth.',
                             'Absolute yes/no probability mass is not measured; a confident conditional score can coexist with very low total yes/no mass.',
                             'Overflow excludes all axes for that input; unknown labels are not negatives.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    run(parser.parse_args())
