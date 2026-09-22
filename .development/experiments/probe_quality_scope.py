"""Fixed development-only prompt comparison; no training, integration or promotion."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import importlib.util


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
probe = load_module('quality_scope_shared_probe', HERE / 'probe_generative_quality.py')
reload_helper = load_module('quality_scope_reload', HERE / 'verify_generative_quality_reload.py')
helper = load_module('quality_scope_inputs', HERE.parents[1] / 'examples/train_response_quality.py')
controls_helper = load_module('quality_scope_controls', HERE / 'evaluate_quality_evidence_controls.py')
gate_helper = load_module('quality_scope_gate', HERE / 'assess_quality_gate.py')
INSTRUCTIONS = {
    'support': 'Using only the supplied evidence, is every factual assertion in the proposed answer supported? Check the exact entities, relationships, categories, quantities and qualifiers. The question is a request, not factual evidence. Do not silently correct names, substitute related categories or transfer a fact from one entity to another. Answer yes only if every assertion is supported; otherwise answer no.',
    'completeness': probe.INSTRUCTIONS['completeness'],
    'constraints': 'Using only the supplied evidence, does the proposed answer correctly satisfy every restriction in the question? Treat claims made by the question as unverified unless the evidence establishes them. Check the exact entities, relationships, categories, quantities and qualifiers. Answer yes only if every restriction is satisfied; otherwise answer no.',
}
EXPECTED_WEIGHTS = '63040d019d9e4bc2d3d3d141e6f3c0085e1857b5135e1a78176185dc926a1561'
EXPECTED_FOUNDATION = '9a2a03d04c00c2cf41e861134d51795e23eb2eda59ed6dffff6021e834f4b85b'


def prompts(row, *, prompt_set):
    if prompt_set not in ('original', 'full_proposition'):
        raise ValueError('only the two fixed prompt sets are permitted')
    inputs = helper.model_inputs(row)
    from tensorcode._internal.response_quality import ResponseQualityAssessor
    ResponseQualityAssessor.validate(inputs)
    instructions = probe.INSTRUCTIONS if prompt_set == 'original' else INSTRUCTIONS
    return {axis: instruction + '\n' + json.dumps(inputs, ensure_ascii=False)
            for axis, instruction in instructions.items()}


def label_ids(model):
    ids = {word: model.tokenizer(word, add_special_tokens=False)['input_ids'] for word in ('yes', 'no')}
    if any(len(value) != 1 for value in ids.values()) or ids['yes'] == ids['no']:
        raise ValueError('distinct single-token yes/no labels required')
    return ids


def assess(model, row, *, prompt_set, yes_id, no_id, workspace_ablation='bypass', autocast_dtype=None):
    """Exactly the shared first-token scoring computation; only prompts vary."""
    import torch
    ids = label_ids(model)
    if ids != {'yes': [yes_id], 'no': [no_id]}:
        raise ValueError('supplied label IDs differ from fixed native yes/no labels')
    if workspace_ablation not in (None, 'bypass'):
        raise ValueError('only native bypass and active workspace are permitted')
    values = prompts(row, prompt_set=prompt_set)
    lengths = {axis: len(model.tokenizer(text, truncation=False)['input_ids']) for axis, text in values.items()}
    result = {'input_token_counts': lengths, 'input_truncated': any(n > model.config['max_input_tokens'] for n in lengths.values()), 'scores': None}
    if result['input_truncated']:
        return result
    with torch.no_grad(), probe.computation_context(model, autocast_dtype):
        state = model.encode_workspace(list(values.values()), workspace_ablation=workspace_ablation)
        labels = torch.full((len(values), 1), yes_id, dtype=torch.long, device=state['conditioning'].device)
        logits = model.decoder(dict(state, labels=labels))['logits'][:, 0, :]
        scores = torch.softmax(logits[:, [no_id, yes_id]].float(), dim=-1)[:, 1].tolist()
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
        raise ValueError('invalid native quality scores')
    result['scores'] = dict(zip(values, scores))
    return result


def evaluate(model, rows, prompt_set, ids, stream, split, *, autocast_dtype):
    records = []
    for row in rows:
        active = assess(model, row, prompt_set=prompt_set, yes_id=ids['yes'][0], no_id=ids['no'][0], workspace_ablation=None, autocast_dtype=autocast_dtype)
        bypass = assess(model, row, prompt_set=prompt_set, yes_id=ids['yes'][0], no_id=ids['no'][0], autocast_dtype=autocast_dtype)
        if any(active[key] != bypass[key] for key in ('input_truncated', 'input_token_counts')):
            raise ValueError('conditioning paths disagree on coverage')
        record = {'id': row['id'], **active, 'bypass_scores': bypass['scores']}
        records.append(record)
        if len(records) % 10 == 0 or len(records) == len(rows):
            print(json.dumps({'prompt_set': prompt_set, 'split': split, 'completed': len(records), 'total': len(rows)}), flush=True)
        stream.write(json.dumps({'prompt_set': prompt_set, 'split': split, **record}) + '\n'); stream.flush()
    eligible = [row for row, record in zip(rows, records) if not record['input_truncated']]
    scored = [record for record in records if not record['input_truncated']]
    return {'records': records, 'excluded': [record['id'] for record in records if record['input_truncated']],
            'metrics': helper.metrics(eligible, [r['scores'] for r in scored]),
            'same_foundation_bypass_metrics': helper.metrics(eligible, [r['bypass_scores'] for r in scored]),
            'gate': {mode: gate_helper.gate(rows, {r['id']: r for r in records}, score_key=key)
                     for mode, key in [('active', 'scores'), ('bypass', 'bypass_scores')]}}


def compare_archived(archived, actual):
    for split in ('calibration', 'development'):
        expected = archived['splits'][split]['records']
        observed = actual[split]['records']
        if len(expected) != len(observed):
            raise ValueError('original baseline record count differs')
        for old, new in zip(expected, observed):
            if any(old.get(key) != new.get(key) for key in ('id', 'input_token_counts', 'input_truncated', 'scores', 'bypass_scores')):
                raise ValueError(f'original baseline receipt differs: {split}/{new["id"]}')


def comparison(rows, original, replacement):
    by_old = {record['id']: record for record in original['records']}
    by_new = {record['id']: record for record in replacement['records']}
    common = [row for row in rows if not by_old[row['id']]['input_truncated'] and not by_new[row['id']]['input_truncated']]
    return {'total': len(rows), 'common_eligible_ids': [row['id'] for row in common],
            'newly_truncated_ids': sorted(set(replacement['excluded']) - set(original['excluded'])),
            'unchanged_truncated_ids': sorted(set(replacement['excluded']) & set(original['excluded'])),
            'newly_eligible_ids': sorted(set(original['excluded']) - set(replacement['excluded'])),
            'common_eligible_gates': {name: {mode: gate_helper.gate(common, {row['id']: records[row['id']] for row in common}, score_key=key)
                for mode, key in [('active', 'scores'), ('bypass', 'bypass_scores')]}
                for name, records in [('original', by_old), ('full_proposition', by_new)]}}


def controls_records(rows, block):
    result = []
    for row, receipt in zip(rows, block['records'], strict=True):
        base = {'input_token_counts': receipt['input_token_counts'], 'input_truncated': receipt['input_truncated']}
        result.append({'id': row['id'], 'role': 'variant' if 'original_id' in row else 'anchor',
                       'targets': copy.deepcopy(row['targets']), 'review': copy.deepcopy(row.get('review', {})),
                       **{key: row[key] for key in ('original_id', 'intervention') if key in row},
                       'active': dict(base, scores=receipt['scores']), 'bypass': dict(base, scores=receipt['bypass_scores'])})
    return result


def run(args):
    # This CLI is deliberately restricted to the already-reviewed checkpoint/data.
    reload_helper.configure_runtime('cuda')
    import torch
    from tensorcode.tools.chatbot import Chatbot
    run_dir, output = Path(args.run), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    archived = json.loads((run_dir / 'report.json').read_text())
    if archived.get('instructions') != probe.INSTRUCTIONS or archived.get('dtype') != 'float32' or archived.get('autocast_dtype') != 'bfloat16':
        raise ValueError('archived prompt/dtype settings differ from fixed protocol')
    manifest, splits = helper.load_data(args.data)
    if [len(splits[name]) for name in ('calibration', 'development')] != [92, 91]:
        raise ValueError('fixed calibration/development counts differ')
    pack = controls_helper.load_controls(args.controls, args.labels)
    provenance = controls_helper.validate_data_provenance(archived, pack, args.data)
    rows_controls = pack['anchors'] + pack['variants']
    expected_controls = json.loads(Path(args.baseline_controls).read_text())
    if (expected_controls.get('instructions') != probe.INSTRUCTIONS
            or expected_controls.get('labels_sha256') != reload_helper.sha256(args.labels)
            or expected_controls.get('controls_manifest_sha256') != reload_helper.sha256(Path(args.controls) / 'manifest.json')):
        raise ValueError('original control baseline labels, prompts or control pack differ')
    weights_hash = reload_helper.sha256(run_dir / 'model/model.safetensors')
    if weights_hash != EXPECTED_WEIGHTS:
        raise ValueError('artifact differs from pinned augmented checkpoint')
    model = Chatbot.from_pretrained(run_dir / 'model', device='cuda').eval().requires_grad_(False)
    if {parameter.dtype for parameter in model.parameters()} != {torch.float32}:
        raise ValueError('fp32 parameters required; no conversion permitted')
    reload_helper.validate_conditioning(archived, model)
    if (archived.get('foundation') != model.configuration().get('foundation')
            or archived.get('max_tokens') != model.config['max_input_tokens']):
        raise ValueError('archived foundation/token budget differs')
    if helper.tensor_digest(model.foundation) != EXPECTED_FOUNDATION:
        raise ValueError('foundation digest differs from pinned protocol')
    ids = label_ids(model)
    if ids != archived.get('label_ids'):
        raise ValueError('archived labels differ')
    for split in ('calibration', 'development'):
        gate_helper.validate_block(archived['splits'][split], splits[split])
    before = probe.state_digest(model.state_dict())
    output.mkdir(parents=True)
    metadata = {'role': 'fixed-weight development-derived prompt diagnostic; no training or qualification',
                'threshold': .5, 'dtype': 'float32', 'autocast_dtype': 'bfloat16', 'device': 'cuda',
                'label_ids': ids, 'max_tokens': model.config['max_input_tokens'],
                'conditioning_paths': {'scores': 'workspace_active', 'bypass_scores': 'native_bypass'},
                'data_manifest_sha256': reload_helper.sha256(Path(args.data) / 'manifest.json'),
                'data_files_sha256': manifest['files'], 'controls_provenance': provenance,
                'controls_manifest_sha256': reload_helper.sha256(Path(args.controls) / 'manifest.json'),
                'controls_files_sha256': pack['manifest']['files'], 'labels_sha256': reload_helper.sha256(args.labels),
                'baseline_report_sha256': reload_helper.sha256(run_dir / 'report.json'),
                'baseline_controls_sha256': reload_helper.sha256(args.baseline_controls),
                'artifact_sha256': {name: reload_helper.sha256(run_dir / 'model' / name) for name in ('tensorcode_config.json', 'model.safetensors')},
                'configuration': model.configuration(), 'state_digest_before': before,
                'script_hashes': {Path(module.__file__).name: reload_helper.sha256(module.__file__) for module in (probe, helper, controls_helper, gate_helper, reload_helper)},
                'script_sha256': reload_helper.sha256(__file__),
                'limitations': ['Known development-derived instructions, not final evaluation.', 'Conditional yes/no scores are not calibrated truth.', 'Longer prompts can overflow; compare common eligible rows.', 'Unknown control labels remain unknown.']}
    results = {}
    with (output / 'progress.jsonl').open('x') as stream:
        for prompt_set in ('original', 'full_proposition'):
            blocks = {split: evaluate(model, splits[split], prompt_set, ids, stream, split, autocast_dtype=torch.bfloat16)
                      for split in ('calibration', 'development')}
            control_block = evaluate(model, rows_controls, prompt_set, ids, stream, 'controls', autocast_dtype=torch.bfloat16)
            control_records = controls_records(rows_controls, control_block)
            report = dict(metadata, instructions=probe.INSTRUCTIONS if prompt_set == 'original' else INSTRUCTIONS,
                          splits=blocks, controls={'records': control_records, 'summary': controls_helper.summarize(control_records),
                                    'gate': control_block['gate'], 'excluded': control_block['excluded']})
            helper.write_json(output / f'{prompt_set}-report.json', report)
            if prompt_set == 'original':
                compare_archived(archived, blocks)
                expected_by_id = {row['id']: row for row in expected_controls['records']}
                if len(expected_by_id) != len(expected_controls['records']) or set(expected_by_id) != {row['id'] for row in control_records}:
                    raise ValueError('original control baseline IDs differ')
                for record in control_records:
                    if expected_by_id[record['id']]['targets'] != record['targets'] or any(expected_by_id[record['id']][mode] != record[mode] for mode in ('active', 'bypass')):
                        raise ValueError(f'original control baseline receipt differs: {record["id"]}')
            results[prompt_set] = {'splits': blocks, 'control_block': control_block}
    after = probe.state_digest(model.state_dict())
    if after != before:
        raise AssertionError('inference mutated model state')
    final = dict(metadata, state_digest_after=after, model_unchanged=True, original_baseline_exact=True,
                 comparisons={split: comparison(splits[split], results['original']['splits'][split], results['full_proposition']['splits'][split]) for split in ('calibration', 'development')},
                 controls_comparison=comparison(rows_controls, results['original']['control_block'], results['full_proposition']['control_block']))
    helper.write_json(output / 'report.json', final)
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'data', 'controls', 'labels', 'baseline-controls', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cuda',), default='cuda')
    run(parser.parse_args())
