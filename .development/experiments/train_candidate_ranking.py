"""Bounded reviewed-candidate ranking adaptation; never an abstention classifier."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import re

AXES = ('support', 'completeness', 'constraints')


def helper_module():
    path = Path(__file__).resolve().parents[2] / 'examples/train_response_quality.py'
    spec = importlib.util.spec_from_file_location('ranking_quality_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def group_rows(rows):
    """Keep explicit failures negative; unresolved axes are never inferred."""
    grouped = {}; unresolved = []; seen = set()
    for row in rows:
        if row['id'] in seen:
            raise ValueError('duplicate candidate ID')
        seen.add(row['id'])
        key = row['question_id']
        group = grouped.setdefault(key, {'question_id': key, 'question': row['question'],
                                         'evidence': copy.deepcopy(row['evidence']), 'rows': []})
        if group['question'] != row['question'] or group['evidence'] != row['evidence']:
            raise ValueError('question group must have identical question and evidence')
        targets = row['targets']
        if set(targets) != set(AXES) or any(v is not None and type(v) is not bool for v in targets.values()):
            raise ValueError('targets must contain three boolean or unknown axes')
        if any(v is False for v in targets.values()):
            good = False
        elif all(v is True for v in targets.values()):
            good = True
        else:
            unresolved.append(row['id']); continue
        group['rows'].append(dict(copy.deepcopy(row), good=good))
    result = []
    for key in sorted(grouped):
        group = grouped[key]
        group['rows'].sort(key=lambda row: row['id'])
        if not group['rows']:
            continue
        group['inputs'] = {'question': group['question'],
                           'evidence': [{'source_id': e['id'], 'text': e['text']} for e in group['evidence']],
                           'hypotheses': [{'id': r['id'], 'text': r['candidate']} for r in group['rows']]}
        positives = sum(r['good'] for r in group['rows'])
        group['mixed'] = 0 < positives < len(group['rows'])
        group['targets'] = [float(r['good']) / positives for r in group['rows']] if positives else None
        result.append(group)
    return result, sorted(unresolved)


def check_lengths(rank, group):
    value = group['inputs']; rank.validate(value)
    texts = [value['question']] + [e['text'] for e in value['evidence']] + [h['text'] for h in value['hypotheses']]
    counts = ([len(rank.tokenizer(text, truncation=False)['input_ids']) for text in texts]
              if rank.tokenizer is not None else [max(1, len(re.findall(r'\w+|[^\w\s]', text.casefold()))) for text in texts])
    return {'segment_token_counts': counts, 'max_tokens': rank.config['max_tokens'],
            'overflow': any(n > rank.config['max_tokens'] for n in counts)}


def extract_rank(path):
    """Read only rank tensors from a complete local artifact; never build its generator."""
    from safetensors import safe_open
    from tensorcode.tools.chatbot import Chatbot
    from tensorcode.tools.investigator import Investigator
    path = Path(path)
    manifest = json.loads((path / 'tensorcode_config.json').read_text())
    for key, expected in [('format', Chatbot.artifact_format), ('version', Chatbot.artifact_version),
                          ('tool', Chatbot._tool_identity())]:
        if manifest.get(key) != expected or type(manifest.get(key)) is not type(expected):
            raise ValueError('expected complete Chatbot artifact')
    config = copy.deepcopy(manifest['config']['cognition']['investigator'])
    for key in list(config):
        if key in ('generator', 'retrieval_encoder') or key.startswith('verifier_'):
            del config[key]
    model = Investigator(config)
    state = {}
    with safe_open(path / 'model.safetensors', framework='pt', device='cpu') as archive:
        keys = set(archive.keys()); aliases = archive.metadata() or {}
        for name in model.state_dict():
            source = 'investigator.' + name
            source = source if source in keys else aliases.get(source)
            if source not in keys:
                raise ValueError(f'missing rank tensor: {name}')
            state[name] = archive.get_tensor(source)
    # Preserve constructor Parameter identities (including native tied embeddings)
    # and shared-storage views while restoring each saved dtype before copy-load.
    layouts = []
    for name, tensor in model.state_dict(keep_vars=True).items():
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes(), tensor.dtype, tensor.device)
        if not storage.nbytes():
            key = (*key, id(tensor))
        layouts.append((tensor, state[name].dtype, key, tensor.shape,
                        tensor.stride(), tensor.storage_offset()))
    converted = {}
    for tensor, dtype, key, shape, stride, offset in layouts:
        if key not in converted:
            flat = tensor.detach().as_strided((key[1] // tensor.element_size(),), (1,), 0)
            converted[key] = flat.to(dtype=dtype)
        storage = converted[key]
        if storage.dtype != dtype:
            raise ValueError('shared rank storage has incompatible artifact dtypes')
        tensor.data = storage.as_strided(shape, stride, offset)
    model.load_state_dict(state, strict=True)
    return model.eval()


def evaluate(model, groups):
    import torch
    records = []
    with torch.no_grad():
        for group in groups:
            modes = {}
            for name, ablation in [('active', None), ('bypass', 'bypass')]:
                scores = model.rank.compute(group['inputs'], workspace_ablation=ablation)[0]
                selected = int(scores.argmax())
                modes[name] = {'scores': scores.tolist(), 'selected_id': group['rows'][selected]['id'],
                               'top_good': group['rows'][selected]['good']}
            records.append({'question_id': group['question_id'], 'inputs': group['inputs'],
                            'labels': [{'id': r['id'], 'targets': r['targets'], 'good': r['good']} for r in group['rows']],
                            'mixed': group['mixed'], 'answerable': group['targets'] is not None, **modes})
    summary = {'groups': len(records), 'mixed_groups': sum(r['mixed'] for r in records),
               'unanswerable_ranking_groups': sum(not r['answerable'] for r in records)}
    for mode in ('active', 'bypass'):
        summary[mode] = {}
        for name, selected in [('answerable', [r for r in records if r['answerable']]),
                               ('mixed', [r for r in records if r['mixed']])]:
            correct = sum(r[mode]['top_good'] for r in selected)
            summary[mode][name] = {'top_good': correct, 'total': len(selected),
                                 'rate': correct / len(selected) if selected else None}
    return {'summary': summary, 'records': records}


def train(model, groups, output, helper, *, epochs=3, lr=.001, seed=20260924):
    import torch
    from tensorcode.training import Trainer
    groups = [g for g in groups if g['mixed']]
    if not groups:
        raise ValueError('no mixed training groups')
    output = Path(output)
    frozen_before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    trainer = Trainer.from_tool(model, optimizer=lambda p: torch.optim.AdamW(p, lr=lr))
    model.eval()  # deterministic dropout-free adapter learning
    rng = random.Random(seed); losses = []
    def step(group):
        experience = trainer.capture(group['inputs'], group['targets'], source='assistant-reviewed response-quality-v2 candidate preferences')
        return trainer.step(experience)
    for epoch in range(epochs):
        order = list(groups); rng.shuffle(order)
        values = [step(group) for group in order]
        losses.append(sum(values) / len(values))
        print(json.dumps({'epoch': epoch + 1, 'loss': losses[-1], 'steps': trainer.steps}), flush=True)
    trainer.save_checkpoint(output / 'training', progress={'epochs': epochs, 'seed': seed, 'groups': [g['question_id'] for g in groups]})
    expected_loss = step(groups[0]); expected = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    trainer.load_checkpoint(output / 'training')
    actual_loss = step(groups[0])
    exact = (expected_loss == actual_loss and helper.state_equal(expected, model.state_dict())
             and helper.state_equal(expected_optimizer, trainer.optimizer.state_dict()))
    if not exact:
        raise AssertionError('optimizer continuation differs')
    trainer.load_checkpoint(output / 'training'); model.eval()
    if any(not torch.equal(p, frozen_before[n]) for n, p in model.named_parameters() if not p.requires_grad):
        raise AssertionError('frozen weights changed')
    model.save_pretrained(output / 'model')
    from tensorcode.tools.investigator import Investigator
    restored = Investigator.from_pretrained(output / 'model', device=next(model.parameters()).device)
    if evaluate(model, groups) != evaluate(restored, groups):
        raise AssertionError('owned artifact predictions differ')
    return {'epochs': epochs, 'lr': lr, 'seed': seed, 'batch_groups': 1, 'steps': trainer.steps,
            'epoch_losses': losses, 'training_groups': [g['question_id'] for g in groups],
            'optimizer_continuation_exact': exact,
            'continuation_scope': 'one fixed next update; epoch shuffle Random state is not checkpointed',
            'owned_reload_exact': True, 'frozen_weights_unchanged': True,
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'trainable_tensor_names': list(before),
            'changed_trainable_tensors': sum(not torch.equal(before[n], p) for n, p in model.named_parameters() if p.requires_grad)}


def run(args):
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('real training requires the authorized CUDA host')
    if args.epochs < 1 or not 0 < args.lr < 1:
        raise ValueError('positive epochs and lr in (0,1) required')
    torch.set_num_threads(8); torch.manual_seed(20260924); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    helper = helper_module(); manifest, splits = helper.load_data(args.data)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    model = extract_rank(args.model).to('cuda')
    report = {'role': 'candidate-ranking development diagnostic; no promotion',
              'limitations': ['Assistant-reviewed supplied candidates, not generated answers or final evaluation.',
                             'Ranking always selects a candidate, including all-bad groups; it cannot establish abstention.',
                             'Small mixed groups; workspace comparisons are same-model ablations, not separately trained controls.'],
              'source_manifest_sha256': helper.sha256(Path(args.model) / 'tensorcode_config.json'),
              'source_weights_sha256': helper.sha256(Path(args.model) / 'model.safetensors'),
              'data_manifest_sha256': helper.sha256(Path(args.data) / 'manifest.json'),
              'script_sha256': helper.sha256(__file__), 'partitions': {}}
    prepared = {}
    for split, rows in splits.items():
        groups, unresolved = group_rows(rows); included = []; excluded = []
        for group in groups:
            coverage = check_lengths(model.rank, group)
            if coverage['overflow']:
                excluded.append({'question_id': group['question_id'], 'candidate_ids': [r['id'] for r in group['rows']], **coverage})
            else:
                included.append(group)
        prepared[split] = included
        report['partitions'][split] = {'unresolved_candidate_ids': unresolved, 'overflow_groups': excluded,
                                        'original': evaluate(model, included)}
    report['training'] = train(model, prepared['train'], output, helper, epochs=args.epochs, lr=args.lr)
    for split, groups in prepared.items():
        report['partitions'][split]['trained'] = evaluate(model, groups)
    report['artifact_sha256'] = {p.name: helper.sha256(p) for p in (output / 'model').iterdir() if p.is_file()}
    helper.write_json(output / 'report.json', report)
    print(json.dumps(report['partitions']['development']['trained']['summary']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True); parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=.001)
    run(parser.parse_args())
