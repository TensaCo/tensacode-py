"""Experiment-only native-foundation training control with frozen adapters.

Uses the same prepared data, prompts, scoring, and schedule as the joint probe.
The saved artifact remains an ordinary Chatbot; the primary experimental path
explicitly requests workspace bypass. No production behavior is monkeypatched.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import random
import weakref

import torch
from tensorcode.ops.base import Operation
from tensorcode.training import ToolTrainer


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load_module('native_control_shared_probe', Path(__file__).with_name('probe_generative_quality.py'))


class _NativeObjective(Operation):
    replayable = True

    def __init__(self, owner):
        self._owner = weakref.ref(owner)

    def parameters(self, recurse=True):
        return self._owner().model.parameters(recurse=recurse)

    def configuration(self):
        return {'operation': 'native_foundation_control.objective.v1',
                'owner': self._owner().configuration()}

    def forward(self, value, *, context=None):
        if context or not isinstance(value, dict) or set(value) != {'inputs', 'targets'}:
            raise ValueError('Native objective requires explicit inputs and targets only')
        return self._owner().model.loss_batch(value['inputs'], value['targets'],
                                              workspace_ablation='bypass').clone()


class NativeFoundationControl(torch.nn.Module):
    """Explicitly reconstructed training adapter; never a pretrained artifact type."""
    replayable = False
    training_inputs_include_targets = True

    def __init__(self, model):
        super().__init__()
        self.model = model
        model.requires_grad_(False)
        model.foundation.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        self.training_operation = _NativeObjective(self)

    def configuration(self):
        return {'experiment': 'native_foundation_control.v1', 'model': self.model.configuration(),
                'objective': 'teacher_forced_cross_entropy', 'workspace_ablation': 'bypass',
                'trainable_scope': 'model.foundation', 'checkpoint_scope': 'complete_owned_chatbot'}

    def operation_bindings(self):
        operations = {f'model.{key}': operation for key, operation in self.model.operation_bindings().items()}
        operations['native_objective'] = self.training_operation
        return operations


def make_trainer(model, *, foundation_lr):
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Native control requires float32 master parameters')
    control = NativeFoundationControl(model)
    trainer = ToolTrainer(control, optimizer=lambda parameters:
                          torch.optim.AdamW(parameters, lr=foundation_lr, foreach=False))
    # Match the joint experiment: deterministic dropout settings, gradients on.
    control.eval()
    return trainer


def adapter_digest(model):
    return probe.state_digest({name: tensor for name, tensor in model.state_dict().items()
                               if not name.startswith('foundation.')})


def train_native(model, rows, helper, output, *, epochs, batch_size,
                 foundation_lr=2e-5, autocast_dtype=None):
    if type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError('epochs and batch_size must be positive integers')
    if not 0 < foundation_lr < 1:
        raise ValueError('foundation learning rate must lie in (0, 1)')
    output = Path(output)
    pairs, excluded = [], []
    for row in rows:
        values = probe.prompts(helper.model_inputs(row))
        if any(len(model.tokenizer(prompt, truncation=False)['input_ids']) > model.config['max_input_tokens']
               for prompt in values.values()):
            excluded.append(row['id'])
            continue
        for axis, prompt in values.items():
            target = row['targets'][axis]
            if target is not None:
                pairs.append((row['id'], axis, prompt, 'yes' if target else 'no'))
    if not pairs:
        raise ValueError('No eligible supervised native-control prompts')
    trainer = make_trainer(model, foundation_lr=foundation_lr)
    foundation_before = helper.tensor_digest(model.foundation)
    adapters_before = adapter_digest(model)

    def step(batch):
        with probe.computation_context(model, autocast_dtype):
            with torch.no_grad():
                experience = trainer.capture([r[2] for r in batch], [r[3] for r in batch],
                                              source='assistant-reviewed response-quality-v2')
            return trainer.step(experience)

    rng = random.Random(20260923)
    losses = []
    for epoch in range(epochs):
        order = list(pairs)
        rng.shuffle(order)
        values = [step(order[start:start + batch_size]) for start in range(0, len(order), batch_size)]
        losses.append(sum(values) / len(values))
        print(json.dumps({'epoch': epoch + 1, 'loss': losses[-1], 'steps': trainer.steps}), flush=True)
    trainer.save_checkpoint(output / 'training', progress={
        'epochs': epochs, 'supervised_axis_examples': len(pairs), 'shuffle_rng_state': rng.getstate(),
        'shuffle_seed': 20260923, 'epoch_complete': True})
    continuation_batch = pairs[:batch_size]
    expected_loss = step(continuation_batch)
    expected_weights = probe.state_digest(model.state_dict())
    expected_optimizer = probe.state_digest(trainer.optimizer.state_dict())
    trainer.optimizer.zero_grad(set_to_none=True)
    del trainer
    gc.collect()
    if next(model.parameters()).device.type == 'cuda':
        torch.cuda.empty_cache()
    trainer = make_trainer(model, foundation_lr=foundation_lr)
    trainer.load_checkpoint(output / 'training')
    actual_loss = step(continuation_batch)
    exact = (actual_loss == expected_loss and expected_weights == probe.state_digest(model.state_dict())
             and expected_optimizer == probe.state_digest(trainer.optimizer.state_dict()))
    if not exact:
        raise AssertionError('Native-control optimizer continuation differed')
    trainer.optimizer.zero_grad(set_to_none=True)
    del trainer
    gc.collect()
    if next(model.parameters()).device.type == 'cuda':
        torch.cuda.empty_cache()
    trainer = make_trainer(model, foundation_lr=foundation_lr)
    trainer.load_checkpoint(output / 'training')
    model.eval()
    foundation_after = helper.tensor_digest(model.foundation)
    adapters_after = adapter_digest(model)
    if foundation_before == foundation_after:
        raise AssertionError('Native foundation did not change')
    if adapters_before != adapters_after:
        raise AssertionError('Frozen adapter state changed')
    model.save_pretrained(output / 'model')
    return {'epochs': epochs, 'batch': batch_size, 'seed': 20260923, 'foundation_lr': foundation_lr,
            'objective_path': 'bypass', 'trainable_scope': 'foundation_only',
            'checkpoint_scope': 'complete_owned_chatbot_via_explicit_training_wrapper',
            'loss': 'native teacher-forced yes/no sequence cross entropy',
            'supervised_axis_examples': len(pairs), 'excluded': excluded, 'epoch_losses': losses,
            'steps': trainer.steps, 'foundation_unchanged': False, 'adapter_unchanged': True,
            'adapter_parameters_changed': 0, 'adapter_sha256_before': adapters_before,
            'adapter_sha256_after': adapters_after, 'foundation_sha256_before': foundation_before,
            'foundation_sha256_after': foundation_after, 'optimizer_continuation_exact': exact,
            'continuation_scope': 'one fixed next batch; epoch-boundary shuffle RNG state recorded in progress',
            'autocast_dtype': str(autocast_dtype),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}


def run(args):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if not torch.cuda.is_available():
        raise RuntimeError('Real native-control runs require the authorized CUDA host')
    if not Path(args.foundation).is_dir():
        raise ValueError('Local foundation required')
    if args.epochs < 1 or args.batch < 1 or not 0 < args.foundation_lr < 1:
        raise ValueError('Invalid native-control schedule')
    torch.set_num_threads(8)
    torch.manual_seed(20260921)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    from tensorcode.tools.chatbot import Chatbot
    root = Path(__file__).resolve().parents[2]
    helper = load_module('native_control_quality_helpers', root / 'examples/train_response_quality.py')
    _, splits = helper.load_data(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model = Chatbot.from_foundation(args.foundation, revision=args.revision, local_files_only=True,
                                   max_input_tokens=512).to('cuda', dtype=torch.float32).eval().requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    ids = {word: model.tokenizer(word, add_special_tokens=False)['input_ids'] for word in ('yes', 'no')}
    if any(len(value) != 1 for value in ids.values()) or ids['yes'] == ids['no']:
        raise ValueError('Control requires distinct single-token yes/no labels')
    report = {'role': 'native-foundation-only development training control; no promotion',
              'primary_path': 'bypass', 'record_score_paths': {'scores': 'active', 'bypass_scores': 'bypass'},
              'instructions': probe.INSTRUCTIONS, 'foundation': model.configuration()['foundation'],
              **{key: model.config[key] for key in ('memory_update', 'memory_mode', 'workspace')},
              'foundation_asset_hashes': {p.name: helper.sha256(p) for p in sorted(Path(args.foundation).glob('*.safetensors'))},
              'script_sha256': helper.sha256(__file__), 'shared_probe_sha256': helper.sha256(probe.__file__),
              'data_manifest_sha256': helper.sha256(Path(args.data) / 'manifest.json'),
              'label_ids': ids, 'dtype': 'float32', 'autocast_dtype': 'bfloat16', 'max_tokens': 512,
              'threshold': .5,
              'score_semantics': 'first decoder-token probability conditioned on yes/no alternatives, not calibrated correctness',
              'limitations': ['Assistant-reviewed development labels; foundation exposure unknown.',
                             'Only native foundation is trained; all workspace/projection/gate parameters remain frozen.',
                             'Primary metrics are same_foundation_bypass_metrics; active-path metrics are an additional control.',
                             'Ordinary saved Chatbot retains its default active behavior; control evaluation explicitly requests bypass.',
                             'No reserved final questions or automatic qualification.']}
    with (output / 'before-progress.jsonl').open('x') as stream:
        report['before_training'] = probe.evaluate_splits(model, splits, helper, ids, stream,
            compare_workspace=True, autocast_dtype=torch.bfloat16)
    helper.write_json(output / 'before-training.json', report)
    report['training'] = train_native(model, splits['train'], helper, output, epochs=args.epochs,
                                     batch_size=args.batch, foundation_lr=args.foundation_lr,
                                     autocast_dtype=torch.bfloat16)
    with (output / 'progress.jsonl').open('x') as stream:
        report['splits'] = probe.evaluate_splits(model, splits, helper, ids, stream,
            compare_workspace=True, autocast_dtype=torch.bfloat16)
    report['cuda_memory'] = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                             'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    helper.write_json(output / 'report.json', report)
    print(json.dumps(report['splits']['development']['same_foundation_bypass_metrics']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foundation', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--foundation-lr', type=float, default=2e-5)
    run(parser.parse_args())
