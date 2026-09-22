"""Tiny random-model checks for an experiment-only native training control."""
import importlib.util
import json
from pathlib import Path
import runpy

import pytest
import torch

ROOT = Path(__file__).parents[2]


def runner():
    spec = importlib.util.spec_from_file_location('native_quality_control', ROOT / '.development/experiments/probe_native_quality.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model():
    from tensorcode.tools.chatbot import Chatbot
    config = runpy.run_path(str(ROOT / 'tests/models/test_chatbot_model.py'))['tiny_config']()
    config['max_input_tokens'] = 256
    tokenizer = json.loads(config['tokenizer_json'])
    tokenizer['model']['vocab'].update(yes=8, no=9)
    config['tokenizer_json'] = json.dumps(tokenizer)
    config['foundation_config']['vocab_size'] = 10
    return Chatbot(config).eval()


def test_native_objective_parity_and_parameter_scope():
    mod = runner()
    owned = model()
    control = mod.NativeFoundationControl(owned).eval()
    value = {'inputs': ['hello world'], 'targets': ['answer']}
    loss = control.training_operation(value)
    assert torch.equal(loss, owned.loss_batch(value['inputs'], value['targets'], workspace_ablation='bypass'))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in owned.foundation.parameters())
    assert all(not p.requires_grad and p.grad is None for name, p in owned.named_parameters()
               if not name.startswith('foundation.'))


def test_native_experience_and_optimizer_resume_include_complete_model(tmp_path):
    from tensorcode.training import load_experience
    mod = runner()
    owned = model()
    trainer = mod.make_trainer(owned, foundation_lr=2e-5)
    frozen = mod.adapter_digest(owned)
    experience = trainer.capture(['hello world'], ['answer'], source='authored mechanism fixture')
    experience.save(tmp_path / 'experience.json', operations=trainer.operations)
    trainer.step(experience)
    assert mod.adapter_digest(owned) == frozen
    trainer.save_checkpoint(tmp_path / 'training', progress={'epochs': 1})
    expected_loss = trainer.step(experience)
    expected_model = mod.probe.state_digest(owned.state_dict())
    expected_optimizer = mod.probe.state_digest(trainer.optimizer.state_dict())
    fresh = model()
    resumed = mod.make_trainer(fresh, foundation_lr=2e-5)
    assert resumed.load_checkpoint(tmp_path / 'training') == {'epochs': 1}
    replay = load_experience(tmp_path / 'experience.json', operations=resumed.operations)
    assert resumed.step(replay) == expected_loss
    assert mod.probe.state_digest(fresh.state_dict()) == expected_model
    assert mod.probe.state_digest(resumed.optimizer.state_dict()) == expected_optimizer
    assert mod.adapter_digest(fresh) == frozen


def test_native_checkpoint_rejects_different_objective_configuration(tmp_path):
    mod = runner()
    trainer = mod.make_trainer(model(), foundation_lr=2e-5)
    trainer.save_checkpoint(tmp_path / 'training')
    fresh = mod.make_trainer(model(), foundation_lr=2e-5)
    original = fresh.tool.configuration
    fresh.tool.configuration = lambda: dict(original(), objective='different')
    with pytest.raises(ValueError):
        fresh.load_checkpoint(tmp_path / 'training')


def test_native_runner_preserves_adapters_and_saves_ordinary_chatbot(tmp_path):
    from tensorcode.tools.chatbot import Chatbot
    mod = runner()
    owned = model()
    helper = mod.load_module('native_test_data_helper', ROOT / 'examples/train_response_quality.py')
    row = {'id': 'fixture', 'question': 'hello', 'candidate': 'world',
           'evidence': [{'id': 'a', 'text': 'hello world'}],
           'targets': {'support': True, 'completeness': False, 'constraints': None}}
    result = mod.train_native(owned, [row], helper, tmp_path, epochs=1, batch_size=2)
    assert result['supervised_axis_examples'] == 2
    assert result['adapter_unchanged'] and not result['foundation_unchanged']
    assert result['optimizer_continuation_exact']
    assert result['objective_path'] == 'bypass'
    restored = Chatbot.from_pretrained(tmp_path / 'model').eval()
    assert torch.equal(owned.loss_batch(['hello'], ['yes'], workspace_ablation='bypass'),
                       restored.loss_batch(['hello'], ['yes'], workspace_ablation='bypass'))
    assert owned.configuration() == restored.configuration()


def test_report_score_fields_keep_generic_reload_verifier_convention():
    import io
    mod = runner()
    owned = model()
    helper = mod.load_module('native_score_test_helper', ROOT / 'examples/train_response_quality.py')
    row = {'id': 'fixture', 'question': 'hello', 'candidate': 'world',
           'evidence': [{'id': 'a', 'text': 'hello world'}],
           'targets': {'support': True, 'completeness': False, 'constraints': None}}
    results = mod.probe.evaluate_splits(owned, {'calibration': [row], 'development': [row]}, helper,
                                       {'yes': [8], 'no': [9]}, io.StringIO(), compare_workspace=True)
    saved = results['development']['records'][0]
    inputs = helper.model_inputs(row)
    assert saved['scores'] == mod.probe.assess(owned, inputs, yes_id=8, no_id=9,
                                               workspace_ablation=None)['scores']
    assert saved['bypass_scores'] == mod.probe.assess(owned, inputs, yes_id=8, no_id=9)['scores']
    assert 'same_foundation_bypass_metrics' in results['development']
