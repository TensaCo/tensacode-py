import json
import random

import pytest
import torch

from tensorcode import training
from tensorcode.ops.vec import Transform


class Objective(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, value):
        noise = torch.rand(()) + random.random()
        return ((self.weight * value['inputs'] + noise - value['targets']) ** 2).mean()


class Tool:
    training_inputs_include_targets = True

    def __init__(self):
        self.training_operation = Transform(Objective())

    def operation_bindings(self):
        return {'objective': self.training_operation}


def trainer():
    return training.ToolTrainer(Tool(), optimizer=lambda params: torch.optim.Adam(params, lr=.01))


def test_persisted_feedback_replays_on_fresh_tool(tmp_path):
    first = trainer()
    session = first.capture(torch.tensor([1., 2.]), torch.tensor([3., 4.]), source='reviewed:42')
    path = tmp_path / 'experience.json'
    session.save(path, operations=first.operations, release=True)
    second = trainer()
    restored = training.load(path, operations=second.operations)
    assert restored.supervisions[0].source == 'reviewed:42'
    assert torch.equal(restored.supervisions[0].target, torch.tensor([3., 4.]))
    before = second.parameters[0].detach().clone()
    second.step(restored)
    assert not torch.equal(before, second.parameters[0])
    assert second.steps == 1


def test_resume_restores_stochastic_next_step_and_optimizer(tmp_path):
    first = trainer()
    session = first.capture(torch.tensor([1., 2.]), torch.tensor([3., 4.]), source='reviewed:42')
    path = tmp_path / 'experience.json'
    session.save(path, operations=first.operations)
    first.step(session)
    first.save_checkpoint(tmp_path / 'resume', progress={'next_example': 8})
    expected_loss = first.step(session)
    expected = first.parameters[0].detach().clone()
    second = trainer()
    restored = training.load(path, operations=second.operations)
    progress = second.load_checkpoint(tmp_path / 'resume')
    assert progress == {'next_example': 8}
    assert second.steps == 1
    assert second.step(restored) == expected_loss
    assert torch.equal(second.parameters[0], expected)
    assert second.steps == 2


def test_invalid_rng_rejected_before_model_mutation(tmp_path):
    first = trainer()
    first.save_checkpoint(tmp_path)
    path = tmp_path / 'training.json'
    payload = json.loads(path.read_text())
    payload['state']['python_rng'] = 0
    path.write_text(json.dumps(payload))
    before = first.parameters[0].detach().clone()
    with pytest.raises((TypeError, ValueError)):
        first.load_checkpoint(tmp_path)
    assert torch.equal(before, first.parameters[0])


def test_feedback_source_required_and_logits_protocol():
    class LogitTool:
        def __init__(self):
            self.training_operation = Transform(torch.nn.Linear(2, 2))
        def operation_bindings(self):
            return {'prediction': self.training_operation}
        def training_loss(self, prediction, targets):
            return torch.nn.functional.cross_entropy(prediction, targets)
    tool = LogitTool()
    learner = training.ToolTrainer(tool)
    with pytest.raises(ValueError, match='source'):
        learner.capture(torch.ones(2), torch.tensor(1), source=' ')
    session = learner.capture(torch.ones(2), torch.tensor(1), source='human')
    assert learner.step(session) > 0


def test_owned_model_checkpoint_and_experience_round_trip(tmp_path):
    from tensorcode.tools.investigator import Investigator
    config = {'vocabulary': ['test', 'yes', 'no'], 'dimensions': 8, 'slots': 2, 'steps': 1}
    first = training.ToolTrainer(Investigator(config))
    inputs = {'question': 'test', 'evidence': [],
              'hypotheses': [{'id': 'y', 'text': 'yes'}, {'id': 'n', 'text': 'no'}]}
    session = first.capture(inputs, 'y', source='test:review')
    first.step(session)
    session.save(tmp_path / 'experience.json', operations=first.operations)
    first.save_checkpoint(tmp_path / 'resume')
    second = training.ToolTrainer(Investigator(config))
    second.load_checkpoint(tmp_path / 'resume')
    for expected, restored in zip(first.parameters, second.parameters):
        assert torch.equal(expected, restored)
    restored = training.load(tmp_path / 'experience.json', operations=second.operations)
    assert second.step(restored) > 0


from tensorcode._internal.pretrained import PretrainedTool


class DropoutTool(PretrainedTool):
    def __init__(self, config):
        super().__init__(config)
        self.prediction = Transform(torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.Dropout(.5), torch.nn.Linear(4, 1)))
        self.frozen = torch.nn.Dropout(.2)

    def train(self, mode=True):
        super().train(mode)
        self.frozen.eval()
        return self

    @property
    def training_operation(self):
        return self.prediction

    def operation_bindings(self):
        return {'prediction': self.prediction}

    def training_loss(self, output, targets):
        return ((output - targets) ** 2).mean()


def test_pretrained_dropout_resume_restores_exact_mixed_modes_and_rng(tmp_path):
    first = training.ToolTrainer(DropoutTool({'width': 4}))
    assert first.tool.training and not first.tool.frozen.training
    session = first.capture(torch.ones(3, 4), torch.zeros(3, 1), source='review')
    first.step(session)
    first.tool.save_pretrained(tmp_path / 'pretrained')
    session.save(tmp_path / 'experience.json', operations=first.operations)
    # Preserve a deliberately mixed mode beyond the tool's frozen policy.
    first.tool.prediction.module[0].eval()
    first.save_checkpoint(tmp_path / 'resume')
    expected_loss = first.step(session)
    expected_params = [parameter.detach().clone() for parameter in first.parameters]
    expected_rng = torch.rand(5)
    expected_python = random.random()

    restored_tool = DropoutTool.from_pretrained(tmp_path / 'pretrained')
    assert not restored_tool.training
    second = training.ToolTrainer(restored_tool)
    assert restored_tool.training and restored_tool.prediction.module[1].training
    assert not restored_tool.frozen.training
    restored_session = training.load(tmp_path / 'experience.json', operations=second.operations)
    second.load_checkpoint(tmp_path / 'resume')
    assert not restored_tool.prediction.module[0].training
    assert restored_tool.prediction.module[1].training
    assert not restored_tool.frozen.training
    assert second.step(restored_session) == expected_loss
    for expected, actual in zip(expected_params, second.parameters):
        assert torch.equal(expected, actual)
    assert torch.equal(expected_rng, torch.rand(5))
    assert expected_python == random.random()


def test_invalid_module_mode_rejected_before_mutation(tmp_path):
    learner = training.ToolTrainer(DropoutTool({'width': 4}))
    learner.save_checkpoint(tmp_path)
    path = tmp_path / 'training.json'
    payload = json.loads(path.read_text())
    payload['state']['modes']['tool']['prediction.module.1'] = 'train'
    path.write_text(json.dumps(payload))
    before = [p.detach().clone() for p in learner.parameters]
    with pytest.raises(ValueError, match='boolean'):
        learner.load_checkpoint(tmp_path)
    assert all(torch.equal(a, b) for a, b in zip(before, learner.parameters))
    assert learner.tool.training


def test_interrupted_resume_rolls_back_weights_modes_and_rng(tmp_path, monkeypatch):
    learner = training.ToolTrainer(DropoutTool({'width': 4}))
    learner.save_checkpoint(tmp_path)
    with torch.no_grad():
        for parameter in learner.parameters:
            parameter.add_(1)
    learner.tool.eval()
    before = [parameter.detach().clone() for parameter in learner.parameters]
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    original = torch.set_rng_state
    calls = 0

    def interrupt_once(state):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt('interrupted restore')
        return original(state)

    monkeypatch.setattr(torch, 'set_rng_state', interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        learner.load_checkpoint(tmp_path)
    assert all(torch.equal(a, b) for a, b in zip(before, learner.parameters))
    assert not learner.tool.training
    assert not learner.tool.prediction.module[1].training
    assert random.getstate() == python_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)


def test_direct_checkpoint_interruption_rolls_back_prior_module(tmp_path, monkeypatch):
    first = Transform(torch.nn.Linear(1, 1))
    second = Transform(torch.nn.Linear(1, 1))
    operations = {'first': first, 'second': second}
    path = tmp_path / 'checkpoint.json'
    training.save_checkpoint(path, operations=operations)
    with torch.no_grad():
        first.module.weight.add_(2)
        second.module.weight.add_(3)
    expected = {name: {key: tensor.clone() for key, tensor in op.state_dict().items()}
                for name, op in operations.items()}
    original = second.load_state_dict
    calls = 0

    def interrupt_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt('second module interrupted')
        return original(*args, **kwargs)

    monkeypatch.setattr(second, 'load_state_dict', interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        training.load_checkpoint(path, operations=operations)
    for name, op in operations.items():
        for key, tensor in op.state_dict().items():
            assert torch.equal(expected[name][key], tensor)
