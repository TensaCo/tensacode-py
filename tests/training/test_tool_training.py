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
        self.training_operation = Transform.from_module(Objective())

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
            self.training_operation = Transform.from_module(torch.nn.Linear(2, 2))
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
        self.prediction = Transform.from_module(torch.nn.Sequential(
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
    first = Transform.from_module(torch.nn.Linear(1, 1))
    second = Transform.from_module(torch.nn.Linear(1, 1))
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


def test_checkpoint_large_tensors_keep_metadata_small(tmp_path, monkeypatch):
    tool = Tool()
    tool.training_operation = Transform.from_module(torch.nn.Linear(1024, 1024))
    learner = training.ToolTrainer(tool)
    def reject_lists(self):
        raise AssertionError('Tensor converted to a Python list')
    monkeypatch.setattr(torch.Tensor, 'tolist', reject_lists)
    learner.save_checkpoint(tmp_path)
    manifest = tmp_path / 'training.json'
    assert manifest.stat().st_size < 50_000
    payload = json.loads(manifest.read_text())
    tensor_file = tmp_path / payload['tensors']['file']
    assert tensor_file.stat().st_size > 4_000_000
    learner.load_checkpoint(tmp_path)


@pytest.mark.parametrize('corruption', ['digest', 'dangling', 'shape', 'dtype', 'nonfinite', 'path'])
def test_binary_corruption_rejected_before_mutation(tmp_path, corruption):
    from safetensors.torch import load_file, save_file
    from tensorcode.training._tensor_store import digest
    learner = training.ToolTrainer(DropoutTool({'width': 4}))
    learner.save_checkpoint(tmp_path)
    path = tmp_path / 'training.json'
    payload = json.loads(path.read_text())
    tensor_path = tmp_path / payload['tensors']['file']
    if corruption == 'digest':
        with tensor_path.open('ab') as stream:
            stream.write(b'bad')
    elif corruption == 'path':
        payload['tensors']['file'] = '../elsewhere.safetensors'
    elif corruption == 'nonfinite':
        tensors = {key: tensor.clone() for key, tensor in load_file(tensor_path).items()}
        floating = next(tensor for tensor in tensors.values() if tensor.is_floating_point())
        floating.flatten()[0] = float('nan')
        save_file(tensors, str(tensor_path))
        payload['tensors']['sha256'] = digest(tensor_path)
    else:
        reference = payload['state']['torch_rng']
        reference[{'dangling': 'key', 'shape': 'shape', 'dtype': 'dtype'}[corruption]] = {
            'dangling': 'absent', 'shape': [123], 'dtype': 'float32'}[corruption]
    path.write_text(json.dumps(payload))
    before = [parameter.detach().clone() for parameter in learner.parameters]
    with pytest.raises(ValueError):
        learner.load_checkpoint(tmp_path)
    assert all(torch.equal(a, b) for a, b in zip(before, learner.parameters))


def test_failed_manifest_switch_preserves_previous_checkpoint(tmp_path, monkeypatch):
    import tensorcode.training.tool as module
    learner = training.ToolTrainer(DropoutTool({'width': 4}))
    learner.save_checkpoint(tmp_path)
    previous = (tmp_path / 'training.json').read_bytes()
    expected = [parameter.detach().clone() for parameter in learner.parameters]
    with torch.no_grad():
        for parameter in learner.parameters:
            parameter.add_(1)
    original = module._write
    def fail_manifest(path, data):
        if path.name == 'training.json':
            raise OSError('interrupted manifest write')
        return original(path, data)
    monkeypatch.setattr(module, '_write', fail_manifest)
    with pytest.raises(OSError):
        learner.save_checkpoint(tmp_path)
    assert (tmp_path / 'training.json').read_bytes() == previous
    learner.load_checkpoint(tmp_path)
    assert all(torch.equal(a, b) for a, b in zip(expected, learner.parameters))


def test_restore_snapshots_original_tensor_storages_only_once(tmp_path, monkeypatch):
    """Nested rollback transactions must not duplicate full model/Adam snapshots."""
    learner = training.ToolTrainer(DropoutTool({'width': 4}),
                                   optimizer=lambda params: torch.optim.Adam(params, lr=.01))
    learner.step(learner.capture(torch.ones(3, 4), torch.zeros(3, 1), source='review'))
    learner.save_checkpoint(tmp_path)
    tensors = list(learner.tool.state_dict().values())
    tensors += [value for slots in learner.optimizer.state.values()
                for value in slots.values() if isinstance(value, torch.Tensor)]
    copies = {value.untyped_storage().data_ptr(): 0 for value in tensors}
    original = torch.Tensor.__deepcopy__

    def count_copies(value, memo):
        pointer = value.untyped_storage().data_ptr()
        if pointer in copies and id(value) not in memo:
            copies[pointer] += 1
        return original(value, memo)

    monkeypatch.setattr(torch.Tensor, '__deepcopy__', count_copies)
    learner.load_checkpoint(tmp_path)
    assert set(copies.values()) == {1}, copies


def test_optimizer_apply_interruption_restores_entire_training_transaction(tmp_path, monkeypatch):
    from copy import deepcopy
    learner = training.ToolTrainer(DropoutTool({'width': 4}),
                                   optimizer=lambda params: torch.optim.Adam(params, lr=.01))
    session = learner.capture(torch.ones(3, 4), torch.zeros(3, 1), source='review')
    learner.step(session)
    learner.save_checkpoint(tmp_path, progress={'cursor': 1})
    learner.step(session)
    learner.optimizer.param_groups[0]['lr'] = .123
    learner.tool.eval()
    learner.tool.prediction.module[0].train()
    learner.progress = {'cursor': 2}
    weights = deepcopy(learner.tool.state_dict())
    optimizer = deepcopy(learner.optimizer.state_dict())
    modes = {name: module.training for name, module in learner.tool.named_modules()}
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    original = learner.optimizer.load_state_dict
    interrupted = False

    def apply_then_interrupt(state):
        nonlocal interrupted
        result = original(state)
        if not interrupted:
            interrupted = True
            random.random()
            torch.rand(3)
            learner.tool.train()
            raise KeyboardInterrupt('optimizer applied before interruption')
        return result

    monkeypatch.setattr(learner.optimizer, 'load_state_dict', apply_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match='optimizer applied'):
        learner.load_checkpoint(tmp_path)
    assert all(torch.equal(value, learner.tool.state_dict()[name]) for name, value in weights.items())
    restored = learner.optimizer.state_dict()
    assert restored['param_groups'] == optimizer['param_groups']
    assert restored['state'].keys() == optimizer['state'].keys()
    for key, slots in optimizer['state'].items():
        assert restored['state'][key].keys() == slots.keys()
        assert all(torch.equal(value, restored['state'][key][name]) for name, value in slots.items())
    assert {name: module.training for name, module in learner.tool.named_modules()} == modes
    assert random.getstate() == python_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert learner.steps == 2
    assert learner.progress == {'cursor': 2}
