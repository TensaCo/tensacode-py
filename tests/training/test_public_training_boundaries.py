import importlib.util
import subprocess
import sys

import pytest
import torch
from tensorcode import trace, training
from tensorcode.ops.vec import Transform


def test_public_trace_and_removed_paths():
    import tensorcode
    assert type(trace()) is tensorcode.Trace
    for name in ('tracing', 'training.tool', 'training.trainer', 'training.checkpoint', 'training.persistence', 'training._tensor_store'):
        assert importlib.util.find_spec('tensorcode.' + name) is None
    for name in ('ToolTrainer', 'load', 'save_checkpoint', 'load_checkpoint'):
        assert not hasattr(training, name)
    subprocess.run([sys.executable, '-c', 'import sys; import tensorcode; import tensorcode.training; assert "torch" not in sys.modules'], check=True)


def test_ops_complete_resume_and_capture_rejection(tmp_path):
    head = Transform.from_module(torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.4)))
    head.eval()
    learner = training.Trainer.from_ops({'head': head}, optimizer=lambda ps: torch.optim.Adam(ps, lr=.01))
    assert not head.training
    with pytest.raises(ValueError, match='trace.*supervise'):
        learner.capture(torch.ones(2), torch.zeros(2), source='review')
    head.train()
    with trace() as experience:
        out = head(torch.ones(2))
    experience.supervise(out, torch.zeros(2), loss='mse')
    learner.step(experience)
    learner.save_checkpoint(tmp_path, progress={'cursor': 3})
    expected_loss = learner.step(experience)
    expected = [p.detach().clone() for p in learner.parameters]
    learner.steps = 50
    assert learner.load_checkpoint(tmp_path) == {'cursor': 3}
    assert learner.steps == 1
    assert learner.step(experience) == expected_loss
    assert all(torch.equal(a, b) for a, b in zip(expected, learner.parameters))


def test_checkpoint_parameterless_external_and_nonmodule_wrapper(tmp_path):
    from tensorcode.ops.base import Operation

    class External(Operation):
        replayable = False
        def forward(self, value, *, context=None):
            return value

    class Wrapper(Operation):
        replayable = True
        def __init__(self):
            self._module = torch.nn.Linear(2, 2)
        def configuration(self):
            return {'width': 2}
        def parameters(self):
            return self._module.parameters()
        def named_parameters(self, **kwargs):
            return self._module.named_parameters(**kwargs)
        def state_dict(self):
            return self._module.state_dict()
        def load_state_dict(self, state, **kwargs):
            return self._module.load_state_dict(state, **kwargs)
        def forward(self, value, *, context=None):
            return self._module(value)

    external, wrapper = External(), Wrapper()
    learner = training.Trainer.from_ops({'external': external, 'wrapper': wrapper})
    with trace() as experience:
        output = wrapper(external(torch.ones(2)))
    experience.supervise(output, torch.zeros(2), loss='mse')
    learner.step(experience)
    learner.save_checkpoint(tmp_path)
    expected = learner.step(experience)
    learner.load_checkpoint(tmp_path)
    assert learner.step(experience) == expected


def test_trainable_without_restore_protocol_rejects_checkpoint(tmp_path):
    from tensorcode.ops.base import Operation
    class Unrestorable(Operation):
        replayable = True
        def __init__(self):
            self._weight = torch.nn.Parameter(torch.ones(2))
        def parameters(self):
            return [self._weight]
        def configuration(self):
            return {}
        def forward(self, value, *, context=None):
            return value * self._weight
    learner = training.Trainer.from_ops({'op': Unrestorable()})
    with pytest.raises(TypeError, match='state_dict|restor'):
        learner.save_checkpoint(tmp_path)


def test_legacy_file_restore_keeps_unavailable_training_state(tmp_path):
    import random
    from tensorcode._internal.training.checkpoint import save_checkpoint
    head = Transform.from_module(torch.nn.Linear(2, 2))
    learner = training.Trainer.from_ops({'head': head})
    path = tmp_path / 'legacy.json'
    save_checkpoint(path, operations=learner.operations, optimizer=learner.optimizer)
    expected = [p.detach().clone() for p in learner.parameters]
    with torch.no_grad():
        for p in learner.parameters:
            p.add_(10)
    head.eval()
    learner.steps = 9
    learner.progress = {'cursor': 12}
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    assert learner.load_checkpoint(path) == {'cursor': 12}
    assert learner.steps == 9 and not head.training
    assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(), torch_rng)
    assert all(torch.equal(a, b) for a, b in zip(expected, learner.parameters))


def test_explicit_factories_defaults_and_reference_identity():
    from tensorcode import Trace, InputRef, OutputRef
    from tensorcode._internal import tracing
    from test_tool_training import Tool
    assert (Trace, InputRef, OutputRef) == (tracing.Trace, tracing.InputRef, tracing.OutputRef)
    tool = training.Trainer.from_tool(Tool())
    ops = training.Trainer.from_ops(tool.operations)
    assert type(tool) is type(ops) is training.Trainer
    assert tool.optimizer.param_groups[0]['lr'] == .001
    assert ops.optimizer.param_groups[0]['lr'] == .01
    with pytest.raises(TypeError, match='from_tool.*from_ops'):
        training.Trainer()


def test_ops_checkpoint_owns_supplied_binding_mapping(tmp_path):
    head = Transform.from_module(torch.nn.Linear(2, 2))
    bindings = {'head': head}
    learner = training.Trainer.from_ops(bindings)
    bindings.clear()
    learner.save_checkpoint(tmp_path)
    learner.load_checkpoint(tmp_path)
    assert learner.operations == {'head': head}
