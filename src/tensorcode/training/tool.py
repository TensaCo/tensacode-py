"""Explicit supervised replay and resumable training for owned tool models."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile

from ..tracing import trace
from .checkpoint import load_checkpoint, save_checkpoint
from .persistence import _read, _write
from .trainer import Trainer
from ._tensor_store import TensorStore


class ToolTrainer(Trainer):
    """Train a tool's declared replayable operation with reviewed feedback.

    Construction activates the tool's training mode through its own ``train``
    policy. Tools expose ``training_operation`` and stable ``operation_bindings()``.
    Normally the operation consumes inputs and ``training_loss(output, targets)``
    computes the objective. Teacher-forced tools explicitly declare
    ``training_inputs_include_targets = True`` and return a scalar objective from
    an ``{'inputs': ..., 'targets': ...}`` envelope. Targets must only condition
    the decoder/objective, never the input encoder or cognitive workspace.
    """

    def __init__(self, tool, *, optimizer=None, lr=0.001):
        self.tool = tool
        self.training_operation = tool.training_operation
        operations = tool.operation_bindings()
        if not any(op is self.training_operation for op in operations.values()):
            raise ValueError('training_operation must appear in operation_bindings()')
        self.checkpoint_operations = ({'tool': tool} if callable(getattr(tool, 'state_dict', None))
                                      and callable(getattr(tool, 'configuration', None)) else operations)
        self.joint_objective = bool(getattr(tool, 'training_inputs_include_targets', False))
        loss = (lambda output, target: output) if self.joint_objective else tool.training_loss
        super().__init__(operations, optimizer=optimizer, lr=lr,
                         losses={'tool_objective': loss})
        self.steps = 0
        self.progress = {}
        if callable(getattr(tool, 'train', None)):
            tool.train()

    def _mode_modules(self):
        return {name: dict(op.named_modules()) for name, op in self.checkpoint_operations.items()}

    def capture(self, inputs, targets, *, source):
        """Capture supervised experience, without performing an optimizer step.

        ``source`` identifies the reviewer/dataset supplying the target. Capture
        stores explicit targets; it does not infer correctness from predictions.
        Use Session.save(..., operations=trainer.operations) for durable replay.
        """
        if not isinstance(source, str) or not source.strip():
            raise ValueError('Feedback source must be a nonempty string')
        value = {'inputs': inputs, 'targets': targets} if self.joint_objective else inputs
        with trace() as session:
            output = self.training_operation(value)
        session.supervise(output, targets, loss='tool_objective', source=source)
        return session

    def step(self, session):
        loss = super().step(session)
        self.steps += 1
        return loss

    def save_checkpoint(self, directory, *, progress=None):
        """Save weights, optimizer, module modes, RNG and progress without pickle.

        An atomic JSON manifest selects immutable, checksummed safetensors.
        Older tensor generations remain valid for concurrent readers.
        Runtime sessions and collected experience are saved separately.
        """
        import torch
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        codec = TensorStore()
        progress = self.progress if progress is None else progress
        if not isinstance(progress, dict):
            raise TypeError('progress must be a dictionary')
        state = {'modes': {name: {key: module.training for key, module in modules.items()}
                           for name, modules in self._mode_modules().items()},
                 'steps': self.steps, 'progress': codec.encode(progress),
                 'python_rng': codec.encode(random.getstate()),
                 'torch_rng': codec.encode(torch.get_rng_state()),
                 'cuda_rng': codec.encode(torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'model.json'
            save_checkpoint(path, operations=self.checkpoint_operations, optimizer=self.optimizer, _codec=codec)
            model = json.loads(path.read_text())
        tensor_file = codec.write(directory)
        _write(directory / 'training.json', {'format': 'tensorcode.tool_training',
               'version': 1, 'model': model, 'state': state, 'tensors': tensor_file})
        self.progress = deepcopy(progress)

    def load_checkpoint(self, directory):
        """Restore an initialized matching tool/optimizer and return progress.

        Exact stochastic continuation requires matching devices and execution
        environment. NumPy RNG and external data-loader state are not captured;
        callers should record their own data cursor in ``progress``.
        """
        import torch
        payload = _read(Path(directory) / 'training.json', 'tensorcode.tool_training')
        if (not isinstance(payload, dict) or set(payload) != {'format', 'version', 'model', 'state', 'tensors'}
                or payload['format'] != 'tensorcode.tool_training' or payload['version'] != 1):
            raise ValueError('Malformed tool training checkpoint')
        state = payload['state']
        if not isinstance(state, dict) or set(state) != {'modes', 'steps', 'progress', 'python_rng', 'torch_rng', 'cuda_rng'}:
            raise ValueError('Malformed training progress')
        if type(state['steps']) is not int or state['steps'] < 0:
            raise ValueError('Invalid training step count')
        modules = self._mode_modules()
        modes = state['modes']
        if not isinstance(modes, dict) or set(modes) != set(modules):
            raise ValueError('Checkpoint module mode topology differs')
        seen_modes = {}
        for name, children in modules.items():
            if not isinstance(modes[name], dict) or set(modes[name]) != set(children):
                raise ValueError('Checkpoint module mode topology differs')
            for key, module in children.items():
                flag = modes[name][key]
                if type(flag) is not bool:
                    raise ValueError('Module training mode must be boolean')
                if id(module) in seen_modes and seen_modes[id(module)] != flag:
                    raise ValueError('Contradictory shared module training modes')
                seen_modes[id(module)] = flag
        codec = TensorStore.read(directory, payload['tensors'], payload)
        progress = codec.decode(state['progress'])
        if not isinstance(progress, dict):
            raise ValueError('Invalid training progress')
        python_rng = codec.decode(state['python_rng'])
        torch_rng = codec.decode(state['torch_rng'])
        cuda_rng = codec.decode(state['cuda_rng'])
        random.Random().setstate(python_rng)
        torch.Generator().set_state(torch_rng)
        if not isinstance(cuda_rng, list):
            raise ValueError('Invalid CUDA RNG states')
        if cuda_rng:
            if not torch.cuda.is_available() or len(cuda_rng) != torch.cuda.device_count():
                raise ValueError('CUDA device topology differs from training checkpoint')
            for index, rng in enumerate(cuda_rng):
                torch.Generator(device=f'cuda:{index}').set_state(rng)
        originals = {name: deepcopy(op.state_dict()) for name, op in self.checkpoint_operations.items()}
        original_optimizer = deepcopy(self.optimizer.state_dict())
        original_modes = [(module, module.training) for children in modules.values() for module in children.values()]
        original_python, original_torch = random.getstate(), torch.get_rng_state()
        original_cuda = torch.cuda.get_rng_state_all() if cuda_rng else []
        try:
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'model.json'
                _write(path, payload['model'])
                load_checkpoint(path, operations=self.checkpoint_operations, optimizer=self.optimizer, _codec=codec)
            # Set exact local flags, without recursively resetting mixed modes or
            # invoking user train() overrides that may force a different policy.
            for name, children in modules.items():
                for key, module in children.items():
                    module.__dict__['training'] = modes[name][key]
            random.setstate(python_rng)
            torch.set_rng_state(torch_rng)
            if cuda_rng:
                torch.cuda.set_rng_state_all(cuda_rng)
        except BaseException:
            for name, op in self.checkpoint_operations.items():
                op.load_state_dict(originals[name], strict=True)
            self.optimizer.load_state_dict(original_optimizer)
            for module, flag in original_modes:
                module.__dict__['training'] = flag
            random.setstate(original_python)
            torch.set_rng_state(original_torch)
            if original_cuda:
                torch.cuda.set_rng_state_all(original_cuda)
            raise
        self.steps, self.progress = state['steps'], progress
        return deepcopy(progress)
