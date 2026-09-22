"""Explicit supervised training and portable experience, with optional tensors."""
from .._internal.training.persistence import load as load_experience


class Trainer:
    """Train a declared tool objective or explicitly supervised operation graph.

    Construct with ``from_tool`` or ``from_ops``. Complete checkpoints preserve
    weights, optimizer, Python/Torch RNG, module modes, step count and progress.
    Experience and model deployment artifacts are saved separately.
    """

    def __init__(self):
        raise TypeError('Use Trainer.from_tool(...) or Trainer.from_ops(...)')

    @classmethod
    def from_tool(cls, tool, *, optimizer=None, lr=.001):
        """Use the tool's declared objective and activate its training policy."""
        from .._internal.training.tool import TrainingEngine
        result = object.__new__(cls)
        result._engine = TrainingEngine(tool.operation_bindings(), tool=tool, optimizer=optimizer, lr=lr)
        return result

    @classmethod
    def from_ops(cls, operations, *, optimizer=None, lr=.01, losses=None):
        """Use explicit trace supervision; retain existing operation modes."""
        from .._internal.training.tool import TrainingEngine
        result = object.__new__(cls)
        result._engine = TrainingEngine(operations, optimizer=optimizer, lr=lr, losses=losses)
        return result

    @property
    def operations(self):
        return self._engine.operations

    @property
    def parameters(self):
        return self._engine.parameters

    @property
    def optimizer(self):
        return self._engine.optimizer

    @property
    def steps(self):
        return self._engine.steps

    @steps.setter
    def steps(self, value):
        if type(value) is not int or value < 0:
            raise ValueError('steps must be a nonnegative integer')
        self._engine.steps = value

    @property
    def progress(self):
        return self._engine.progress

    @progress.setter
    def progress(self, value):
        if not isinstance(value, dict):
            raise TypeError('progress must be a dictionary')
        self._engine.progress = value

    @property
    def tool(self):
        return self._engine.tool

    def capture(self, inputs, targets, *, source):
        """Capture reviewed tool feedback; ops graphs use trace().supervise()."""
        return self._engine.capture(inputs, targets, source=source)

    def step(self, experience):
        """Apply one optimizer step to one captured experience."""
        return self._engine.step(experience)

    def fit(self, experiences, *, epochs=1):
        """Step through the experiences for ``epochs`` passes, in order."""
        return self._engine.fit(experiences, epochs=epochs)

    def save_checkpoint(self, path, *, progress=None):
        """Save complete continuation state; deploy weights with ``save_pretrained``."""
        return self._engine.save_checkpoint(path, progress=progress)

    def load_checkpoint(self, path):
        """Restore complete directories, or legacy model/optimizer-only files.

        Legacy files leave RNG, modes, steps and progress unchanged. Complete
        continuation requires matching devices/environment; NumPy and external
        data-loader state are not captured.
        """
        return self._engine.load_checkpoint(path)


__all__ = ['load_experience', 'Trainer', 'TemperatureCalibration',
           'evaluate_calibration', 'fit_threshold']


def __getattr__(name):
    if name not in {'TemperatureCalibration', 'evaluate_calibration', 'fit_threshold'}:
        raise AttributeError(name)
    from importlib import import_module
    value = getattr(import_module(f'{__name__}.calibration'), name)
    globals()[name] = value
    return value
