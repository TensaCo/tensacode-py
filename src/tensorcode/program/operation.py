"""Replayable operation integration for trace() and Trainer.from_ops()."""
import torch
from tensorcode.ops.base import Operation
from .core import Program


class ProgramOperation(Operation, torch.nn.Module):
    replayable = True

    def __init__(self, artifact):
        torch.nn.Module.__init__(self)
        program = Program(artifact)
        if any(n['op'] in ('call','state') for n in artifact['nodes']):
            raise ValueError('replayable operations require pure, stateless programs')
        self.program = program.to_tensor_module()

    def forward(self, inputs, *, context=None):
        if context: raise ValueError('program inputs must be explicit; context is unsupported')
        return self.program(inputs)

    def configuration(self): return self.program.program.graph()
    def artifact(self): return self.program.artifact().graph()

ProgramOperation.__module__ = "tensorcode.program"
