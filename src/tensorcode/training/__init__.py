"""Portable supervised experience and explicit optional tensor training.

Importing this module does not import torch. Tensor decoding, checkpoint loading
and Trainer construction require the vec extra.
"""
from .persistence import load
from .trainer import Trainer
from .tool import ToolTrainer
from .checkpoint import save_checkpoint, load_checkpoint

__all__ = ['load', 'Trainer', 'ToolTrainer', 'save_checkpoint', 'load_checkpoint',
           'TemperatureCalibration', 'evaluate_calibration', 'fit_threshold']


def __getattr__(name):
    if name not in {'TemperatureCalibration', 'evaluate_calibration', 'fit_threshold'}:
        raise AttributeError(name)
    from importlib import import_module
    value = getattr(import_module(f'{__name__}.calibration'), name)
    globals()[name] = value
    return value
