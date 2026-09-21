"""Portable supervised experience and explicit optional tensor training.

Importing this module does not import torch. Tensor decoding, checkpoint loading
and Trainer construction require the vec extra.
"""
from .persistence import load
from .trainer import Trainer
from .tool import ToolTrainer
from .checkpoint import save_checkpoint, load_checkpoint

__all__ = ['load', 'Trainer', 'ToolTrainer', 'save_checkpoint', 'load_checkpoint']
