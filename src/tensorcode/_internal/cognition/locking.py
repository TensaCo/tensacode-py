"""Explicit source-grounded session composition; generated claims remain hypotheses."""
from __future__ import annotations

import hashlib
import json
import threading
from functools import wraps

import torch



_LOCK_CREATION = threading.Lock()


def _ensure_lock(investigator):
    with _LOCK_CREATION:
        if not hasattr(investigator, '_cognition_lock'):
            investigator._cognition_lock = threading.RLock()
        if not hasattr(investigator, '_cognition_fingerprint'):
            investigator._cognition_fingerprint = _ModelFingerprint()


def _model_locked(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self.investigator._cognition_lock:
            return method(self, *args, **kwargs)
    return locked


class _ModelFingerprint:
    """Cache content hashes by tensor identity/version and configuration.

    Standard optimizer/no_grad mutations invalidate the hash. Mutating via .data
    bypasses PyTorch version counters and is unsupported; call invalidate after
    any such external mutation. Concurrent weight updates are unsupported.
    """
    def __init__(self):
        self._key = None
        self._digest = None

    def invalidate(self):
        self._key = None

    def __call__(self, modules, config):
        serialized = json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False)
        tensors = [(f'{prefix}.{name}', tensor) for prefix, module in modules
                   for name, tensor in list(module.named_parameters()) + list(module.named_buffers())]
        key = (serialized, tuple((name, id(t), t._version, str(t.dtype), str(t.device), tuple(t.shape)) for name, t in tensors))
        if key != self._key:
            digest = hashlib.sha256(serialized.encode())
            for name, tensor in tensors:
                digest.update(name.encode())
                digest.update(str(tensor.dtype).encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            self._digest = digest.hexdigest()
            self._key = key
        return self._digest
