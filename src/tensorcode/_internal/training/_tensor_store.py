"""Private safetensors codec for bounded-size training checkpoint metadata."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import uuid

from .persistence import Codec
from tensorcode._internal.tracing import _tensor


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


class TensorStore(Codec):
    def __init__(self, tensors=None):
        super().__init__()
        self.tensors = {} if tensors is None else tensors
        self.aliases = {}
        self.sources = []

    def encode(self, value):
        if not _tensor(value):
            return super().encode(value)
        import torch
        if value.layout != torch.strided or value.is_complex():
            raise TypeError('Only dense real tensors have a checkpoint codec')
        alias = (str(value.device), value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype)
        key = self.aliases.get(alias)
        if key is None:
            key = f'tensor_{len(self.tensors)}'
            tensor = value.detach().cpu().contiguous().clone()
            if not torch.isfinite(tensor).all():
                raise ValueError('Nonfinite tensor in checkpoint')
            self.tensors[key] = tensor
            self.aliases[alias] = key
            self.sources.append(value)
        return {'type': 'tensor_ref', 'key': key, 'shape': list(value.shape),
                'dtype': str(value.dtype).removeprefix('torch.')}

    def decode(self, value):
        if isinstance(value, dict) and value.get('type') == 'tensor_ref':
            if (set(value) != {'type', 'key', 'shape', 'dtype'} or not isinstance(value['key'], str)
                    or not isinstance(value['shape'], list)
                    or any(type(size) is not int or size < 0 for size in value['shape'])
                    or not isinstance(value['dtype'], str)):
                raise ValueError('Malformed checkpoint tensor reference')
            tensor = self.tensors.get(value['key'])
            if tensor is None:
                raise ValueError('Dangling checkpoint tensor reference')
            if value['shape'] != list(tensor.shape) or value['dtype'] != str(tensor.dtype).removeprefix('torch.'):
                raise ValueError('Checkpoint tensor reference shape or dtype differs')
            return tensor
        return super().decode(value)

    def write(self, directory):
        from safetensors.torch import save_file
        filename = f'tensors-{uuid.uuid4().hex}.safetensors'
        path = Path(directory) / filename
        save_file(self.tensors, str(path))
        return {'file': filename, 'sha256': digest(path)}

    @classmethod
    def read(cls, directory, record, payload):
        from safetensors.torch import load_file
        import torch
        if (not isinstance(record, dict) or set(record) != {'file', 'sha256'}
                or not isinstance(record['file'], str)
                or re.fullmatch(r'tensors-[0-9a-f]{32}\.safetensors', record['file']) is None
                or not isinstance(record['sha256'], str)
                or re.fullmatch(r'[0-9a-f]{64}', record['sha256']) is None):
            raise ValueError('Invalid checkpoint tensor file reference')
        path = Path(directory) / record['file']
        if path.is_symlink():
            raise ValueError('Checkpoint tensor file must not be a symlink')
        if digest(path) != record['sha256']:
            raise ValueError('Checkpoint tensor file digest mismatch')
        tensors = load_file(str(path), device='cpu')
        codec, used = cls(tensors), set()
        for tensor in tensors.values():
            if not torch.isfinite(tensor).all():
                raise ValueError('Nonfinite tensor in checkpoint')
        def visit(value):
            if isinstance(value, dict):
                if value.get('type') == 'tensor_ref':
                    codec.decode(value)
                    used.add(value['key'])
                else:
                    for item in value.values():
                        visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        visit(payload)
        if used != set(tensors):
            raise ValueError('Unreferenced checkpoint tensors')
        return codec
