"""Versioned JSON artifacts. Loading never imports types named by an artifact.

Dataclass codecs are an explicit mapping of stable names to trusted classes.
Tensor payloads use dtype/shape/data, never pickle. Operation bindings are supplied
by the caller, and configuration identity is separate from mutable model weights.
"""
from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

from ..tracing import Call, InputRef, OutputRef, Session, Supervision, Tree, _tensor


def _identity(value):
    cls = value if isinstance(value, type) else type(value)
    return f'{cls.__module__}.{cls.__qualname__}'


class Codec:
    def __init__(self, codecs=None):
        self.types = dict(codecs or {})
        for name, cls in self.types.items():
            if not isinstance(name, str) or not name or not isinstance(cls, type) or not is_dataclass(cls):
                raise TypeError('codecs must map nonempty names to trusted dataclass types')
        if len(set(self.types.values())) != len(self.types):
            raise ValueError('A dataclass must have exactly one codec name')

    def name(self, cls):
        for name, candidate in self.types.items():
            if candidate is cls:
                return name
        raise TypeError(f'No allowlisted codec for {_identity(cls)}')

    def encode(self, value):
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError('Nonfinite values cannot be persisted')
            return value
        if isinstance(value, bytes):
            return {'type': 'bytes', 'data': base64.b64encode(value).decode('ascii')}
        if _tensor(value):
            value = value.detach().cpu()
            if value.layout != sys.modules['torch'].strided or value.is_complex():
                raise TypeError('Only dense real tensors have a persistence codec')
            return {'type': 'tensor', 'dtype': str(value.dtype).removeprefix('torch.'),
                    'shape': list(value.shape), 'data': self.encode(value.tolist())}
        if isinstance(value, (list, tuple)):
            return {'type': 'tuple' if isinstance(value, tuple) else 'list', 'items': [self.encode(v) for v in value]}
        if isinstance(value, Mapping):
            return {'type': 'dict', 'items': [[self.encode(k), self.encode(v)] for k, v in value.items()]}
        if is_dataclass(value) and not isinstance(value, type):
            return {'type': 'dataclass', 'codec': self.name(type(value)),
                    'fields': {f.name: self.encode(getattr(value, f.name)) for f in fields(value)}}
        raise TypeError(f'No safe codec for {_identity(value)}')

    def decode(self, value):
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float and math.isfinite(value):
            return value
        if not isinstance(value, dict):
            raise ValueError('Malformed encoded value')
        kind = value.get('type')
        if kind == 'bytes' and set(value) == {'type', 'data'}:
            return base64.b64decode(value['data'], validate=True)
        if kind in ('list', 'tuple') and set(value) == {'type', 'items'} and isinstance(value['items'], list):
            items = [self.decode(v) for v in value['items']]
            return tuple(items) if kind == 'tuple' else items
        if kind == 'dict' and set(value) == {'type', 'items'}:
            items = value['items']
            if not isinstance(items, list) or any(not isinstance(p, list) or len(p) != 2 for p in items):
                raise ValueError('Malformed mapping')
            result = {}
            for key, item in items:
                key = self.decode(key)
                if key in result:
                    raise ValueError('Duplicate mapping key')
                result[key] = self.decode(item)
            return result
        if kind == 'dataclass' and set(value) == {'type', 'codec', 'fields'}:
            cls = self.types.get(value['codec'])
            if cls is None:
                raise ValueError(f"Unknown allowlisted codec: {value['codec']}")
            if set(value['fields']) != {f.name for f in fields(cls)}:
                raise ValueError('Dataclass fields differ from codec')
            return cls(**{k: self.decode(v) for k, v in value['fields'].items()})
        if kind == 'tensor' and set(value) == {'type', 'dtype', 'shape', 'data'}:
            import torch
            dtypes = {name: getattr(torch, name) for name in ('float16', 'bfloat16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'bool')}
            if value['dtype'] not in dtypes or not isinstance(value['shape'], list) or any(type(n) is not int or n < 0 for n in value['shape']):
                raise ValueError('Unsupported tensor dtype or shape')
            data = self.decode(value['data'])
            tensor = torch.tensor(data, dtype=dtypes[value['dtype']])
            shape = value['shape']
            # Empty multidimensional tensors need their shape restored explicitly.
            if tensor.numel() == 0 and math.prod(shape) == 0:
                return tensor.reshape(shape)
            if list(tensor.shape) != shape:
                raise ValueError('Tensor shape does not match payload')
            return tensor
        raise ValueError('Unknown or malformed safe codec payload')


def _configuration_value(value):
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, (list, tuple)):
        return [_configuration_value(v) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _configuration_value(v) for k, v in value.items()}
    raise TypeError('Operation configuration requires JSON data; custom callbacks need explicit configuration() metadata')


def configuration(operation):
    protocol = getattr(operation, 'configuration', None)
    if protocol is not None:
        config = _configuration_value(protocol())
    else:
        config = {}
        backend = sys.modules.get('torch')
        is_module = backend is not None and isinstance(operation, backend.nn.Module)
        modules = operation.named_modules() if is_module else [('', operation)]
        for name, module in modules:
            attributes = {k: _configuration_value(v) for k, v in vars(module).items()
                          if not k.startswith('_') and k != 'training'}
            config[name] = {'type': _identity(module), 'attributes': attributes}
    backend = sys.modules.get('torch')
    if backend is not None and isinstance(operation, backend.nn.Module):
        topology = {name: {'shape': list(value.shape), 'dtype': str(value.dtype)}
                    for name, value in operation.state_dict().items() if _tensor(value)}
    else:
        topology = None
    return {'type': _identity(operation), 'config': config, 'state_topology': topology,
            'replayable': bool(operation.replayable)}


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def bindings(operations):
    if not isinstance(operations, dict) or not operations or any(not isinstance(k, str) or not k for k in operations):
        raise ValueError('operations must provide nonempty named bindings')
    records = {}
    for name, op in operations.items():
        config = configuration(op)
        records[name] = {'configuration': config, 'fingerprint': fingerprint(config)}
    return records


def validate_bindings(saved, operations):
    if not isinstance(saved, dict) or not isinstance(operations, dict):
        raise ValueError('Malformed operation bindings')
    for name, record in saved.items():
        if name not in operations:
            raise ValueError(f'Missing operation binding: {name}')
        if set(record) != {'configuration', 'fingerprint'} or record['fingerprint'] != fingerprint(record['configuration']):
            raise ValueError('Corrupt operation configuration fingerprint')
        if record['fingerprint'] != fingerprint(configuration(operations[name])):
            raise ValueError(f'Incompatible operation configuration: {name}')


def _write(path, data):
    # Build the entire JSON before touching the destination, then atomic replace.
    import os
    import tempfile
    content = json.dumps(data, sort_keys=True, allow_nan=False)
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read(path, artifact):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    with open(path) as stream:
        data = json.load(stream, object_pairs_hook=object_pairs,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON value')))
    if not isinstance(data, dict) or data.get('format') != artifact or type(data.get('version')) is not int or data['version'] != 1:
        raise ValueError('Unknown artifact format or version')
    return data


def save(session, path, *, operations, codecs=None):
    if session._token is not None:
        raise RuntimeError('Close the trace before saving')
    codec = Codec(codecs)
    names = {}
    for name, op in operations.items():
        if id(op) in names:
            raise ValueError('Each operation instance needs one unambiguous binding name')
        names[id(op)] = name
    def bound(value):
        if isinstance(value, InputRef):
            return {'kind': 'input', 'key': value.key}
        if isinstance(value, OutputRef):
            return {'kind': 'output', 'call': value.call, 'path': list(value.path)}
        kind = {dict: 'dict', tuple: 'tuple', list: 'list'}.get(value.kind, 'dataclass')
        result = {'kind': kind, 'children': {k: bound(v) for k, v in value.children.items()} if isinstance(value.children, dict) else [bound(v) for v in value.children]}
        if kind == 'dataclass':
            result['codec'] = codec.name(value.kind)
        return result
    calls = []
    used = {}
    for index, call in enumerate(session.calls):
        if call.error:
            raise ValueError('Cannot persist failed calls')
        if not session._released:
            session._get(call.output)  # Reject silently mutated captured outputs.
        if id(call.operation) not in names:
            raise ValueError('Missing named operation binding for traced call')
        name = names[id(call.operation)]
        used[name] = call.operation
        record = {'operation': name, 'value': bound(call.value), 'context': bound(call.context)}
        if not call.operation.replayable:
            record['boundary'] = codec.encode(session._boundaries.get(index, call.result))
        calls.append(record)
    data = {'format': 'tensorcode.experience', 'version': 1, 'operations': bindings(used),
            'inputs': {str(k): codec.encode(v) for k, v in session.inputs.items()}, 'calls': calls,
            'supervisions': [{'output': bound(s.output), 'target': codec.encode(s.target), 'loss': s.loss, 'source': s.source}
                             for s in session.supervisions]}
    _write(path, data)


def load(path, *, operations, codecs=None):
    """Load data using explicitly bound operations and allowlisted dataclass codecs.

    Weights may differ from capture: only stable configuration is compared. Replay
    intentionally uses the supplied current parameters, which enables training.
    """
    try:
        return _load(path, operations=operations, codecs=codecs)
    except (KeyError, TypeError, IndexError, AttributeError, RecursionError) as exc:
        raise ValueError(f'Malformed experience artifact: {exc}') from exc


def _load(path, *, operations, codecs=None):
    data = _read(path, 'tensorcode.experience')
    if set(data) != {'format', 'version', 'operations', 'inputs', 'calls', 'supervisions'}:
        raise ValueError('Malformed experience fields')
    validate_bindings(data['operations'], operations)
    codec = Codec(codecs)
    session = Session()
    session._closed = True
    session._released = True
    session.inputs = {int(k): codec.decode(v) for k, v in data['inputs'].items()}
    if set(data['inputs']) != {str(k) for k in session.inputs} or any(k < 0 for k in session.inputs):
        raise ValueError('Malformed input keys')
    def bound(value, before):
        kind = value['kind']
        if kind == 'input' and set(value) == {'kind', 'key'}:
            if type(value['key']) is not int or value['key'] not in session.inputs:
                raise ValueError('Unknown input root')
            return InputRef(value['key'])
        if kind == 'output' and set(value) == {'kind', 'call', 'path'}:
            if type(value['call']) is not int or not 0 <= value['call'] < before or not isinstance(value['path'], list) or any(type(k) not in (int, str) for k in value['path']):
                raise ValueError('Invalid or forward output dependency')
            return OutputRef(session.id, value['call'], tuple(value['path']))
        if kind in ('dict', 'dataclass'):
            expected = {'kind', 'children', 'codec'} if kind == 'dataclass' else {'kind', 'children'}
            if set(value) != expected or not isinstance(value['children'], dict):
                raise ValueError('Malformed tree')
            cls = dict if kind == 'dict' else codec.types.get(value['codec'])
            if cls is None:
                raise ValueError('Unknown allowlisted dataclass codec')
            if kind == 'dataclass' and set(value['children']) != {f.name for f in fields(cls)}:
                raise ValueError('Malformed dataclass tree fields')
            return Tree(cls, {k: bound(v, before) for k, v in value['children'].items()})
        if kind in ('tuple', 'list') and set(value) == {'kind', 'children'} and isinstance(value['children'], list):
            return Tree(tuple if kind == 'tuple' else list, tuple(bound(v, before) for v in value['children']))
        raise ValueError('Unknown dependency node')
    if not isinstance(data['calls'], list) or not isinstance(data['supervisions'], list):
        raise ValueError('Malformed calls or supervision')
    for index, record in enumerate(data['calls']):
        name = record['operation']
        if name not in data['operations']:
            raise ValueError('Unvalidated operation binding')
        op = operations[name]
        expected = {'operation', 'value', 'context'} | ({'boundary'} if not op.replayable else set())
        if set(record) != expected:
            raise ValueError('Malformed call or missing external boundary')
        session.calls.append(Call(op, bound(record['value'], index), bound(record['context'], index), OutputRef(session.id, index)))
        if 'boundary' in record:
            session._boundaries[index] = codec.decode(record['boundary'])
    for record in data['supervisions']:
        if set(record) != {'output', 'target', 'loss', 'source'}:
            raise ValueError('Malformed supervision')
        ref = bound(record['output'], len(session.calls))
        if not isinstance(ref, OutputRef):
            raise ValueError('Supervision must target an output')
        session.supervise(ref, codec.decode(record['target']), loss=record['loss'], source=record['source'])
    return session
