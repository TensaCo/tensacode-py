"""Safe JSON tensor-module checkpoints with explicit binding and alias checks."""
from copy import deepcopy

from .persistence import Codec, _identity, _read, _write, bindings, validate_bindings
from .trainer import parameters, validate_optimizer
from tensorcode._internal.tracing import _tensor


def _stateful(operation):
    protocol = ('state_dict', 'load_state_dict', 'named_parameters')
    present = [callable(getattr(operation, name, None)) for name in protocol]
    if all(present):
        return True
    if any(present) or list(getattr(operation, 'parameters', lambda: ())()):
        raise TypeError('Checkpoint restoration requires state_dict, load_state_dict and named_parameters')
    return False


def _state_dict(operation):
    return operation.state_dict() if _stateful(operation) else {}


def _load_state_dict(operation, state):
    if _stateful(operation):
        operation.load_state_dict(state, strict=True)
    elif state:
        raise ValueError('Parameterless operation checkpoint must have empty state')


def _named_parameters(operation, **kwargs):
    return operation.named_parameters(**kwargs) if _stateful(operation) else ()


def _aliases(operations):
    groups = {}
    for name, operation in operations.items():
        for key, parameter in _named_parameters(operation, remove_duplicate=False):
            groups.setdefault(id(parameter), []).append([name, key])
    return sorted([sorted(group) for group in groups.values()])


def _optimizer_layout(optimizer, operations):
    names = {}
    for name, op in operations.items():
        for key, param in _named_parameters(op):
            names.setdefault(id(param), []).append([name, key])
    return [[sorted(names[id(p)]) for p in group['params']] for group in optimizer.param_groups]


def _validate_optimizer_state(optimizer, state):
    import torch
    if type(optimizer) not in (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW):
        raise TypeError('Optimizer checkpoints support SGD, Adam and AdamW')
    if not isinstance(state, dict) or set(state) != {'state', 'param_groups'} or not isinstance(state['state'], dict) or not isinstance(state['param_groups'], list):
        raise ValueError('Malformed optimizer checkpoint state')
    groups = state['param_groups']
    if len(groups) != len(optimizer.param_groups):
        raise ValueError('Checkpoint optimizer parameter groups differ')
    owners = {}
    for saved, current in zip(groups, optimizer.param_groups):
        if not isinstance(saved, dict) or set(saved) != set(current) or not isinstance(saved.get('params'), list) or len(saved['params']) != len(current['params']):
            raise ValueError('Checkpoint optimizer parameter group fields differ')
        for key, parameter in zip(saved['params'], current['params']):
            if type(key) is not int or key in owners:
                raise ValueError('Invalid optimizer parameter identity')
            owners[key] = parameter
    if not state['state'].keys() <= owners.keys():
        raise ValueError('Unknown optimizer state parameter')
    for key, slots in state['state'].items():
        if not isinstance(slots, dict):
            raise ValueError('Malformed optimizer parameter slots')
        if type(optimizer) is torch.optim.SGD:
            if set(slots) - {'momentum_buffer'}:
                raise ValueError('Unknown SGD optimizer state slot')
        elif slots and (not {'step', 'exp_avg', 'exp_avg_sq'} <= slots.keys() or set(slots) - {'step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'}):
            raise ValueError('Malformed Adam optimizer state slots')
        for name, value in slots.items():
            if name == 'step':
                if not _tensor(value) or value.ndim != 0 or not torch.isfinite(value) or value < 0:
                    raise ValueError('Malformed optimizer step counter')
            elif not _tensor(value) or value.shape != owners[key].shape or value.dtype != owners[key].dtype:
                raise ValueError('Checkpoint optimizer slot shape or dtype differs from parameter')


def save_checkpoint(path, *, operations, optimizer=None, _codec=None):
    codec = Codec() if _codec is None else _codec
    config = bindings(operations)
    aliases = _aliases(operations)
    payload = {'format': 'tensorcode.checkpoint', 'version': 1, 'operations': config,
               'aliases': aliases, 'states': {name: codec.encode(_state_dict(op)) for name, op in operations.items()},
               'optimizer': None}
    if optimizer is not None:
        validate_optimizer(optimizer, parameters(operations))
        _validate_optimizer_state(optimizer, optimizer.state_dict())
        payload['optimizer'] = {'type': _identity(optimizer), 'layout': _optimizer_layout(optimizer, operations),
                                'state': codec.encode(optimizer.state_dict())}
    _write(path, payload)


def _prepare_checkpoint(path, *, operations, optimizer=None, _codec=None):
    """Read, decode and validate without mutating or snapshotting bound state."""
    import torch
    payload = _read(path, 'tensorcode.checkpoint')
    if set(payload) != {'format', 'version', 'operations', 'aliases', 'states', 'optimizer'}:
        raise ValueError('Malformed checkpoint fields')
    if set(payload['operations']) != set(operations) or set(payload['states']) != set(operations):
        raise ValueError('Checkpoint operation bindings must match exactly')
    validate_bindings(payload['operations'], operations)
    if payload['aliases'] != _aliases(operations):
        raise ValueError('Shared-parameter alias topology differs from checkpoint')
    codec = Codec() if _codec is None else _codec
    states = {name: codec.decode(value) for name, value in payload['states'].items()}
    for name, state in states.items():
        current = _state_dict(operations[name])
        if not isinstance(state, dict) or set(state) != set(current):
            raise ValueError('Checkpoint state keys differ from bound module')
        for key, value in current.items():
            other = state[key]
            if _tensor(value) and (not _tensor(other) or value.shape != other.shape or value.dtype != other.dtype):
                raise ValueError('Checkpoint tensor shape or dtype differs')
            if not _tensor(value) and _tensor(other):
                raise ValueError('Checkpoint state type differs')
    for group in payload['aliases']:
        first_name, first_key = group[0]
        first = states[first_name][first_key]
        if any(not torch.equal(first, states[name][key]) for name, key in group[1:]):
            raise ValueError('Contradictory shared-parameter alias values')
    optimizer_state = None
    if optimizer is not None:
        validate_optimizer(optimizer, parameters(operations))
        record = payload['optimizer']
        if not isinstance(record, dict) or set(record) != {'type', 'layout', 'state'} or record['type'] != _identity(optimizer) or record['layout'] != _optimizer_layout(optimizer, operations):
            raise ValueError('Checkpoint optimizer type or parameter layout differs')
        optimizer_state = codec.decode(record['state'])
        _validate_optimizer_state(optimizer, optimizer_state)
    return states, optimizer_state


def _apply_checkpoint(states, optimizer_state, *, operations, optimizer=None):
    """Apply prepared states inside the caller's rollback transaction."""
    for name, state in states.items():
        _load_state_dict(operations[name], state)
    if optimizer is not None:
        optimizer.load_state_dict(optimizer_state)


def load_checkpoint(path, *, operations, optimizer=None, _codec=None):
    """Restore supported module states after configuration/alias validation.

    A supplied optimizer also restores its saved state. Omit it to restore model
    weights only. The artifact never creates modules, optimizers or classes.
    """
    states, optimizer_state = _prepare_checkpoint(
        path, operations=operations, optimizer=optimizer, _codec=_codec)
    originals = {name: deepcopy(_state_dict(op)) for name, op in operations.items()}
    original_optimizer = deepcopy(optimizer.state_dict()) if optimizer is not None else None
    try:
        _apply_checkpoint(states, optimizer_state, operations=operations, optimizer=optimizer)
    except BaseException:
        for name, state in originals.items():
            _load_state_dict(operations[name], state)
        if optimizer is not None:
            optimizer.load_state_dict(original_optimizer)
        raise
