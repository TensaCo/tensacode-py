"""In-memory operation provenance and opt-in replay, separate from autograd.

Only captured operation boundaries are observable. Plain scalar dependencies need
explicit OutputRef handles. Inputs are snapshotted; outputs stay live so native
gradients survive. Portable persistence is opt-in through Trace.save.
"""
from __future__ import annotations
from collections.abc import Mapping
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from typing import Any
from uuid import uuid4
import sys

_active: ContextVar[Trace | None] = ContextVar('tensorcode_trace', default=None)
_SCALARS = (str, bytes, int, float, bool, type(None))


def _tensor(value):
    backend = sys.modules.get('torch')
    return backend is not None and isinstance(value, backend.Tensor)


def _snapshot(value):
    if _tensor(value):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_snapshot(v) for v in value)
    if isinstance(value, list):
        return [_snapshot(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _snapshot(v) for k, v in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{f.name: _snapshot(getattr(value, f.name)) for f in fields(value)})
    return deepcopy(value)


def _stamp(value):
    if _tensor(value):
        if value.is_inference():
            # Inference tensors have no version counter. This conservative
            # content stamp costs a device sync/copy; it is not a fast path.
            return ('inference', id(value), str(value.dtype), tuple(value.shape), repr(value.detach().cpu().tolist()))
        return ('tensor', id(value), value._version)
    if isinstance(value, _SCALARS):
        return (type(value), repr(value))
    if isinstance(value, Mapping):
        return ('dict', tuple((k, _stamp(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return (type(value), tuple(_stamp(v) for v in value))
    if is_dataclass(value) and not isinstance(value, type):
        return (type(value), tuple((f.name, _stamp(getattr(value, f.name))) for f in fields(value)))
    raise TypeError(f'Unsupported trace value {type(value).__name__}; use tensors, dataclasses or Python containers')


@dataclass(frozen=True)
class OutputRef:
    session: str
    call: int
    path: tuple = ()


@dataclass(frozen=True)
class InputRef:
    key: int


@dataclass(frozen=True)
class Tree:
    kind: type
    children: Any


@dataclass
class Call:
    operation: Any
    value: Any
    context: Any
    output: OutputRef
    result: Any = None
    error: str | None = None
    pending: bool = False


@dataclass(frozen=True)
class Example:
    """A dependency closure: external roots, call IDs, and the target port."""
    inputs: dict[int, Any]
    calls: tuple[int, ...]
    target: OutputRef


@dataclass(frozen=True)
class Supervision:
    """An explicitly supplied target, never inferred from a model prediction."""
    output: OutputRef
    target: Any
    loss: str
    source: str


class Trace:
    """In-memory record of operation calls, inputs and explicit supervision.

    Create with ``tensorcode.trace()`` and enter once as a context manager.
    It records local calls; it does not make remote or discrete calls
    differentiable.
    """
    def __init__(self):
        self.id = uuid4().hex
        self.calls: list[Call] = []
        self.inputs: dict[int, Any] = {}
        self._objects: dict[int, list[tuple[Any, OutputRef, Any]]] = {}
        self._stamps: dict[OutputRef, Any] = {}
        self._token = None
        self._closed = False
        self.supervisions: list[Supervision] = []
        self._released = False
        self._boundaries: dict[int, Any] = {}

    def __enter__(self):
        if self._token is not None or self._closed:
            raise RuntimeError('A trace session can be entered only once')
        self._token = _active.set(self)
        return self

    def __exit__(self, *exc):
        _active.reset(self._token)
        self._token = None
        self._closed = True

    def _check(self, ref):
        if not isinstance(ref, OutputRef) or ref.session != self.id:
            raise ValueError('Output reference belongs to another session')
        if ref.call < 0 or ref.call >= len(self.calls):
            raise ValueError('Unknown output reference')
        if self.calls[ref.call].pending:
            raise RuntimeError('Call is still pending; await it before using or persisting its output')
        if self.calls[ref.call].error:
            raise ValueError('Failed call has no usable output')

    def _get(self, ref, results=None):
        self._check(ref)
        value = self.calls[ref.call].result if results is None else results[ref.call]
        for key in ref.path:
            if is_dataclass(value):
                if key not in {f.name for f in fields(value)}:
                    raise ValueError('Output path must name a dataclass field')
                value = getattr(value, key)
            else:
                value = value[key]
        if results is None and ref in self._stamps and _stamp(value) != self._stamps[ref]:
            raise ValueError('A traced intermediate was mutated; represent state changes explicitly')
        return value

    def ref(self, value):
        if isinstance(value, OutputRef):
            self._check(value) if self._released else self._get(value)
            return value
        if isinstance(value, _SCALARS):
            raise ValueError('Scalar lineage requires an explicit call.output reference')
        matches = self._objects.get(id(value), ())
        if len(matches) != 1:
            raise ValueError('Unknown or aliased value: use an explicit call.output reference')
        return matches[0][1]

    def _bind(self, value):
        if isinstance(value, OutputRef):
            self._check(value)
            return value
        matches = self._objects.get(id(value), ()) if not isinstance(value, _SCALARS) else ()
        if matches:
            if len(matches) != 1:
                raise ValueError('Aliased value requires an explicit output reference')
            original, ref, stamp = matches[0]
            if _stamp(original) != stamp:
                raise ValueError('A traced intermediate was mutated; represent state changes explicitly')
            return ref
        if isinstance(value, Mapping):
            if not all(isinstance(k, str) for k in value):
                raise TypeError('Traced mapping keys must be strings')
            return Tree(dict, {k: self._bind(v) for k, v in value.items()})
        if isinstance(value, (list, tuple)):
            return Tree(type(value), tuple(self._bind(v) for v in value))
        if is_dataclass(value) and self._has_producer(value):
            return Tree(type(value), {f.name: self._bind(getattr(value, f.name)) for f in fields(value)})
        _stamp(value)  # reject unsupported opaque/mutable objects
        key = len(self.inputs)
        self.inputs[key] = _snapshot(value)
        return InputRef(key)

    def _has_producer(self, value):
        if isinstance(value, OutputRef):
            return True
        if not isinstance(value, _SCALARS) and id(value) in self._objects:
            return True
        if is_dataclass(value) and not isinstance(value, type):
            return any(self._has_producer(getattr(value, f.name)) for f in fields(value))
        if isinstance(value, Mapping):
            return any(self._has_producer(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return any(self._has_producer(v) for v in value)
        return False

    def _resolve(self, bound, results=None, inputs=None):
        if isinstance(bound, OutputRef):
            return self._get(bound, results)
        if isinstance(bound, InputRef):
            return _snapshot((self.inputs if inputs is None else inputs)[bound.key])
        if isinstance(bound.children, dict):
            resolved = {k: self._resolve(v, results, inputs) for k, v in bound.children.items()}
            return resolved if bound.kind is dict else bound.kind(**resolved)
        return bound.kind(self._resolve(v, results, inputs) for v in bound.children)

    def _register(self, value, ref):
        stamp = _stamp(value)
        self._stamps[ref] = stamp
        if not isinstance(value, _SCALARS):
            existing = self._objects.setdefault(id(value), [])
            # A child carried through another container retains its original
            # producer. Returning the same object as a new root stays ambiguous.
            if not ref.path or not existing:
                existing.append((value, ref, stamp))
        if is_dataclass(value) and not isinstance(value, type):
            children = ((f.name, getattr(value, f.name)) for f in fields(value))
        elif isinstance(value, Mapping):
            children = value.items()
        elif isinstance(value, (tuple, list)):
            children = enumerate(value)
        else:
            return
        for key, child in children:
            self._register(child, OutputRef(self.id, ref.call, ref.path + (key,)))

    def capture(self, operation, value, context, forward):
        if self._closed:
            raise RuntimeError('Cannot capture into a closed trace session')
        bound_value = self._bind(value)
        bound_context = self._bind(dict(context or {}))
        ref = OutputRef(self.id, len(self.calls))
        call = Call(operation, bound_value, bound_context, ref, pending=True)
        self.calls.append(call)
        try:
            # Preserve the original tensors/objects and gradient graph on live execution.
            result = forward(_unwrap(value, self), context=_unwrap(context or {}, self))
            call.result = result
            self._register(result, ref)
            return result
        except Exception as exc:
            call.error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            call.pending = False

    async def capture_async(self, operation, value, context, forward):
        if self._closed:
            raise RuntimeError('Cannot capture into a closed trace session')
        bound_value = self._bind(value)
        bound_context = self._bind(dict(context or {}))
        ref = OutputRef(self.id, len(self.calls))
        call = Call(operation, bound_value, bound_context, ref, pending=True)
        self.calls.append(call)
        try:
            live_value = _unwrap(value, self)
            live_context = _unwrap(context or {}, self)
            # Awaited execution still consumes the original objects so native
            # gradients survive. Their root snapshots must describe those same
            # inputs; reject changes before publishing a replayable output.
            before_value, before_context = _stamp(live_value), _stamp(live_context)
            result = await forward(live_value, context=live_context)
            if _stamp(live_value) != before_value or _stamp(live_context) != before_context:
                raise ValueError('Inputs or context mutated during async capture; represent state changes explicitly')
            call.result = result
            self._register(result, ref)
            return result
        except BaseException as exc:
            # Cancellation also leaves a failed call rather than a usable null output.
            call.error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            call.pending = False

    def example(self, target):
        target = self.ref(target)
        required_calls, roots = set(), set()
        visiting = set()
        def visit(bound):
            if isinstance(bound, InputRef):
                roots.add(bound.key)
            elif isinstance(bound, OutputRef):
                self._check(bound)
                if bound.call in visiting:
                    raise ValueError('Cyclic or composite call graph is not replayable')
                if bound.call not in required_calls:
                    visiting.add(bound.call)
                    call = self.calls[bound.call]
                    visit(call.value)
                    visit(call.context)
                    visiting.remove(bound.call)
                    required_calls.add(bound.call)
            else:
                values = bound.children.values() if isinstance(bound.children, dict) else bound.children
                for child in values:
                    visit(child)
        visit(target)
        return Example({k: _snapshot(self.inputs[k]) for k in sorted(roots)}, tuple(sorted(required_calls)), target)

    def replay(self, target, *, inputs=None, boundary='error'):
        """Recompute pure operations; recorded external outputs require opt-in."""
        if boundary not in ('error', 'recorded'):
            raise ValueError("boundary must be 'error' or 'recorded'")
        example = self.example(target)
        if inputs and not inputs.keys() <= example.inputs.keys():
            raise ValueError('Replacement inputs must name roots of this example')
        for index in example.calls:
            if not self.calls[index].operation.replayable and boundary == 'error':
                raise ValueError('Operation has not opted into effect-free replay')
        roots = {**example.inputs, **(inputs or {})}
        results = {}
        token = _active.set(None)
        try:
            for index in example.calls:
                call = self.calls[index]
                if not call.operation.replayable:
                    if inputs:
                        raise ValueError('Cannot replace inputs across a recorded external boundary')
                    results[index] = _snapshot(self._boundaries[index] if index in self._boundaries else call.result)
                    continue
                results[index] = call.operation(
                    self._resolve(call.value, results, roots),
                    context=self._resolve(call.context, results, roots),
                )
            return self._get(example.target, results)
        finally:
            _active.reset(token)

    def supervise(self, output_or_ref, target, *, loss='cross_entropy', source='human'):
        if not isinstance(source, str) or not source.strip():
            raise ValueError('Supervision source must be a nonempty explicit provenance string')
        if not isinstance(loss, str) or not loss:
            raise ValueError('Loss must be a nonempty name; supply custom callbacks to Trainer')
        output = self.ref(output_or_ref)
        self.example(output)
        supervision = Supervision(output, _snapshot(target), loss, source)
        self.supervisions.append(supervision)
        return supervision

    def save(self, path, *, operations, codecs=None, release=False):
        from .training.persistence import save
        save(self, path, operations=operations, codecs=codecs)
        if release:
            self.release()

    def release(self):
        """Drop live outputs/autograd graphs; retain roots, DAG and external boundaries.

        Keep OutputRef handles before release; object-based lookup is unavailable
        afterwards. Pure operations remain replayable from the retained roots.
        """
        if self._token is not None:
            raise RuntimeError('Cannot release an active trace session')
        if not self._released:
            for call in self.calls:
                if not call.error:
                    self._get(call.output)
        boundaries = dict(self._boundaries)
        for index, call in enumerate(self.calls):
            if not call.operation.replayable and not call.error and index not in boundaries:
                boundaries[index] = _snapshot(call.result)
        self._boundaries = boundaries
        for call in self.calls:
            call.result = None
        self._objects.clear()
        self._stamps.clear()
        self._released = True


def _unwrap(value, session):
    if isinstance(value, OutputRef):
        return session._get(value)
    if isinstance(value, Mapping):
        resolved = {k: _unwrap(v, session) for k, v in value.items()}
        return value if all(resolved[k] is v for k, v in value.items()) else resolved
    if isinstance(value, (tuple, list)):
        resolved = type(value)(_unwrap(v, session) for v in value)
        return value if all(a is b for a, b in zip(value, resolved)) else resolved
    if is_dataclass(value) and session._has_producer(value):
        resolved = {f.name: _unwrap(getattr(value, f.name), session) for f in fields(value)}
        return value if all(resolved[f.name] is getattr(value, f.name) for f in fields(value)) else type(value)(**resolved)
    return value


def invoke(operation, value, context, forward):
    session = _active.get()
    if session is None:
        if isinstance(value, OutputRef):
            raise ValueError('Output references require their active trace session')
        return forward(value, context=context)
    return session.capture(operation, value, context, forward)


async def invoke_async(operation, value, context, forward):
    session = _active.get()
    if session is None:
        if isinstance(value, OutputRef):
            raise ValueError('Output references require their active trace session')
        return await forward(value, context=context)
    return await session.capture_async(operation, value, context, forward)


def trace():
    """Create a fresh in-memory trace; no persistence or model calls on entry."""
    return Trace()
