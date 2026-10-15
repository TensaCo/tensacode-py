"""Typed, bounded, data-only algorithms, independent of their execution substrate."""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass

import numpy as np

FORMAT = "tensorcode.algorithm"
VERSION = 1
MAX_ELEMENTS = 1_000_000
MAX_NODES = 10_000
LEAVES = {"input", "constant", "parameter", "state"}
UNARY = {"neg", "relu", "sigmoid", "tanh", "sin", "cos", "exp", "sqrt", "log", "abs"}
BINARY = {
    "add",
    "sub",
    "mul",
    "div",
    "gt",
    "ge",
    "lt",
    "le",
    "eq",
    "ne",
    "maximum",
    "minimum",
}
ARITIES = {
    **{x: 1 for x in UNARY},
    **{x: 2 for x in BINARY},
    "where": 3,
    "matmul": 2,
    "transpose": 1,
    "reshape": 1,
    "sum": 1,
    "mean": 1,
    "softmax": 1,
    "masked_softmax": 2,
    "slice": 1,
    "concat": None,
}
ATTRS = {
    "transpose": {"axes"},
    "reshape": {"shape"},
    "sum": {"axis", "keepdims"},
    "mean": {"axis", "keepdims"},
    "softmax": {"axis"},
    "masked_softmax": {"axis"},
    "slice": {"axis", "start", "stop"},
    "concat": {"axis"},
}


def _name(name):
    if (
        name in ("__proto__", "constructor", "prototype")
        or not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
    ):
        raise ValueError("names must be identifiers")
    return name


def _shape(shape):
    if not isinstance(shape, (list, tuple)) or len(shape) > 6:
        raise ValueError("shape must have at most six dimensions")
    if (
        any(type(d) is not int or d < 1 for d in shape)
        or math.prod(shape) > MAX_ELEMENTS
    ):
        raise ValueError("invalid or oversized shape")
    return tuple(shape)


def _axis(axis, rank):
    if type(axis) is not int or not -rank <= axis < rank:
        raise ValueError("axis out of range")
    return axis % rank


def infer(op, shapes, attrs):
    """Validate instruction attributes and infer shape without running user data."""
    if op not in ARITIES or (ARITIES[op] is not None and len(shapes) != ARITIES[op]):
        raise ValueError(f"unsupported operation or arity: {op}")
    if set(attrs) != ATTRS.get(op, set()):
        raise ValueError(f"invalid attributes for {op}")
    if op in UNARY:
        return shapes[0]
    if op in BINARY or op == "where":
        return _shape(np.broadcast_shapes(*shapes))
    if op == "matmul":
        a, b = shapes
        if len(a) < 2 or len(b) < 2 or a[-1] != b[-2]:
            raise ValueError("matmul requires compatible rank >= 2 operands")
        return _shape((*np.broadcast_shapes(a[:-2], b[:-2]), a[-2], b[-1]))
    if op == "transpose":
        axes = attrs["axes"]
        if (
            not isinstance(axes, list)
            or any(type(i) is not int for i in axes)
            or sorted(axes) != list(range(len(shapes[0])))
        ):
            raise ValueError("transpose requires an axis permutation")
        return tuple(shapes[0][i] for i in axes)
    if op == "reshape":
        target = _shape(attrs["shape"])
        if math.prod(target) != math.prod(shapes[0]):
            raise ValueError("reshape changes element count")
        return target
    if op in ("sum", "mean"):
        axis = attrs["axis"]
        if type(attrs["keepdims"]) is not bool:
            raise ValueError("keepdims must be boolean")
        axes = (
            tuple(range(len(shapes[0])))
            if axis is None
            else (_axis(axis, len(shapes[0])),)
        )
        return tuple(
            1 if i in axes else d
            for i, d in enumerate(shapes[0])
            if attrs["keepdims"] or i not in axes
        )
    if op in ("softmax", "masked_softmax"):
        _axis(attrs["axis"], len(shapes[0]))
        if op == "masked_softmax" and np.broadcast_shapes(*shapes) != shapes[0]:
            raise ValueError("mask must broadcast to logits")
        return shapes[0]
    if op == "slice":
        a = _axis(attrs["axis"], len(shapes[0]))
        start, stop = attrs["start"], attrs["stop"]
        if (
            type(start) is not int
            or type(stop) is not int
            or not 0 <= start < stop <= shapes[0][a]
        ):
            raise ValueError("slice requires a nonempty in-range interval")
        out = list(shapes[0])
        out[a] = stop - start
        return tuple(out)
    if op == "concat":
        if not shapes:
            raise ValueError("concat requires operands")
        a = _axis(attrs["axis"], len(shapes[0]))
        out = list(shapes[0])
        out[a] = 0
        for s in shapes:
            if len(s) != len(out) or any(
                s[i] != shapes[0][i] for i in range(len(out)) if i != a
            ):
                raise ValueError("incompatible concat operands")
            out[a] += s[a]
        return _shape(out)
    raise ValueError(op)


def _data(value, dtype):
    a = np.asarray(value, dtype=dtype)
    _shape(a.shape)
    if not np.isfinite(a).all():
        raise ValueError("nonfinite data")
    return a.tolist()


@dataclass(frozen=True, eq=False)
class Variable:
    builder: AlgorithmBuilder
    id: int

    @property
    def shape(self):
        return tuple(self.builder._nodes[self.id]["shape"])

    def _op(self, op, *others, **attrs):
        return self.builder.op(op, self, *others, **attrs)

    def __add__(self, x):
        return self._op("add", x)

    def __radd__(self, x):
        return self.builder.op("add", x, self)

    def __sub__(self, x):
        return self._op("sub", x)

    def __rsub__(self, x):
        return self.builder.op("sub", x, self)

    def __mul__(self, x):
        return self._op("mul", x)

    def __rmul__(self, x):
        return self.builder.op("mul", x, self)

    def __truediv__(self, x):
        return self._op("div", x)

    def __rtruediv__(self, x):
        return self.builder.op("div", x, self)

    def __matmul__(self, x):
        return self._op("matmul", x)

    def __neg__(self):
        return self._op("neg")

    def __gt__(self, x):
        return self._op("gt", x)

    def __ge__(self, x):
        return self._op("ge", x)

    def __lt__(self, x):
        return self._op("lt", x)

    def __le__(self, x):
        return self._op("le", x)

    def __eq__(self, x):
        return self._op("eq", x)

    def __ne__(self, x):
        return self._op("ne", x)

    def __bool__(self):
        raise TypeError("symbolic variable: use builder.where")

    def relu(self):
        return self._op("relu")

    def tanh(self):
        return self._op("tanh")

    def sigmoid(self):
        return self._op("sigmoid")

    def sin(self):
        return self._op("sin")

    def cos(self):
        return self._op("cos")

    def exp(self):
        return self._op("exp")

    def sqrt(self):
        return self._op("sqrt")

    def log(self):
        return self._op("log")

    def abs(self):
        return self._op("abs")

    def clip(self, lo, hi):
        return self.builder.op("minimum", self.builder.op("maximum", self, lo), hi)

    def reshape(self, *shape):
        return self._op("reshape", shape=list(shape))

    def transpose(self, *axes):
        return self._op(
            "transpose",
            axes=list(axes) if axes else list(reversed(range(len(self.shape)))),
        )

    def sum(self, axis=None, keepdims=False):
        return self._op("sum", axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        return self._op("mean", axis=axis, keepdims=keepdims)

    def softmax(self, axis=-1):
        return self._op("softmax", axis=axis)

    def slice(self, start, stop, axis=-1):
        return self._op("slice", axis=axis, start=start, stop=stop)


class AlgorithmBuilder:
    """Author a bounded tensor algorithm; Python loops statically unroll its graph."""

    def __init__(self, *, name="algorithm", dtype="float64"):
        if dtype not in ("float32", "float64"):
            raise ValueError("unsupported dtype")
        self.name = _name(name)
        self.dtype = dtype
        self._nodes = []
        self._updates = {}
        self._names = set()

    def _append(self, node):
        if len(self._nodes) >= MAX_NODES:
            raise ValueError("node budget exceeded")
        node["id"] = len(self._nodes)
        self._nodes.append(node)
        return Variable(self, node["id"])

    def _leaf(self, op, name=None, shape=None, value=None):
        attrs = {}
        if name is not None:
            _name(name)
            if name in self._names:
                raise ValueError(f"duplicate name: {name}")
            self._names.add(name)
            attrs["name"] = name
        if op != "input":
            value = _data(value, self.dtype)
            shape = np.asarray(value).shape
            attrs["value"] = value
        return self._append(
            {"op": op, "args": [], "shape": list(_shape(shape)), "attrs": attrs}
        )

    def input(self, name, shape=()):
        return self._leaf("input", name, shape)

    def state(self, name, initial):
        return self._leaf("state", name, value=initial)

    def parameter(self, name, initial):
        return self._leaf("parameter", name, value=initial)

    def constant(self, value):
        return self._leaf("constant", value=value)

    def value(self, value):
        if isinstance(value, Variable):
            if value.builder is not self:
                raise ValueError("variable belongs to another builder")
            return value
        return self.constant(value)

    def op(self, op, *args, **attrs):
        vs = [self.value(x) for x in args]
        shape = infer(op, [x.shape for x in vs], attrs)
        return self._append(
            {
                "op": op,
                "args": [x.id for x in vs],
                "shape": list(shape),
                "attrs": copy.deepcopy(attrs),
            }
        )

    def where(self, condition, yes, no):
        return self.op("where", condition, yes, no)

    def concat(self, values, axis=-1):
        return self.op("concat", *values, axis=axis)

    def update(self, state, value):
        state = self.value(state)
        value = self.value(value)
        if self._nodes[state.id]["op"] != "state" or state.shape != value.shape:
            raise ValueError("update requires a state and matching shape")
        self._updates[self._nodes[state.id]["attrs"]["name"]] = value.id

    def attention(self, query, key, value, *, bias=None, allowed=None):
        q, k, v = map(self.value, (query, key, value))
        if len(q.shape) < 2 or len(k.shape) < 2:
            raise ValueError("attention requires rank >= 2")
        axes = list(range(len(k.shape)))
        axes[-1], axes[-2] = axes[-2], axes[-1]
        logits = (q @ k.transpose(*axes)) / math.sqrt(q.shape[-1])
        if bias is not None:
            logits = logits + bias
        weights = (
            logits.softmax()
            if allowed is None
            else self.op("masked_softmax", logits, allowed, axis=-1)
        )
        return weights @ v

    def include(self, algorithm, inputs, *, prefix="included"):
        """Compose a stateless algorithm while retaining its operation structure."""
        _name(prefix)
        if algorithm._ir["dtype"] != self.dtype:
            raise ValueError("include dtype mismatch")
        if algorithm.state_names:
            raise ValueError("stateful include requires explicit state wiring")
        if set(inputs) != set(algorithm.input_names):
            raise ValueError("include input mismatch")
        values = []
        for n in algorithm._ir["nodes"]:
            op, a = n["op"], n["attrs"]
            if op == "input":
                v = self.value(inputs[a["name"]])
                if v.shape != tuple(n["shape"]):
                    raise ValueError("include shape mismatch")
            elif op == "constant":
                v = self.constant(a["value"])
            elif op == "parameter":
                v = self.parameter(prefix + "_" + a["name"], a["value"])
            else:
                v = self.op(op, *(values[i] for i in n["args"]), **a)
            values.append(v)
        return {k: values[i] for k, i in algorithm._ir["outputs"].items()}

    def build(self, outputs):
        outputs = outputs if isinstance(outputs, dict) else {"result": outputs}
        ids = {_name(k): self.value(v).id for k, v in outputs.items()}
        return Algorithm(
            {
                "format": FORMAT,
                "version": VERSION,
                "name": self.name,
                "dtype": self.dtype,
                "nodes": self._nodes,
                "outputs": ids,
                "updates": self._updates,
            }
        )


class Algorithm:
    """Validated ALG.0 artifact. No executable Python or external callbacks are stored."""

    def __init__(self, artifact):
        ir = copy.deepcopy(artifact)
        if not isinstance(ir, dict) or set(ir) != {
            "format",
            "version",
            "name",
            "dtype",
            "nodes",
            "outputs",
            "updates",
        }:
            raise ValueError("invalid algorithm artifact fields")
        if ir["format"] != FORMAT or (
            type(ir["version"]) is not int or ir["version"] != VERSION
        ):
            raise ValueError("unsupported algorithm format")
        _name(ir["name"])
        if ir["dtype"] not in ("float32", "float64"):
            raise ValueError("unsupported dtype")
        if not isinstance(ir["nodes"], list) or not 0 < len(ir["nodes"]) <= MAX_NODES:
            raise ValueError("invalid node count")
        shapes = []
        names = set()
        states = {}
        for i, n in enumerate(ir["nodes"]):
            if (
                not isinstance(n, dict)
                or set(n) != {"id", "op", "args", "shape", "attrs"}
                or type(n["id"]) is not int
                or n["id"] != i
            ):
                raise ValueError("invalid node schema or ordering")
            shape = _shape(n["shape"])
            op = n["op"]
            a = n["attrs"]
            if not isinstance(a, dict) or not isinstance(n["args"], list):
                raise TypeError("invalid node data")
            if any(type(j) is not int or not 0 <= j < i for j in n["args"]):
                raise ValueError("invalid dependency")
            if op in LEAVES:
                expected = (
                    {"name"}
                    if op == "input"
                    else {"value"}
                    if op == "constant"
                    else {"name", "value"}
                )
                if n["args"] or set(a) != expected:
                    raise ValueError("invalid leaf")
                if "name" in a:
                    _name(a["name"])
                    if a["name"] in names:
                        raise ValueError("duplicate name")
                    names.add(a["name"])
                if "value" in a:
                    v = np.asarray(_data(a["value"], ir["dtype"]))
                    if v.shape != shape:
                        raise ValueError("value shape mismatch")
                if op == "state":
                    states[a["name"]] = shape
            elif infer(op, [shapes[j] for j in n["args"]], a) != shape:
                raise ValueError("inferred shape mismatch")
            shapes.append(shape)
        if (
            sum(math.prod(s) for s in shapes) > 32_000_000
            or sum(len(n["args"]) for n in ir["nodes"]) > 1_000_000
        ):
            raise ValueError("algorithm aggregate budget exceeded")
        if (
            not isinstance(ir["outputs"], dict)
            or not ir["outputs"]
            or not isinstance(ir["updates"], dict)
        ):
            raise ValueError("invalid outputs or updates")
        for k, i in list(ir["outputs"].items()) + list(ir["updates"].items()):
            _name(k)
            if type(i) is not int or not 0 <= i < len(shapes):
                raise ValueError("invalid output reference")
        for k, i in ir["updates"].items():
            if k not in states or states[k] != shapes[i]:
                raise ValueError("state update shape mismatch")
        self._ir = ir

    @property
    def input_names(self):
        return tuple(
            n["attrs"]["name"] for n in self._ir["nodes"] if n["op"] == "input"
        )

    @property
    def state_names(self):
        return tuple(
            n["attrs"]["name"] for n in self._ir["nodes"] if n["op"] == "state"
        )

    def graph(self):
        return copy.deepcopy(self._ir)

    def to_json(self):
        return json.dumps(
            self._ir, allow_nan=False, separators=(",", ":"), sort_keys=True
        )

    @classmethod
    def from_json(cls, text):
        return cls(json.loads(text))

    def run(self, inputs, *, parameters=None):
        if self.state_names:
            raise ValueError("stateful algorithm: call step with explicit state")
        return self.step(inputs, parameters=parameters)[0]

    def step(self, inputs, *, state=None, parameters=None):
        from .runtime import execute_numpy

        return execute_numpy(self, inputs, state=state, parameters=parameters)

    def initial_state(self):
        return {
            n["attrs"]["name"]: np.array(n["attrs"]["value"], dtype=self._ir["dtype"])
            for n in self._ir["nodes"]
            if n["op"] == "state"
        }

    def compile(self, target="alg1", **options):
        from .runtime import compile_algorithm

        return compile_algorithm(self, target, **options)

    def freeze(self, parameters=None):
        ir = self.graph()
        parameters = {} if parameters is None else parameters
        valid = {n["attrs"]["name"] for n in ir["nodes"] if n["op"] == "parameter"}
        if set(parameters) - valid:
            raise ValueError("unknown parameter")
        for n in ir["nodes"]:
            if n["op"] == "parameter":
                a = n["attrs"]
                n["op"] = "constant"
                n["attrs"] = {
                    "value": _data(parameters.get(a["name"], a["value"]), ir["dtype"])
                }
        return Algorithm(ir)

    def optimize(self):
        """Remove unreachable nodes without algebraic changes to rounding or execution."""
        live = set(self._ir["outputs"].values()) | set(self._ir["updates"].values())
        # Keep the external input/state/parameter signature stable.
        live.update(
            n["id"]
            for n in self._ir["nodes"]
            if n["op"] in {"input", "state", "parameter"}
        )
        for n in reversed(self._ir["nodes"]):
            if n["id"] in live:
                live.update(n["args"])
        ir = self.graph()
        remap = {old: new for new, old in enumerate(sorted(live))}
        ir["nodes"] = [n for n in ir["nodes"] if n["id"] in live]
        for n in ir["nodes"]:
            n["id"] = remap[n["id"]]
            n["args"] = [remap[x] for x in n["args"]]
        ir["outputs"] = {k: remap[v] for k, v in ir["outputs"].items()}
        ir["updates"] = {k: remap[v] for k, v in ir["updates"].items()}
        return Algorithm(ir)

    def to_python(self, function_name="run"):
        """Export readable, standalone NumPy source; never execute generated source on load."""
        from .runtime import python_source

        return python_source(self, function_name)
