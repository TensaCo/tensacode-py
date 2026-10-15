"""Reference execution, substrate dispatch and standalone symbolic code export."""

from __future__ import annotations

import numpy as np

from .ir import LEAVES, Algorithm, _name


def numpy_op(op, xs, a):
    unary = {
        "neg": np.negative,
        "relu": lambda x: np.maximum(x, 0),
        "sigmoid": lambda x: np.exp(-np.logaddexp(0, -x)),
        "tanh": np.tanh,
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "sqrt": np.sqrt,
        "log": np.log,
        "abs": np.abs,
    }
    binary = {
        "add": np.add,
        "sub": np.subtract,
        "mul": np.multiply,
        "div": np.divide,
        "gt": np.greater,
        "ge": np.greater_equal,
        "lt": np.less,
        "le": np.less_equal,
        "eq": np.equal,
        "ne": np.not_equal,
        "maximum": np.maximum,
        "minimum": np.minimum,
        "matmul": np.matmul,
    }
    if op in unary:
        return unary[op](*xs)
    if op in binary:
        return binary[op](*xs)
    if op == "where":
        return np.where(xs[0] != 0, xs[1], xs[2])
    if op == "reshape":
        return xs[0].reshape(a["shape"])
    if op == "transpose":
        return np.transpose(xs[0], a["axes"])
    if op in ("sum", "mean"):
        return getattr(np, op)(xs[0], axis=a["axis"], keepdims=a["keepdims"])
    if op == "concat":
        return np.concatenate(xs, axis=a["axis"])
    if op == "slice":
        idx = [slice(None)] * xs[0].ndim
        idx[a["axis"]] = slice(a["start"], a["stop"])
        return xs[0][tuple(idx)]
    if op in ("softmax", "masked_softmax"):
        x = xs[0]
        axis = a["axis"]
        if op == "masked_softmax":
            if not np.isin(xs[1], [0, 1]).all():
                raise ValueError("attention mask must be binary")
            mask = np.broadcast_to(xs[1] != 0, x.shape)
            if not np.all(mask.any(axis=axis)):
                raise ValueError("attention row has no allowed source")
            x = np.where(mask, x, -np.inf)
        z = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return z / z.sum(axis=axis, keepdims=True)
    raise ValueError(op)


def _signature(algorithm, inputs, state, parameters):
    if set(inputs) != set(algorithm.input_names):
        raise ValueError("input signature mismatch")
    if state is not None and set(state) != set(algorithm.state_names):
        raise ValueError("state signature mismatch")
    expected = {
        n["attrs"]["name"] for n in algorithm._ir["nodes"] if n["op"] == "parameter"
    }
    if parameters is not None and set(parameters) - expected:
        raise ValueError("unknown parameter")


def execute_numpy(algorithm, inputs, *, state=None, parameters=None):
    _signature(algorithm, inputs, state, parameters)
    values = []
    state = {} if state is None else state
    parameters = {} if parameters is None else parameters
    dtype = algorithm._ir["dtype"]
    with np.errstate(all="ignore"):
        for n in algorithm._ir["nodes"]:
            op, a = n["op"], n["attrs"]
            if op == "input":
                v = inputs[a["name"]]
            elif op == "state":
                v = state.get(a["name"], a["value"])
            elif op == "parameter":
                v = parameters.get(a["name"], a["value"])
            elif op == "constant":
                v = a["value"]
            else:
                v = numpy_op(op, [values[i] for i in n["args"]], a)
            v = np.asarray(v, dtype=dtype)
            if v.shape != tuple(n["shape"]) or not np.isfinite(v).all():
                raise ValueError(
                    f"invalid shape or nonfinite value at node {n['id']} ({op})"
                )
            values.append(v)
    outputs = {k: values[i].copy() for k, i in algorithm._ir["outputs"].items()}
    next_state = {
        n["attrs"]["name"]: values[
            algorithm._ir["updates"].get(n["attrs"]["name"], n["id"])
        ].copy()
        for n in algorithm._ir["nodes"]
        if n["op"] == "state"
    }
    return outputs, next_state


class SymbolicExecutable:
    target = "alg0"

    def __init__(self, algorithm):
        self.algorithm = algorithm

    def __call__(self, inputs):
        return self.algorithm.run(inputs)

    def step(self, inputs, *, state=None):
        return self.algorithm.step(inputs, state=state)

    def artifact(self):
        return Algorithm(self.algorithm.graph())

    def convert(self, target, **options):
        return self.artifact().compile(target, **options)

    def manifest(self):
        return {
            "target": self.target,
            "contract": "reference",
            "dtype": self.algorithm._ir["dtype"],
        }


def compile_algorithm(algorithm, target="alg1", **options):
    target = str(target).lower().replace(".", "")
    target = {
        "symbolic": "alg0",
        "circuit": "alg1",
        "tensor": "alg1",
        "attention": "alg2",
    }.get(target, target)
    if target == "alg0":
        if options:
            raise ValueError("ALG.0 has no compiler options")
        return SymbolicExecutable(algorithm)
    if target == "alg2_bias":
        from .structure import StructuralBias

        return StructuralBias.from_algorithm(algorithm, **options)
    if target not in ("alg1", "alg2"):
        raise ValueError("target must be alg0, alg1, alg2 or alg2_bias")
    from .tensor import AlgorithmModule, AttentionModule

    return (AlgorithmModule if target == "alg1" else AttentionModule)(
        algorithm.graph(), **options
    )


def convert(source, target, **options):
    """Convert authored algorithms or framework-owned compiled artifacts."""
    algorithm = source if isinstance(source, Algorithm) else source.artifact()
    if not isinstance(algorithm, Algorithm):
        raise TypeError("source must expose a validated Algorithm artifact")
    return compile_algorithm(algorithm, target, **options)


def topology(algorithm):
    """Sparse directed query-row/key-column relations, one per argument role.

    Dense matrices are materialized only by adapters with their own bounded budget.
    State-write edges describe recurrence separately from the execution schedule.
    """
    nodes = algorithm._ir["nodes"]
    arity = max((len(x["args"]) for x in nodes), default=0)
    roles = [f"operand_{i}" for i in range(max(1, arity))]
    edges = [
        {"query": node["id"], "key": source, "role": role}
        for node in nodes
        for role, source in enumerate(node["args"])
    ]
    if algorithm._ir["updates"]:
        role = len(roles)
        roles.append("next_state")
        for node in nodes:
            if (
                node["op"] == "state"
                and node["attrs"]["name"] in algorithm._ir["updates"]
            ):
                edges.append(
                    {
                        "query": node["id"],
                        "key": algorithm._ir["updates"][node["attrs"]["name"]],
                        "role": role,
                    }
                )
    return {
        "format": "tensorcode.algorithm.topology",
        "version": 1,
        "algorithm": algorithm._ir["name"],
        "roles": roles,
        "nodes": [
            {
                "id": x["id"],
                "op": x["op"],
                "shape": x["shape"],
                "name": x["attrs"].get("name"),
            }
            for x in nodes
        ],
        "edges": edges,
        "schedule": [x["id"] for x in nodes if x["op"] not in LEAVES],
        "updates": dict(algorithm._ir["updates"]),
    }


def python_source(algorithm, function_name):
    _name(function_name)
    lines = [
        "import numpy as np",
        "",
        "def _softmax(x, axis=-1, mask=None):",
        "    if mask is not None:",
        '        if not np.isin(mask, [0, 1]).all(): raise ValueError("mask must be binary")',
        "        mask = np.broadcast_to(mask != 0, x.shape)",
        '        if not np.all(mask.any(axis=axis)): raise ValueError("empty attention row")',
        "        x = np.where(mask, x, -np.inf)",
        "    z = np.exp(x - np.max(x, axis=axis, keepdims=True))",
        "    return z / z.sum(axis=axis, keepdims=True)",
        "",
        f"def {function_name}(inputs, state=None, parameters=None):",
        f'    if set(inputs) != set({list(algorithm.input_names)!r}): raise ValueError("input signature mismatch")',
        f'    if state is not None and set(state) != set({list(algorithm.state_names)!r}): raise ValueError("state signature mismatch")',
        "    state = {} if state is None else state",
        "    parameters = {} if parameters is None else parameters",
    ]
    params = [
        n["attrs"]["name"] for n in algorithm._ir["nodes"] if n["op"] == "parameter"
    ]
    lines.append(
        f'    if set(parameters) - set({params!r}): raise ValueError("unknown parameter")'
    )
    binary = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
        "matmul": "@",
        "gt": ">",
        "ge": ">=",
        "lt": "<",
        "le": "<=",
        "eq": "==",
        "ne": "!=",
    }
    for n in algorithm._ir["nodes"]:
        op, a = n["op"], n["attrs"]
        xs = [f"v{i}" for i in n["args"]]
        i = n["id"]
        if op == "input":
            expr = f"inputs[{a['name']!r}]"
        elif op in ("state", "parameter"):
            expr = f"{'state' if op == 'state' else 'parameters'}.get({a['name']!r}, {a['value']!r})"
        elif op == "constant":
            expr = repr(a["value"])
        elif op in binary:
            expr = f"({xs[0]} {binary[op]} {xs[1]})"
        elif op == "neg":
            expr = f"-{xs[0]}"
        elif op == "relu":
            expr = f"np.maximum({xs[0]}, 0)"
        elif op == "sigmoid":
            expr = f"np.exp(-np.logaddexp(0, -{xs[0]}))"
        elif op in ("tanh", "sin", "cos", "exp", "sqrt", "log", "abs"):
            expr = f"np.{op}({xs[0]})"
        elif op in ("minimum", "maximum"):
            expr = f"np.{op}({xs[0]}, {xs[1]})"
        elif op == "where":
            expr = f"np.where({xs[0]} != 0, {xs[1]}, {xs[2]})"
        elif op == "transpose":
            expr = f"np.transpose({xs[0]}, {a['axes']!r})"
        elif op == "reshape":
            expr = f"{xs[0]}.reshape({a['shape']!r})"
        elif op in ("sum", "mean"):
            expr = f"np.{op}({xs[0]}, axis={a['axis']!r}, keepdims={a['keepdims']!r})"
        elif op == "concat":
            expr = f"np.concatenate([{', '.join(xs)}], axis={a['axis']})"
        elif op == "slice":
            rank = len(algorithm._ir["nodes"][n["args"][0]]["shape"])
            axis = a["axis"] % rank
            idx = [":"] * rank
            idx[axis] = f"{a['start']}:{a['stop']}"
            expr = f"{xs[0]}[{', '.join(idx)}]"
        elif op in ("softmax", "masked_softmax"):
            expr = f"_softmax({xs[0]}, axis={a['axis']}, mask={xs[1] if len(xs) > 1 else 'None'})"
        else:
            raise ValueError(op)
        lines += [
            f"    v{i} = np.asarray({expr}, dtype={algorithm._ir['dtype']!r})",
            f'    if v{i}.shape != {tuple(n["shape"])!r} or not np.isfinite(v{i}).all(): raise ValueError("invalid node {i}")',
        ]
    outs = ", ".join(f"{k!r}: v{i}.copy()" for k, i in algorithm._ir["outputs"].items())
    states = ", ".join(
        f"{n['attrs']['name']!r}: v{algorithm._ir['updates'].get(n['attrs']['name'], n['id'])}.copy()"
        for n in algorithm._ir["nodes"]
        if n["op"] == "state"
    )
    lines += [f"    return {{{outs}}}, {{{states}}}", ""]
    return "\n".join(lines)
