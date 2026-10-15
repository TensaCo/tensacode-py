"""ALG.1 native tensor circuits and ALG.2 scheduled attention machines."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .ir import Algorithm
from .runtime import _signature, topology


def torch_op(op, xs, a):
    unary = {
        "neg": torch.neg,
        "relu": torch.relu,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "sin": torch.sin,
        "cos": torch.cos,
        "exp": torch.exp,
        "sqrt": torch.sqrt,
        "log": torch.log,
        "abs": torch.abs,
    }
    binary = {
        "add": torch.add,
        "sub": torch.sub,
        "mul": torch.mul,
        "div": torch.div,
        "gt": torch.gt,
        "ge": torch.ge,
        "lt": torch.lt,
        "le": torch.le,
        "eq": torch.eq,
        "ne": torch.ne,
        "maximum": torch.maximum,
        "minimum": torch.minimum,
        "matmul": torch.matmul,
    }
    if op in unary:
        return unary[op](*xs)
    if op in binary:
        return binary[op](*xs)
    if op == "where":
        return torch.where(xs[0] != 0, xs[1], xs[2])
    if op == "transpose":
        return xs[0].permute(*a["axes"])
    if op == "reshape":
        return xs[0].reshape(a["shape"])
    if op in ("sum", "mean"):
        axis = a["axis"] if a["axis"] is not None else tuple(range(xs[0].ndim))
        return getattr(torch, op)(xs[0], dim=axis, keepdim=a["keepdims"])
    if op == "concat":
        return torch.cat(xs, dim=a["axis"])
    if op == "slice":
        return xs[0].narrow(a["axis"], a["start"], a["stop"] - a["start"])
    if op in ("softmax", "masked_softmax"):
        x = xs[0]
        if op == "masked_softmax":
            if not ((xs[1] == 0) | (xs[1] == 1)).all():
                raise ValueError("attention mask must be binary")
            mask = torch.broadcast_to(xs[1] != 0, x.shape)
            if not mask.any(dim=a["axis"]).all():
                raise ValueError("attention row has no allowed source")
            x = x.masked_fill(~mask, -torch.inf)
        return x.softmax(dim=a["axis"])
    raise ValueError(op)


class AlgorithmModule(torch.nn.Module):
    """Owned parameters and differentiable state; construction uses JSON data only."""

    target = "alg1"

    def __init__(self, artifact):
        super().__init__()
        self.algorithm = Algorithm(artifact)
        dtype = getattr(torch, self.algorithm._ir["dtype"])
        self._parameter_ids = {}
        self.register_buffer("_anchor", torch.empty(0, dtype=dtype))
        for n in self.algorithm._ir["nodes"]:
            if n["op"] == "parameter":
                key = f"p{n['id']}"
                self._parameter_ids[n["id"]] = key
                self.register_parameter(
                    key,
                    torch.nn.Parameter(torch.tensor(n["attrs"]["value"], dtype=dtype)),
                )
            elif n["op"] in ("constant", "state"):
                self.register_buffer(
                    f"c{n['id']}", torch.tensor(n["attrs"]["value"], dtype=dtype)
                )

    def _tensor(self, x):
        if isinstance(x, torch.Tensor):
            if x.device != self._anchor.device:
                raise ValueError("input and module devices differ")
            return x.to(dtype=self._anchor.dtype)
        return torch.as_tensor(x, dtype=self._anchor.dtype, device=self._anchor.device)

    def initial_state(self):
        return {
            n["attrs"]["name"]: getattr(self, f"c{n['id']}").clone()
            for n in self.algorithm._ir["nodes"]
            if n["op"] == "state"
        }

    def _read(self, n, values, context, trace):
        return [values[i] for i in n["args"]]

    def _execute(self, inputs, state=None, context=None, capture=False):
        _signature(self.algorithm, inputs, state, None)
        values = []
        state = {} if state is None else state
        trace = [] if capture else None
        for n in self.algorithm._ir["nodes"]:
            op, a = n["op"], n["attrs"]
            if op == "input":
                v = self._tensor(inputs[a["name"]])
            elif op == "state":
                v = (
                    self._tensor(state[a["name"]])
                    if a["name"] in state
                    else getattr(self, f"c{n['id']}")
                )
            elif op == "constant":
                v = getattr(self, f"c{n['id']}")
            elif op == "parameter":
                v = getattr(self, self._parameter_ids[n["id"]])
            else:
                v = torch_op(op, self._read(n, values, context, trace), a)
            v = v.to(dtype=self._anchor.dtype)
            if tuple(v.shape) != tuple(n["shape"]) or not torch.isfinite(v).all():
                raise ValueError(
                    f"invalid shape or nonfinite value at node {n['id']} ({op})"
                )
            values.append(v)
        outputs = {
            k: values[i].clone() for k, i in self.algorithm._ir["outputs"].items()
        }
        next_state = {
            n["attrs"]["name"]: values[
                self.algorithm._ir["updates"].get(n["attrs"]["name"], n["id"])
            ].clone()
            for n in self.algorithm._ir["nodes"]
            if n["op"] == "state"
        }
        return outputs, next_state, trace

    def forward(self, inputs, *, context=None):
        if self.algorithm.state_names:
            raise ValueError("stateful algorithm: use step or unroll")
        return self._execute(inputs, context=context)[0]

    def step(self, inputs, *, state=None, context=None):
        return self._execute(inputs, state, context)[:2]

    def unroll(self, sequence, *, state=None, detach_every=None):
        """Backpropagation through explicit state; truncation is opt-in."""
        if detach_every is not None and (
            type(detach_every) is not int or detach_every < 1
        ):
            raise ValueError("invalid detach interval")
        outputs = []
        for i, inputs in enumerate(sequence):
            out, state = self.step(inputs, state=state)
            outputs.append(out)
            if detach_every and (i + 1) % detach_every == 0:
                state = {k: v.detach() for k, v in state.items()}
        return outputs, self.initial_state() if state is None else state

    def inspect(self, inputs, *, state=None, context=None):
        outputs, state, trace = self._execute(inputs, state, context, capture=True)
        return {
            "outputs": outputs,
            "state": state,
            "routing": trace,
            "manifest": self.manifest(),
        }

    def artifact(self):
        ir = self.algorithm.graph()
        ir["dtype"] = str(self._anchor.dtype).removeprefix("torch.")
        for n in ir["nodes"]:
            if n["op"] == "parameter":
                n["attrs"]["value"] = (
                    getattr(self, self._parameter_ids[n["id"]]).detach().cpu().tolist()
                )
            elif n["op"] in ("constant", "state"):
                n["attrs"]["value"] = (
                    getattr(self, f"c{n['id']}").detach().cpu().tolist()
                )
        return Algorithm(ir)

    def convert(self, target, **options):
        return self.artifact().compile(target, **options)

    def manifest(self):
        return {
            "target": self.target,
            "contract": "same_operations",
            "dtype": str(self._anchor.dtype),
            "gradient": "native_autograd; comparisons discrete",
            "state": "explicit_simultaneous",
            "rounding": "backend floating-point; validate tolerance",
        }


class AttentionModule(AlgorithmModule):
    """Compile operand dependencies into an attention machine.

    Every instruction reads operand registers through scaled dot-product attention,
    then executes its declared local tensor operation. Hard mode is exact routing;
    soft mode is a structural prior, with empirical validation required. This is
    a scheduled neural/symbolic machine, NOT arbitrary source -> vanilla Transformer.
    """

    target = "alg2"

    def __init__(self, artifact, *, routing="hard", strength=12.0, context_dim=16):
        super().__init__(artifact)
        if routing not in ("hard", "soft"):
            raise ValueError("routing must be hard or soft")
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("invalid routing strength")
        if type(context_dim) is not int or not 1 <= context_dim <= 4096:
            raise ValueError("invalid context dimension")
        self.routing = routing
        self.strength = float(strength)
        self.context_dim = context_dim
        self.topology = topology(self.algorithm)
        # Zero content logits isolate the supplied algorithmic routing at initialization.
        self.register_buffer(
            "default_context",
            torch.zeros(len(artifact["nodes"]), context_dim, dtype=self._anchor.dtype),
        )

    def _read(self, n, values, context, trace):
        if not n["args"]:
            return []
        ctx = self.default_context if context is None else self._tensor(context)
        if ctx.shape != self.default_context.shape or not torch.isfinite(ctx).all():
            raise ValueError("invalid routing context")
        width = max(v.numel() for v in values)
        if width * len(values) > 16_000_000:
            raise ValueError("attention memory budget exceeded")
        memory = torch.stack(
            [F.pad(v.reshape(-1), (0, width - v.numel())) for v in values]
        )
        q = ctx[n["id"]].expand(len(n["args"]), -1)
        k = ctx[: len(values)]
        legal = torch.zeros(
            len(n["args"]), len(values), dtype=torch.bool, device=memory.device
        )
        legal[torch.arange(len(n["args"]), device=memory.device), n["args"]] = True
        bias = torch.zeros_like(legal, dtype=memory.dtype)
        bias = bias.masked_fill(
            ~legal, -torch.inf if self.routing == "hard" else -self.strength
        )
        # Actual attention contraction; V is the padded register content, Q/K are context.
        reads = F.scaled_dot_product_attention(
            q.unsqueeze(0),
            k.unsqueeze(0),
            memory.unsqueeze(0),
            attn_mask=bias.unsqueeze(0),
        ).squeeze(0)
        if trace is not None:
            weights = (q @ k.T / math.sqrt(self.context_dim) + bias).softmax(-1)
            trace.append(
                {
                    "node": n["id"],
                    "sources": list(n["args"]),
                    "weights": weights.detach().cpu().tolist(),
                }
            )
        return [
            reads[i, : values[source].numel()].reshape(values[source].shape)
            for i, source in enumerate(n["args"])
        ]

    def artifact(self):
        if self.routing != "hard":
            raise ValueError(
                "soft routing is an approximation; export its checkpoint and validate a symbolic replacement explicitly"
            )
        return super().artifact()

    def manifest(self):
        return {
            **super().manifest(),
            "routing": self.routing,
            "strength": self.strength,
            "contract": "exact_operand_routing"
            if self.routing == "hard"
            else "structural_prior",
            "local_operations": "supplied tensor instructions",
            "schedule": "supplied topological order",
            "context_binding": "explicit register-aligned features; default zero",
            "vanilla_transformer": False,
        }

    def checkpoint(self):
        return {
            "format": "tensorcode.attention_machine",
            "version": 1,
            "algorithm": super().artifact().graph(),
            "options": {
                "routing": self.routing,
                "strength": self.strength,
                "context_dim": self.context_dim,
            },
        }

    @classmethod
    def restore(cls, checkpoint):
        if (
            set(checkpoint) != {"format", "version", "algorithm", "options"}
            or checkpoint["format"] != "tensorcode.attention_machine"
            or checkpoint["version"] != 1
        ):
            raise ValueError("invalid attention checkpoint")
        return cls(checkpoint["algorithm"], **checkpoint["options"])
