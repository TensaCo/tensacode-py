"""Compile algorithm relations into contextual, zero-initialized attention biases.

This is an inductive-bias target: it does not claim that a pretrained model executes
an algorithm merely because these relations were made available to it.
"""

from __future__ import annotations

import copy
import math

import torch

from .ir import Algorithm
from .runtime import topology


def _save(module):
    return {k: v.detach().cpu().tolist() for k, v in module.state_dict().items()}


def _load(module, state):
    expected = module.state_dict()
    if set(state) != set(expected):
        raise ValueError("checkpoint tensor names differ")
    tensors = {k: torch.as_tensor(state[k], dtype=v.dtype) for k, v in expected.items()}
    if any(
        tensors[k].shape != v.shape or not torch.isfinite(tensors[k]).all()
        for k, v in expected.items()
    ):
        raise ValueError("invalid checkpoint tensor")
    module.load_state_dict(tensors, strict=True)


class StructuralBias(torch.nn.Module):
    """Pq A_role Pk^T, with independent query/key grounding and a null binding.

    Inputs [batch,tokens,hidden_dim]; optional explicit bindings have shape
    [batch,tokens,nodes+1], are nonnegative and sum to one. Last slot is null.
    Learned grounding is untrained initially; users train it or supply bindings.
    """

    def __init__(self, config):
        super().__init__()
        if set(config) != {"algorithm", "hidden_dim", "heads", "binding_dim", "seed"}:
            raise ValueError("invalid structural config")
        algorithm = Algorithm(config["algorithm"])
        h, heads, d = config["hidden_dim"], config["heads"], config["binding_dim"]
        if any(type(x) is not int or not 1 <= x <= 4096 for x in (h, heads, d)):
            raise ValueError("invalid structural dimensions")
        n = len(algorithm._ir["nodes"])
        if n > 512:
            raise ValueError("structural adapter supports at most 512 algorithm nodes")
        if type(config["seed"]) is not int:
            raise ValueError("seed must be an integer")
        self.config = copy.deepcopy(config)
        self.node_count = n
        self.heads = heads
        graph = topology(algorithm)
        if len(graph["roles"]) * n * n > 4_000_000:
            raise ValueError("relation budget exceeded")
        relations = torch.zeros(len(graph["roles"]), n, n, dtype=torch.float32)
        for edge in graph["edges"]:
            relations[edge["role"], edge["query"], edge["key"]] = 1
        self.register_buffer("relations", relations)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config["seed"])
            self.query_projection = torch.nn.Linear(h, d, bias=False)
            self.key_projection = torch.nn.Linear(h, d, bias=False)
            self.entities = torch.nn.Parameter(torch.randn(n, d) / math.sqrt(d))
        self.null_logits = torch.nn.Parameter(torch.zeros(2))
        self.strengths = torch.nn.Parameter(torch.zeros(heads, relations.shape[0]))

    @classmethod
    def from_algorithm(cls, algorithm, *, hidden_dim, heads, binding_dim=32, seed=0):
        return cls(
            {
                "algorithm": algorithm.graph(),
                "hidden_dim": hidden_dim,
                "heads": heads,
                "binding_dim": binding_dim,
                "seed": seed,
            }
        )

    def _binding(self, x, role, supplied):
        if (
            x.ndim != 3
            or x.shape[-1] != self.config["hidden_dim"]
            or not torch.isfinite(x).all()
        ):
            raise ValueError("invalid latent input")
        if supplied is None:
            proj = self.query_projection if role == 0 else self.key_projection
            logits = proj(x) @ self.entities.T / math.sqrt(self.entities.shape[-1])
            logits = torch.cat(
                (logits, self.null_logits[role].expand(*logits.shape[:-1], 1)), -1
            )
            return logits.softmax(-1)
        p = supplied.to(device=x.device, dtype=x.dtype)
        if (
            p.shape != (*x.shape[:2], self.node_count + 1)
            or not torch.isfinite(p).all()
            or (p < 0).any()
            or not torch.allclose(
                p.sum(-1), torch.ones_like(p[..., 0]), atol=1e-6, rtol=1e-6
            )
        ):
            raise ValueError("invalid query/key grounding distribution")
        return p

    def bindings(self, query, key, *, query_binding=None, key_binding=None):
        if query.shape[0] != key.shape[0]:
            raise ValueError("batch mismatch")
        return self._binding(query, 0, query_binding), self._binding(
            key, 1, key_binding
        )

    def forward(self, query, key, *, query_binding=None, key_binding=None):
        pq, pk = self.bindings(
            query, key, query_binding=query_binding, key_binding=key_binding
        )
        induced = torch.einsum(
            "bti,rij,bsj->brts", pq[..., :-1], self.relations, pk[..., :-1]
        )
        return torch.einsum("hr,brts->bhts", self.strengths, induced)

    def checkpoint(self):
        return {
            "format": "tensorcode.structural_bias",
            "version": 1,
            "config": copy.deepcopy(self.config),
            "dtype": str(next(self.parameters()).dtype).removeprefix("torch."),
            "state": _save(self),
        }

    @classmethod
    def restore(cls, data):
        if (
            set(data) != {"format", "version", "config", "dtype", "state"}
            or data["format"] != "tensorcode.structural_bias"
            or data["version"] != 1
        ):
            raise ValueError("invalid structural checkpoint")
        if data["dtype"] not in ("float32", "float64"):
            raise ValueError("unsupported checkpoint dtype")
        model = cls(data["config"]).to(dtype=getattr(torch, data["dtype"]))
        _load(model, data["state"])
        return model

    def manifest(self):
        return {
            "target": "alg2_bias",
            "contract": "structural_prior",
            "initialization": "zero relational logit contribution",
            "binding": "learned unless explicitly supplied",
            "relations": "authored algorithm operand dependencies",
            "algorithm_execution_guaranteed": False,
        }


class BiasedMultiheadAttention(torch.nn.Module):
    """Adapt an existing batch-first torch MultiheadAttention via an explicit factory.

    The pretrained projection weights are copied, not reinitialized. No source model
    is mutated. Causal/key padding masks retain PyTorch's True=blocked convention.
    """

    def __init__(self, config):
        super().__init__()
        if set(config) != {
            "embed_dim",
            "num_heads",
            "bias",
            "structural",
            "freeze_base",
        }:
            raise ValueError("invalid adapter config")
        self.config = copy.deepcopy(config)
        self.base = torch.nn.MultiheadAttention(
            config["embed_dim"],
            config["num_heads"],
            bias=config["bias"],
            batch_first=True,
            dropout=0.0,
        )
        self.structure = StructuralBias(config["structural"])
        if (
            self.structure.config["hidden_dim"] != config["embed_dim"]
            or self.structure.heads != config["num_heads"]
        ):
            raise ValueError("structural dimensions do not match attention")
        if config["freeze_base"]:
            for p in self.base.parameters():
                p.requires_grad_(False)

    @classmethod
    def from_module(
        cls, module, algorithm, *, binding_dim=32, seed=0, freeze_base=True
    ):
        if (
            not isinstance(module, torch.nn.MultiheadAttention)
            or not module.batch_first
            or not module._qkv_same_embed_dim
            or module.bias_k is not None
            or module.bias_v is not None
            or module.add_zero_attn
        ):
            raise ValueError(
                "requires batch-first equal-width MHA without extra key/value tokens"
            )
        if module.dropout != 0:
            raise ValueError("dropout must be zero for deterministic conversion")
        config = {
            "embed_dim": module.embed_dim,
            "num_heads": module.num_heads,
            "bias": module.in_proj_bias is not None,
            "freeze_base": bool(freeze_base),
            "structural": {
                "algorithm": algorithm.graph(),
                "hidden_dim": module.embed_dim,
                "heads": module.num_heads,
                "binding_dim": binding_dim,
                "seed": seed,
            },
        }
        result = cls(config).to(
            device=module.in_proj_weight.device, dtype=module.in_proj_weight.dtype
        )
        result.base.load_state_dict(module.state_dict())
        result.train(module.training)
        return result

    @classmethod
    def from_projections(
        cls, query, key, value, output, algorithm, *, heads, **options
    ):
        """Adapt standard scaled-dot-product attention from four pretrained Linear maps.

        Assumes the caller uses 1/sqrt(head_dim) scoring and equal embedding widths.
        Positional biases/masks must be passed explicitly to forward.
        """
        modules = (query, key, value, output)
        if not all(isinstance(m, torch.nn.Linear) for m in modules):
            raise ValueError("projections must be Linear modules")
        dim = query.in_features
        if any(m.in_features != dim or m.out_features != dim for m in modules):
            raise ValueError("projection widths must be equal")
        if any(
            m.weight.dtype != query.weight.dtype
            or m.weight.device != query.weight.device
            for m in modules
        ):
            raise ValueError("projection device/dtype mismatch")
        base = torch.nn.MultiheadAttention(
            dim, heads, batch_first=True, dropout=0.0, bias=True
        ).to(query.weight)
        with torch.no_grad():
            base.in_proj_weight.copy_(torch.cat([m.weight for m in modules[:3]]))
            base.in_proj_bias.copy_(
                torch.cat(
                    [
                        m.bias
                        if m.bias is not None
                        else torch.zeros(
                            dim, device=m.weight.device, dtype=m.weight.dtype
                        )
                        for m in modules[:3]
                    ]
                )
            )
            base.out_proj.weight.copy_(output.weight)
            base.out_proj.bias.copy_(
                output.bias
                if output.bias is not None
                else torch.zeros_like(base.out_proj.bias)
            )
        base.train(query.training)
        return cls.from_module(base, algorithm, **options)

    def forward(
        self,
        query,
        key,
        value,
        *,
        query_binding=None,
        key_binding=None,
        key_padding_mask=None,
        attn_mask=None,
        need_weights=True,
        average_attn_weights=True,
        is_causal=False,
    ):
        bias = self.structure(
            query, key, query_binding=query_binding, key_binding=key_binding
        )
        b, h, t, s = bias.shape
        bias = bias.reshape(b * h, t, s)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                extra = torch.zeros_like(attn_mask, dtype=bias.dtype).masked_fill(
                    attn_mask, -torch.inf
                )
            else:
                extra = attn_mask.to(dtype=bias.dtype)
            bias = bias + extra
        if is_causal:
            causal = (
                torch.arange(s, device=bias.device)[None, :]
                > torch.arange(t, device=bias.device)[:, None]
            )
            bias = bias.masked_fill(causal, -torch.inf)
        if key_padding_mask is not None:
            if key_padding_mask.dtype == torch.bool:
                key_padding_mask = torch.zeros_like(
                    key_padding_mask, dtype=bias.dtype
                ).masked_fill(key_padding_mask, -torch.inf)
            effective = bias.reshape(b, h, t, s) + key_padding_mask[:, None, None, :]
        else:
            effective = bias
        if (
            torch.isnan(effective).any()
            or torch.isposinf(effective).any()
            or not torch.isfinite(effective).any(-1).all()
        ):
            raise ValueError("invalid or fully masked attention row")
        return self.base(
            query,
            key,
            value,
            attn_mask=bias,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )

    def checkpoint(self):
        return {
            "format": "tensorcode.biased_mha",
            "version": 1,
            "config": copy.deepcopy(self.config),
            "dtype": str(next(self.parameters()).dtype).removeprefix("torch."),
            "state": _save(self),
        }

    @classmethod
    def restore(cls, data):
        if (
            set(data) != {"format", "version", "config", "dtype", "state"}
            or data["format"] != "tensorcode.biased_mha"
            or data["version"] != 1
        ):
            raise ValueError("invalid adapter checkpoint")
        if data["dtype"] not in ("float32", "float64"):
            raise ValueError("unsupported checkpoint dtype")
        model = cls(data["config"]).to(dtype=getattr(torch, data["dtype"]))
        _load(model, data["state"])
        return model
