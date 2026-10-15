"""Exact structural import of supported native modules, not general algorithm discovery."""

from __future__ import annotations

import numpy as np

from .ir import AlgorithmBuilder


def from_module(module, *, input_shape, name="imported_network"):
    """Import supported torch Sequential/Linear/activation/LayerNorm modules.

    Reads current weights, not compiler provenance. Unsupported modules fail closed.
    The result retains the neural computation symbolically; no claim of semantic
    simplification or discovery of the author's intended algorithm is made.
    """
    import torch

    parameters = list(module.parameters())
    dtype = (
        "float64" if parameters and parameters[0].dtype == torch.float64 else "float32"
    )
    if any(p.dtype != getattr(torch, dtype) for p in parameters):
        raise ValueError("mixed model dtypes unsupported")
    b = AlgorithmBuilder(name=name, dtype=dtype)
    x = b.input("x", input_shape)
    counter = [0]
    shared = {}

    def weight(p, suffix):
        if id(p) in shared:
            return shared[id(p)]
        value = p.detach().cpu().numpy()
        key = f"layer{counter[0]}_{suffix}"
        result = b.parameter(key, value) if p.requires_grad else b.constant(value)
        shared[id(p)] = result
        return result

    def visit(m, x):
        counter[0] += 1
        if isinstance(m, torch.nn.Sequential):
            for child in m:
                x = visit(child, x)
            return x
        if isinstance(m, torch.nn.Linear):
            shape = x.shape
            if not shape or shape[-1] != m.in_features:
                raise ValueError("linear input shape mismatch")
            xx = x.reshape(1, shape[-1]) if len(shape) == 1 else x
            out = xx @ weight(m.weight, "weight").transpose(1, 0)
            if m.bias is not None:
                out = out + weight(m.bias, "bias")
            return out.reshape(m.out_features) if len(shape) == 1 else out
        for cls, op in (
            (torch.nn.ReLU, "relu"),
            (torch.nn.Sigmoid, "sigmoid"),
            (torch.nn.Tanh, "tanh"),
        ):
            if isinstance(m, cls):
                if getattr(m, "inplace", False):
                    raise ValueError("in-place source modules are unsupported")
                return b.op(op, x)
        if isinstance(m, torch.nn.Softmax):
            if m.dim is None:
                raise ValueError("softmax dimension must be explicit")
            return x.softmax(m.dim)
        if isinstance(m, torch.nn.LayerNorm):
            if len(m.normalized_shape) != 1 or x.shape[-1] != m.normalized_shape[0]:
                raise ValueError("only last-axis layer normalization supported")
            centered = x - x.mean(-1, True)
            variance = (centered * centered).mean(-1, True)
            out = centered / (variance + m.eps).sqrt()
            if m.elementwise_affine:
                out = out * weight(m.weight, "weight")
                if m.bias is not None:
                    out = out + weight(m.bias, "bias")
            return out
        if isinstance(m, torch.nn.Identity):
            return x
        if isinstance(m, torch.nn.Dropout) and (m.p == 0 or not m.training):
            return x
        raise ValueError(f"unsupported native module: {type(m).__name__}")

    return b.build(visit(module, x))


def from_attention(module, *, tokens, causal=False, name="imported_attention"):
    """Import a native MHA self-attention block as explicit QK/OV operations.

    Input is [tokens,embed_dim], corresponding to one batch item. Dropout, extra
    tokens, unequal widths and dynamic padding are rejected. This reconstructs
    attention computation from trained weights; it does not label its algorithm.
    """
    import torch

    if (
        not isinstance(module, torch.nn.MultiheadAttention)
        or not module._qkv_same_embed_dim
        or module.add_zero_attn
        or module.bias_k is not None
        or module.bias_v is not None
        or module.dropout != 0
    ):
        raise ValueError("unsupported attention configuration")
    if type(tokens) is not int or tokens < 1:
        raise ValueError("invalid token count")
    dtype = "float64" if module.in_proj_weight.dtype == torch.float64 else "float32"
    b = AlgorithmBuilder(name=name, dtype=dtype)
    d = module.embed_dim
    hd = d // module.num_heads
    x = b.input("x", (tokens, d))

    def w(p, name):
        return (
            b.parameter(name, p.detach().cpu().numpy())
            if p.requires_grad
            else b.constant(p.detach().cpu().numpy())
        )

    projections = x @ w(module.in_proj_weight, "qkv_weight").transpose(1, 0)
    if module.in_proj_bias is not None:
        projections = projections + w(module.in_proj_bias, "qkv_bias")
    q, k, v = [projections.slice(i * d, (i + 1) * d) for i in range(3)]
    mask = np.tril(np.ones((tokens, tokens))) if causal else None
    heads = [
        b.attention(
            q.slice(i * hd, (i + 1) * hd),
            k.slice(i * hd, (i + 1) * hd),
            v.slice(i * hd, (i + 1) * hd),
            allowed=mask,
        )
        for i in range(module.num_heads)
    ]
    result = b.concat(heads) @ w(module.out_proj.weight, "output_weight").transpose(
        1, 0
    )
    if module.out_proj.bias is not None:
        result = result + w(module.out_proj.bias, "output_bias")
    return b.build(result)
