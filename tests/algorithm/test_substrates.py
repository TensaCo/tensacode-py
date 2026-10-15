import json

import numpy as np
import pytest
import torch
from tensorcode.algorithm import (
    Algorithm,
    compare,
    compile,
    convert,
    topology,
)
from tensorcode.algorithm import (
    AlgorithmBuilder as Builder,
)
from tensorcode.algorithm.structure import BiasedMultiheadAttention, StructuralBias
from tensorcode.algorithm.tensor import AttentionModule


def example():
    b = Builder()
    x = b.input("x", (3, 2))
    w = b.parameter("weight", [[1.0, 2.0], [3.0, -1.0]])
    y = (x @ w).tanh()
    s = b.state("memory", np.zeros((3, 2)))
    b.update(s, 0.5 * s + y)
    return b.build({"value": y + s})


def test_roundtrip_all_substrates_and_state():
    a = example()
    cases = [{"x": np.random.default_rng(i).normal(size=(3, 2))} for i in range(12)]
    for target in ("alg0", "alg1", "alg2"):
        executable = a.compile(target)
        assert compare(a, executable, cases, rollout=True)["passed"]
        for other in ("alg0", "alg1", "alg2"):
            assert compare(a, convert(executable, other), cases, rollout=True)["passed"]
    restored = Algorithm.from_json(a.to_json())
    assert compare(a, restored, cases, rollout=True)["passed"]
    assert compare(a, a.optimize(), cases, rollout=True)["passed"]


def test_gradient_through_attention_and_state():
    b = Builder()
    x = b.input("x")
    w = b.parameter("w", 0.7)
    s = b.state("s", 0.0)
    b.update(s, s * w + x)
    a = b.build(s * w + x)
    grads = []
    for target in ("alg1", "alg2"):
        m = a.compile(target)
        xs = [
            {"x": torch.tensor(1.0, dtype=torch.float64, requires_grad=True)}
            for _ in range(4)
        ]
        ys, _ = m.unroll(xs)
        ys[-1]["result"].backward()
        grads.append([next(m.parameters()).grad.item(), xs[0]["x"].grad.item()])
    np.testing.assert_allclose(grads[0], grads[1])
    assert grads[0][1] > 0


def test_live_parameter_export_and_training():
    b = Builder()
    x = b.input("x", (2,))
    w = b.parameter("w", [0.0, 0.0])
    a = b.build(x * w)
    m = a.compile("alg2")
    opt = torch.optim.Adam(m.parameters(), lr=0.15)
    for _ in range(100):
        opt.zero_grad()
        loss = (
            (m({"x": [1.0, 2.0]})["result"] - torch.tensor([2.0, 6.0])).square().mean()
        )
        loss.backward()
        opt.step()
    assert loss.item() < 0.001
    q = m.convert("alg0")
    assert compare(m, q, [{"x": [1.0, 2.0]}, {"x": [-3.0, 1.0]}])["passed"]
    restored = AttentionModule.restore(json.loads(json.dumps(m.checkpoint())))
    assert compare(m, restored, [{"x": [1.0, 2.0]}])["passed"]


def test_source_compiler_and_python_export():
    a = compile(
        """def controller(q, dq, target, gain):
 error = target - q
 tau = gain * error - 2 * dq
 for i in range(2):
  tau = tau + 0.1
 return {"torque": tau.clip(-5, 5)}
""",
        inputs={"q": (3,), "dq": (3,), "target": (3,)},
        parameters={"gain": 3.0},
    )
    case = {"q": [1.0, 2.0, 3.0], "dq": [0.1, -0.2, 0.3], "target": [0.0, 1.0, 2.0]}
    assert compare(a, a.compile("alg2"), [case])["passed"]
    namespace = {}
    exec(a.to_python("policy"), namespace)  # noqa: S102 - test our own generated source
    np.testing.assert_allclose(
        namespace["policy"](case)[0]["torque"], a.run(case)["torque"]
    )
    with pytest.raises(ValueError):
        compile('def f(x):\n return __import__("os")', inputs={"x": ()})
    with pytest.raises(ValueError):
        compile("def f(x):\n while x:\n  x=x+1\n return x", inputs={"x": ()})


def test_attention_primitive_masks_and_vector_ops():
    b = Builder()
    q = b.input("q", (2, 3))
    k = b.input("k", (4, 3))
    v = b.input("v", (4, 2))
    mask = b.input("mask", (2, 4))
    y = b.attention(q, k, v, allowed=mask)
    z = b.concat([y, y], axis=0).slice(0, 2, axis=0)
    a = b.build({"y": z, "stats": z.mean(axis=0), "total": z.sum()})
    rng = np.random.default_rng(3)
    case = {
        "q": rng.normal(size=(2, 3)),
        "k": rng.normal(size=(4, 3)),
        "v": rng.normal(size=(4, 2)),
        "mask": [[1, 0, 1, 0], [0, 1, 0, 0]],
    }
    assert compare(a, a.compile("alg1"), [case])["passed"]
    assert compare(a, a.compile("alg2"), [case])["passed"]
    for impl in (a, a.compile("alg1"), a.compile("alg2")):
        with pytest.raises(ValueError):
            impl.run({**case, "mask": np.zeros((2, 4))}) if isinstance(
                impl, Algorithm
            ) else impl({**case, "mask": np.zeros((2, 4))})


def test_soft_routing_is_not_silently_exported():
    b = Builder()
    x = b.input("x")
    a = b.build(x - 3)
    m = a.compile("alg2", routing="soft", strength=0)
    assert not compare(a, m, [{"x": 10.0}])["passed"]
    with pytest.raises(ValueError, match="approximation"):
        m.convert("alg0")
    r = AttentionModule.restore(m.checkpoint())
    assert compare(m, r, [{"x": 10.0}])["passed"]


def test_structural_bias_roles_null_and_gradients():
    b = Builder()
    x = b.input("x")
    y = b.input("y")
    a = b.build(x - y)
    s = StructuralBias.from_algorithm(a, hidden_dim=4, heads=2, binding_dim=3).double()
    latent = torch.randn(1, 3, 4, dtype=torch.float64)
    p = torch.cat(
        (torch.eye(3, dtype=torch.float64), torch.zeros(3, 1, dtype=torch.float64)), 1
    )[None]
    out = s(latent, latent, query_binding=p, key_binding=p)
    assert torch.count_nonzero(out) == 0
    out.sum().backward()
    assert s.strengths.grad.abs().sum() > 0
    with torch.no_grad():
        s.strengths[0, 0] = 2
    out = s(latent, latent, query_binding=p, key_binding=p)
    assert out[0, 0, 2, 0] == 2 and out[0, 0, 2, 1] == 0
    null = torch.zeros_like(p)
    null[..., -1] = 1
    assert (
        torch.count_nonzero(s(latent, latent, query_binding=null, key_binding=p)) == 0
    )
    wrong = p.clone()
    wrong[0, 0, 0] = 2
    with pytest.raises(ValueError):
        s(latent, latent, query_binding=wrong)


def test_pretrained_mha_adapter_zero_identity_training_and_reload():
    torch.manual_seed(2)
    base = (
        torch.nn.MultiheadAttention(8, 2, batch_first=True, dropout=0).double().eval()
    )
    b = Builder()
    x = b.input("x")
    a = b.build(x + 2)
    adapter = BiasedMultiheadAttention.from_module(base, a, binding_dim=4).eval()
    x = torch.randn(2, 4, 8, dtype=torch.float64)
    mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    expected = base(x, x, x, attn_mask=mask)[0]
    actual = adapter(x, x, x, attn_mask=mask)[0]
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual.square().sum().backward()
    assert adapter.structure.strengths.grad is not None
    assert all(p.grad is None for p in base.parameters())
    with torch.no_grad():
        adapter.structure.strengths.fill_(2)
    restored = (
        BiasedMultiheadAttention.restore(json.loads(json.dumps(adapter.checkpoint())))
        .double()
        .eval()
    )
    torch.testing.assert_close(
        adapter(x, x, x)[0], restored(x, x, x)[0], atol=1e-6, rtol=1e-6
    )
    with pytest.raises(ValueError):
        adapter(x, x, x, attn_mask=torch.ones(4, 4, dtype=torch.bool))


def test_rejection_atomicity_and_validation_contract():
    a = example()
    state = a.initial_state()
    before = {k: v.copy() for k, v in state.items()}
    with pytest.raises(ValueError):
        a.step({"x": np.ones(3)}, state=state)
    np.testing.assert_array_equal(state["memory"], before["memory"])
    for field, value in [("format", "bad"), ("version", 9)]:
        bad = a.graph()
        bad[field] = value
        with pytest.raises(ValueError):
            Algorithm(bad)
    bad = a.graph()
    bad["nodes"][-1]["args"][0] = 999
    with pytest.raises(ValueError):
        Algorithm(bad)
    bad = a.graph()
    bad["nodes"][0]["shape"] = [1_000_001]
    with pytest.raises(ValueError):
        Algorithm(bad)
    with pytest.raises(ValueError):
        compare(a, a, [])
    with pytest.raises(ValueError):
        a.run({"x": np.ones((3, 2))})
    with pytest.raises(ValueError):
        a.compile("unknown")
    with pytest.raises(ValueError):
        Builder().input("bad.name")


def test_context_changes_soft_routing_but_not_hard():
    b = Builder()
    x = b.input("x")
    a = b.build(x + 3)
    ctx = torch.randn(len(a.graph()["nodes"]), 8, dtype=torch.float64) * 100
    hard = a.compile("alg2", context_dim=8)
    torch.testing.assert_close(
        hard({"x": 7.0})["result"], hard({"x": 7.0}, context=ctx)["result"]
    )
    trace = hard.inspect({"x": 7.0})["routing"]
    assert trace and all(np.array(row["weights"]).max() == 1 for row in trace)
    assert topology(a)["roles"] == ["operand_0", "operand_1"]


def test_import_trained_network_and_attention_without_provenance():
    from tensorcode.algorithm import from_attention, from_module

    torch.manual_seed(7)
    net = (
        torch.nn.Sequential(
            torch.nn.Linear(3, 5),
            torch.nn.Tanh(),
            torch.nn.LayerNorm(5),
            torch.nn.Linear(5, 2),
        )
        .double()
        .eval()
    )
    a = from_module(net, input_shape=(3,))
    x = torch.randn(3, dtype=torch.float64)
    torch.testing.assert_close(
        a.compile("alg2")({"x": x})["result"], net(x), atol=1e-10, rtol=1e-10
    )
    mha = (
        torch.nn.MultiheadAttention(8, 2, batch_first=True, dropout=0.0).double().eval()
    )
    a = from_attention(mha, tokens=4, causal=True)
    x = torch.randn(4, 8, dtype=torch.float64)
    mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), 1)
    expected = mha(x[None], x[None], x[None], attn_mask=mask)[0][0]
    torch.testing.assert_close(
        a.compile("alg1")({"x": x})["result"], expected, atol=1e-10, rtol=1e-10
    )
    torch.testing.assert_close(
        a.compile("alg2")({"x": x})["result"], expected, atol=1e-10, rtol=1e-10
    )
    with pytest.raises(ValueError):
        from_module(torch.nn.Conv1d(2, 2, 3), input_shape=(2, 5))


def test_all_elementary_ops_and_eager_selection():
    b = Builder()
    x = b.input("x", (2, 3))
    z = x.abs() + 0.2
    a = b.build(
        {
            "nonlinear": z.sin()
            + z.cos()
            + z.sqrt()
            + z.log()
            + (-z).exp()
            + z.sigmoid()
            + x.relu(),
            "compare": b.where(x == 0, 3, b.where(x != 1, x, 2)),
            "axes": x.transpose(1, 0).reshape(6).sum(),
        }
    )
    cases = [{"x": np.random.default_rng(i).normal(size=(2, 3))} for i in range(10)]
    for t in ("alg1", "alg2"):
        assert compare(a, a.compile(t), cases)["passed"]
    ns = {}
    exec(a.to_python(), ns)  # noqa: S102 - test our own generated source
    for case in cases:
        expected = a.run(case)
        actual = ns["run"](case)[0]
        for k in expected:
            np.testing.assert_allclose(actual[k], expected[k])
    b = Builder()
    x = b.input("x")
    a = b.build(b.where(x > 0, 1 / x, 0))
    for t in ("alg0", "alg1", "alg2"):
        with pytest.raises(ValueError):
            a.compile(t)({"x": 0.0})


def test_shared_native_parameters_and_gradient_binding():
    from tensorcode.algorithm import from_module

    layer = torch.nn.Linear(2, 2, bias=False).double()
    net = torch.nn.Sequential(layer, torch.nn.Tanh(), layer)
    a = from_module(net, input_shape=(2,))
    assert len(list(a.compile("alg1").parameters())) == 1
    bias = a.compile("alg2_bias", hidden_dim=4, heads=2, binding_dim=3).double()
    with torch.no_grad():
        bias.strengths.fill_(1.0)
    q = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
    bias(q, q).square().sum().backward()
    assert bias.query_projection.weight.grad.abs().sum() > 0
    assert bias.key_projection.weight.grad.abs().sum() > 0
    restored = StructuralBias.restore(json.loads(json.dumps(bias.checkpoint())))
    torch.testing.assert_close(bias(q, q), restored(q, q), atol=0, rtol=0)


def test_no_torch_import_for_symbolic_api_and_sparse_topology():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'import sys; import tensorcode.algorithm; assert "torch" not in sys.modules',
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    b = Builder()
    x = b.input("x")
    s = b.state("s", 0)
    b.update(s, s + x)
    a = b.build(s)
    graph = topology(a)
    assert "next_state" in graph["roles"]
    assert "relations" not in graph
    assert len(graph["edges"]) == 3


def test_transactional_state_outputs_and_counterexamples():
    a = example()
    case = {"x": np.ones((3, 2))}
    state = a.initial_state()
    for t in ("alg0", "alg1", "alg2"):
        m = a.compile(t)
        _out, next_state = m.step(case, state=state)
        if t == "alg0":
            next_state["memory"][:] = 100
        else:
            next_state["memory"].detach().fill_(100)
        np.testing.assert_array_equal(state["memory"], np.zeros((3, 2)))
    report = compare(a, lambda inputs: {"bad": 0}, [case])
    assert not report["passed"] and report["counterexample"]["case"] == 0
    assert report["exhaustive"] is False
