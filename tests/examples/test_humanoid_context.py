"""Runtime demonstration values and sensor-anchor invariance across substrates."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "examples/robotics/transfer")
)
from experiment import Context, build_context_algorithm


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value


def fixture_inputs():
    return {
        "phase": 0.4,
        "times": [0.0, 0.5, 1.0],
        "values": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.1, 0.3, 0.0, 0.0, 0.0, 0.028],
            [0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
        ],
        "anchor": [1.0, -0.2, 0.8],
    }


@pytest.mark.parametrize("substrate", ["alg0", "alg1", "alg2"])
def test_runtime_context_and_anchor_translation(substrate):
    machine = build_context_algorithm(3).compile(substrate)
    inputs = fixture_inputs()
    reference = {k: array(v) for k, v in machine(inputs).items()}
    shifted = dict(inputs, anchor=np.array(inputs["anchor"]) + [0.1, -0.2, 0.03])
    moved = {k: array(v) for k, v in machine(shifted).items()}
    np.testing.assert_allclose(
        moved["hand"] - reference["hand"], [0.1, -0.2, 0.03], atol=1e-10
    )
    np.testing.assert_allclose(moved["rotation"], reference["rotation"])
    changed = np.array(inputs["values"])
    changed[:, 0] += 0.4
    replacement = machine(dict(inputs, values=changed))
    np.testing.assert_allclose(
        array(replacement["hand"]) - reference["hand"], [0.4, 0, 0], atol=1e-10
    )


def test_substrate_parity_and_fixed_replay_anchor():
    algorithm = build_context_algorithm(3)
    inputs = fixture_inputs()
    expected = algorithm.compile("alg0")(inputs)
    for substrate in ["alg1", "alg2"]:
        actual = algorithm.compile(substrate)(inputs)
        for key in expected:
            np.testing.assert_allclose(
                array(actual[key]), array(expected[key]), atol=1e-10
            )
    context = {
        "times": inputs["times"],
        "values": inputs["values"],
        "origin": inputs["anchor"],
    }
    fixed = Context(context, "fixed")
    adaptive = Context(context, "context")
    anchor = np.array(context["origin"]) + [0.05, 0.08, 0.0]
    f = fixed(0.4, anchor, None)[0]
    a = adaptive(0.4, anchor, None)[0]
    np.testing.assert_allclose(a - f, [0.05, 0.08, 0.0], atol=1e-10)
