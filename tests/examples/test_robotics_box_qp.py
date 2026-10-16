"""Portable bounded IK subproblem versus an independent constrained solver."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import lsq_linear

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "examples/robotics/transfer")
)
from box_qp import BoxQPSolver


def cases():
    rng = np.random.default_rng(104)
    for index in range(12):
        j = rng.normal(size=(6, 7))
        if index % 3 == 0:
            j[3:] *= 1e-5
        if index % 3 == 1:
            j[:, 2] = j[:, 1]
        rhs = rng.normal(size=6) * 0.08
        lo = rng.uniform(-0.03, -0.01, 7)
        hi = rng.uniform(0.01, 0.03, 7)
        yield j, rhs, lo, hi


def test_box_qp_against_independent_lsq():
    solver = BoxQPSolver(iterations=192)
    rejected = 0
    for j, rhs, lo, hi in cases():
        x = solver.solve(j, rhs, lo, hi)
        ref = lsq_linear(
            np.vstack([j, 0.025 * np.eye(7)]),
            np.r_[rhs, np.zeros(7)],
            bounds=(lo, hi),
            tol=1e-13,
        ).x
        objective = lambda v, j=j, rhs=rhs: (
            np.linalg.norm(j @ v - rhs) ** 2 + 0.025**2 * np.linalg.norm(v) ** 2
        )
        diagnostics = solver.last_diagnostics
        assert (objective(x) - objective(ref)) / 2 <= diagnostics[
            "objective_gap_bound"
        ] + 1e-12
        assert np.linalg.norm(x - ref) <= diagnostics["solution_distance_bound"] + 1e-7
        if diagnostics["converged"]:
            np.testing.assert_allclose(x, ref, atol=1e-4)
        else:
            rejected += 1
        assert np.all(x >= lo) and np.all(x <= hi)
        assert solver.last_diagnostics["projected_gradient_inf"] < 2e-5
    # Flat directions must be flagged for the robust backend, not silently accepted.
    assert rejected >= 2


@pytest.mark.parametrize("substrate", ["alg1", "alg2"])
def test_substrate_parity_and_runtime_problem(substrate):
    reference = BoxQPSolver(iterations=16)
    machine = BoxQPSolver(iterations=16, substrate=substrate)
    for j, rhs, lo, hi in list(cases())[:2]:
        expected = reference.solve(j, rhs, lo, hi)
        actual = machine.solve(j, rhs, lo, hi)
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_invalid_inputs_rejected_and_fixed_coordinates_supported():
    solver = BoxQPSolver(iterations=16)
    j, rhs, lo, hi = next(cases())
    lo[0] = hi[0] = 0.001
    assert solver.solve(j, rhs, lo, hi)[0] == 0.001
    with pytest.raises(ValueError):
        solver.solve(j, rhs, hi + 1, lo)
    with pytest.raises(ValueError):
        solver.solve(j * np.nan, rhs, lo, hi)
