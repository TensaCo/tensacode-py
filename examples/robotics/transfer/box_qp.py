"""Portable bounded least-squares iterations, compiled through ALG.0/1/2.

This fixed-work approximate solver reports a conservative distance bound from
convexity and the numerical Hessian eigenvalues. Callers must check ``converged``
and fall back to a reference solver when false; this is not a formal proof.
"""

from __future__ import annotations

import numpy as np

from tensorcode.algorithm import AlgorithmBuilder


def build_box_qp(size=7, iterations=128):
    if not 1 <= size <= 64 or not 1 <= iterations <= 192:
        raise ValueError("size must be 1..64 and iterations 1..192")
    b = AlgorithmBuilder(name="bounded_quadratic_iterations", dtype="float64")
    h = b.input("hessian", (size, size))
    g = b.input("linear", (size,))
    lo = b.input("lower", (size,))
    hi = b.input("upper", (size,))
    step = b.input("step")
    beta = b.input("momentum")
    x = b.input("initial", (size,)).clip(lo, hi)
    y = x

    def mv(v):
        return (h @ v.reshape(size, 1)).reshape(size)

    def objective(v):
        return (v * mv(v)).sum() * 0.5 + (g * v).sum()

    for _ in range(iterations):
        candidate = (y - step * (mv(y) + g)).clip(lo, hi)
        ordinary = (x - step * (mv(x) + g)).clip(lo, hi)
        improved = objective(candidate) <= objective(ordinary)
        next_x = b.where(improved, candidate, ordinary)
        y = b.where(improved, next_x + beta * (next_x - x), next_x)
        x = next_x
    return b.build({"solution": x})


class BoxQPSolver:
    """Solve min ||J delta-rhs||² + damping² ||delta||² inside a box."""

    def __init__(self, size=7, iterations=128, substrate="alg0", damping=0.025):
        if not np.isfinite(damping) or damping <= 0:
            raise ValueError("damping must be positive and finite")
        self.size, self.iterations, self.damping = size, iterations, damping
        self.algorithm = build_box_qp(size, iterations)
        self.machine = self.algorithm.compile(substrate)
        self.last_diagnostics = {}

    def solve(self, jacobian, rhs, lower, upper, initial=None):
        j, rhs, lo, hi = (
            np.asarray(x, dtype=float) for x in (jacobian, rhs, lower, upper)
        )
        n = self.size
        if (
            j.ndim != 2
            or j.shape[1] != n
            or rhs.shape != (j.shape[0],)
            or lo.shape != (n,)
            or hi.shape != (n,)
        ):
            raise ValueError("Incompatible least-squares shapes")
        if not all(np.isfinite(v).all() for v in (j, rhs, lo, hi)) or np.any(lo > hi):
            raise ValueError("Finite ordered bounds and inputs required")
        h = j.T @ j + self.damping**2 * np.eye(n)
        g = -j.T @ rhs
        eigenvalues = np.linalg.eigvalsh(h)
        mu, lipschitz = eigenvalues[0], eigenvalues[-1]
        if mu <= 0 or not np.isfinite(lipschitz):
            raise ValueError("Numerically positive definite Hessian required")
        beta = (np.sqrt(lipschitz) - np.sqrt(mu)) / (np.sqrt(lipschitz) + np.sqrt(mu))
        x0 = np.zeros(n) if initial is None else np.asarray(initial, dtype=float)
        if x0.shape != (n,) or not np.isfinite(x0).all():
            raise ValueError("Invalid initial iterate")
        out = self.machine(
            {
                "hessian": h,
                "linear": g,
                "lower": lo,
                "upper": hi,
                "step": 1 / lipschitz,
                "momentum": beta,
                "initial": x0,
            }
        )["solution"]
        x = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
        projected = x - np.clip(x - (h @ x + g) / lipschitz, lo, hi)
        gradient = h @ x + g
        linear_minimizer = np.where(gradient >= 0, lo, hi)
        gap = float(max(0.0, gradient @ (x - linear_minimizer)))
        # Convexity lower-bounds f* by the minimum affine tangent on the box.
        # Strong convexity then bounds distance to the unique minimizer.
        distance_bound = float(np.sqrt(2 * gap / mu))
        self.last_diagnostics = {
            "objective_gap_bound": gap,
            "solution_distance_bound": distance_bound,
            "converged": distance_bound <= 1e-4,
            "iterations": self.iterations,
            "objective": float(0.5 * x @ h @ x + g @ x),
            "residual_norm": float(np.linalg.norm(j @ x - rhs)),
            "projected_gradient_inf": float(lipschitz * np.max(np.abs(projected))),
            "condition_number": float(lipschitz / mu),
            "bound_violation": float(max(0, np.max(lo - x), np.max(x - hi))),
        }
        return x
