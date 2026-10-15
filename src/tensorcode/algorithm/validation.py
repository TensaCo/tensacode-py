"""Explicit empirical conversion contracts; samples are never called a proof."""

from __future__ import annotations

import math

import numpy as np


def _array(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _step(source, inputs, state):
    if hasattr(source, "step"):
        return source.step(inputs, state=state)
    if state is not None:
        raise ValueError("callable does not expose state transitions")
    result = source(inputs)
    return result if isinstance(result, tuple) and len(result) == 2 else (result, {})


def compare(
    reference,
    candidate,
    cases,
    *,
    atol=1e-8,
    rtol=1e-6,
    initial_state=None,
    rollout=False,
):
    """Compare outputs AND next state. Rollouts keep each implementation's own state.

    Cases are explicit input mappings. No generated data or training split is hidden.
    A failure returns the first counterexample; errors never count as agreement.
    """
    if not all(math.isfinite(x) and x >= 0 for x in (atol, rtol)):
        raise ValueError("invalid tolerance")
    report = {
        "passed": True,
        "contract": "empirical",
        "exhaustive": False,
        "cases": 0,
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": 0.0,
        "rollout": bool(rollout),
        "counterexample": None,
    }
    rs = initial_state
    cs = initial_state
    for i, inputs in enumerate(cases):
        report["cases"] += 1
        try:
            ro, rn = _step(reference, inputs, rs)
            co, cn = _step(candidate, inputs, cs)
            for category, left, right in (("output", ro, co), ("state", rn, cn)):
                if set(left) != set(right):
                    raise ValueError(f"{category} signature mismatch")
                for name in left:
                    a, b = _array(left[name]), _array(right[name])
                    if (
                        a.shape != b.shape
                        or not np.isfinite(a).all()
                        or not np.isfinite(b).all()
                    ):
                        raise ValueError(f"{category} {name} invalid shape or value")
                    error = float(np.max(np.abs(a - b)))
                    report["max_abs_error"] = max(report["max_abs_error"], error)
                    if not np.allclose(a, b, atol=atol, rtol=rtol):
                        raise ValueError(f"{category} {name} differs")
            if rollout:
                rs = rn
                cs = cn
        except (ValueError, TypeError, RuntimeError, KeyError) as e:
            report["passed"] = False
            report["counterexample"] = {
                "case": i,
                "inputs": {k: _array(v).tolist() for k, v in inputs.items()},
                "reason": str(e),
            }
            break
    if not report["cases"]:
        raise ValueError("validation requires at least one case")
    return report


def validate_replacement(reference, candidate, cases, **options):
    report = compare(reference, candidate, cases, **options)
    if not report["passed"]:
        raise ValueError(f"replacement rejected: {report['counterexample']}")
    return report
