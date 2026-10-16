"""Reconcile only roundoff from reconstructing sensor time as now minus age."""

from math import isfinite, ulp


def observation_time(now, age, previous=None):
    """Keep cached timestamps monotonic within the subtraction's ULP bound.

    Genuine older observations are returned unchanged so SceneGraph's strict
    ordering check still rejects them. This does not refresh stale evidence.
    """
    if not isfinite(now) or not isfinite(age) or age < 0:
        raise ValueError("Invalid observation clock/age")
    if previous is not None and not isfinite(previous):
        raise ValueError("Invalid previous observation clock")
    reconstructed = now - age
    if not isfinite(reconstructed):
        raise ValueError("Nonfinite reconstructed observation clock")
    if previous is None or reconstructed >= previous:
        return reconstructed
    roundoff = 2 * (ulp(now) + ulp(age) + ulp(previous))
    return previous if previous - reconstructed <= roundoff else reconstructed
