"""Long outages must not turn floating-point subtraction into stale rewrites."""

import math
from dataclasses import replace

import pytest

from examples.robotics.warehouse.physics_timestamps import observation_time
from examples.robotics.warehouse.representation import Entity, SceneGraph


def test_long_outage_reconstruction_regression_keeps_actual_timestamp():
    now = 0.0
    last = previous = None
    raw_regressions = 0
    scene = SceneGraph()
    for step in range(42501):
        if step % 10 == 0:
            if now < 15:
                last = now
            raw = now - (now - last)
            if previous is not None:
                raw_regressions += raw < previous
            observed = observation_time(now, now - last, previous)
            scene.upsert(Entity("parcel", "parcel", {}, (0, 0, 0), observed, "rgbd"))
            assert abs(observed - last) <= 2 * (math.ulp(now) + math.ulp(last))
            previous = observed
        now += 0.002
    assert raw_regressions > 0
    assert now - previous > 69  # Cached evidence never becomes fresh.


def test_real_old_evidence_still_rejected():
    scene = SceneGraph()
    entity = Entity("parcel", "parcel", {}, (0, 0, 0), 15.0, "rgbd")
    scene.upsert(entity)
    for older_by in [1e-9, 0.02, 1.0]:
        stamp = observation_time(79.0, 79.0 - (15.0 - older_by), 15.0)
        with pytest.raises(ValueError, match="Out-of-order"):
            scene.upsert(replace(entity, observed_at=stamp))
    assert observation_time(79.0, 63.0, 15.0) == 16.0


@pytest.mark.parametrize("now,age", [(float("nan"), 1), (1, float("inf")), (1, -1)])
def test_invalid_clock_rejected(now, age):
    with pytest.raises(ValueError):
        observation_time(now, age)


@pytest.mark.parametrize("previous", [float("-inf"), float("inf"), float("nan")])
def test_invalid_previous_rejected_before_early_return(previous):
    with pytest.raises(ValueError, match="previous"):
        observation_time(79.0, 63.0, previous)


def test_subtraction_overflow_rejected():
    with pytest.raises(ValueError, match="reconstructed"):
        observation_time(-1e308, 1e308)
