from types import SimpleNamespace

import numpy as np
import pytest

from examples.robotics.warehouse.physics_actuation import ArmSlewLimiter


def test_slew_initializes_from_own_sensors_and_tracks_executed_target():
    sensors = np.arange(31, dtype=float) / 10
    requested = np.ones(29) * 10
    controller = SimpleNamespace(
        target=requested.copy(),
        observer=SimpleNamespace(
            _joint_map=[(i, 100 + i, i, 200 + i) for i in range(31)],
            proprio=SimpleNamespace(qpos=sensors),
        ),
    )
    limiter = ArmSlewLimiter(0.018)
    first = limiter.apply(controller)
    assert np.allclose(controller.target[22:29], sensors[22:29] + 0.018)
    assert np.array_equal(controller.target[:22], requested[:22])
    previous = controller.target[22:29].copy()
    controller.observer.proprio.qpos[:] = 1000  # Later targets use executed state.
    controller.target[22:29] = -10
    second = limiter.apply(controller)
    assert np.allclose(controller.target[22:29], previous - 0.018)
    assert first["clipped_joints"] == second["clipped_joints"] == 7
    assert limiter.summary["clipped_step_fraction"] == 1


def test_slew_preserves_commands_inside_bound_and_rejects_invalid_limits():
    controller = SimpleNamespace(
        target=np.zeros(29),
        observer=SimpleNamespace(
            _joint_map=[(i, i, i, i) for i in range(31)],
            proprio=SimpleNamespace(qpos=np.zeros(31)),
        ),
    )
    limiter = ArmSlewLimiter(0.018)
    limiter.apply(controller)
    controller.target[22:29] = 0.01
    assert limiter.apply(controller)["clipped_joints"] == 0
    assert np.all(controller.target[22:29] == 0.01)
    for bad in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            ArmSlewLimiter(bad)
