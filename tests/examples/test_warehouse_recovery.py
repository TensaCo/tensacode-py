"""Tactile reacquisition bounds and own-pad correction semantics."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROBOTICS = Path(__file__).parents[2] / "examples/robotics"
sys.path.insert(0, str(ROBOTICS / "transfer"))
_spec = importlib.util.spec_from_file_location(
    "recovery", ROBOTICS / "warehouse/physics_recovery.py"
)
recovery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recovery)


def controller(attempts=0):
    c = object.__new__(recovery.TactilePickPlace)
    c.phase = "descend"
    c.failed = False
    c.done = False
    c.ik = SimpleNamespace(site=0, geoms=[0, 1, 2], qa=np.arange(7))
    c.recovery_attempts = attempts
    c.recovery_events = []
    c.rotation = np.eye(3)
    c.stable = 0
    c.events = []
    return c


def fake_contact_failure(self, *args, **kwargs):
    self.failed = True
    self.transition_reason = "unexpected_contact"
    self.phase = "failed"
    return np.zeros(7), np.zeros(3), np.eye(3), 0, {}


def estimated():
    return SimpleNamespace(
        site_xpos=np.array([[1.0, 2.0, 3.0]]),
        geom_xpos=np.array([[1, 2, 3], [1, 2.045, 3], [1, 1.955, 3]]),
        qpos=np.arange(7.0),
    )


def test_corrects_toward_contacting_own_pad_not_contact_world_position(monkeypatch):
    monkeypatch.setattr(recovery.PickPlace, "step", fake_contact_failure)
    c = controller()
    result = c.step(
        estimated(), {"pad_force_n": [2, 0], "object_measurement_age_s": 0.1}, 10
    )
    assert c.phase == "reacquire_retract" and not c.failed
    assert c.recovery_shift == pytest.approx([0, 0.005, 0])
    assert c.recovery_target == pytest.approx([1, 2, 3.06])
    assert result[3] == 0 and c.recovery_attempts == 1
    assert c.recovery_events[0]["pad_index"] == 0


@pytest.mark.parametrize(
    "forces,age,attempts",
    [
        ([2, 2], 0.1, 0),
        ([9, 0], 0.1, 0),
        ([2, 0], None, 0),
        ([2, 0], 0.5, 0),
        ([2, 0], 0.1, 2),
    ],
)
def test_recovery_rejects_ambiguous_high_force_stale_or_exhausted(
    monkeypatch, forces, age, attempts
):
    monkeypatch.setattr(recovery.PickPlace, "step", fake_contact_failure)
    c = controller(attempts)
    c.step(estimated(), {"pad_force_n": forces, "object_measurement_age_s": age}, 10)
    assert c.failed and c.phase == "failed"
    assert c.recovery_attempts == attempts


def test_pad_side_is_kinematic_and_horizontal():
    assert recovery.recovery_correction([[0, 0, 1], [0, 2, 3]], 0) == pytest.approx(
        [0, -0.005, 0]
    )
    with pytest.raises(ValueError):
        recovery.recovery_correction([[0, 0, 1], [0, 0, 3]], 0)
