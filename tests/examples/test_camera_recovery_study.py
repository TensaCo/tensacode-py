"""Factorial audit must reject mismatched execution, not silently compare it."""

import hashlib
import json

import pytest

from examples.robotics.warehouse import camera_recovery_study as study


def fixture(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    (tmp_path / "source.zip").write_bytes(b"frozen")
    scenario = code / "scenario.json"
    scenario.write_text("{}")
    run = tmp_path / "run"
    run.mkdir()
    case = {
        "id": "cell",
        "condition": "nominal",
        "run_path": str(run),
        "scenario_path": "scenario.json",
        "scenario_sha256": hashlib.sha256(scenario.read_bytes()).hexdigest(),
        "grasp_recovery": False,
        "offscreen_samples": 4,
    }
    protocol = {
        "source_directory": str(code),
        "source_zip_sha256": hashlib.sha256(b"frozen").hexdigest(),
        "cases": [case],
        "robots": 2,
        "requested_duration_s": 85,
        "driver": "580.126",
        "opengl_vendor": "NVIDIA",
        "scope": "development",
        "replication": "one episode",
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    return path, run


def test_missing_cells_remain_explicit(tmp_path):
    protocol, _ = fixture(tmp_path)
    result = study.summarize(protocol)
    assert not result["complete"]
    assert result["cells"][0]["status"] == "pending"
    assert result["cells"][0]["successful_robots"] is None


def test_cap_cannot_silently_enter_uncapped_factorial(tmp_path):
    protocol, run = fixture(tmp_path)
    probe = {
        "robots": 2,
        "policy": "authored",
        "grasp_recovery": False,
        "offscreen_samples": 4,
        "full_horizon": True,
        "requested_duration_s": 85,
        "arm_slew_rad": 0.018,
        "scenario": {},
    }
    (run / "probe.json").write_text(json.dumps(probe))
    with pytest.raises(ValueError, match="arm_slew_rad"):
        study.summarize(protocol)


def test_modified_frozen_archive_rejected_even_before_runs(tmp_path):
    protocol, _ = fixture(tmp_path)
    (tmp_path / "source.zip").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source archive"):
        study.summarize(protocol)
