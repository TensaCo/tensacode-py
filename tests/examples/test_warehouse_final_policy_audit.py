"""Final evaluation must retain missing cells and reject mismatched executions."""

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.robotics.warehouse.final_policy_audit import audit, check_contract

PLAN = ROOT / "docs/results/robotics/warehouse/learned-comparison/final-plan"


def test_no_completed_runs_is_not_success_and_keeps_all_cells(tmp_path):
    result = audit(
        PLAN / "execution-manifest.json", tmp_path, PLAN / "source.zip", PLAN / "cases"
    )
    assert not result["complete_and_matched"]
    assert len(result["runs"]) == result["expected_runs"] == 40
    assert all(row["status"] == "unrun" for row in result["runs"])
    assert all(
        row["observed_robot_placements"] == 0 for row in result["summary"].values()
    )


def test_incomplete_and_invalid_are_not_counted_as_physical_failures(tmp_path):
    missing_probe = tmp_path / "authored-recovery/payload-200g-compensated"
    missing_probe.mkdir(parents=True)
    invalid = tmp_path / "gru-primary/payload-200g-compensated"
    invalid.mkdir(parents=True)
    (invalid / "probe.json").write_text("{}")
    result = audit(
        PLAN / "execution-manifest.json", tmp_path, PLAN / "source.zip", PLAN / "cases"
    )
    assert len(result["runs"]) == 40
    assert result["summary"]["authored-recovery"]["incomplete"] == 1
    assert result["summary"]["gru-primary"]["invalid"] == 1
    assert result["summary"]["gru-primary"]["observed_robot_placements"] == 0


def test_modified_protocol_binding_rejected(tmp_path):
    manifest = json.loads((PLAN / "execution-manifest.json").read_text())
    manifest["protocol"]["cases"] = manifest["protocol"]["cases"][:1]
    changed = tmp_path / "manifest.json"
    changed.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Embedded protocol"):
        audit(changed, tmp_path, PLAN / "source.zip", PLAN / "cases")


@pytest.mark.parametrize(
    "field,value",
    [
        ("learned_policy_sha256", {"weights.pt": "different"}),
        ("offscreen_samples", 0),
        ("arm_slew_rad", None),
        ("full_horizon", False),
        ("requested_duration_s", 40),
        ("scenario", {"id": "different"}),
        ("source_sha256", {"warehouse/physics.py": "different"}),
    ],
)
def test_contract_rejects_changed_candidate_or_conditions(field, value):
    protocol = {
        "robots_per_run": 2,
        "physical_seconds": 85,
        "arm_slew_rad_per_20ms": 0.018,
    }
    policy = {"kind": "bc"}
    candidate = {"files": {"weights.pt": {"sha256": "frozen"}}}
    scenario = {"id": "declared"}
    source = {"examples/robotics/warehouse/physics.py": "source"}
    probe = {
        "policy": "learned_bc",
        "learned_policy_sha256": {"weights.pt": "frozen"},
        "scenario": scenario,
        "robots": 2,
        "full_horizon": True,
        "requested_duration_s": 85,
        "offscreen_samples": 4,
        "grasp_recovery": True,
        "arm_slew_rad": 0.018,
        "source_sha256": {"warehouse/physics.py": "source"},
    }
    check_contract(probe, policy, protocol, candidate, scenario, source)
    changed = deepcopy(probe)
    changed[field] = value
    with pytest.raises(ValueError):
        check_contract(changed, policy, protocol, candidate, scenario, source)
