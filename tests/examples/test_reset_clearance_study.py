"""Smoke admission must never enter full-episode evidence."""

import json

import pytest

from examples.robotics.warehouse.reset_clearance_study import summarize


@pytest.mark.parametrize("smoke", [True, None])
def test_smoke_or_missing_kind_is_rejected(tmp_path, smoke):
    run = tmp_path / "run"
    run.mkdir()
    (run / "probe.json").write_text("{}")
    (run / "clearance-proof.json").write_text(json.dumps({"smoke_only": smoke}))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {"reference_run": str(run), "cases": [{"id": "test", "run_path": str(run)}]}
        )
    )
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({"cases": [{"id": "test"}]}))
    with pytest.raises(ValueError, match="Smoke/missing"):
        summarize(protocol, preflight)
