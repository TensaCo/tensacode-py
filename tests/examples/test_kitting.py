import importlib.util
from pathlib import Path
import numpy as np
import pytest


def load(monkeypatch):
    pytest.importorskip("mujoco")
    path = Path(__file__).parents[2] / "examples/robotics/kitting"
    monkeypatch.syspath_prepend(str(path))
    spec = importlib.util.spec_from_file_location(
        "kitting_evaluation", path / "evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_combined_order_quality_recovery_and_contact_physics(monkeypatch):
    m = load(monkeypatch)
    good, cell, _ = m.run(0, "combined", "alg2")
    baseline, _, _ = m.run(0, "combined", "fixed_recipe")
    native, _, _ = m.run(0, "combined", "native_feedback")
    assert good["success"] and native["success"]
    assert not baseline["success"] and baseline["false_completion"]
    assert good["retries"] == 1 and good["correct_slots"] == 3 and good["quarantined"]
    assert cell.model.neq == 0 and cell.model.nmocap == 0
    assert cell.model.nu == 5
    np.testing.assert_allclose(
        good["final_positions"], native["final_positions"], atol=1e-9
    )


def test_bad_grasp_does_not_count_as_success(monkeypatch):
    m = load(monkeypatch)
    result, _, _ = m.run(0, "missed_grasp", "no_retry")
    assert result["failed_transfers"] == 1 and not result["success"]


def test_recipe_inventory_shortage_is_not_released(monkeypatch):
    module = load(monkeypatch)
    import run_recipe

    recipe = run_recipe.parse_recipe({"order": ["B", "B"]})
    row, _, _ = module.run(0, "custom", "alg0", recipe=recipe)
    assert not row["success"] and not row["sealed"]
    assert any(e["event"] == "inventory_shortage" for e in row["events"])
    for invalid in (
        {"order": []},
        {"order": ["C"]},
        {"order": ["A"], "rejected_parts": [True]},
        {"order": ["A"], "unknown": 1},
    ):
        with pytest.raises(ValueError):
            run_recipe.parse_recipe(invalid)
