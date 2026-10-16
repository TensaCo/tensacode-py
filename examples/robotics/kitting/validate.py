"""Independently audit reported order outcomes, recorded states and source hashes."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import mujoco
import numpy as np
from tensorcode.algorithm import Algorithm
from cell import SLOTS, REJECT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    a = p.parse_args()
    root = a.results
    report = json.loads((root / "report.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    checked = 0
    with zipfile.ZipFile(root / "execution-source.zip") as source:
        for name, digest in protocol["source_sha256"].items():
            assert hashlib.sha256(source.read(name)).hexdigest() == digest, name
    for row in report["episodes"]:
        spec = report["scenarios"][row["scenario"]]
        positions = np.array(row["final_positions"])
        quality = [not spec["reject"], True, True, True]
        kinds = [0, 0, 1, 0]
        order = spec.get("new_order", [1, 0, 0]) if spec["revise"] else [0, 1]
        assert row["final_order"] == order
        correct = 0
        for slot, kind in enumerate(order):
            ids = [
                i
                for i in range(4)
                if np.linalg.norm(positions[i, :2] - SLOTS[slot]) < 0.05
                and 0.40 < positions[i, 2] < 0.46
            ]
            correct += int(len(ids) == 1 and kinds[ids[0]] == kind and quality[ids[0]])
        quarantine = not spec["reject"] or (
            np.linalg.norm(positions[0, :2] - REJECT) < 0.06
            and 0.40 < positions[0, 2] < 0.50
        )
        sealed = any(e["event"] == "seal_requested" for e in row["events"])
        success = correct == len(order) and quarantine and sealed
        assert row["success"] == success and row["correct_slots"] == correct
        assert row["false_completion"] == bool(sealed and not success)
        checked += 1
    # Pair every neural substrate outcome with the independent native implementation.
    max_position_error = 0.0
    for seed in report["seeds"]:
        for scenario in report["scenarios"]:
            rows = [
                r
                for r in report["episodes"]
                if r["seed"] == seed and r["scenario"] == scenario
            ]
            reference = next(r for r in rows if r["controller"] == "native_feedback")
            for row in rows:
                if row["controller"] in ("alg0", "alg1", "alg2"):
                    error = float(
                        np.max(
                            np.abs(
                                np.array(reference["final_positions"])
                                - row["final_positions"]
                            )
                        )
                    )
                    max_position_error = max(max_position_error, error)
                    assert error < 1e-9 and row["success"] == reference["success"]
    artifact = Algorithm.from_json((root / "policy.json").read_text())
    decisions = 0
    for key in ("alg2", "fixed_recipe"):
        m = mujoco.MjModel.from_xml_path(str(root / f"{key}.xml"))
        d = mujoco.MjData(m)
        assert m.neq == 0 and m.nmocap == 0 and m.nu == 5
        assert sum(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE) == 4
        q = np.load(root / f"{key}.npz")
        d.qpos[:] = q["qpos"][-1]
        mujoco.mj_forward(m, d)
        receipt = json.loads((root / f"{key}.json").read_text())
        np.testing.assert_allclose(
            [d.body(f"part{i}").xpos for i in range(4)],
            receipt["final_positions"],
            atol=1e-7,
        )
        for case in json.loads((root / f"{key}-decisions.json").read_text()):
            for name, value in artifact.run(case["inputs"]).items():
                np.testing.assert_array_equal(value, case["outputs"][name])
            decisions += 1
    result = {
        "episodes_audited": checked,
        "recorded_decisions_replayed": decisions,
        "max_native_substrate_final_position_error": max_position_error,
        "free_objects_and_no_attachment_constraints": True,
        "frozen_source_hashes": "passed",
    }
    (root / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
