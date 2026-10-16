"""Check the committed G1 evidence bundle against traces and exported computation."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from tensorcode.algorithm import Algorithm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    a = p.parse_args()
    h = a.evidence
    report = json.loads((h / "report.json").read_text())
    for name, digest in json.loads((h / "artifact-sha256.json").read_text()).items():
        assert hashlib.sha256((h / name).read_bytes()).hexdigest() == digest, name
    artifact = Algorithm.from_json(
        gzip.decompress((h / "walker.json.gz").read_bytes()).decode()
    )
    checked = 0
    for name, metrics in report["rollouts"].items():
        records = json.loads(
            gzip.decompress((h / f"{name}-trace.json.gz").read_bytes())
        )
        qpos = np.load(h / f"{name}-state.npz")["qpos"]
        assert len(qpos) == len(records) == metrics["steps"]
        assert np.isfinite(qpos).all()
        maximum = max(x["object"][2] - 0.805 for x in records)
        contact = (
            sum("pad_a" in x["contacts"] and "pad_b" in x["contacts"] for x in records)
            * 0.02
        )
        held = (
            sum(
                "pad_a" in x["contacts"]
                and "pad_b" in x["contacts"]
                and x["object"][2] > 0.905
                for x in records
            )
            * 0.02
        )
        assert np.isclose(maximum, metrics["max_lift"], atol=1e-12)
        assert np.isclose(contact, metrics["bilateral_contact_seconds"], atol=1e-12)
        assert np.isclose(held, metrics["held_above_10cm_seconds"], atol=1e-12)
        assert metrics["min_height"] <= qpos[:, 2].min() + 1e-12
        if name == "alg2":
            for rec in records[::100]:
                actual = artifact.run(
                    {"observation": np.array(rec["observation"], np.float32)}
                )["action"]
                assert np.allclose(actual, rec["action"], atol=2e-5, rtol=2e-5)
                checked += 1
    xml = (
        (h / "scene.xml")
        .read_text()
        .replace(
            "assets/src/assets/robots/unitree_g1/xmls/assets",
            str(a.assets / "src/assets/robots/unitree_g1/xmls/assets"),
        )
    )
    m = mujoco.MjModel.from_xml_string(xml)
    assert m.neq == 0 and m.nmocap == 0 and m.nu == 31
    assert m.joint("floating_base_joint").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert m.joint("object_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
    result = {
        "artifact_hashes": "passed",
        "trace_metric_recalculation": "passed",
        "rollouts_checked": len(report["rollouts"]),
        "frozen_artifact_action_checks": checked,
        "no_equality_constraints": True,
        "no_mocap_bodies": True,
        "free_root_and_object": True,
        "actuated_joints": m.nu,
    }
    (h / "evidence-validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
