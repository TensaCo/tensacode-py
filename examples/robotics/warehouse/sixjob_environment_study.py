"""Independent physical/environment audit of the frozen six-job sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .validate_jobs import validate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def telemetry(records, robot, job):
    pickup_errors, destination_errors = [], []
    first_pickup = first_destination = None
    first_failure = None
    previous_phase = None
    assigned = False
    phases = []
    for row in records:
        state = row["robots"][robot]
        sensed = state["job_sensors"][job]
        truth = state["job_evaluation"][job]
        for key, age_key, truth_key, collection in (
            (
                "object_position",
                "object_measurement_age_s",
                "object_position",
                pickup_errors,
            ),
            (
                "destination_position",
                "destination_measurement_age_s",
                "receiving_surface_xyz",
                destination_errors,
            ),
        ):
            age = sensed.get(age_key)
            if sensed.get(key) is None or age is None or age > 0.11:
                continue
            error = float(np.linalg.norm(np.asarray(sensed[key]) - truth[truth_key]))
            sample = {"time": row["t"], "error_m": error}
            if key == "object_position" and first_pickup is None:
                first_pickup = sample
            if key == "destination_position" and first_destination is None:
                first_destination = sample
            if state["current_job"] == job and (
                key == "destination_position"
                or state["phase"] in ("walk", "settle", "approach", "align", "descend")
            ):
                collection.append(error)
        if state["current_job"] != job:
            continue
        assigned |= state["task_id"] is not None
        phase = state["phase"]
        if phase != previous_phase:
            phases.append({"time": row["t"], "phase": phase})
        if phase == "failed" and first_failure is None:
            first_failure = {
                "time": row["t"],
                "from_phase": previous_phase,
                "reason": state["controller_state"].get("transition_reason"),
            }
        previous_phase = phase

    def stats(values):
        return (
            {
                "samples": len(values),
                "median_error_m": float(np.median(values)),
                "p95_error_m": float(np.percentile(values, 95)),
            }
            if values
            else None
        )

    return {
        "assigned": bool(assigned),
        "first_pickup_observation": first_pickup,
        "first_destination_observation": first_destination,
        "fresh_active_pregrasp_pickup_error": stats(pickup_errors),
        "fresh_active_destination_error": stats(destination_errors),
        "failure": first_failure,
        "phases": phases,
        "pose_error_scope": "Scoring-only comparison to retained truth; never supplied to controller. Fresh≤110ms observations, pickup active before close only; destination all active phases.",
    }


def summarize(protocol_path, preflight_path):
    protocol = json.loads(protocol_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    reference_path = Path(protocol["reference_run"])
    reference = json.loads((reference_path / "probe.json").read_text())
    if (
        sha(reference_path / "probe.json") != protocol["reference_probe_sha256"]
        or sha(protocol["frozen_source_zip"]) != protocol["frozen_source_zip_sha256"]
    ):
        raise ValueError("Frozen reference evidence changed")
    baseline = mujoco.MjModel.from_xml_path(str(reference_path / "scene.xml"))
    expected_scenes = {c["id"]: c for c in preflight["cases"]}
    rows = []
    for case in protocol["cases"]:
        path = Path(case["run_path"])
        row = dict(case)
        if not (path / "environment-proof.json").is_file():
            row["status"] = "pending"
            rows.append(row)
            continue
        probe = json.loads((path / "probe.json").read_text())
        proof = json.loads((path / "environment-proof.json").read_text())
        if sha(path / "environment-harness.py") != protocol[
            "environment_harness_sha256"
        ] or sha(path / "environment-protocol.json") != sha(protocol_path):
            raise ValueError("Environment source/protocol changed")
        if (
            proof["protocol_sha256"] != sha(protocol_path)
            or proof["case"] != case
            or json.loads((path / "environment-case.json").read_text()) != case
        ):
            raise ValueError("Environment case binding changed")
        if (
            proof["environment_harness_sha256"]
            != protocol["environment_harness_sha256"]
            or proof["frozen_source_zip_sha256"] != protocol["frozen_source_zip_sha256"]
        ):
            raise ValueError("Environment provenance proof mismatch")
        if probe["source_sha256"] != reference["source_sha256"]:
            raise ValueError("Controller source changed")
        expected = expected_scenes[case["id"]]
        if (
            sha(path / "scene.xml") != expected["scene_sha256"]
            or probe["initial_qpos_sha256"] != expected["initial_qpos_sha256"]
        ):
            raise ValueError("Physical scene/initial state differs from preflight")
        if (
            probe["robots"] != 3
            or probe["jobs_per_robot"] != 2
            or probe["dispatch_substrate"] != "alg2"
            or probe["requested_duration_s"] != 170
        ):
            raise ValueError("Execution configuration changed")
        for name, digest in protocol["mission_sha256"].items():
            if sha(path / "mission" / name) != digest:
                raise ValueError("Replayed model mission changed")
        model = mujoco.MjModel.from_xml_path(str(path / "scene.xml"))
        actual_data = mujoco.MjData(model)
        baseline_data = mujoco.MjData(baseline)
        mujoco.mj_forward(model, actual_data)
        mujoco.mj_forward(baseline, baseline_data)
        for i in range(3):
            for suffix in ("", "_b"):
                body = model.body(f"r{i}/object{suffix}").id
                old = baseline.body(f"r{i}/object{suffix}").id
                np.testing.assert_allclose(
                    actual_data.xpos[body] - baseline_data.xpos[old],
                    case["pickup_delta_xyz"],
                    atol=1e-12,
                )
                for geometry, delta in (
                    ("receiving_marker", case["receiver_delta_xyz"]),
                    (
                        "receiving_support",
                        np.asarray(case["receiver_delta_xyz"]) * [1, 1, 0.5],
                    ),
                    ("pedestal", np.asarray(case["pickup_delta_xyz"]) * [1, 1, 0.5]),
                ):
                    current_geom = model.geom(f"r{i}/{geometry}{suffix}").id
                    old_geom = baseline.geom(f"r{i}/{geometry}{suffix}").id
                    np.testing.assert_allclose(
                        actual_data.geom_xpos[current_geom]
                        - baseline_data.geom_xpos[old_geom],
                        delta,
                        atol=1e-12,
                    )
                    if geometry != "receiving_marker":
                        np.testing.assert_allclose(
                            actual_data.geom_xpos[current_geom, 2]
                            - model.geom_size[current_geom, 2],
                            baseline_data.geom_xpos[old_geom, 2]
                            - baseline.geom_size[old_geom, 2],
                            atol=1e-12,
                        )
                np.testing.assert_allclose(
                    model.body_mass[body], case["payload_mass_kg"], atol=1e-12
                )
                np.testing.assert_allclose(
                    model.body_inertia[body],
                    baseline.body_inertia[old] * case["payload_mass_kg"] / 0.08,
                    atol=1e-12,
                )
        audit = validate(path)
        records = json.loads((path / "records.json").read_text())
        diagnostics = {
            r: {j: telemetry(records, r, j) for j in ("a", "b")}
            for r in audit["robots"]
        }
        succeeded = sum(
            s["physical_success"]
            for robot in audit["robots"].values()
            for s in robot.values()
        )
        row.update(
            status="complete",
            successful_placements=succeeded,
            placements=6,
            all_jobs_success=audit["success"],
            validation=audit,
            telemetry=diagnostics,
            actual_masses_and_inertias_verified=True,
            actual_pickup_receiver_and_grounded_support_geometry_verified=True,
            original_controller_sources_verified=True,
        )
        rows.append(row)
    return {
        "schema": "tensorcode.warehouse.sixjob.environment.results.v1",
        "protocol_sha256": sha(protocol_path),
        "scope": protocol["scope"],
        "audit_mujoco_version": mujoco.__version__,
        "runtime_scope": "Run commands use the same host interpreter and frozen v5 source; v5 does not directly record per-run GL/MuJoCo versions. Resource receipts remain separate.",
        "complete": all(r["status"] == "complete" for r in rows),
        "completed_cases": sum(r["status"] == "complete" for r in rows),
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.protocol, args.preflight)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "completed_cases": result["completed_cases"],
            }
        )
    )


if __name__ == "__main__":
    main()
