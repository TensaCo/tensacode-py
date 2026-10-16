"""Environment-only perturbations around an immutable repeated-job controller.

The wrapper changes initial physical XML only. No perturbation parameters are
passed to controllers, perception, scheduling or waypoint generation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def perturb_both_jobs(xml, case, robots):
    pickup = np.asarray(case["pickup_delta_xyz"], dtype=float)
    receiver = np.asarray(case["receiver_delta_xyz"], dtype=float)
    mass = float(case["payload_mass_kg"])
    if (
        pickup.shape != (3,)
        or receiver.shape != (3,)
        or not np.isfinite(np.r_[pickup, receiver, mass]).all()
    ):
        raise ValueError("Invalid environment geometry")
    if max(abs(pickup)) > 0.02 or max(abs(receiver)) > 0.01 or mass not in (0.08, 0.2):
        raise ValueError("Environment outside frozen sweep bounds")
    if not pickup.any() and not receiver.any() and mass == 0.08:
        return xml
    root = ET.fromstring(xml)
    for i in range(robots):
        for suffix in ("", "_b"):

            def named(tag, name, i=i, suffix=suffix):
                element = root.find(f".//{tag}[@name='r{i}/{name}{suffix}']")
                if element is None:
                    raise ValueError(f"Missing physical geometry r{i}/{name}{suffix}")
                return element

            def move(element, delta):
                position = np.fromstring(element.get("pos", "0 0 0"), sep=" ")
                element.set("pos", " ".join(map(str, position + delta)))

            move(named("body", "object"), pickup)
            pedestal = named("geom", "pedestal")
            move(pedestal, pickup * [1, 1, 0.5])
            size = np.fromstring(pedestal.get("size"), sep=" ")
            size[2] += pickup[2] / 2
            pedestal.set("size", " ".join(map(str, size)))
            named("geom", "cube").set("mass", str(mass))
            move(named("geom", "receiving_marker"), receiver)
            support = named("geom", "receiving_support")
            move(support, receiver * [1, 1, 0.5])
            size = np.fromstring(support.get("size"), sep=" ")
            size[2] += receiver[2] / 2
            support.set("size", " ".join(map(str, size)))
    return ET.tostring(root, encoding="unicode")


def load_frozen(protocol):
    code = Path(protocol["frozen_code"])
    if digest(protocol["frozen_source_zip"]) != protocol["frozen_source_zip_sha256"]:
        raise ValueError("Frozen controller archive changed")
    reference = Path(protocol["reference_run"])
    if digest(reference / "probe.json") != protocol["reference_probe_sha256"]:
        raise ValueError("Reference evidence changed")
    probe = json.loads((reference / "probe.json").read_text())
    for name, expected in probe["source_sha256"].items():
        if digest(code / "examples/robotics" / name) != expected:
            raise ValueError(f"Frozen runtime file changed: {name}")
    sys.path.insert(0, str(code / "examples/robotics"))
    sys.path.insert(0, str(code / "src"))
    frozen = importlib.import_module("warehouse.physics_jobs")
    if (
        Path(frozen.__file__).resolve()
        != (code / "examples/robotics/warehouse/physics_jobs.py").resolve()
    ):
        raise ValueError("Unexpected executor import")
    return frozen, probe


def preflight(protocol):
    import mujoco

    frozen, reference = load_frozen(protocol)
    xml, _, _ = frozen.compose_jobs_scene(3, 2.5, frozen.ASSETS)
    baseline = mujoco.MjModel.from_xml_string(xml)
    initial = frozen.initialize(baseline, 3, frozen.ASSETS)
    initial_hash = hashlib.sha256(initial.qpos.tobytes()).hexdigest()
    if initial_hash != reference["initial_qpos_sha256"]:
        raise ValueError("Nominal initial physics differs from qualified v5")
    if xml != (Path(protocol["reference_run"]) / "scene.xml").read_text():
        raise ValueError("Nominal scene XML differs from qualified v5")
    checks = []
    for case in protocol["cases"]:
        changed = perturb_both_jobs(xml, case, 3)
        model = mujoco.MjModel.from_xml_string(changed)
        data = frozen.initialize(model, 3, frozen.ASSETS)
        for i in range(3):
            for suffix in ("", "_b"):
                name = f"r{i}/object{suffix}"
                body, original = model.body(name).id, baseline.body(name).id
                np.testing.assert_allclose(
                    model.body_mass[body], case["payload_mass_kg"], atol=1e-12
                )
                np.testing.assert_allclose(
                    model.body_inertia[body],
                    baseline.body_inertia[original] * case["payload_mass_kg"] / 0.08,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    data.xpos[body] - initial.xpos[original],
                    case["pickup_delta_xyz"],
                    atol=1e-12,
                )
                for geom_name in ("receiving_marker", "receiving_support", "pedestal"):
                    geom = model.geom(f"r{i}/{geom_name}{suffix}").id
                    before = baseline.geom(f"r{i}/{geom_name}{suffix}").id
                    if geom_name == "receiving_marker":
                        np.testing.assert_allclose(
                            data.geom_xpos[geom] - initial.geom_xpos[before],
                            case["receiver_delta_xyz"],
                            atol=1e-12,
                        )
                    else:
                        np.testing.assert_allclose(
                            data.geom_xpos[geom, 2] - model.geom_size[geom, 2],
                            initial.geom_xpos[before, 2]
                            - baseline.geom_size[before, 2],
                            atol=1e-12,
                        )
        checks.append(
            {
                "id": case["id"],
                "scene_sha256": hashlib.sha256(changed.encode()).hexdigest(),
                "initial_qpos_sha256": hashlib.sha256(data.qpos.tobytes()).hexdigest(),
                "six_payloads_and_supports_checked": True,
            }
        )
    return {
        "nominal_matches_qualified_scene_and_initial_state": True,
        "controller_files_unchanged": True,
        "cases": checks,
    }


def run(protocol_path, case_id, output):
    protocol = json.loads(protocol_path.read_text())
    if digest(__file__) != protocol["environment_harness_sha256"]:
        raise ValueError("Environment harness changed after declaration")
    frozen, reference = load_frozen(protocol)
    case = next(c for c in protocol["cases"] if c["id"] == case_id)
    mission = Path(protocol["reference_run"]) / "mission"
    for name, expected in protocol["mission_sha256"].items():
        if digest(mission / name) != expected:
            raise ValueError("Certified mission input changed")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(mission, output / "mission", dirs_exist_ok=True)
    shutil.copyfile(__file__, output / "environment-harness.py")
    shutil.copyfile(protocol_path, output / "environment-protocol.json")
    (output / "environment-case.json").write_text(json.dumps(case, indent=2) + "\n")
    compose = frozen.compose_jobs_scene

    def environment_scene(*args, **kwargs):
        xml, local, offsets = compose(*args, **kwargs)
        return perturb_both_jobs(xml, case, 3), local, offsets

    frozen.compose_jobs_scene = environment_scene
    try:
        report = frozen.run(
            SimpleNamespace(
                output=output,
                robots=3,
                spacing=2.5,
                assets=frozen.ASSETS,
                seconds=170,
                instruction=reference["instruction"],
                instruction_json=output / "mission/instruction.json",
                dispatch_substrate="alg2",
            )
        )
    finally:
        frozen.compose_jobs_scene = compose
    if report["source_sha256"] != reference["source_sha256"]:
        raise ValueError("Controller sources differ from qualified v5")
    proof = {
        "protocol_sha256": digest(protocol_path),
        "environment_harness_sha256": digest(__file__),
        "frozen_source_zip_sha256": protocol["frozen_source_zip_sha256"],
        "case": case,
        "controller_source_hashes_equal_reference": True,
        "model_mission_reused_not_fresh_inference": True,
        "scope": "Only initial physical XML changes; no environment parameters passed into controller or sensors",
    }
    (output / "environment-proof.json").write_text(json.dumps(proof, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        args.output.write_text(
            json.dumps(preflight(json.loads(args.protocol.read_text())), indent=2)
            + "\n"
        )
    else:
        run(args.protocol, args.case, args.output)


if __name__ == "__main__":
    main()
