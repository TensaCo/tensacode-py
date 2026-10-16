"""Versioned v6 development wrapper; frozen v5 core plus explicit extensions."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    spec = importlib.util.spec_from_file_location(
        "frozen_environment", protocol["environment_harness"]
    )
    environment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(environment)
    for path, digest in protocol["extension_sources"].items():
        if environment.digest(path) != digest:
            raise ValueError("Frozen v6 extension changed")
    if (
        environment.digest(protocol["environment_harness"])
        != protocol["environment_harness_sha256"]
    ):
        raise ValueError("Frozen environment helper changed")
    frozen, reference = environment.load_frozen(protocol)
    package = importlib.import_module("warehouse")
    package.__path__.append(str(Path(__file__).parent))
    controller = importlib.import_module(
        "warehouse.physics_jobs_clearance"
    ).ClearanceJobController
    frozen.RepeatedJobController = controller
    case = next(c for c in protocol["cases"] if c["id"] == args.case)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    mission = Path(protocol["reference_run"]) / "mission"
    for name, digest in protocol["mission_sha256"].items():
        if environment.digest(mission / name) != digest:
            raise ValueError("Certified mission changed")
    shutil.copytree(mission, output / "mission", dirs_exist_ok=True)
    shutil.copyfile(args.protocol, output / "clearance-protocol.json")
    for path in protocol["extension_sources"]:
        shutil.copyfile(path, output / Path(path).name)
    shutil.copyfile(protocol["environment_harness"], output / "environment-harness.py")
    compose = frozen.compose_jobs_scene

    def scene(*a, **kw):
        xml, local, offsets = compose(*a, **kw)
        return environment.perturb_both_jobs(xml, case, 3), local, offsets

    frozen.compose_jobs_scene = scene
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
    proof = {
        "schema": "tensorcode.warehouse.reset-clearance.v6",
        "protocol_sha256": environment.digest(args.protocol),
        "case": case,
        "base_controller_sources_equal_v5": report["source_sha256"]
        == reference["source_sha256"],
        "extension_sources": protocol["extension_sources"],
        "scope": "New v6 sampled reset-clearance controller explicitly replaces v5 back-out gate; no scene truth is supplied to the controller.",
        "mission_reused_not_fresh_inference": True,
    }
    (output / "clearance-proof.json").write_text(json.dumps(proof, indent=2) + "\n")


if __name__ == "__main__":
    main()
