"""Versioned v6 development wrapper; frozen v5 core plus explicit extensions."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
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
    for module_name in ("physics_reset_clearance", "physics_jobs_clearance"):
        qualified = "warehouse." + module_name
        source_path = next(
            Path(path)
            for path in protocol["extension_sources"]
            if Path(path).name == module_name + ".py"
        )
        extension_spec = importlib.util.spec_from_file_location(qualified, source_path)
        extension = importlib.util.module_from_spec(extension_spec)
        sys.modules[qualified] = extension
        extension_spec.loader.exec_module(extension)
        if Path(extension.__file__).resolve() != source_path.resolve():
            raise ValueError("Unexpected extension import path")
    controller = sys.modules["warehouse.physics_jobs_clearance"].ClearanceJobController
    if args.preflight:
        result = environment.preflight(protocol)
        result["extension_imports_verified"] = list(protocol["extension_sources"])
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return
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
            seconds=3 if args.smoke else 170,
            instruction=reference["instruction"],
            instruction_json=output / "mission/instruction.json",
            dispatch_substrate="alg2",
        )
    )
    proof = {
        "schema": "tensorcode.warehouse.reset-clearance.v6",
        "smoke_only": args.smoke,
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
