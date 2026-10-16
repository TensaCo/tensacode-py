"""Verify packaged hashes and optionally run portable CPU checkpoint inference."""

import argparse
import hashlib
import importlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--variant", default="original-primary")
parser.add_argument("--inference", action="store_true")
args = parser.parse_args()
expected = json.loads((root / "bundle-sha256.json").read_text())
for name, digest in expected.items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != digest:
        raise ValueError(f"Artifact hash mismatch: {name}")
result = {"files_verified": len(expected), "inference_run": False}
if args.inference:
    models = {
        m["id"]: m
        for m in json.loads((root / "model-index.json").read_text())["models"]
    }
    directory = root / models[args.variant]["directory"]
    with tempfile.TemporaryDirectory(
        prefix="tensorcode-diffusion-inference-"
    ) as temporary:
        source = Path(temporary) / "warehouse"
        source.mkdir()
        with zipfile.ZipFile(directory / "source.zip") as archive:
            for name in archive.namelist():
                (source / name).resolve().relative_to(source.resolve())
            archive.extractall(source)
        sys.path.insert(0, temporary)
        Runtime = importlib.import_module(
            "warehouse.diffusion_policy.model"
        ).PolicyRuntime
        runtime = Runtime(directory)
        observation = json.loads((root / "inference-observation.json").read_text())
        first = runtime.step(observation)
        runtime.reset()
        assert first == runtime.step(observation), (
            "Reset must reproduce the local seeded result"
        )
        for _ in range(7):
            runtime.step(observation)
        result.update(
            inference_run=True, variant=args.variant, cpu_steps=9, reset_reproduced=True
        )
print(json.dumps(result, indent=2))
