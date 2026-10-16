"""Invoke pinned upstream JAX-to-PyTorch conversion without compiling kernels."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

from openpi.models.pi0_config import Pi0Config

REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.openpi), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Expected openpi {REVISION}, got {revision}")
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.openpi / "examples/convert_jax_model_to_pytorch.py"
    spec = importlib.util.spec_from_file_location("official_pi0_conversion", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.convert_pi0_checkpoint(
        str(args.checkpoint),
        "bfloat16",
        str(args.output),
        Pi0Config(pytorch_compile_mode=None),
    )
    # The pinned upstream converter looks in the checkpoint's parent for assets.
    # This checkpoint stores assets inside itself; preserve the exact norm stats.
    shutil.copytree(
        args.checkpoint / "assets", args.output / "assets", dirs_exist_ok=True
    )
    digest = hashlib.sha256()
    with (args.output / "model.safetensors").open("rb") as stream:
        while chunk := stream.read(4 * 1024**2):
            digest.update(chunk)
    (args.output / "conversion.json").write_text(
        json.dumps(
            {
                "openpi_revision": revision,
                "converter_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "precision": "bfloat16",
                "torch_compile": False,
                "model_sha256": digest.hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
