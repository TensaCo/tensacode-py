"""Cycles still of one actual shared-scene state, using the shared mesh exporter.

Pass --run, --output, --time, --camera-location and --lookat. This creates a
path-traced presentation still; it neither steps physics nor changes robot poses.
"""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    renderer = Path(__file__).resolve().parents[1] / "transfer/warehouse_photoreal.py"
    sys.argv.append("--shared-scene")
    runpy.run_path(str(renderer), run_name="__main__")
