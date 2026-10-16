"""Package actual released-Ψ0 recordings; reject missing or mislabelled evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    episodes = []
    for directory in a.runs:
        rows = [
            json.loads(line)
            for line in (directory / "psi0_video_released.jsonl")
            .read_text()
            .splitlines()
        ]
        for row in rows:
            if not row["source"].startswith("upstream:psi0-released/"):
                raise ValueError("Evidence does not use released upstream weights")
            candidates = list(
                directory.glob(f"*_released_s{row['seed']}_{row['outcome']}.mp4")
            )
            if len(candidates) != 1:
                raise ValueError("Missing or ambiguous recording")
            video = candidates[0]
            diag = video.with_suffix(".diag.json")
            detail = json.loads(diag.read_text())
            assert detail["policy_source"] == row["source"]
            assert detail["outcome"] == row["outcome"] and detail["seed"] == row["seed"]
            for f in (video, diag):
                shutil.copy2(f, a.output / f.name)
            episodes.append(
                {
                    "id": f"released-psi0-c{row['seed']}",
                    "video": video.name,
                    "diagnostics": diag.name,
                    "seed": row["seed"],
                    "outcome": row["outcome"],
                    "success": bool(row["success_privileged"]),
                    "steps": row["steps"],
                    "source": row["source"],
                    "row": row,
                }
            )
    (a.output / "results.json").write_text(
        json.dumps(
            {
                "policy": "Released Ψ0",
                "native_task": "G1WholebodyTabletopGraspMP-v0",
                "fresh_recordings": True,
                "comparison": "Independent native-task reference; not a matched warehouse benchmark",
                "episodes": episodes,
                "attempted": len(episodes),
                "successes": sum(e["success"] for e in episodes),
            },
            indent=2,
        )
        + "\n"
    )
    (a.output / "sha256.json").write_text(
        json.dumps(
            {
                f.name: sha(f)
                for f in sorted(a.output.iterdir())
                if f.is_file() and f.name != "sha256.json"
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
