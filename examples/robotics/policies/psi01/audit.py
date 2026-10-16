"""Validate imported Psi-0.1 evidence and regenerate its small dashboard manifest.

No model loading, new evaluation, or network access. Original rollout rows and
videos remain immutable; this checks their recorded identities and pairing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def audit(bundle: Path) -> dict:
    provenance = json.loads((bundle / "provenance.json").read_text())
    protocol_hash = hashlib.sha256(
        json.dumps(provenance["protocol"], sort_keys=True).encode()
    ).hexdigest()[:16]
    assert protocol_hash == provenance["protocol_hash"]
    conditions, lookup = [], {}
    for arm in ("psi01", "bc"):
        for level in (0, 1):
            filename = f"{'bc_' if arm == 'bc' else ''}L{level}_step10000.jsonl"
            rows = [
                json.loads(line)
                for line in (bundle / filename).read_text().splitlines()
                if line
            ]
            assert len(rows) == 20
            assert {(r["config"], r["repeat"]) for r in rows} == {
                (c, r) for c in range(10) for r in range(2)
            }
            for row in rows:
                assert (
                    row["arm"] == arm
                    and row["level"] == level
                    and row["env_level"] == level
                )
                assert (
                    row["task"] == provenance["task"]
                    and row["step"] == provenance["step"]
                )
                assert (
                    row["seed"] == provenance["train_seed"]
                    and row["protocol_hash"] == protocol_hash
                )
                assert (
                    row["ckpt_sha256_16"] == provenance["checkpoint_sha256"][arm][:16]
                )
                assert row["render"] == provenance["protocol"][
                    "render"
                ] and not row.get("smoke")
                assert type(row["success"]) is bool
                lookup[arm, level, row["config"], row["repeat"]] = row
            conditions.append(
                {
                    "id": f"{arm}-L{level}",
                    "arm": arm,
                    "level": level,
                    "successes": sum(r["success"] for r in rows),
                    "attempted": len(rows),
                    "rows": filename,
                }
            )
    videos = []
    pattern = re.compile(
        r"L([01])_G1WholebodyTabletopGraspMP-v0_c(\d+)r([01])_(success|FAIL)\.mp4"
    )
    for path in sorted((bundle / "video").glob("*.mp4")):
        match = pattern.fullmatch(path.name)
        assert match, path
        level, config, repeat = map(int, match.groups()[:3])
        row = lookup["psi01", level, config, repeat]
        assert row["success"] == (match[4] == "success")
        videos.append(
            {
                "id": f"psi01-L{level}-c{config}-r{repeat}",
                "path": str(path.relative_to(bundle)),
                "level": level,
                "config": config,
                "repeat": repeat,
                "success": row["success"],
                "outcome": row["outcome"],
                "control_steps": row["steps"],
                "source": row["source"],
                "rows": f"L{level}_step10000.jsonl",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    assert any(v["success"] for v in videos) and any(not v["success"] for v in videos)
    return {
        "schema": "tensorcode.external-policy-results.v1",
        "policy": "Psi-0.1",
        "qualification": provenance["qualification"],
        "conditions": conditions,
        "videos": videos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path(__file__).resolve().parents[4]
        / "docs/results/robotics/baselines/psi01",
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = audit(args.bundle)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.write:
        (args.bundle / "results.json").write_text(rendered)
        hashes = {
            str(p.relative_to(args.bundle)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.bundle.rglob("*"))
            if p.is_file() and p.name != "hashes.json"
        }
        (args.bundle / "hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    else:
        assert (args.bundle / "results.json").read_text() == rendered
        for name, expected in json.loads(
            (args.bundle / "hashes.json").read_text()
        ).items():
            assert (
                hashlib.sha256((args.bundle / name).read_bytes()).hexdigest()
                == expected
            ), name
    print(
        json.dumps(
            {
                "conditions": result["conditions"],
                "verified_videos": len(result["videos"]),
            }
        )
    )


if __name__ == "__main__":
    main()
