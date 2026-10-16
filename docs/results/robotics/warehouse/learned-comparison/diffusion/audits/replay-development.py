import argparse
import gzip
import json
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Replay frozen diffusion checkpoints on local recorded sensor inputs"
)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--diagnostics", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--case", action="append")
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve()))
from pathlib import Path

import numpy as np

from examples.robotics.warehouse.diffusion_policy.model import PolicyRuntime
from examples.robotics.warehouse.learned_policy.data import sha256
from examples.robotics.warehouse.validate_v4 import validate

root = args.diagnostics
results = []
for case in sorted(root.glob("diffusion-*")):
    if not case.is_dir() or not (case / "probe.json").exists():
        continue
    if args.case and case.name not in args.case:
        continue
    started = time.monotonic()
    scores = validate(case)
    records = json.loads((case / "records.json").read_text())
    probe = json.loads((case / "probe.json").read_text())
    policy = case / "learned-policy"
    assert {
        name: sha256(policy / name) for name in ("weights.pt", "policy.json")
    } == probe["learned_policy_sha256"]
    rows = []
    for robot in ("r0", "r1"):
        episode = json.loads(
            gzip.decompress((case / robot / "bc-episode.json.gz").read_bytes())
        )
        runtime = PolicyRuntime(policy)
        full = True
        selected = episode["steps"] if full else episode["steps"][:256]
        maximum = 0.0
        active = 0
        closed = 0
        contact = 0.0
        for index, step in enumerate(selected):
            action, raw = runtime.step(step["observation"])
            record = records[index]["robots"][robot]
            maximum = max(
                maximum,
                float(
                    np.max(
                        np.abs(
                            np.asarray(raw)
                            - record["controller_state"]["raw_prediction"]
                        )
                    )
                ),
            )
            active += int(step["action"]["arm_active"])
            closed += int(step["action"]["grip"] >= 0.014)
            contact = max(
                contact, max(step["observation"]["sensor_state"]["pad_force_n"])
            )
        rows.append(
            {
                "robot": robot,
                "replayed_steps": len(selected),
                "recorded_steps": len(episode["steps"]),
                "full_trace_replay": full,
                "max_replayed_raw_prediction_error": maximum,
                "replayed_active_steps": active,
                "replayed_grip_closed_steps": closed,
                "maximum_replayed_pad_force_n": contact,
                "strict_metrics": scores["robots"][robot],
            }
        )
    result = {
        "case": case.name,
        "checkpoint_hashes_verified": True,
        "weights_sha256": sha256(policy / "weights.pt"),
        "robots": rows,
        "audit_wall_seconds": time.monotonic() - started,
        "final_data_opened": False,
        "tuned": False,
    }
    results.append(result)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {
                "case": case.name,
                "seconds": result["audit_wall_seconds"],
                "robots": [
                    {
                        "robot": r["robot"],
                        "strict_success": r["strict_metrics"][
                            "strict_current_settle_success"
                        ],
                        "max_raw_error": r["max_replayed_raw_prediction_error"],
                        "replayed_steps": r["replayed_steps"],
                    }
                    for r in rows
                ],
            }
        ),
        flush=True,
    )
