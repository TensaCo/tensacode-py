"""Build the offline humanoid report from measured artifacts and rendered media."""

from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/results/robotics/humanoid"


def main():
    h = OUT
    report = json.loads((h / "report.json").read_text())
    resources = json.loads((h / "evaluation-resources.json").read_text())
    r = report["rollouts"]["alg2"]
    template = Path(__file__).with_name("humanoid_report.html").read_text()
    style = "<style>" + Path(__file__).parents[1].joinpath("report.css").read_text() + "</style>"

    def embedded(name, mime):
        return (
            "data:"
            + mime
            + ";base64,"
            + base64.b64encode((h / name).read_bytes()).decode()
        )

    labels = {
        "native": "Native ONNX + authored controller",
        "alg0": "TensorCode ALG.0",
        "alg1": "TensorCode ALG.1",
        "alg2": "TensorCode ALG.2",
        "open-gripper": "Open-gripper control (native)",
        "shifted-light": "Shifted + 60 g (ALG.0)",
        "shifted-heavy": "Shifted + 100 g (ALG.0)",
    }
    rows = "".join(
        f"<tr><th>{labels[name]}</th><td>{x['walking_distance_m']:.3f} m</td><td>{x['max_lift'] * 100:.2f} cm</td><td>{x['held_above_10cm_seconds']:.2f} s</td><td>{x['min_height']:.3f} m</td><td>{'Passed' if x['success'] else 'No lift (expected)'}</td></tr>"
        for name, x in report["rollouts"].items()
    )
    values = {
        "STYLE": style,
        "DISTANCE": f"{r['walking_distance_m']:.2f}",
        "LIFT": f"{r['max_lift'] * 100:.1f}",
        "HELD": f"{r['held_above_10cm_seconds']:.2f}",
        "ROWS": rows,
        "REPORT": json.dumps(report, indent=2),
        "CPU_MAX": f"{max(x['total_cpu_fraction'] for x in resources['samples']) * 100:.1f}",
        "MEM_FREE": f"{resources['minimum_memory_available_fraction'] * 100:.1f}",
        "MAIN_VIDEO": embedded("humanoid-walk-grasp.mp4", "video/mp4"),
        "MAIN_POSTER": embedded("humanoid-poster.png", "image/png"),
        "CONTROL_VIDEO": embedded("grip-control.mp4", "video/mp4"),
        "CONTROL_POSTER": embedded("control-poster.png", "image/png"),
    }
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    if "{{" in template:
        raise RuntimeError("Unresolved template placeholder")
    (OUT / "index.html").write_text(template)
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
