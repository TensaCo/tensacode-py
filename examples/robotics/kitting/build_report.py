"""Assemble the offline report from committed measurements and rendered video."""

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/results/robotics/kitting"
NAMES = {
    "fixed_recipe": "Fixed recipe (authored ablation)",
    "no_retry": "Revised recipe, no retry",
    "native_feedback": "Native symbolic feedback",
    "alg0": "TensorCode ALG.0",
    "alg1": "TensorCode ALG.1",
    "alg2": "TensorCode ALG.2",
}


def main():
    report = json.loads((OUT / "report.json").read_text())
    resources = json.loads((OUT / "evaluation-resources.json").read_text())
    rows = "".join(
        f"<tr><th>{NAMES[c]}</th><td>{s['successes']}/{s['episodes']}</td><td>{s['false_completions']}</td><td>{s['retries']}</td><td>{s['mean_seconds']:.2f} s</td></tr>"
        for c, s in report["summary"].items()
    )
    scenario_rows = []
    for scenario in report["scenarios"]:
        cells = []
        for controller in ("fixed_recipe", "no_retry", "native_feedback", "alg2"):
            episodes = [
                r
                for r in report["episodes"]
                if r["scenario"] == scenario and r["controller"] == controller
            ]
            success = sum(r["success"] for r in episodes)
            cells.append(
                f'<td class="{"good" if success == len(episodes) else "bad"}">{success}/{len(episodes)}</td>'
            )
        scenario_rows.append(
            "<tr><th>" + scenario.replace("_", " ") + "</th>" + "".join(cells) + "</tr>"
        )

    def embed(name, mime):
        return (
            "data:"
            + mime
            + ";base64,"
            + base64.b64encode((OUT / name).read_bytes()).decode()
        )

    full = report["summary"]["alg2"]
    values = {
        "SUCCESS": f"{full['successes']}/{full['episodes']}",
        "RETRIES": str(full["retries"]),
        "ROWS": rows,
        "SCENARIOS": "".join(scenario_rows),
        "CASES": str(report["decision_cases_checked"]),
        "ERROR": str(report["decision_max_abs_error"]),
        "CPU": f"{max(s['total_cpu_fraction'] for s in resources['samples']) * 100:.1f}",
        "MEM": f"{resources['minimum_memory_available_fraction'] * 100:.1f}",
        "VIDEO": embed("comparison.mp4", "video/mp4"),
        "POSTER": embed("poster.png", "image/png"),
    }
    text = Path(__file__).with_name("report.html").read_text()
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", v)
    if "{{" in text:
        raise ValueError("Unresolved report field")
    (OUT / "index.html").write_text(text)


if __name__ == "__main__":
    main()
