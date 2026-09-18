"""Fit and measure the parser's abstention: a probability that its own reading is right.

    python eval/training/calibrate_parser.py

The held-out paraphrase set is split in two: one half fits the temperature and picks a
threshold, the other half reports what that threshold actually buys. Nothing is fitted on
the half it is measured on. Writes the threshold into the artifact so the runtime uses it,
and records a reliability table, the expected calibration error, and a risk-coverage curve.
"""

from __future__ import annotations

import os

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

SP = Path(os.environ.get("TENSACODE_SCRATCH", os.path.expanduser("~/.cache/tensacode")))


def reliability(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> tuple[list[dict], float]:
    """Bin by confidence; the gap between confidence and accuracy in each bin is the calibration error."""
    table, ece = [], 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        keep = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= 1.0)
        if not keep.any():
            continue
        acc, mean_conf, share = float(correct[keep].mean()), float(conf[keep].mean()), float(keep.mean())
        table.append({"bin": [round(lo, 2), round(hi, 2)], "n": int(keep.sum()),
                      "mean_confidence": round(mean_conf, 4), "accuracy": round(acc, 4)})
        ece += share * abs(acc - mean_conf)
    return table, ece


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, default=SP / "artifacts" / "request-parser")
    ap.add_argument("--eval", type=Path, default=SP / "training" / "paraphrase_eval.jsonl")
    ap.add_argument("--target", type=float, default=0.95, help="selective accuracy to aim for on the fitting half")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "learned_parser_calibration.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from eval.training.calibration import fit_threshold  # noqa: PLC0415
    from eval.training.eval_parser import score  # noqa: PLC0415
    from eval.training.parser_data import read  # noqa: PLC0415
    from tensacode.backends.neural import NeuralRequestParser  # noqa: PLC0415

    rows = read(args.eval)
    random.Random(args.seed).shuffle(rows)
    half = len(rows) // 2
    fit_rows, test_rows = rows[:half], rows[half:]
    parser = NeuralRequestParser(args.artifact, threshold=0.0)

    def measure(batch):
        texts = [r.text for r in batch]
        parsed = parser.parse(texts)
        want = [(r.act, {k: r.value(k) for k in r.spans} | {k: v for k, v in r.flags.items()}
                 | ({"aspect": r.closed["aspect"]} if "aspect" in r.closed else {})) for r in batch]
        got = [(p.act, dict(p.slots)) for p in parsed]
        correct = np.array([score(g, a, s)["slots"] for g, (a, s) in zip(got, want)], dtype=float)
        conf = np.array([p.confidence or 0.0 for p in parsed])
        return conf, correct

    fit_conf, fit_correct = measure(fit_rows)
    test_conf, test_correct = measure(test_rows)

    # temperature on the act logit margin is already a softmax probability; fit a scalar power
    # (monotone, so it cannot change the ranking) to bring mean confidence onto accuracy
    grid = np.exp(np.linspace(math.log(0.25), math.log(6.0), 120))
    losses = [abs(float((fit_conf**t).mean()) - float(fit_correct.mean())) for t in grid]
    power = float(grid[int(np.argmin(losses))])
    fit_scaled, test_scaled = fit_conf**power, test_conf**power

    fit = fit_threshold(fit_scaled, fit_correct, target=args.target)
    threshold = fit.threshold

    attempted = test_scaled >= threshold
    curve = []
    for thr in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        keep = test_scaled >= thr
        curve.append({"threshold": thr, "coverage": round(float(keep.mean()), 4),
                      "accuracy_over_attempted": round(float(test_correct[keep].mean()) if keep.any() else 0.0, 4),
                      "correct_overall": round(float((test_correct * keep).mean()), 4)})
    table, ece = reliability(test_scaled, test_correct)

    report = {
        "fitted_on": {"n": len(fit_rows), "set": str(args.eval) + " (half, disjoint from the reported half)"},
        "measured_on": {"n": len(test_rows)},
        "power": round(power, 4),
        "target_selective_accuracy": args.target,
        "target_reachable": fit.reachable,
        "best_selective_accuracy_on_fitting_half": round(fit.best_selective_accuracy, 4),
        "note": fit.note,
        "threshold": round(threshold, 4),
        "unthresholded_accuracy": round(float(test_correct.mean()), 4),
        "at_threshold": {"coverage": round(float(attempted.mean()), 4),
                         "accuracy_over_attempted": round(float(test_correct[attempted].mean()) if attempted.any() else 0.0, 4),
                         "correct_overall": round(float((test_correct * attempted).mean()), 4)},
        "expected_calibration_error": round(ece, 4),
        "reliability": table,
        "risk_coverage": curve,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))

    # Reaching the target is not the same as being worth shipping. Refusing costs one right
    # answer for every wrong one it avoids, so a threshold that lowers the share of inputs
    # answered correctly is a loss however good its selective accuracy looks. Ship 0.0 and
    # keep the measurement, rather than quietly making the runtime refuse a fifth of its work.
    shipped, refused_reason = threshold, None
    if report["at_threshold"]["correct_overall"] < report["unthresholded_accuracy"]:
        shipped, refused_reason = 0.0, (
            f"fitted threshold {threshold:.4f} reaches {report['at_threshold']['accuracy_over_attempted']:.4f} "
            f"selective accuracy at {report['at_threshold']['coverage']:.4f} coverage, but that answers "
            f"{report['at_threshold']['correct_overall']:.4f} of inputs correctly against "
            f"{report['unthresholded_accuracy']:.4f} for answering everything: abstention loses "
            f"{report['unthresholded_accuracy'] - report['at_threshold']['correct_overall']:.4f}. "
            f"Shipping 0.0.")
        print(f"!! {refused_reason}")
    report["shipped_threshold"] = shipped
    report["not_shipped_because"] = refused_reason
    args.out.write_text(json.dumps(report, indent=1))

    config_path = args.artifact / "config.json"
    config = json.loads(config_path.read_text())
    config["threshold"], config["confidence_power"] = round(shipped, 4), round(power, 4)
    config["threshold_fitted"] = round(threshold, 4)
    config["threshold_not_shipped_because"] = refused_reason
    config["quality"] = dict(config.get("quality", {})) | {
        "paraphrase_exact_accuracy": report["unthresholded_accuracy"],
        "expected_calibration_error": report["expected_calibration_error"],
    }
    config_path.write_text(json.dumps(config, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "reliability"}, indent=1))


if __name__ == "__main__":
    main()
