"""Measure the trained answerer against the rule tier and the local model, on the audit's splits.

    python eval/training/eval_span.py --benchmarks squad2,hotpot

Same loaders, same seed, same graders as ``eval/open_domain`` — so these rows sit beside
the ones in docs/revival/12-open-domain.md rather than replacing them. Calibration is
fitted on the audit's disjoint calibration slice and reported on the test slice, and the
honest bar is stated in the report: does any operating point beat "always ask the local
model" at equal accuracy while saving calls?
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

SP = Path("/tmp/claude-1000/-home-brandonin-Documents-tensacode-tensacode-python/c572c14b-5662-4c07-8a7a-1ba7821d2bfa/scratchpad")

#: the numbers this is being compared against, from eval/results/open_domain.json
AUDIT = {
    "squad2": {"rules": {"coverage": 0.183, "accuracy_over_attempted": 0.091, "correct_overall": 0.420},
               "model": {"coverage": 0.713, "accuracy_over_attempted": 0.551, "correct_overall": 0.670}},
    "hotpot": {"rules": {"coverage": 0.913, "accuracy_over_attempted": 0.058, "correct_overall": 0.053},
               "model": {"coverage": 1.0, "accuracy_over_attempted": 0.513, "correct_overall": 0.513}},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, default=SP / "artifacts" / "span-answerer")
    ap.add_argument("--benchmarks", default="squad2,hotpot")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n-cal", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", type=float, default=0.8, help="selective accuracy to aim for on the calibration slice")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "learned_answerer.json")
    args = ap.parse_args()

    from eval.open_domain.data import LOADERS  # noqa: PLC0415
    from eval.training.calibration import fit_threshold  # noqa: PLC0415
    from eval.open_domain.score import grade  # noqa: PLC0415
    from eval.selection.selector import Truncation, truncation_of  # noqa: PLC0415
    from tensacode.backends.neural import NeuralAnswerer, QuestionOverPassages  # noqa: PLC0415

    answerer = NeuralAnswerer(args.artifact, threshold=0.0)
    report: dict = {
        "artifact": str(args.artifact), "parameters": answerer.config["parameters"],
        "trained_on": f"SQuAD 2.0 train, {answerer.config['train_size']} questions, "
                      f"{answerer.config['epochs']} epoch(s), {answerer.config['train_seconds']}s",
        "compared_against": "eval/results/open_domain.json (same loaders, seed and graders)",
        "benchmarks": {},
    }

    for benchmark in args.benchmarks.split(","):
        window = Truncation()
        pool = LOADERS[benchmark](args.n + args.n_cal, args.seed)
        cal_items, items = pool[args.n :], pool[: args.n]

        def run(batch):
            questions = [QuestionOverPassages(it.question, tuple(it.passages)) for it in batch]
            # Per item, does what we just handed the model fit the window it reads through? This
            # evaluation once published 0.093 for HotpotQA as a capability result while discarding
            # about forty sentences per item unannounced, and nothing in the output said so.
            window.update(truncation_of(answerer.tokenizer,
                                        [(it.question, " ".join(s for _, s in it.passages)) for it in batch],
                                        answerer.max_length))
            t0 = time.perf_counter()
            answers = answerer.answer(questions)
            seconds = time.perf_counter() - t0
            return [{"item": it, "pred": got.text, "confidence": got.confidence or 0.0} for it, got in zip(batch, answers)], \
                   seconds / max(1, len(batch))

        def summarize(rows, thr):
            """Re-grade at this threshold with the audit's own grader: below it, the answer is Unknown."""
            import tensacode as tc  # noqa: PLC0415

            graded = [grade(benchmark, r["item"],
                            r["pred"] if (r["pred"] and r["confidence"] >= thr) else tc.Unknown("not_in_passage"))
                      for r in rows]
            attempted = [g for g in graded if g["attempted"]]
            return {
                "coverage": round(len(attempted) / len(graded), 4),
                "accuracy_over_attempted": round(sum(g["correct"] for g in attempted) / max(1, len(attempted)), 4),
                "correct_overall": round(sum(g["correct"] for g in graded) / len(graded), 4),
            }

        cal_rows, _ = run(cal_items)
        test_rows, seconds_per_item = run(items)

        # a threshold from the calibration slice only, never from the reported slice
        import tensacode as tc

        cal_graded = [grade(benchmark, r["item"], r["pred"] if r["pred"] else tc.Unknown("not_in_passage")) for r in cal_rows]
        fit = fit_threshold(np.array([r["confidence"] for r in cal_rows]),
                            np.array([float(g["correct"]) for g in cal_graded]), target=args.target)
        threshold = fit.threshold
        if not fit.reachable:
            print(f"   !! {benchmark}: {fit.note}", flush=True)

        curve = [{"threshold": round(t, 3), **summarize(test_rows, t)} for t in np.linspace(0.0, 0.95, 12)]
        block = {
            "n": len(items),
            "calibration_n": len(cal_items),
            "threshold_from_calibration": round(threshold, 4),
            "target_reachable": fit.reachable,
            "best_selective_accuracy_on_calibration": round(fit.best_selective_accuracy, 4),
            "calibration_note": fit.note,
            "unthresholded": summarize(test_rows, 0.0),
            "at_threshold": summarize(test_rows, threshold),
            "risk_coverage": curve,
            "seconds_per_item": round(seconds_per_item, 4),
            "model_calls": 0,
            "audit_rows": AUDIT.get(benchmark, {}),
        }
        best = max(curve, key=lambda r: r["correct_overall"])
        model_overall = AUDIT.get(benchmark, {}).get("model", {}).get("correct_overall")
        block["beats_always_asking_the_model"] = (
            None if model_overall is None else bool(best["correct_overall"] > model_overall)
        )
        block["best_operating_point"] = best
        block["truncation"] = window.report()
        report["benchmarks"][benchmark] = block
        print(f"\n== {benchmark} (n={len(items)})")
        print(f"   learned  {block['unthresholded']}  |  at threshold {block['at_threshold']}")
        print(f"   rules    {AUDIT.get(benchmark, {}).get('rules')}")
        print(f"   model    {AUDIT.get(benchmark, {}).get('model')}")
        print(f"   beats always-ask-the-model on overall correctness: {block['beats_always_asking_the_model']}")
        print(f"   window   {block['truncation']['verdict']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
