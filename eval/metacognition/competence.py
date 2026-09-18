"""Does knowing my own competence beat looking at the input?

The question this settles. ``docs/revival/13`` measured two input-based abstention signals
and both failed: a BM25 threshold on the question (AUC 0.508 on SQuAD) and a logistic
capability router trained on item features (AUC 0.548 on SQuAD, 0.457 — below chance — on
HotpotQA). Both asked "will I get *this item* right?".

A competence prior asks a different and much cheaper question: "how do I do at *this kind*
of thing?" It needs no per-item signal at all. It can only pay where two conditions hold:

1. accuracy actually varies across kinds (measured here first, as ``spread``), and
2. the kind is knowable before attempting (it is: the dataset, and the answer type read off
   the question by ``eval/open_domain/rules.answer_type``).

Prediction registered before running (docs/revival/23):
  (a) pooled across the four benchmarks, competence gating beats the input-based BM25 gate
      by >= 5 points of selective accuracy at equal coverage;
  (b) within SQuAD 2.0 alone it gains < 2 points, because there the kinds are all alike.
If (a) fails, our own history carries no competence signal either. If (b) *succeeds*, my
reading of why the earlier routers failed is wrong.

Provenance: environment and labels are public datasets (SQuAD 2.0, HotpotQA distractor,
GSM8K, ARC-Easy); the grader is the datasets' own labels via ``eval/open_domain/score``;
the arm under test is the repo's no-model rule tier. Competence is fitted on one half of
the items and every number reported is from the other half.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tensorcode as tc  # noqa: E402
from eval.open_domain import rules as R  # noqa: E402
from eval.open_domain.data import LOADERS  # noqa: E402
from eval.open_domain.score import grade, wilson  # noqa: E402
from tensorcode.metacognition import SelfModel  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "eval" / "results" / "metacognition_competence.json"
EXTRACTIVE = ("squad2", "hotpot")


def kinds_of(benchmark: str, question: str) -> list[str]:
    """Specific to general: dataset+answer-type, dataset, family. Knowable before attempting."""
    family = "extractive_qa" if benchmark in EXTRACTIVE else ("arithmetic" if benchmark == "gsm8k" else "multiple_choice")
    return [f"{benchmark}/{R.answer_type(question)}", benchmark, family]


def attempt(benchmark: str, item, *, min_score: float) -> dict:
    """Run the rule tier on one item, with abstention disabled so every item has an outcome.

    Abstention is what we are trying to *decide*, so the arm must answer everything and let
    each gate abstain for itself; otherwise the arm's own threshold is baked into the data.
    """
    t0 = time.perf_counter()
    if benchmark in EXTRACTIVE:
        ans = R.answer_extractive(item, min_score=0.0)
    elif benchmark == "gsm8k":
        ans = R.answer_arithmetic(item)
    else:
        ans = R.answer_multiple_choice(item, guess_by_overlap=True)
    text = ans.text if isinstance(ans.text, str) else ""
    g = grade(benchmark, item, text if text else tc.Unknown("abstain"))
    return {"id": item.id, "benchmark": benchmark, "question": item.question, "kinds": kinds_of(benchmark, item.question),
            "pred": text, "correct": bool(g["correct"]), "bm25": float(ans.score), "seconds": round(time.perf_counter() - t0, 4)}


def selective(rows: list[dict], answer: list[bool]) -> dict:
    """Accuracy over what was attempted, and over everything (an abstention scores 0)."""
    n = len(rows)
    taken = [r for r, a in zip(rows, answer) if a]
    correct = sum(r["correct"] for r in taken)
    return {"coverage": round(len(taken) / n, 4) if n else 0.0,
            "selective_accuracy": round(correct / len(taken), 4) if taken else None,
            "overall_correct": round(correct / n, 4) if n else 0.0,
            "attempted": len(taken), "n": n}


def at_coverage(rows: list[dict], strength: list[float], target: int) -> dict:
    """Answer the ``target`` highest-strength items; ties broken by index for determinism."""
    order = sorted(range(len(rows)), key=lambda i: (-strength[i], i))
    keep = set(order[:target])
    return selective(rows, [i in keep for i in range(len(rows))])


def run(benchmarks: list[str], n: int, seed: int) -> dict:
    fitted, held = [], []
    for b in benchmarks:
        items = LOADERS[b](n, seed)
        rows = [attempt(b, it, min_score=0.0) for it in items]
        rng = random.Random(seed)
        rng.shuffle(rows)
        half = len(rows) // 2
        fitted += rows[:half]
        held += rows[half:]

    model = SelfModel(name="open-domain/rule-tier")
    for r in fitted:
        model.record(r["kinds"], r["correct"])

    report: dict = {
        "note": __doc__.strip().splitlines()[0],
        "provenance": {"environment_author": "public datasets", "grader": "dataset labels via eval/open_domain/score",
                       "held_out": "competence fitted on one random half, all numbers from the other half",
                       "arm": "the repo's no-model rule tier, abstention disabled so every item has an outcome"},
        "fitted_on": len(fitted), "held_out_on": len(held),
        "competence_table": [c.describe() for c in model.table()],
        "spread": {}, "gates": {}, "per_benchmark": {},
    }

    # (1) would a prior even help? Answer this before building any gate.
    report["spread"]["across_benchmarks"] = round(model.spread(benchmarks), 4)
    for b in benchmarks:
        fine = sorted({k for r in fitted if r["benchmark"] == b for k in r["kinds"] if k.startswith(f"{b}/")})
        report["spread"][f"within_{b}"] = round(model.spread(fine), 4)
        report["per_benchmark"][b] = {"kinds": fine}

    def strengths(rows: list[dict]) -> tuple[list[float], list[float], list[float]]:
        comp, bm25, oracle = [], [], []
        for r in rows:
            got = model.competence(r["kinds"])
            comp.append(0.0 if isinstance(got, tc.Unknown) else got.lower)
            bm25.append(r["bm25"])
            oracle.append(1.0 if r["correct"] else 0.0)
        return comp, bm25, oracle

    comp, bm25, oracle = strengths(held)
    # compare the three gates at the same number of attempted items, sweeping coverage
    report["gates"]["pooled"] = {}
    for frac in (0.25, 0.5, 0.75):
        target = int(len(held) * frac)
        report["gates"]["pooled"][f"coverage_{frac}"] = {
            "competence_prior": at_coverage(held, comp, target),
            "bm25_input_signal": at_coverage(held, bm25, target),
            "oracle": at_coverage(held, oracle, target),
            "answer_everything": selective(held, [True] * len(held)),
        }
    report["gates"]["pooled"]["auc"] = {"competence_prior": auc(comp, [r["correct"] for r in held]),
                                       "bm25_input_signal": auc(bm25, [r["correct"] for r in held])}

    # (2) the falsification case: within one benchmark, where kinds are alike
    for b in benchmarks:
        sub = [r for r in held if r["benchmark"] == b]
        if len(sub) < 20:
            continue
        c, s, o = strengths(sub)
        target = len(sub) // 2
        report["gates"][b] = {
            "competence_prior": at_coverage(sub, c, target),
            "bm25_input_signal": at_coverage(sub, s, target),
            "oracle": at_coverage(sub, o, target),
            "answer_everything": selective(sub, [True] * len(sub)),
            "auc": {"competence_prior": auc(c, [r["correct"] for r in sub]),
                    "bm25_input_signal": auc(s, [r["correct"] for r in sub])},
        }

    pooled = report["gates"]["pooled"]["coverage_0.5"]
    gain = (pooled["competence_prior"]["selective_accuracy"] or 0) - (pooled["bm25_input_signal"]["selective_accuracy"] or 0)
    squad = report["gates"].get("squad2", {})
    squad_gain = ((squad.get("competence_prior", {}).get("selective_accuracy") or 0)
                  - (squad.get("bm25_input_signal", {}).get("selective_accuracy") or 0))
    report["prediction"] = {
        "a_pooled_gain_at_least_0.05": round(gain, 4),
        "a_holds": gain >= 0.05,
        "b_squad_gain_below_0.02": round(squad_gain, 4),
        "b_holds": abs(squad_gain) < 0.02,
        "reading": ("a competence prior pays where competence varies by kind and the kind is knowable in advance; "
                    "it buys task selection, not item selection"),
    }
    return report


def auc(strength: list[float], correct: list[bool]) -> float | None:
    """Probability that a correct item outranks an incorrect one (ties count a half)."""
    pos = [s for s, c in zip(strength, correct) if c]
    neg = [s for s, c in zip(strength, correct) if not c]
    if not pos or not neg:
        return None
    wins = sum((1.0 if p > q else 0.5 if p == q else 0.0) for p in pos for q in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks", default="squad2,hotpot,gsm8k,arc_easy")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    report = run(args.benchmarks.split(","), args.n, args.seed)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(f"wrote {OUT}")
    print("spread:", json.dumps(report["spread"]))
    for name, block in report["gates"].items():
        if name == "pooled":
            half = block["coverage_0.5"]
            print(f"  pooled @50% coverage: competence {half['competence_prior']['selective_accuracy']} "
                  f"vs bm25 {half['bm25_input_signal']['selective_accuracy']} vs oracle {half['oracle']['selective_accuracy']}")
            print("  auc:", json.dumps(block["auc"]))
        else:
            print(f"  {name} @50%: competence {block['competence_prior']['selective_accuracy']} "
                  f"vs bm25 {block['bm25_input_signal']['selective_accuracy']} (auc {json.dumps(block['auc'])})")
    print("prediction:", json.dumps(report["prediction"]))


if __name__ == "__main__":
    main()
