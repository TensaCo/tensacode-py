"""Is comparison-in-prose the same faculty as arithmetic-in-word-problems? The transfer test.

    PYTHONPATH=src:. venv-eval/bin/python eval/relations/eval_gsm8k_transfer.py

Doc 21 concluded that the structures which help on HotpotQA and the structures which help on
GSM8K are "two coincidences, not one". Doc 27 built a relation operation that moved HotpotQA
comparison questions from 0.1935 to 0.4677 EM. If that operation is a general faculty, it should
move GSM8K, where the current solver scores 0.0 accuracy at 0.0267 coverage. If it moves only the
handful of GSM8K problems that are literally comparisons — "how many more X than Y" — then the
shared part is a step, not a faculty, and the two-faculties conclusion is strengthened.

The operation is applied exactly as it is: no GSM8K-specific solver, no chaining, no unstated
constants. ``relation.difference_in`` reads two quantities of one kind out of the problem text and
returns the gap. That is the whole of the transfer.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

OUT = ROOT / "eval" / "results" / "relation_gsm8k.json"
#: eval/results/structures_gsm8k.json, the same 300 items
BASELINE = {"coverage": 0.0267, "accuracy": 0.0, "wrong_when_answered": 8,
            "stages": {"compose": 217, "question": 70, "computed": 8, "read": 5}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    from eval.open_domain.data import gsm8k
    from eval.open_domain.score import numeric_match, wilson
    from tensacode.outcomes import Unknown
    from tensacode.relation import Resolved, difference_in, read

    items = gsm8k(args.n, seed=0)
    answered, correct, reasons = [], [], Counter()
    comparisons_read = 0
    examples: list[dict] = []
    for it in items:
        # the question-side reader first: does GSM8K even contain two-candidate comparisons?
        if not isinstance(read(it.question), Unknown):
            comparisons_read += 1
        out = difference_in(it.question, it.question)
        if isinstance(out, Unknown):
            reasons[out.reason] += 1
            continue
        good = numeric_match(out.text, it.gold)
        answered.append(it.id)
        correct.append(good)
        if len(examples) < 20:
            examples.append({"question": it.question, "gold": it.gold[0], "answer": out.text,
                             "steps": list(out.steps), "correct": good})

    n = len(items)
    lo, hi = wilson(sum(correct), n)
    body = {
        "measured_on": f"GSM8K test, gsm8k({args.n}, seed=0) — the same items as eval/results/structures_gsm8k.json",
        "operation": "tensacode.relation.difference_in, unchanged from the HotpotQA faculty",
        "coverage": round(len(answered) / n, 4),
        "accuracy_overall": round(sum(correct) / n, 4),
        "accuracy_overall_ci": [round(lo, 4), round(hi, 4)],
        "accuracy_when_answered": round(sum(correct) / max(1, len(answered)), 4),
        "answered": len(answered), "correct": sum(correct),
        "comparison_questions_read_by_the_question_side": comparisons_read,
        "refusal_reasons": dict(reasons),
        "baseline_structures_gsm8k": BASELINE,
        "examples": examples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=1))
    print(f"coverage {body['coverage']:.4f} (baseline {BASELINE['coverage']})")
    print(f"accuracy overall {body['accuracy_overall']:.4f} (baseline {BASELINE['accuracy']}) "
          f"CI {body['accuracy_overall_ci']}")
    print(f"accuracy when answered {body['accuracy_when_answered']:.4f} on {len(answered)} items")
    print(f"two-candidate comparisons read in GSM8K questions: {comparisons_read}/{n}")
    print(f"refusals {dict(reasons)}\nwrote {args.out}")


if __name__ == "__main__":
    main()
