"""Measure the decision service on public data. Three arms, honest provenance labels.

    python -m eval.decisions.measure [--limit 3080] [--hotpot 300] [--out eval/results]

What is measured, and whose labels decide it:

    intent            Banking77 test split          PUBLIC LABEL, nothing of ours in the path
    department        Banking77 test + our mapping   public text + public label + OUR mapping
    urgency           Banking77 test + our mapping   same
    refund asked      Banking77 test + our mapping   same
    rerank            HotpotQA distractor            PUBLIC LABEL (gold supporting paragraphs)
    citation support  HotpotQA distractor            PUBLIC LABEL (gold vs distractor paragraph)

The derived rows are reported separately from the intent row for exactly this reason: a
mapping we wrote cannot be evidence about the framework, only about the mapping.

Arms: rules only, learned only, cascade (rules then learned). Our own audit found cascades
bought nothing on open-domain QA; bounded business classification is a different regime, and
this says whether it helps here.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import tensorcode as tc
from tensorcode.expectation import calibration
from tensorcode.outcomes import Unknown

from examples.decisions import tiers
from examples.decisions.decisions import classify_with_confidence, rerank, supports
from examples.decisions.domain import (
    DEPARTMENT_OF_INTENT,
    REFUND_INTENTS,
    Passage,
    urgency_of,
)
from examples.decisions.gating import Gate, Thresholds
from examples.support_router.domain import Intent

REPO = Path(__file__).resolve().parents[2]


@dataclass
class ArmResult:
    arm: str
    n: int
    # intent: public label only
    intent_attempted: int
    intent_correct: int
    intent_accuracy_over_attempted: float | None
    intent_coverage: float
    intent_accuracy_overall: float
    # derived (our mapping)
    department_correct: int
    department_accuracy_over_attempted: float | None
    urgency_correct: int
    urgency_accuracy_over_attempted: float | None
    refund_asked_correct: int
    refund_asked_accuracy_over_attempted: float | None
    # gating
    gate_counts: dict[str, int]
    auto_accuracy: float | None
    confirm_accuracy: float | None
    # calibration
    ece: float | None
    reliability: list[list[float]]
    # cost / speed
    ms_p50: float
    ms_p95: float
    usd_per_1000: float | None
    usd_basis: str
    model_calls: int
    answered_by: dict[str, int]
    # floors
    majority_baseline: float
    random_baseline: float


def load_banking77(path: Path, limit: int | None, *, seed: int = 0) -> list[tuple[str, Intent]]:
    """The test split. A ``limit`` SAMPLES at random rather than taking a prefix.

    The file is ordered by category, so a prefix is a biased slice — taking the first 400 rows
    gave a subset containing none of the intents the keyword tier covers, which made the rules
    arm look like it had zero coverage for a reason that was entirely my sampling.
    """
    import random

    with path.open(newline="") as f:
        rows = [(r["text"], Intent(r["category"])) for r in csv.DictReader(f)]
    if limit and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    return rows


def measure_arm(name: str, bindings: Sequence[object], rows: Sequence[tuple[str, Intent]], gate: Gate) -> ArmResult:
    runtime = tiers.runtime(bindings)
    attempted = correct = dept_ok = urg_ok = refund_ok = 0
    gate_counts: Counter[str] = Counter()
    gate_correct: Counter[str] = Counter()
    latencies: list[float] = []
    pairs: list[tuple[float, bool]] = []
    answered_by: Counter[str] = Counter()
    usd_known: list[float] = []
    unmetered = 0

    with tc.use(runtime):
        for text, gold in rows:
            t0 = time.perf_counter()
            answer, score, span = classify_with_confidence(text, Intent)
            latencies.append((time.perf_counter() - t0) * 1e3)
            decided = gate.decide(answer, score)
            gate_counts[decided.action] += 1
            if span is not None:
                answered_by[span.answered_by or "none"] += 1
                for a in span.attempts:
                    if a.outcome in ("answer",):
                        if a.usd is None:
                            unmetered += 1
                        else:
                            usd_known.append(a.usd)
            if isinstance(answer, Unknown):
                continue
            attempted += 1
            hit = answer is gold
            correct += hit
            gate_correct[decided.action] += hit
            dept_ok += DEPARTMENT_OF_INTENT[answer.name] == DEPARTMENT_OF_INTENT[gold.name]
            urg_ok += urgency_of(answer) == urgency_of(gold)
            refund_ok += (answer.name in REFUND_INTENTS) == (gold.name in REFUND_INTENTS)
            if score is not None and score.kind == "probability":
                pairs.append((score.value, hit))

    n = len(rows)
    cal = calibration(pairs) if pairs else None
    counts = Counter(gold.name for _, gold in rows)
    majority = max(counts.values()) / n if n else 0.0
    model_calls = sum(1 for impl, c in answered_by.items() if impl.startswith("chat:") for _ in range(c))

    def over_attempted(k: int) -> float | None:
        return round(k / attempted, 4) if attempted else None

    return ArmResult(
        arm=name,
        n=n,
        intent_attempted=attempted,
        intent_correct=correct,
        intent_accuracy_over_attempted=over_attempted(correct),
        intent_coverage=round(attempted / n, 4) if n else 0.0,
        intent_accuracy_overall=round(correct / n, 4) if n else 0.0,
        department_correct=dept_ok,
        department_accuracy_over_attempted=over_attempted(dept_ok),
        urgency_correct=urg_ok,
        urgency_accuracy_over_attempted=over_attempted(urg_ok),
        refund_asked_correct=refund_ok,
        refund_asked_accuracy_over_attempted=over_attempted(refund_ok),
        gate_counts=dict(gate_counts),
        auto_accuracy=round(gate_correct["auto"] / gate_counts["auto"], 4) if gate_counts["auto"] else None,
        confirm_accuracy=round(gate_correct["confirm"] / gate_counts["confirm"], 4) if gate_counts["confirm"] else None,
        ece=None if cal is None or isinstance(cal, Unknown) else cal.ece,
        reliability=[] if cal is None or isinstance(cal, Unknown) else [[s, o, n_] for s, o, n_ in cal.bins],
        ms_p50=round(statistics.median(latencies), 3) if latencies else 0.0,
        ms_p95=round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 3) if latencies else 0.0,
        usd_per_1000=round(sum(usd_known) / len(usd_known) * 1000, 6) if usd_known else (0.0 if not unmetered else None),
        usd_basis="declared 0.0 by every in-process implementation" if not unmetered else f"{unmetered} answers had unknown cost",
        model_calls=model_calls,
        answered_by=dict(answered_by),
        majority_baseline=round(majority, 4),
        random_baseline=round(1 / len(Intent), 6),
    )


def selective_curve(bindings: Sequence[object], rows: Sequence[tuple[str, Intent]]) -> list[tuple[float, float, int]]:
    """(threshold, accuracy above it, n) on a held-out slice — what a gate should be built from."""
    runtime = tiers.runtime(bindings)
    observed: list[tuple[float, bool]] = []
    with tc.use(runtime):
        for text, gold in rows:
            answer, score, _ = classify_with_confidence(text, Intent)
            if isinstance(answer, Unknown) or score is None or score.kind != "probability":
                continue
            observed.append((score.value, answer is gold))
    curve = []
    for threshold in [i / 20 for i in range(21)]:
        kept = [hit for p, hit in observed if p >= threshold]
        curve.append((threshold, sum(kept) / len(kept) if kept else 0.0, len(kept)))
    return curve


def measure_rerank(path: Path, limit: int) -> dict:
    """Rank the 10 distractor paragraphs; gold supporting titles are the public label."""
    import pandas as pd

    df = pd.read_parquet(path).head(limit)
    runtime = tiers.runtime(tiers.rules_only())
    hits_at_1 = hits_at_2 = both_at_2 = 0
    rr: list[float] = []
    latencies: list[float] = []
    with tc.use(runtime):
        for _, row in df.iterrows():
            titles = list(row["context"]["title"])
            sentences = row["context"]["sentences"]
            gold = set(row["supporting_facts"]["title"])
            passages = [Passage(f"P{i}", t, " ".join(list(s))) for i, (t, s) in enumerate(zip(titles, sentences))]
            t0 = time.perf_counter()
            ranked = rerank(str(row["question"]), passages)
            latencies.append((time.perf_counter() - t0) * 1e3)
            if isinstance(ranked, Unknown):
                continue
            order = [p.title for p, _ in ranked]
            hits_at_1 += order[0] in gold
            top2 = set(order[:2])
            hits_at_2 += bool(top2 & gold)
            both_at_2 += top2 >= gold
            rank_of_first = next((i + 1 for i, t in enumerate(order) if t in gold), None)
            rr.append(1 / rank_of_first if rank_of_first else 0.0)
    n = len(df)
    return {
        "n": n,
        "labels": "PUBLIC: HotpotQA distractor gold supporting_facts titles",
        "recall_at_1": round(hits_at_1 / n, 4),
        "recall_at_2_any": round(hits_at_2 / n, 4),
        "both_gold_in_top_2": round(both_at_2 / n, 4),
        "mrr": round(sum(rr) / len(rr), 4) if rr else None,
        "random_recall_at_1": round(2 / 10, 4),
        "ms_p50": round(statistics.median(latencies), 3),
        "implementation": "bm25-rerank@1 (Score kind=relevance)",
    }


def measure_support(path: Path, limit: int) -> dict:
    """Does a paragraph support the question's answer? Gold vs distractor is the label."""
    import pandas as pd

    df = pd.read_parquet(path).head(limit)
    runtime = tiers.runtime(tiers.rules_only())
    stats = {"gold": Counter(), "distractor": Counter()}
    with tc.use(runtime):
        for _, row in df.iterrows():
            titles = list(row["context"]["title"])
            sentences = row["context"]["sentences"]
            gold = set(row["supporting_facts"]["title"])
            claim = f"{row['question']} {row['answer']}"
            for i, (title, sents) in enumerate(zip(titles, sentences)):
                passage = Passage(f"P{i}", title, " ".join(list(sents)))
                verdict = supports(claim, passage)
                stats["gold" if title in gold else "distractor"][verdict.status] += 1
    g, d = stats["gold"], stats["distractor"]
    gn, dn = sum(g.values()), sum(d.values())
    return {
        "n_questions": len(df),
        "labels": "PUBLIC: gold supporting paragraph vs distractor paragraph",
        "gold": {k: v for k, v in g.items()},
        "distractor": {k: v for k, v in d.items()},
        "holds_on_gold": round(g["holds"] / gn, 4) if gn else None,
        "holds_on_distractor": round(d["holds"] / dn, 4) if dn else None,
        "unknown_share": round((g["unknown"] + d["unknown"]) / (gn + dn), 4) if gn + dn else None,
        "implementation": "overlap-support@1 (three-valued, explicit unknown band)",
        "note": "a deliberately weak checker; the point measured here is that its unknown band is used rather than collapsed to false",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--limit", type=int, default=None, help="Banking77 test rows (default: all 3080)")
    ap.add_argument("--hotpot", type=int, default=300)
    ap.add_argument("--out", type=Path, default=REPO / "eval/results")
    args = ap.parse_args()

    train_csv, test_csv = args.data / "banking77_train.csv", args.data / "banking77_test.csv"
    rows = load_banking77(test_csv, args.limit)
    hotpot = args.data / "hotpot_distractor_validation.parquet"

    learned_bindings, report = tiers.learned_only(train_csv)
    cascade_bindings, _ = tiers.cascade(train_csv)

    # A gate built from measurement, on the fit's own validation split — never on test.
    from examples.support_router.config import load_banking77 as load_train, split_train_validation

    _, validation = split_train_validation(load_train(train_csv))
    curve = selective_curve(cascade_bindings, validation)
    measured_gate = Gate.from_measurement(
        "routing",
        curve,
        target_accuracy=0.95,
        confirm_at=0.30,
        basis=f"selective curve on the fit's own validation holdout, n={len(validation)}",
    )

    scored_bindings, _ = tiers.cascade_scored(train_csv)
    arms = [
        measure_arm("rules", tiers.rules_only(), rows, Gate(Thresholds("routing", 0.60, 0.30, "hand-set default"))),
        measure_arm("learned", learned_bindings, rows, measured_gate),
        measure_arm("cascade", cascade_bindings, rows, measured_gate),
        measure_arm("cascade+scored-rules", scored_bindings, rows, measured_gate),
    ]

    out = {
        "date": time.strftime("%Y-%m-%d"),
        "what": "a decision layer in a normal request path: triage, refund eligibility, routing, rerank, citation support",
        "provenance": {
            "intent": "PUBLIC LABEL (Banking77 test split); nothing of ours in the path",
            "department/urgency/refund_asked": "public text + public intent label + OUR mapping (examples/decisions/domain.py)",
            "rerank": "PUBLIC LABEL (HotpotQA distractor gold supporting paragraphs)",
            "citation_support": "PUBLIC LABEL (gold vs distractor paragraph)",
            "refund_eligibility": "NOT MEASURED against public data: the policy and charge records are ours; it is a deterministic walk, covered by unit tests instead",
        },
        "gate_from_measurement": {
            "auto": measured_gate.t.auto,
            "confirm": measured_gate.t.confirm,
            "basis": measured_gate.t.basis,
            "curve": [[round(t, 3), round(a, 4), n] for t, a, n in curve],
        },
        "learned_fit": asdict(report),
        "arms": [asdict(a) for a in arms],
        "rerank": measure_rerank(hotpot, args.hotpot),
        "citation_support": measure_support(hotpot, min(args.hotpot, 100)),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "decisions_measure.json").write_text(json.dumps(out, indent=1, default=str))

    print(f"{'arm':<9} {'cover':>6} {'acc|att':>8} {'dept':>6} {'urg':>6} {'refund':>7} {'ECE':>6} {'p50ms':>6} {'auto/conf/esc':>16} {'model':>6}")
    for a in arms:
        gc = a.gate_counts
        print(
            f"{a.arm:<9} {a.intent_coverage:>6.3f} {(a.intent_accuracy_over_attempted or 0):>8.4f} "
            f"{(a.department_accuracy_over_attempted or 0):>6.3f} {(a.urgency_accuracy_over_attempted or 0):>6.3f} "
            f"{(a.refund_asked_accuracy_over_attempted or 0):>7.3f} {(a.ece if a.ece is not None else float('nan')):>6.3f} "
            f"{a.ms_p50:>6.2f} {gc.get('auto', 0):>5}/{gc.get('confirm', 0):>4}/{gc.get('escalate', 0):>4} {a.model_calls:>6}"
        )
    print("\nrerank:", json.dumps(out["rerank"], default=str))
    print("support:", json.dumps(out["citation_support"], default=str))
    print(f"\nwrote {args.out / 'decisions_measure.json'}")


if __name__ == "__main__":
    main()
