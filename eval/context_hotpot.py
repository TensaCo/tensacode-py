"""Evaluate retrieve -> rank -> dedupe -> pack on HotpotQA (distractor setting, validation split).

    python eval/context_hotpot.py --parquet hotpot_distractor_validation.parquet

Candidates are the ~40 sentences of the 10 paragraphs given per question (2 gold,
8 distractors); the question's supporting-fact sentences are the evidence we want
inside the budget. This measures a lexical ranker plus deterministic packing on
real text; it says nothing about answer quality downstream.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import tensacode as tc  # noqa: E402
from examples.context_select.program import Snippet  # noqa: E402
from tensacode.backends.builtin import BM25Ranker  # noqa: E402

BUDGETS = (64, 128, 256)


def cost(s: Snippet) -> int:
    return tc.approx_tokens(s.text)


def similar(a: Snippet, b: Snippet) -> tc.Score:
    return tc.shingle_similarity(a.text, b.text)


def pct(xs: list[float]) -> dict:
    a = np.asarray(xs)
    return {"p50": round(float(np.percentile(a, 50)), 4), "p95": round(float(np.percentile(a, 95)), 4), "mean": round(float(a.mean()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "eval/results/context_hotpot.json")
    args = ap.parse_args()
    rows = pq.read_table(args.parquet).to_pylist()[: args.limit]

    rt = tc.Runtime([BM25Ranker(text_of=lambda s: s.text)], policy=tc.Policy(cache=False))
    stats = {f"{strategy}@{b}": {"recall": [], "complete": [], "used": [], "feasible": []} for strategy in ("document_order", "bm25", "bm25_dedupe") for b in BUDGETS}
    required_checks = {"included_when_feasible": 0, "feasible": 0, "unknown_when_infeasible": 0, "infeasible": 0}
    dup = {"bm25": {"recall": [], "dup_tokens": []}, "bm25_dedupe": {"recall": [], "dup_tokens": []}}
    timing = {"rank_total_ms": [], "rank_backend_ms": [], "dedupe_ms": [], "pack_ms": []}
    skipped = 0

    with tc.use(rt):
        for row in rows:
            ctx = row["context"]
            cands = [Snippet(f"{t}#{k}", f"{t}: {sent.strip()}", tc.Ref(f"wiki:{i}")) for i, (t, sents) in enumerate(zip(ctx["title"], ctx["sentences"])) for k, sent in enumerate(sents)]
            ids = {c.id for c in cands}
            gold = {f"{t}#{k}" for t, k in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"])} & ids
            if not gold:
                skipped += 1
                continue
            gold_cost = sum(cost(c) for c in cands if c.id in gold)

            n0 = len(rt.trace.spans)
            ranked = tc.rank(row["question"], cands)
            span = rt.trace.spans[n0]
            timing["rank_total_ms"].append(span.total_ms)
            timing["rank_backend_ms"].append(span.backend_ms)
            t0 = time.perf_counter()
            deduped, _ = tc.dedupe(ranked, similarity=similar, threshold=0.5, key=lambda s: s.id)
            timing["dedupe_ms"].append(1e3 * (time.perf_counter() - t0))
            orders = {"document_order": [(c, tc.Score(0.0, "relevance")) for c in cands], "bm25": ranked, "bm25_dedupe": deduped}

            for b in BUDGETS:
                for strategy, order in orders.items():
                    t0 = time.perf_counter()
                    packed = tc.pack(order, budget=b, cost=cost, key=lambda s: s.id)
                    if strategy == "bm25" and b == 128:
                        timing["pack_ms"].append(1e3 * (time.perf_counter() - t0))
                    got = {s.id for s in packed.items}
                    st = stats[f"{strategy}@{b}"]
                    st["recall"].append(len(got & gold) / len(gold))
                    st["complete"].append(gold <= got)
                    st["used"].append(packed.used)
                    st["feasible"].append(gold_cost <= b)

                required = [c for c in cands if c.id == sorted(gold)[0]]
                packed = tc.pack(ranked, budget=b, cost=cost, required=required, key=lambda s: s.id)
                if cost(required[0]) <= b:
                    required_checks["feasible"] += 1
                    required_checks["included_when_feasible"] += int(not isinstance(packed, tc.Unknown) and required[0] in packed.items)
                else:
                    required_checks["infeasible"] += 1
                    required_checks["unknown_when_infeasible"] += int(isinstance(packed, tc.Unknown))

            # duplicate injection: a lightly edited copy of every gold sentence
            copies = [Snippet(c.id + "~copy", c.text.replace(": ", " - ", 1) + " (via mirror)", c.source) for c in cands if c.id in gold]
            ranked_d = tc.rank(row["question"], cands + copies)
            for strategy, order in (("bm25", ranked_d), ("bm25_dedupe", tc.dedupe(ranked_d, similarity=similar, threshold=0.5, key=lambda s: s.id)[0])):
                packed = tc.pack(order, budget=128, cost=cost, key=lambda s: s.id)
                got = {s.id for s in packed.items}
                dup[strategy]["recall"].append(len(got & gold) / len(gold))
                dup[strategy]["dup_tokens"].append(sum(cost(s) for s in packed.items if s.id.endswith("~copy")))

    n = len(rows) - skipped
    results = {
        "dataset": "hotpotqa/hotpot_qa distractor validation (CC-BY-SA-4.0)",
        "questions": n,
        "skipped_no_valid_supporting_fact": skipped,
        "token_counter": "tensacode.approx_tokens (words and punctuation, not a model tokenizer)",
        "strategies": {
            k: {
                "mean_supporting_fact_recall": round(float(np.mean(v["recall"])), 4),
                "all_supporting_facts_included": round(float(np.mean(v["complete"])), 4),
                "mean_tokens_used": round(float(np.mean(v["used"])), 1),
                "gold_fits_budget": round(float(np.mean(v["feasible"])), 4),
            }
            for k, v in stats.items()
        },
        "required_evidence_invariant": required_checks,
        "duplicate_injection_budget_128": {k: {"mean_recall": round(float(np.mean(v["recall"])), 4), "mean_tokens_spent_on_copies": round(float(np.mean(v["dup_tokens"])), 2)} for k, v in dup.items()},
        "latency_ms_per_question": {k: pct(v) for k, v in timing.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
