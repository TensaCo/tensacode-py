"""Run three arms on public open-domain benchmarks and write eval/results/open_domain.json.

    arm 1  rules      tensacode only: grammar/patterns + BM25 rank + Unknown. No model.
    arm 2  model      the same local model, prompted plainly, no tensacode structure.
    arm 3  cascade    rules answer what they can; only abstentions escalate to the model.

    python eval/open_domain/run.py --benchmarks squad2,hotpot,gsm8k,arc_easy --n 300 --arms rules,model,cascade

Calibration and test samples are disjoint slices of the same shuffled pool, so the
abstention threshold is never chosen on the items it is scored on.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import tensacode as tc

from . import model as M
from . import rules as R
from .data import LOADERS, Item
from .score import floors, grade, supporting_overlap, wilson

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval" / "results" / "open_domain.json"
EXTRACTIVE = {"squad2", "hotpot"}


def load_split(benchmark: str, n: int, n_cal: int, seed: int) -> tuple[list[Item], list[Item]]:
    pool = LOADERS[benchmark](n + n_cal, seed)
    return pool[:n_cal], pool[n_cal:]


def calibrate(benchmark: str, cal: list[Item]) -> dict:
    """Choose the BM25 abstention threshold on the calibration slice only."""
    if benchmark not in EXTRACTIVE:
        return {"min_score": 0.0, "chosen_on": len(cal), "note": "no threshold applies"}
    best = (0.0, -1.0)
    for thr in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]:
        rows = [grade(benchmark, it, R.answer_extractive(it, min_score=thr).text) for it in cal]
        score = sum(r["correct"] for r in rows) / max(1, len(rows))
        if score > best[1]:
            best = (thr, score)
    return {"min_score": best[0], "calibration_score": round(best[1], 4), "chosen_on": len(cal),
            "note": "threshold maximising overall correctness on the calibration slice"}


def run_rules(benchmark: str, items: list[Item], cal: dict, *, guess_mc: bool = False) -> list[dict]:
    rows = []
    for it in items:
        t0 = time.perf_counter()
        store = tc.Store() if benchmark in EXTRACTIVE else None
        if benchmark in EXTRACTIVE:
            ans = R.answer_extractive(it, min_score=cal["min_score"], store=store)
        elif benchmark == "gsm8k":
            ans = R.answer_arithmetic(it)
        else:
            ans = R.answer_multiple_choice(it, guess_by_overlap=guess_mc)
        dt = time.perf_counter() - t0
        g = grade(benchmark, it, ans.text)
        row = {"id": it.id, "pred": ans.text if isinstance(ans.text, str) else f"Unknown({ans.text.reason})",
               "seconds": round(dt, 4), "model_calls": 0, **g}
        if benchmark == "hotpot":
            row["provenance"] = supporting_overlap(it, ans.evidence)
        if store is not None:
            row["claims"] = len(store.claims())
        rows.append(row)
    return rows


def run_model(benchmark: str, items: list[Item], llm: M.LocalModel) -> list[dict]:
    t0 = time.perf_counter()
    if benchmark == "arc_easy":
        preds = [llm.score_options(it) for it in items]
    else:
        preds = [M.parse_reply(benchmark, r) for r in llm.generate(benchmark, [M.prompt_for(benchmark, it) for it in items])]
    dt = (time.perf_counter() - t0) / max(1, len(items))
    rows = []
    for it, p in zip(items, preds):
        pred: str | tc.Unknown = p
        if benchmark == "squad2" and p.strip().lower().startswith("unanswerable"):
            pred = tc.Unknown("model_says_unanswerable", "the model declined")
        g = grade(benchmark, it, pred)
        rows.append({"id": it.id, "pred": p[:120], "seconds": round(dt, 3), "model_calls": 1, **g})
    return rows


def run_cascade(benchmark: str, items: list[Item], cal: dict, llm: M.LocalModel, rule_rows: list[dict]) -> list[dict]:
    """Rules first; escalate only their abstentions. Answers already computed are reused (the cache)."""
    by_id = {r["id"]: r for r in rule_rows}
    escalate = [it for it in items if by_id[it.id]["abstained"]]
    got = {}
    if escalate:
        if benchmark == "arc_easy":
            preds = [llm.score_options(it) for it in escalate]
        else:
            preds = [M.parse_reply(benchmark, r) for r in llm.generate(benchmark, [M.prompt_for(benchmark, it) for it in escalate])]
        got = dict(zip([it.id for it in escalate], preds))
    rows = []
    for it in items:
        base = by_id[it.id]
        if not base["abstained"]:
            rows.append({**base, "tier": "rules"})
            continue
        p = got[it.id]
        pred: str | tc.Unknown = p
        if benchmark == "squad2" and p.strip().lower().startswith("unanswerable"):
            pred = tc.Unknown("model_says_unanswerable", "the model declined")
        g = grade(benchmark, it, pred)
        rows.append({"id": it.id, "pred": p[:120], "seconds": base["seconds"], "model_calls": 1, "tier": "model", **g})
    return rows


def summarize(benchmark: str, rows: list[dict], items: list[Item]) -> dict:
    n = len(rows)
    attempted = [r for r in rows if r["attempted"]]
    correct = sum(r["correct"] for r in rows)
    acc_attempted = sum(r["correct"] for r in attempted) / len(attempted) if attempted else 0.0
    lo, hi = wilson(sum(r["correct"] for r in attempted), len(attempted))
    out = {
        "n": n,
        "coverage": round(len(attempted) / n, 4),
        "accuracy_over_attempted": round(acc_attempted, 4),
        "accuracy_over_attempted_ci95": [round(lo, 4), round(hi, 4)],
        "correct_overall": round(correct / n, 4),
        "mean_f1_over_attempted": round(sum(r["f1"] for r in attempted) / len(attempted), 4) if attempted else 0.0,
        "model_calls": sum(r["model_calls"] for r in rows),
        "seconds_per_item": round(sum(r["seconds"] for r in rows) / n, 4),
    }
    if benchmark == "squad2":
        ans = [r for r in rows if r.get("kind") == "answerable"]
        una = [r for r in rows if r.get("kind") == "unanswerable"]
        out["answerable"] = {"n": len(ans), "attempted": sum(r["attempted"] for r in ans),
                            "em_over_attempted": round(sum(r["correct"] for r in ans) / max(1, sum(r["attempted"] for r in ans)), 4),
                            "em_over_all": round(sum(r["correct"] for r in ans) / max(1, len(ans)), 4)}
        out["unanswerable"] = {"n": len(una),
                               "correctly_abstained": round(sum(r["correct"] for r in una) / max(1, len(una)), 4)}
    if benchmark == "hotpot":
        prov = [r["provenance"] for r in rows if r.get("provenance")]
        if prov:
            out["provenance"] = {
                "items_with_gold": len(prov),
                "all_gold_cited": round(sum(p["all_gold_cited"] for p in prov) / len(prov), 4),
                "mean_gold_hit_rate": round(sum(p["hit"] / p["gold"] for p in prov) / len(prov), 4),
            }
    if any("tier" in r for r in rows):
        by_tier: dict[str, list[dict]] = {}
        for r in rows:
            by_tier.setdefault(r.get("tier", "rules"), []).append(r)
        out["tiers"] = {t: {"n": len(v), "attempted": sum(x["attempted"] for x in v),
                            "accuracy_over_attempted": round(sum(x["correct"] for x in v) / max(1, sum(x["attempted"] for x in v)), 4)}
                        for t, v in by_tier.items()}
    return out


def equal_coverage(benchmark: str, rules_rows: list[dict], model_rows: list[dict]) -> dict:
    """The comparison that matters: how does the model do on exactly the items the rules answered,
    and on exactly the items the rules refused?"""
    by_id = {r["id"]: r for r in model_rows}
    answered = [r for r in rules_rows if r["attempted"]]
    refused = [r for r in rules_rows if not r["attempted"]]
    def acc(rows):
        m = [by_id[r["id"]] for r in rows if r["id"] in by_id]
        att = [x for x in m if x["attempted"]]
        return {"n": len(m), "model_attempted": len(att),
                "model_accuracy": round(sum(x["correct"] for x in att) / max(1, len(att)), 4)}
    return {
        "items_rules_answered": {"n": len(answered),
                                 "rules_accuracy": round(sum(r["correct"] for r in answered) / max(1, len(answered)), 4),
                                 "model_on_same_items": acc(answered)},
        "items_rules_refused": {"n": len(refused), "model_on_same_items": acc(refused)},
    }


def risk_coverage(benchmark: str, items: list[Item]) -> list[dict]:
    """Sweep the abstention threshold on the TEST slice, reported as a curve, not a tuned number.

    It shows what the rule arm's abstention is actually buying: on SQuAD 2.0 roughly half
    the sample is unanswerable, so refusing everything already scores about half.
    """
    curve = []
    for thr in [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]:
        rows = [grade(benchmark, it, R.answer_extractive(it, min_score=thr).text) for it in items]
        att = [r for r in rows if r["attempted"]]
        curve.append({
            "min_score": thr,
            "coverage": round(len(att) / len(rows), 4),
            "accuracy_over_attempted": round(sum(r["correct"] for r in att) / max(1, len(att)), 4),
            "correct_overall": round(sum(r["correct"] for r in rows) / len(rows), 4),
        })
    return curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default="squad2,hotpot,gsm8k,arc_easy")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n-cal", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="rules,model,cascade")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    arms = args.arms.split(",")
    llm = None
    if "model" in arms or "cascade" in arms:
        llm = M.LocalModel(args.model, batch_size=args.batch_size)
        print(f"loading {args.model} ...", flush=True)
        llm.load()
        print(f"loaded in {llm.load_seconds:.0f}s", flush=True)

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "model": args.model if llm else None},
        "design": {
            "arms": {"rules": "tensacode only: patterns + BM25 rank + Unknown; no model",
                     "model": "the same local model prompted plainly, no tensacode structure",
                     "cascade": "rules answer what they can; only their abstentions reach the model"},
            "graders": "public dataset labels only; no model judges anything",
            "splits": f"disjoint calibration ({args.n_cal}) and test ({args.n}) slices of one shuffled pool, seed {args.seed}",
        },
        "benchmarks": {},
    }

    for benchmark in args.benchmarks.split(","):
        print(f"\n=== {benchmark}", flush=True)
        cal_items, items = load_split(benchmark, args.n, args.n_cal, args.seed)
        cal = calibrate(benchmark, cal_items)
        entry = {"n_test": len(items), "n_calibration": len(cal_items), "calibration": cal,
                 "floors": floors(benchmark, items, args.seed), "arms": {}}
        if benchmark in EXTRACTIVE:
            entry["risk_coverage_sweep_on_test"] = risk_coverage(benchmark, items)
        rule_rows = run_rules(benchmark, items, cal)
        entry["arms"]["rules"] = summarize(benchmark, rule_rows, items)
        print(f"  rules: {entry['arms']['rules']['accuracy_over_attempted']:.3f} over {entry['arms']['rules']['coverage']:.3f} coverage", flush=True)
        if benchmark == "arc_easy":
            guess_rows = run_rules(benchmark, items, cal, guess_mc=True)
            entry["arms"]["rules_lexical_overlap_guess"] = summarize(benchmark, guess_rows, items)
            print(f"  rules(lexical guess): {entry['arms']['rules_lexical_overlap_guess']['accuracy_over_attempted']:.3f}", flush=True)
        if llm is not None and "model" in arms:
            model_rows = run_model(benchmark, items, llm)
            entry["arms"]["model"] = summarize(benchmark, model_rows, items)
            entry["equal_coverage"] = equal_coverage(benchmark, rule_rows, model_rows)
            print(f"  model: {entry['arms']['model']['accuracy_over_attempted']:.3f} over {entry['arms']['model']['coverage']:.3f} coverage", flush=True)
        if llm is not None and "cascade" in arms:
            casc_rows = run_cascade(benchmark, items, cal, llm, rule_rows)
            entry["arms"]["cascade"] = summarize(benchmark, casc_rows, items)
            print(f"  cascade: {entry['arms']['cascade']['accuracy_over_attempted']:.3f} over {entry['arms']['cascade']['coverage']:.3f} coverage, "
                  f"{entry['arms']['cascade']['model_calls']} model calls", flush=True)
        report["benchmarks"][benchmark] = entry
        args.out.write_text(json.dumps(report, indent=1, default=str))

    if llm is not None:
        report["model_cost"] = {"calls": llm.calls, "prompt_tokens": llm.prompt_tokens, "new_tokens": llm.new_tokens,
                                "generate_seconds": round(llm.seconds, 1),
                                "tokens_per_second": round(llm.new_tokens / llm.seconds, 1) if llm.seconds else None,
                                "load_seconds": round(llm.load_seconds, 1)}
    args.out.write_text(json.dumps(report, indent=1, default=str))
    print("\nwrote", args.out, flush=True)


if __name__ == "__main__":
    main()
