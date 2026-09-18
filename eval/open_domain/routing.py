"""Can any routing beat "always ask the model" at equal accuracy?

The previous run showed the cheap tier's abstention is capability-blind: it fires on weak
lexical overlap, not on whether the tier can answer, so the cascade came out worse than the
model alone. This script replaces the routing signal with one trained to predict exactly
that — "will the cheap tier be right on this item?" — and prices it honestly:

* features come only from the item and the cheap tier's own output (no gold, no model);
* the router is trained on a slice disjoint from both the calibration and test slices;
* an oracle router is included as the headroom bound, so we can tell "our router is bad"
  from "there is nothing here to route on";
* the curve is accuracy against model calls, because a router that saves no calls is just
  the model with extra steps.

    python -m eval.open_domain.routing --benchmarks squad2,hotpot --n 300 --n-cal 150 --n-router 400
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
from .score import grade, wilson

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval" / "results" / "open_domain_routing.json"


def splits(benchmark: str, n: int, n_cal: int, n_router: int, seed: int) -> tuple[list[Item], list[Item], list[Item]]:
    """cal / test / router-train, in that order, from one shuffle. `test` is byte-identical to the earlier run."""
    pool = LOADERS[benchmark](n + n_cal + n_router, seed)
    return pool[:n_cal], pool[n_cal : n_cal + n], pool[n_cal + n :]


def features(item: Item, min_score: float) -> tuple[dict, R.Answer]:
    """Signals available before knowing the answer: evidence strength, agreement, span availability."""
    ans = R.answer_extractive(item, min_score=0.0)          # unthresholded, so features are always defined
    thresholded = R.answer_extractive(item, min_score=min_score)
    sentences = [s for _, s in item.passages]
    with tc.use(R.RUNTIME):
        ranked = tc.rank(item.question, sentences, limit=3)
    scores = [float(s.value) for _, s in ranked] if not isinstance(ranked, tc.Unknown) else [0.0]
    kind = R.answer_type(item.question)
    qwords = set(R._words(item.question))
    top_sent = ranked[0][0] if not isinstance(ranked, tc.Unknown) else ""
    # a cheap self-consistency signal: do the top two sentences yield the same span?
    spans = []
    for sent, _ in (ranked[:2] if not isinstance(ranked, tc.Unknown) else []):
        cands = R._candidates(sent, kind, item.question)
        spans.append(cands[0].lower() if cands else None)
    agree = 1.0 if len(spans) == 2 and spans[0] is not None and spans[0] == spans[1] else 0.0
    n_cands = len(R._candidates(top_sent, kind, item.question)) if top_sent else 0
    f = {
        "bm25_top": scores[0],
        "bm25_margin": scores[0] - (scores[1] if len(scores) > 1 else 0.0),
        "bm25_mean3": sum(scores[:3]) / max(1, len(scores[:3])),
        "n_candidates": float(n_cands),
        "has_candidate": 1.0 if n_cands else 0.0,
        "two_readings_agree": agree,
        "q_words": float(len(qwords)),
        "overlap_ratio": len(qwords & set(R._words(top_sent))) / max(1, len(qwords)),
        "n_passage_sentences": float(len(sentences)),
        "answered_at_threshold": 0.0 if isinstance(thresholded.text, tc.Unknown) else 1.0,
    }
    for k in ("number", "date", "person", "place", "entity", "boolean"):
        f[f"type_{k}"] = 1.0 if kind == k else 0.0
    return f, ans


def train_router(rows: list[dict]) -> object:
    """Logistic regression: P(the cheap tier's unthresholded answer is correct)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    keys = sorted(rows[0]["features"])
    X = [[r["features"][k] for k in keys] for r in rows]
    y = [int(r["cheap_correct"]) for r in rows]
    if len(set(y)) < 2:
        return None
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
    clf.fit(X, y)
    clf._keys = keys
    return clf


def predict(clf: object, row: dict) -> float:
    if clf is None:
        return 0.0
    X = [[row["features"][k] for k in clf._keys]]
    return float(clf.predict_proba(X)[0][1])


def policy_curve(benchmark: str, test_rows: list[dict], *, key: str, thresholds: list[float]) -> list[dict]:
    """Route by `key` >= tau: keep the cheap answer, else use the model's."""
    out = []
    for tau in thresholds:
        correct = calls = 0
        for r in test_rows:
            if r[key] >= tau:
                correct += int(r["cheap_correct"])
            else:
                correct += int(r["model_correct"])
                calls += 1
        n = len(test_rows)
        lo, hi = wilson(correct, n)
        out.append({"tau": tau, "accuracy": round(correct / n, 4), "ci95": [round(lo, 4), round(hi, 4)],
                    "model_calls": calls, "calls_saved": round(1 - calls / n, 4)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default="squad2,hotpot")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n-cal", type=int, default=150)
    ap.add_argument("--n-router", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    llm = M.LocalModel(args.model, batch_size=args.batch_size)
    print(f"loading {args.model} ...", flush=True)
    llm.load()
    print(f"loaded in {llm.load_seconds:.0f}s", flush=True)

    report = {"date": time.strftime("%Y-%m-%d %H:%M"),
              "environment": {"python": platform.python_version(), "platform": platform.platform(), "model": args.model},
              "design": {
                  "question": "does any routing beat 'always ask the model' at equal accuracy?",
                  "router": "logistic regression predicting whether the cheap tier's answer is correct, from item and cheap-tier features only",
                  "router_training": f"a {args.n_router}-item slice disjoint from both the calibration and test slices",
                  "test": f"the same {args.n} test items as eval/results/open_domain.json",
                  "policies": ["always_model", "always_rules", "bm25_threshold (the old capability-blind signal)",
                               "capability_router (learned)", "oracle (headroom bound)"],
              },
              "benchmarks": {}}

    for benchmark in args.benchmarks.split(","):
        print(f"\n=== {benchmark}", flush=True)
        cal, test, router_train = splits(benchmark, args.n, args.n_cal, args.n_router, args.seed)

        # threshold for the old signal, chosen on calibration only (as before)
        best = (0.0, -1.0)
        for thr in [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
            rows = [grade(benchmark, it, R.answer_extractive(it, min_score=thr).text) for it in cal]
            sc = sum(r["correct"] for r in rows) / max(1, len(rows))
            if sc > best[1]:
                best = (thr, sc)
        min_score = best[0]

        def build(items: list[Item]) -> list[dict]:
            rows = []
            for it in items:
                f, ans = features(it, min_score)
                pred = ans.text
                g = grade(benchmark, it, pred)
                # "cheap_correct" is whether keeping the cheap tier's unthresholded answer scores a point
                rows.append({"id": it.id, "features": f, "cheap_pred": pred if isinstance(pred, str) else None,
                             "cheap_correct": bool(g["correct"]), "unanswerable": bool(it.unanswerable)})
            return rows

        tr_rows, te_rows = build(router_train), build(test)
        clf = train_router(tr_rows)

        # model predictions on the test slice only
        t0 = time.perf_counter()
        preds = [M.parse_reply(benchmark, r) for r in llm.generate(benchmark, [M.prompt_for(benchmark, it) for it in test])]
        model_s = time.perf_counter() - t0
        for it, row, p in zip(test, te_rows, preds):
            mp: str | tc.Unknown = p
            if benchmark == "squad2" and p.strip().lower().startswith("unanswerable"):
                mp = tc.Unknown("model_says_unanswerable", "the model declined")
            row["model_correct"] = bool(grade(benchmark, it, mp)["correct"])
            row["router_p"] = predict(clf, row)
            row["bm25_top"] = row["features"]["bm25_top"]
            row["oracle"] = 1.0 if row["cheap_correct"] and not row["model_correct"] else 0.0

        n = len(te_rows)
        always_model = sum(r["model_correct"] for r in te_rows) / n
        always_rules = sum(r["cheap_correct"] for r in te_rows) / n
        oracle = sum(max(r["cheap_correct"], r["model_correct"]) for r in te_rows) / n
        # oracle that only keeps the cheap answer when it is right AND the model is wrong
        oracle_calls = sum(1 for r in te_rows if not (r["cheap_correct"] and not r["model_correct"]))

        entry = {
            "n_test": n, "n_router_train": len(tr_rows), "min_score_old_signal": min_score,
            "model_seconds_for_test": round(model_s, 1),
            "baselines": {
                "always_model": {"accuracy": round(always_model, 4), "model_calls": n, "calls_saved": 0.0,
                                 "ci95": [round(x, 4) for x in wilson(sum(r['model_correct'] for r in te_rows), n)]},
                "always_rules": {"accuracy": round(always_rules, 4), "model_calls": 0, "calls_saved": 1.0,
                                 "ci95": [round(x, 4) for x in wilson(sum(r['cheap_correct'] for r in te_rows), n)]},
                "oracle_router": {"accuracy": round(oracle, 4), "model_calls": oracle_calls,
                                  "calls_saved": round(1 - oracle_calls / n, 4),
                                  "note": "upper bound: keeps the cheap answer exactly when it is right and the model is wrong"},
            },
            "curves": {
                "bm25_threshold_old_signal": policy_curve(benchmark, te_rows, key="bm25_top",
                                                          thresholds=[0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 999.0]),
                "capability_router": policy_curve(benchmark, te_rows, key="router_p",
                                                  thresholds=[0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]),
            },
            "router_quality": {},
        }

        # how discriminative is each signal about the cheap tier being right?
        def auc(key: str) -> float:
            pos = [r[key] for r in te_rows if r["cheap_correct"]]
            neg = [r[key] for r in te_rows if not r["cheap_correct"]]
            if not pos or not neg:
                return float("nan")
            wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
            return round(wins / (len(pos) * len(neg)), 4)

        entry["router_quality"] = {
            "auc_capability_router": auc("router_p"),
            "auc_bm25_old_signal": auc("bm25_top"),
            "cheap_correct_rate": round(always_rules, 4),
            "note": "AUC 0.5 means the signal carries no information about whether the cheap tier will be right",
        }

        # (b) the coordinator's hypothesis: is the cheap tier's abstention a good answer-not-present detector?
        if benchmark == "squad2":
            abst = [r for r in te_rows if r["features"]["answered_at_threshold"] == 0.0]
            una = [r for r in te_rows if r["unanswerable"]]
            ansr = [r for r in te_rows if not r["unanswerable"]]
            p_abstain_given_unans = sum(r["features"]["answered_at_threshold"] == 0.0 for r in una) / max(1, len(una))
            p_abstain_given_ans = sum(r["features"]["answered_at_threshold"] == 0.0 for r in ansr) / max(1, len(ansr))
            # hybrid: rules decide answerability, model answers the rest
            correct = calls = 0
            for r, it in zip(te_rows, test):
                if r["features"]["answered_at_threshold"] == 0.0:
                    correct += int(it.unanswerable)          # declared unanswerable
                else:
                    correct += int(r["model_correct"])
                    calls += 1
            entry["answer_not_present_detector"] = {
                "p_abstain_given_unanswerable": round(p_abstain_given_unans, 4),
                "p_abstain_given_answerable": round(p_abstain_given_ans, 4),
                "lift": round(p_abstain_given_unans - p_abstain_given_ans, 4),
                "hybrid_rules_decide_answerability_model_answers": {
                    "accuracy": round(correct / n, 4), "model_calls": calls, "calls_saved": round(1 - calls / n, 4)},
                "note": ("the 84% 'correctly abstained' figure is mostly a high abstention rate, not detection: "
                         "compare P(abstain | unanswerable) with P(abstain | answerable)"),
            }
        report["benchmarks"][benchmark] = entry
        print(f"  always_model={always_model:.3f} always_rules={always_rules:.3f} oracle={oracle:.3f} "
              f"auc_router={entry['router_quality']['auc_capability_router']} auc_bm25={entry['router_quality']['auc_bm25_old_signal']}", flush=True)
        best_router = max(entry["curves"]["capability_router"], key=lambda r: r["accuracy"])
        print(f"  best router point: acc={best_router['accuracy']:.3f} at {best_router['calls_saved']:.1%} calls saved", flush=True)
        args.out.write_text(json.dumps(report, indent=1, default=str))

    report["model_cost"] = {"calls": llm.calls, "new_tokens": llm.new_tokens,
                            "generate_seconds": round(llm.seconds, 1),
                            "tokens_per_second": round(llm.new_tokens / llm.seconds, 1) if llm.seconds else None}
    args.out.write_text(json.dumps(report, indent=1, default=str))
    print("\nwrote", args.out, flush=True)


if __name__ == "__main__":
    main()
