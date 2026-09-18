"""Does a trained selector close the multi-hop gap, and does an answer type catch what it can't?

    PYTHONPATH=src:. venv-eval/bin/python eval/selection/eval_selection.py

Doc 21 bracketed the gap: a perfect selector over a BM25-limited pool was worth +0.087 EM, and
three lexical criteria all failed to reach it. This measures a trained one over ALL sentences,
reports the recall x conditional-accuracy decomposition that is the actual trade being managed,
classifies the oracle arm's failures (boundary convention vs wrong referent vs bridge entity),
and puts the answer-type gate against both sides of the ledger: failures removed, correct lost.

Every arm records how often its input overflowed the answerer's window. The published 0.093 was
an arm whose evidence was two thirds discarded; no accuracy number here is reportable without
that count beside it.
"""

from __future__ import annotations

import os

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

SP = Path(os.environ.get("TENSACODE_SCRATCH", os.path.expanduser("~/.cache/tensacode")))
OUT = ROOT / "eval" / "results" / "selection_hotpot.json"


def contains(pred: str, golds: list[str], *, budget: int | None = None) -> bool:
    """One extent inside the other, either direction, as whole words.

    Words rather than characters because "no" is a character substring of "northeastern Ontario",
    and a character test scores a yes/no question answered with a place name as a near miss.

    ``budget`` bounds the disagreement: with it, the two strings must differ by at most that many
    words. Unbounded containment is too generous to be evidence of a convention dispute — the
    answerer sometimes returns a whole sentence, which swallows the gold string without being a
    dispute about extent at all ("Glenn Ford, Vince Edwards, Shirley Jones... Edward Albert
    Heimberger" contains the gold 'Edward Albert Heimberger' and is not the same answer).
    """
    from tensacode.answer_type import _words, contains_words
    if not (pred or "").strip():
        return False
    for g in golds:
        if not (contains_words(g, pred) or contains_words(pred, g)):
            continue
        if budget is None or abs(len(_words(pred)) - len(_words(g))) <= budget:
            return True
    return False


def classify(pred: str, item, question: str) -> str:
    """Why a wrong answer was wrong. Categories follow doc 21 so the counts are comparable."""
    from eval.open_domain.score import f1
    from tensacode.answer_type import from_question
    if not pred:
        return "abstained"
    if contains(pred, item.gold):
        return "span_boundary"
    if from_question(pred, question):
        return "bridge_entity_returned"
    if f1(pred, item.gold) > 0.0:
        return "partial_overlap"
    return "wrong_span"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--selector", type=Path, default=SP / "artifacts" / "sentence-selector")
    ap.add_argument("--artifact", type=Path, default=SP / "artifacts" / "span-answerer")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8, 12])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    from eval.open_domain.data import hotpot
    from eval.open_domain.score import em, f1, wilson
    from eval.schema.multihop import gold_recall, gold_sentences, question_types, rank, split_of
    from eval.selection.selector import Selector, truncation_of
    from tensacode.answer_type import AnswerType, Rejection, asked_for, mismatch, shape
    from tensacode.backends.neural import NeuralAnswerer, QuestionOverPassages

    items = hotpot(args.n, seed=0)
    kinds = question_types()
    selector = Selector(args.selector)
    answerer = NeuralAnswerer(args.artifact)
    limit = answerer.max_length

    t0 = time.time()
    scores = [selector.score(it.question, it.passages) for it in items]
    print(f"scored {sum(len(s) for s in scores)} sentences in {time.time() - t0:.0f}s", flush=True)

    def ranked(i: int, k: int) -> list[tuple[str, str]]:
        it = items[i]
        keep = {j for j, _ in sorted(enumerate(scores[i]), key=lambda p: -p[1])[:k]}
        return [s for j, s in enumerate(it.passages) if j in keep]

    arms: dict[str, list[list[tuple[str, str]]]] = {}
    for k in args.ks:
        arms[f"selector_k{k}"] = [ranked(i, k) for i in range(len(items))]
    arms["bm25_k4"] = [rank(it.question, it.passages, 4) for it in items]
    arms["bm25_k12"] = [rank(it.question, it.passages, 12) for it in items]
    arms["oracle_gold_only"] = [gold_sentences(it) for it in items]
    arms["no_retrieval"] = [list(it.passages) for it in items]

    report: dict[str, dict] = {}
    predictions: dict[str, list[str]] = {}
    for name, evidence in arms.items():
        pairs = [(it.question, " ".join(s for _, s in ev)) for it, ev in zip(items, evidence)]
        trunc = truncation_of(answerer.tokenizer, pairs, limit)
        t0 = time.time()
        answers = answerer.answer([QuestionOverPassages(it.question, tuple(ev))
                                   for it, ev in zip(items, evidence)])
        preds = [a.text for a in answers]
        predictions[name] = preds

        rows = []
        for it, ev, pred in zip(items, evidence, preds):
            recall, full = gold_recall(it, ev)
            n_sel = len(ev) or 1
            hits = round(recall * max(1, len(it.supporting)))
            rows.append({
                "id": it.id, "split": split_of(it.id), "kind": kinds.get(it.id, "?"),
                "recall": recall, "full": full, "precision": hits / n_sel,
                "em": em(pred, it.gold) if pred else False, "f1": f1(pred, it.gold) if pred else 0.0,
                "contained": contains(pred, it.gold),
                "contained_tight": contains(pred, it.gold, budget=3), "answered": bool(pred),
            })
        n = len(rows)
        got_all = [r for r in rows if r["full"]]
        lost = [r for r in rows if not r["full"]]
        correct = sum(r["em"] for r in rows)
        lo, hi = wilson(correct, n)
        report[name] = {
            "n": n, "em": round(correct / n, 4), "em_ci": [round(lo, 4), round(hi, 4)],
            "f1": round(sum(r["f1"] for r in rows) / n, 4),
            "em_containment_tolerant": round(sum(r["contained"] for r in rows) / n, 4),
            "em_same_referent": round(sum(r["contained_tight"] for r in rows) / n, 4),
            "coverage": round(sum(r["answered"] for r in rows) / n, 4),
            "sentences_per_item": round(sum(len(e) for e in evidence) / n, 2),
            "selection_precision": round(sum(r["precision"] for r in rows) / n, 4),
            "gold_recall": round(sum(r["recall"] for r in rows) / n, 4),
            "all_gold_share": round(len(got_all) / n, 4),
            "accuracy_given_all_gold": round(sum(r["em"] for r in got_all) / max(1, len(got_all)), 4),
            "accuracy_when_gold_missing": round(sum(r["em"] for r in lost) / max(1, len(lost)), 4),
            "em_design": round(sum(r["em"] for r in rows if r["split"] == "design")
                               / max(1, sum(r["split"] == "design" for r in rows)), 4),
            "em_heldout": round(sum(r["em"] for r in rows if r["split"] == "heldout")
                                / max(1, sum(r["split"] == "heldout" for r in rows)), 4),
            "em_bridge": round(sum(r["em"] for r in rows if r["kind"] == "bridge")
                               / max(1, sum(r["kind"] == "bridge" for r in rows)), 4),
            "em_comparison": round(sum(r["em"] for r in rows if r["kind"] == "comparison")
                                   / max(1, sum(r["kind"] == "comparison" for r in rows)), 4),
            "truncation": trunc.report(),
            "seconds": round(time.time() - t0, 1),
        }
        r = report[name]
        print(f"{name:18s} EM {r['em']:.4f} cover {r['coverage']:.3f} recall {r['gold_recall']:.3f} "
              f"all-gold {r['all_gold_share']:.3f} acc|gold {r['accuracy_given_all_gold']:.3f} "
              f"trunc {r['truncation']['share_truncated']:.2f} ({r['seconds']:.0f}s)", flush=True)

    # --- failure attribution and the boundary verdict, on the oracle arm (P3) ---
    oracle = predictions["oracle_gold_only"]
    failures = Counter()
    boundary = Counter()
    for it, pred in zip(items, oracle):
        from eval.open_domain.score import em as _em
        if pred and _em(pred, it.gold):
            continue
        why = classify(pred, it, it.question)
        failures[why] += 1
        if why == "span_boundary":
            from tensacode.answer_type import contains_words
            boundary["pred_inside_gold" if contains_words(it.gold[0], pred) else "gold_inside_pred"] += 1
            boundary["within_3_words" if contains(pred, it.gold, budget=3) else "beyond_3_words"] += 1

    # --- the answer-type gate, both sides (P2) ---
    gate: dict[str, dict] = {}
    best = max((k for k in report if k.startswith("selector_k")), key=lambda k: report[k]["em"])
    for name in ("oracle_gold_only", best):
        removed, cost, reasons = Counter(), 0, Counter()
        for it, pred in zip(items, predictions[name]):
            if not pred:
                continue
            rej = mismatch(it.question, pred)
            if rej is None:
                continue
            reasons[rej.reason] += 1
            if em(pred, it.gold):
                cost += 1
            else:
                removed[classify(pred, it, it.question)] += 1
        n_wrong = sum(1 for it, p in zip(items, predictions[name]) if p and not em(p, it.gold))
        n_right = sum(1 for it, p in zip(items, predictions[name]) if p and em(p, it.gold))
        gate[name] = {
            "rejected": int(sum(reasons.values())), "by_reason": dict(reasons),
            "failures_removed": dict(removed), "failures_removed_total": int(sum(removed.values())),
            "correct_answers_lost": cost,
            "wrong_answers_before": n_wrong, "correct_answers_before": n_right,
            "share_of_failures_removed": round(sum(removed.values()) / max(1, n_wrong), 4),
            "share_of_correct_lost": round(cost / max(1, n_right), 4),
            "em_after_gate": round(sum(em(p, it.gold) for it, p in zip(items, predictions[name])
                                       if p and mismatch(it.question, p) is None) / len(items), 4),
        }

    # how well the shape and type derivations themselves do, on the reported items
    derived = Counter((shape(it.question).value, kinds.get(it.id, "?")) for it in items)
    tp = derived[("comparison", "comparison")]
    body = {
        "measured_on": "HotpotQA distractor validation, hotpot(300, seed=0); selector trained on the official train split",
        "selector": selector.config, "answerer": answerer.config.get("encoder"),
        "answerer_max_length": limit,
        "arms": report,
        "best_selector_arm": best,
        "oracle_failure_attribution": dict(failures),
        "boundary_direction": dict(boundary),
        "answer_type_gate": gate,
        "derivation_quality": {
            "comparison_precision": round(tp / max(1, tp + derived[("comparison", "bridge")]), 4),
            "comparison_recall": round(tp / max(1, tp + derived[("bridge", "comparison")]), 4),
            "asked_for_distribution": dict(Counter(asked_for(it.question).value for it in items)),
        },
        "doc21_arms_for_comparison": {"best_real": 0.263, "oracle_selection": 0.350,
                                      "oracle_gold_only": 0.433, "qwen3_8b": 0.5133},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=1))
    print(f"\nfailures(oracle) {dict(failures)}\nboundary {boundary}")
    for k, v in gate.items():
        print(f"gate[{k}] removed {v['failures_removed_total']}/{v['wrong_answers_before']} "
              f"({v['share_of_failures_removed']:.3f}) cost {v['correct_answers_lost']}/"
              f"{v['correct_answers_before']} ({v['share_of_correct_lost']:.3f}) EM {v['em_after_gate']:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
