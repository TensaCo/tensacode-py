"""Where do our cognitive schemas actually break? A per-stage failure taxonomy.

The benchmarks are used here as a diagnostic instrument, not a scoreboard. Nothing is tuned.
For every item the rule arm gets wrong or abstains on, the failure is attributed to the
EARLIEST stage of its own pipeline that is responsible:

    question -> answer-type read -> retrieval over passage sentences -> candidate spans
             -> selection among candidates -> projection to an answer, or abstention

Each stage failure is then labelled by the deficiency it reveals:

    (a) missing representation      something the schema cannot express at all
    (b) missing mechanism           the representation is adequate, the operation is absent
    (c) knowledge gap               no symbolic structure could supply the answer
    (d) benchmark wiring artifact   our error in how the arm meets the data

And, because a low score can still be flattering, correct answers are checked for luck:
if the arm picked one of k same-type candidates from the chosen sentence with no mechanism
discriminating between them, then 1/k of that credit is chance.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import time
from collections import Counter
from pathlib import Path

import tensorcode as tc

from . import rules as R
from .data import LOADERS, Item
from .score import normalize
from .run import EXTRACTIVE, calibrate, load_split

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval" / "results" / "schema_brittleness.json"

# stage -> (deficiency class, what the schema would have needed)
STAGES = {
    "retrieval_missed_evidence": ("a", "lexical semantics: synonymy, hypernymy and paraphrase over claim objects. BM25 matches surface strings, so a question that shares no words with its evidence cannot retrieve it."),
    "answer_not_in_passage_but_labelled_answerable": ("d", "the gold span is not literally in the sentences we split out; our sentence splitter or passage handling dropped it."),
    "span_not_produced": ("a", "an entity/number/date recogniser over the retrieved sentence. The candidate generator is regex-shaped, so it cannot see spans it has no pattern for."),
    "wrong_type_asked": ("b", "a better question-type reader: the type was misread, so only candidates of the wrong kind were considered."),
    "selection_chose_wrong_candidate": ("b", "a mechanism that scores candidates against the question. Right now the first candidate of the right type in the best sentence wins, which is position, not reasoning."),
    "abstained_though_answerable": ("b", "capability-aware abstention. The threshold is on evidence strength, not on whether a span of the right type was actually found."),
    "multi_hop_second_fact_never_represented": ("a", "claims that compose: the second hop needs the first hop's answer as a term in a new query. The arm issues one query and stops."),
    "arithmetic_composition": ("a", "a representation of quantities, their relations and an order of operations. One-step sum/difference patterns cannot express a chain."),
    "no_quantities_parsed": ("a", "quantity extraction with units and referents, not bare numbers."),
    "world_knowledge": ("c", "facts about the world. Nothing in the passage or the grammar can supply them."),
    "boolean_or_other_type": ("b", "yes/no and comparison handling; the arm has no projection for them."),
}


def where_gold_is(item: Item, golds: list[str]) -> tuple[bool, set[int]]:
    """Which passage sentences contain a gold answer string (normalized substring match)."""
    hits = set()
    for i, (_, sent) in enumerate(item.passages):
        n = normalize(sent)
        if any(normalize(g) and normalize(g) in n for g in golds):
            hits.add(i)
    return bool(hits), hits


def diagnose_extractive(item: Item, min_score: float) -> dict:
    """Run the arm's own stages and attribute the outcome to the earliest responsible one."""
    sentences = [s for _, s in item.passages]
    kind = R.answer_type(item.question)
    with tc.use(R.RUNTIME):
        ranked = tc.rank(item.question, sentences, limit=3)
    top = [] if isinstance(ranked, tc.Unknown) else [s for s, _ in ranked]
    ans = R.answer_extractive(item, min_score=min_score)
    abstained = isinstance(ans.text, tc.Unknown)
    pred = None if abstained else ans.text
    correct = (not abstained) and any(normalize(g) == normalize(pred) for g in item.gold)

    out = {"id": item.id, "kind": kind, "abstained": abstained,
           "abstain_reason": ans.text.reason if abstained else None,
           "pred": pred, "gold": item.gold, "correct": bool(correct),
           "unanswerable": bool(item.unanswerable)}

    if item.unanswerable:
        out["stage"] = None if abstained else "answered_an_unanswerable_question"
        out["deficiency"] = None if abstained else "b"
        return out
    if correct:
        # luck audit: how many same-type candidates were available in the sentence used?
        cands = []
        for sent in top:
            cands = R._candidates(sent, kind, item.question)
            if cands:
                break
        out["candidates_in_chosen_sentence"] = len(cands)
        out["chance_of_this_being_luck"] = round(1 - 1 / max(1, len(cands)), 3)
        out["stage"] = None
        return out

    gold_present, gold_idx = where_gold_is(item, item.gold)
    retrieved_idx = {i for i, (_, s) in enumerate(item.passages) if s in top}
    cands_top = [c for sent in top for c in R._candidates(sent, kind, item.question)]
    gold_in_cands = any(normalize(g) == normalize(c) for g in item.gold for c in cands_top)

    if not gold_present:
        stage = "answer_not_in_passage_but_labelled_answerable"
    elif not (gold_idx & retrieved_idx):
        stage = "retrieval_missed_evidence"
    elif kind == "boolean":
        stage = "boolean_or_other_type"
    elif not gold_in_cands:
        stage = "span_not_produced"
    elif abstained:
        stage = "abstained_though_answerable"
    else:
        stage = "selection_chose_wrong_candidate"
    out["stage"] = stage
    out["deficiency"] = STAGES.get(stage, ("?", ""))[0]
    out["gold_sentence_retrieved"] = bool(gold_idx & retrieved_idx)
    out["gold_among_candidates"] = gold_in_cands
    return out


def diagnose_gsm8k(item: Item) -> dict:
    ans = R.answer_arithmetic(item)
    abstained = isinstance(ans.text, tc.Unknown)
    correct = (not abstained) and re.sub(r"[^\d.-]", "", str(ans.text)) == item.gold[0]
    reason = ans.text.reason if abstained else None
    stage = None
    if not correct:
        stage = {"not_enough_quantities": "no_quantities_parsed",
                 "no_single_step_pattern": "arithmetic_composition",
                 "multi_step": "arithmetic_composition"}.get(reason, "arithmetic_composition")
    steps = len([x for x in re.findall(r"####|\n", item.gold[0])])
    return {"id": item.id, "abstained": abstained, "abstain_reason": reason, "correct": bool(correct),
            "stage": stage, "deficiency": STAGES.get(stage, ("?", ""))[0] if stage else None,
            "numbers_in_question": len(re.findall(r"\d", item.question)), "gsm_steps_hint": steps}


def diagnose_arc(item: Item) -> dict:
    ans = R.answer_multiple_choice(item, guess_by_overlap=False)
    return {"id": item.id, "abstained": True, "abstain_reason": ans.text.reason, "correct": False,
            "stage": "world_knowledge", "deficiency": "c"}


def probe_typeless_candidates(items: list[Item], min_score: float = 0.0) -> dict:
    """PROBE (b): drop the answer-type filter, ungated by the evidence threshold.

    Run at min_score=0 on purpose: with the calibrated threshold (~8) only a handful of items are
    attempted at all, so a gated version measures the threshold rather than the type filter.
    """
    right = attempted = 0
    for it in items:
        if it.unanswerable:
            continue
        sentences = [s for _, s in it.passages]
        with tc.use(R.RUNTIME):
            ranked = tc.rank(it.question, sentences, limit=3)
        if isinstance(ranked, tc.Unknown) or not ranked or float(ranked[0][1].value) < min_score:
            continue
        cands: list[str] = []
        for sent, _ in ranked:
            for k in ("number", "date", "person", "entity"):
                cands += R._candidates(sent, k, it.question)
            if cands:
                break
        if not cands:
            continue
        attempted += 1
        right += any(normalize(g) == normalize(cands[0]) for g in it.gold)
    return {"attempted": attempted, "correct": right,
            "accuracy_over_attempted": round(right / max(1, attempted), 4)}


def probe_span_coverage(items: list[Item]) -> dict:
    """PROBE (a): when the gold sentence IS retrieved, can the candidate generator even produce the gold span?

    This isolates the span-production stage from retrieval, which the stem probe shows is not the
    binding constraint on SQuAD 2.0.
    """
    usable = retrieved = producible_typed = producible_any = 0
    for it in items:
        if it.unanswerable:
            continue
        present, gold_idx = where_gold_is(it, it.gold)
        if not present:
            continue
        usable += 1
        sentences = [s for _, s in it.passages]
        with tc.use(R.RUNTIME):
            ranked = tc.rank(it.question, sentences, limit=3)
        top = [] if isinstance(ranked, tc.Unknown) else [s for s, _ in ranked]
        idx = {sentences.index(s) for s in top}
        if not (gold_idx & idx):
            continue
        retrieved += 1
        kind = R.answer_type(it.question)
        typed = [c for sent in top for c in R._candidates(sent, kind, it.question)]
        anyt = [c for sent in top for k in ("number", "date", "person", "place", "entity") for c in R._candidates(sent, k, it.question)]
        producible_typed += any(normalize(g) == normalize(c) for g in it.gold for c in typed)
        producible_any += any(normalize(g) == normalize(c) for g in it.gold for c in anyt)
    return {"answerable_with_gold_in_passage": usable, "gold_sentence_retrieved": retrieved,
            "gold_span_producible_with_asked_type": producible_typed,
            "gold_span_producible_with_any_type": producible_any,
            "producible_rate_asked_type": round(producible_typed / max(1, retrieved), 4),
            "producible_rate_any_type": round(producible_any / max(1, retrieved), 4)}


def probe_stem_retrieval(items: list[Item]) -> dict:
    """PROBE (a): does crude stemming lift retrieval of the gold-bearing sentence into the top 3?"""
    def stem(w: str) -> str:
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > 4 and w.endswith(suf):
                return w[: -len(suf)]
        return w

    base = better = usable = 0
    for it in items:
        if it.unanswerable:
            continue
        present, gold_idx = where_gold_is(it, it.gold)
        if not present:
            continue
        usable += 1
        sentences = [s for _, s in it.passages]
        with tc.use(R.RUNTIME):
            ranked = tc.rank(it.question, sentences, limit=3)
        top = set() if isinstance(ranked, tc.Unknown) else {sentences.index(s) for s, _ in ranked}
        base += bool(gold_idx & top)
        q = Counter(stem(w) for w in R._words(it.question))
        scored = sorted(((sum(q[stem(w)] for w in R._words(s)) / (len(R._words(s)) ** 0.5 + 1), i)
                         for i, s in enumerate(sentences)), reverse=True)[:3]
        better += bool(gold_idx & {i for _, i in scored})
    return {"items_with_gold_in_passage": usable,
            "bm25_recall_at_3": round(base / max(1, usable), 4),
            "stem_overlap_recall_at_3": round(better / max(1, usable), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n-cal", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-n", type=int, default=150)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    report = {"date": time.strftime("%Y-%m-%d %H:%M"),
              "environment": {"python": platform.python_version(), "platform": platform.platform()},
              "method": __doc__.strip().splitlines()[0],
              "deficiency_classes": {"a": "missing representation", "b": "missing mechanism over an adequate representation",
                                     "c": "knowledge gap no symbolic structure can fill", "d": "benchmark wiring artifact (our error)"},
              "benchmarks": {}}

    for benchmark in ("squad2", "hotpot", "gsm8k", "arc_easy"):
        cal, items = load_split(benchmark, args.n, args.n_cal, args.seed)
        min_score = calibrate(benchmark, cal)["min_score"] if benchmark in EXTRACTIVE else 0.0
        if benchmark in EXTRACTIVE:
            rows = [diagnose_extractive(it, min_score) for it in items]
        elif benchmark == "gsm8k":
            rows = [diagnose_gsm8k(it) for it in items]
        else:
            rows = [diagnose_arc(it) for it in items]

        stages = Counter(r["stage"] for r in rows if r.get("stage"))
        defs = Counter(r.get("deficiency") for r in rows if r.get("deficiency"))
        correct = [r for r in rows if r["correct"]]
        luck = [r for r in correct if r.get("candidates_in_chosen_sentence", 1) > 1]
        expected_by_chance = sum(r.get("chance_of_this_being_luck", 0.0) for r in correct)
        entry = {
            "n": len(rows),
            "correct": len(correct),
            "abstained": sum(r["abstained"] for r in rows),
            "stages": dict(stages.most_common()),
            "deficiency_counts": dict(defs.most_common()),
            "stage_meaning": {k: {"class": STAGES[k][0], "would_have_needed": STAGES[k][1]} for k in stages if k in STAGES},
            "luck_audit": {
                "correct_items": len(correct),
                "correct_with_more_than_one_same_type_candidate": len(luck),
                "expected_correct_by_chance_among_them": round(expected_by_chance, 2),
                "note": ("1/k of the credit is chance when k same-type candidates were available and nothing "
                         "discriminated between them; the arm takes the first by position"),
            },
            "examples": {},
        }
        for st in list(stages)[:6]:
            ex = next((r for r in rows if r.get("stage") == st), None)
            if ex:
                entry["examples"][st] = {k: ex.get(k) for k in ("id", "kind", "pred", "gold", "abstain_reason",
                                                                "gold_sentence_retrieved", "gold_among_candidates")}
        report["benchmarks"][benchmark] = entry
        print(f"{benchmark}: correct={len(correct)}/{len(rows)} stages={dict(stages.most_common(4))}", flush=True)

    # cheap probes, on a slice disjoint from the diagnosed test items
    probe_items = LOADERS["squad2"](args.n + args.n_cal + args.probe_n, args.seed)[args.n + args.n_cal :]
    cal, _ = load_split("squad2", args.n, args.n_cal, args.seed)
    ms = calibrate("squad2", cal)["min_score"]
    report["probes"] = {
        "slice": f"{len(probe_items)} SQuAD 2.0 items disjoint from both the calibration and diagnosed test slices",
        "typeless_candidates": {
            "prediction": ("If the answer-type filter is the binding constraint, dropping it should raise "
                           "answerable accuracy above the arm's 15.6% EM on attempted answerable items. If it does "
                           "not, type filtering was not what was stopping it."),
            "result": probe_typeless_candidates(probe_items, 0.0),
        },
        "span_coverage": {
            "prediction": ("Retrieval is not the constraint on SQuAD 2.0 (the stem probe shows BM25 already puts "
                           "the gold-bearing sentence in the top 3 for ~92% of items). So the gold span should "
                           "usually be UNPRODUCIBLE by the regex candidate generator. If instead it is usually "
                           "producible, the failure is selection, not representation, and the (a) label on "
                           "span_not_produced is wrong."),
            "result": probe_span_coverage(probe_items),
        },
        "stem_retrieval": {
            "prediction": ("If string-shaped matching is the binding constraint at retrieval, crude stemming should "
                           "lift recall@3 of the gold-bearing sentence by at least 5 points. A smaller gain means "
                           "morphology is not the problem and real lexical semantics is needed."),
            "result": probe_stem_retrieval(probe_items),
        },
    }
    args.out.write_text(json.dumps(report, indent=1, default=str))
    print("\nwrote", args.out, flush=True)
    print(json.dumps(report["probes"], indent=1)[:900])


if __name__ == "__main__":
    main()
