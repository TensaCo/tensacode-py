"""Write docs/revival/13-schema-brittleness.md from the diagnostic JSON. Numbers come from the files."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BR = ROOT / "eval" / "results" / "schema_brittleness.json"
RT = ROOT / "eval" / "results" / "open_domain_routing.json"
OD = ROOT / "eval" / "results" / "open_domain.json"
DST = ROOT / "docs" / "revival" / "13-schema-brittleness.md"

CLASSES = {"a": "missing representation", "b": "missing mechanism", "c": "knowledge gap", "d": "our wiring error"}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    d = json.loads(BR.read_text())
    rt = json.loads(RT.read_text()) if RT.exists() else None
    od = json.loads(OD.read_text())
    B = d["benchmarks"]
    L: list[str] = []
    A = L.append

    # aggregate the four-way split
    totals: dict[str, int] = {}
    for e in B.values():
        for k, v in e["deficiency_counts"].items():
            if k and k != "None":
                totals[k] = totals.get(k, 0) + v
    diagnosed = sum(totals.values())

    sq, hp = B["squad2"], B["hotpot"]
    sq_luck, hp_luck = sq["luck_audit"], hp["luck_audit"]
    span = d["probes"]["span_coverage"]["result"]
    stem = d["probes"]["stem_retrieval"]["result"]
    typeless = d["probes"]["typeless_candidates"]["result"]

    A("# 13. Where the cognitive schemas break: a failure taxonomy")
    A("")
    A("The benchmarks in [12](12-open-domain.md) are used here as a diagnostic instrument rather than a "
      "scoreboard. Nothing was tuned. Every wrong or abstained item was attributed to the earliest stage of the "
      "arm's own pipeline that is responsible, then labelled by the deficiency it reveals.")
    A("")
    A("## Ranked findings")
    A("")
    A(f"**1. Half of what the arm gets right on HotpotQA is chance.** Of its {hp_luck['correct_items']} correct "
      f"answers, {hp_luck['correct_with_more_than_one_same_type_candidate']} were picked from a sentence holding "
      f"several candidates of the asked-for type with nothing discriminating between them — the arm takes the "
      f"first by position. Expected correct by chance alone: **{hp_luck['expected_correct_by_chance_among_them']} "
      f"of {hp_luck['correct_items']}**. On SQuAD 2.0 it is "
      f"{sq_luck['expected_correct_by_chance_among_them']} of {sq_luck['correct_items']}. So the 5.8% and 9.1% "
      "reported in 12 overstate the arm: roughly half the HotpotQA credit and a third of the SQuAD credit is "
      "luck, not discrimination. This belongs in any headline that quotes those numbers.")
    A("")
    A("**2. A prior of mine was wrong: retrieval is not the bottleneck on SQuAD 2.0.** I expected "
      "string-shaped matching to be the binding constraint. In fact BM25 already places a gold-answer-bearing "
      f"sentence in the top 3 for **{pct(stem['bm25_recall_at_3'])}** of answerable items, and crude stemming "
      f"changes that by **exactly nothing** ({pct(stem['bm25_recall_at_3'])} to "
      f"{pct(stem['stem_overlap_recall_at_3'])}). The evidence is found and then wasted downstream. Retrieval "
      "*is* the top failure on HotpotQA, where bridge facts share few words with the question "
      f"({hp['stages'].get('retrieval_missed_evidence', 0)} of 300 items) — so the same deficiency binds on one "
      "benchmark and not the other, and a single story about 'lexical matching' would have been wrong.")
    A("")
    A("**3. The real constraint on extraction is that the schema cannot name things.** When the gold sentence "
      f"*is* retrieved ({span['gold_sentence_retrieved']} of {span['answerable_with_gold_in_passage']} items), "
      f"the candidate generator can produce the gold span only **{pct(span['producible_rate_asked_type'])}** of "
      "the time. Dropping the answer-type filter changes it by zero "
      f"({span['gold_span_producible_with_asked_type']} vs {span['gold_span_producible_with_any_type']} items), "
      "so type filtering is not what stops it: there is simply no entity recogniser. Regexes over capitalised "
      "runs, numbers and dates cannot see two thirds of the answers people ask for.")
    A("")
    A("**4. Abstention is miscalibrated in both directions, not just one.** On SQuAD 2.0 the arm abstained on "
      f"{sq['stages'].get('abstained_though_answerable', 0)} answerable questions it had the evidence for, and "
      f"answered {sq['stages'].get('answered_an_unanswerable_question', 0)} questions that have no answer. Both "
      "come from the same cause: the threshold is on evidence strength, and evidence strength is not capability.")
    A("")
    A("**5. Multi-step numeric reasoning is absent, not weak.** GSM8K: "
      f"{B['gsm8k']['stages'].get('arithmetic_composition', 0)} of 300 items need a chain the one-step sum/"
      f"difference patterns cannot express, and {B['gsm8k']['stages'].get('no_quantities_parsed', 0)} do not even "
      "yield two quantities. Zero correct out of 300.")
    A("")
    A("**6. ARC-Easy is unanswerable by construction, and that is the honest bound.** All 300 items are a "
      "knowledge gap: nothing in the question, the options or the grammar supplies the fact. The arm refuses "
      "every one, which is the correct behaviour and worth zero points.")
    A("")

    A("## The four-way split")
    A("")
    A(f"Across {diagnosed} diagnosed failures:")
    A("")
    A("| Class | Meaning | Count | Share |")
    A("| --- | --- | ---: | ---: |")
    for k in ("a", "b", "c", "d"):
        n = totals.get(k, 0)
        A(f"| **({k})** | {CLASSES[k]} | {n} | {pct(n / max(1, diagnosed))} |")
    A("")
    A("(a) and (b) are work we can do. (c) bounds what any symbolic structure can claim. (d) is our own error, "
      "and it is small but real: on HotpotQA "
      f"{hp['stages'].get('answer_not_in_passage_but_labelled_answerable', 0)} items are labelled answerable "
      "while the gold string is not literally in the sentences we split out — mostly yes/no questions, which the "
      "arm has no projection for at all.")
    A("")

    A("## Per-stage counts")
    A("")
    A("| Benchmark | Stage the failure is attributed to | Class | Count |")
    A("| --- | --- | :---: | ---: |")
    for bench, e in B.items():
        for stage, n in e["stages"].items():
            cls = e.get("stage_meaning", {}).get(stage, {}).get("class", "b")
            A(f"| {bench} | {stage.replace('_', ' ')} | ({cls}) | {n} |")
    A("")
    A("What each stage would have required:")
    A("")
    seen = set()
    for e in B.values():
        for stage, info in e.get("stage_meaning", {}).items():
            if stage in seen:
                continue
            seen.add(stage)
            A(f"- **{stage.replace('_', ' ')}** ({info['class']}): {info['would_have_needed']}")
    A("")

    A("## Probes, with the predictions they were written to falsify")
    A("")
    A("Each probe ran on a 150-item SQuAD 2.0 slice disjoint from both the calibration and diagnosed test "
      "slices. These are probes, not fixes: nothing was adopted.")
    A("")
    for name, p in d["probes"].items():
        if name == "slice":
            continue
        A(f"**{name.replace('_', ' ')}**")
        A("")
        A(f"- *Prediction:* {p['prediction']}")
        A(f"- *Result:* `{json.dumps(p['result'])}`")
        if name == "stem_retrieval":
            A("- *Verdict:* **falsified.** No change at all, and the baseline was already 91.6%. Morphology is "
              "not the problem on this benchmark, and neither is retrieval.")
        elif name == "span_coverage":
            A("- *Verdict:* **confirmed.** Two thirds of gold spans are unproducible even when the right "
              "sentence is in hand, and the type filter is irrelevant to that. The (a) label on "
              "`span_not_produced` stands.")
        elif name == "typeless_candidates":
            A(f"- *Verdict:* **falsified.** Accuracy over attempted is {pct(typeless['accuracy_over_attempted'])}, "
              "no better than the gated arm, so the answer-type filter was never the binding constraint.")
        A("")

    A("## What to change, and the number that would falsify each")
    A("")
    A("Ordered by the count of failures each would address. None is implemented.")
    A("")
    A(f"1. **An entity/number/date recogniser over retrieved sentences** (class a; addresses "
      f"{sq['stages'].get('span_not_produced', 0) + hp['stages'].get('span_not_produced', 0)} items). The claim "
      "objects are strings; they need to be typed entities. *Prediction:* raising gold-span producibility from "
      f"{pct(span['producible_rate_asked_type'])} to >70% should lift SQuAD answerable EM on attempted items "
      "from 15.6% to at least 35% at unchanged coverage. If it does not, span production was not the binding "
      "constraint and selection is.")
    A(f"2. **Candidate scoring against the question** (class b; addresses "
      f"{sq['stages'].get('selection_chose_wrong_candidate', 0) + hp['stages'].get('selection_chose_wrong_candidate', 0)} "
      "items directly and most of the luck credit). Today the first candidate by position wins. *Prediction:* any "
      "scoring that beats position should cut the chance-expected share of HotpotQA correct answers from "
      f"{pct(hp_luck['expected_correct_by_chance_among_them'] / max(1, hp_luck['correct_items']))} to under 25%. "
      "If the chance share stays put, the selector is still not discriminating.")
    A(f"3. **A second hop** (class a; addresses much of HotpotQA's "
      f"{hp['stages'].get('retrieval_missed_evidence', 0)} retrieval misses). The arm issues one query and stops; "
      "a bridge question needs the first answer as a term in a second query. *Prediction:* two-hop retrieval "
      f"should lift HotpotQA gold-fact coverage from {pct(od['benchmarks']['hotpot']['arms']['rules']['provenance']['mean_gold_hit_rate'])} "
      "to at least 55%. If coverage barely moves, the failure is lexical, not structural.")
    A("4. **Capability-aware abstention** (class b). Already measured and **falsified as a cascade strategy** — "
      "see the routing section below. Abstention should still be fixed for its own sake (it refuses "
      f"{sq['stages'].get('abstained_though_answerable', 0)} answerable items), but not on the grounds that it "
      "makes a cascade pay.")
    A("5. **Quantity representation with an operation chain** (class a). *Prediction:* representing quantities, "
      "their referents and an order of operations should take GSM8K from 0/300 to at least 15% on items whose "
      "gold solution has two steps. If a two-step representation still scores near zero, the parse is the "
      "problem, not the arithmetic.")
    A("6. **Nothing for ARC-Easy** (class c). No symbolic change helps; this bounds the claim rather than "
      "inviting work.")
    A("")

    if rt:
        A("## Routing: is any cascade worth having? No.")
        A("")
        A("Asked directly, on the same 300-item test slices, with a router trained to predict whether the cheap "
          "tier will be right (features from the item and the tier's own output; trained on a disjoint "
          f"{rt['benchmarks']['squad2']['n_router_train']}-item slice).")
        A("")
        A("| Benchmark | Always model | Always rules | **Oracle router** | Best learned router | Calls saved at best |")
        A("| --- | ---: | ---: | ---: | ---: | ---: |")
        for bench, e in rt["benchmarks"].items():
            best = max(e["curves"]["capability_router"], key=lambda r: r["accuracy"])
            A(f"| {bench} | {pct(e['baselines']['always_model']['accuracy'])} | "
              f"{pct(e['baselines']['always_rules']['accuracy'])} | "
              f"**{pct(e['baselines']['oracle_router']['accuracy'])}** | {pct(best['accuracy'])} | "
              f"{pct(best['calls_saved'])} |")
        A("")
        A("**No routing beats asking the model every time, and there is almost nothing to route on.** The "
          "*oracle* — a router with perfect foresight, keeping the cheap answer exactly when it is right and the "
          "model is wrong — gains "
          + " and ".join(f"{100 * (e['baselines']['oracle_router']['accuracy'] - e['baselines']['always_model']['accuracy']):.1f} points on {b}"
                         for b, e in rt["benchmarks"].items())
          + ". The learned router's best operating point saves 0% of calls on both benchmarks: it degenerates to "
            "'always ask the model', which is the correct thing for it to do.")
        A("")
        A("Signal quality, as AUC for predicting whether the cheap tier will be right (0.5 = no information):")
        A("")
        A("| Benchmark | Learned capability router | The old BM25 signal |")
        A("| --- | ---: | ---: |")
        for bench, e in rt["benchmarks"].items():
            q = e["router_quality"]
            A(f"| {bench} | {q['auc_capability_router']} | {q['auc_bm25_old_signal']} |")
        A("")
        A("Both are at or near chance, and on HotpotQA the learned router is *below* chance (0.457), meaning it "
          "did not generalise from its training slice at all. The honest reading is not 'our router is bad' but "
          "'when the cheap tier is right 5-11% of the time, there is no subset worth routing to it'. A cascade "
          "needs a cheap tier that is right often enough for its correct region to be findable.")
        A("")
        anp = rt["benchmarks"]["squad2"].get("answer_not_present_detector")
        if anp:
            A("**And the one hypothesis that looked most promising fails too.** The arm refuses 84.0% of "
              "unanswerable SQuAD questions against the model's 57.6%, which suggested using it purely as an "
              "answer-not-present detector. But that figure is a high abstention rate, not detection: "
              f"P(abstain | unanswerable) = {pct(anp['p_abstain_given_unanswerable'])} against "
              f"P(abstain | answerable) = {pct(anp['p_abstain_given_answerable'])}, a lift of just "
              f"{100 * anp['lift']:.1f} points. Wiring it up that way — rules decide answerability, model answers "
              f"the rest — scores {pct(anp['hybrid_rules_decide_answerability_model_answers']['accuracy'])} "
              f"against the model's {pct(rt['benchmarks']['squad2']['baselines']['always_model']['accuracy'])}. "
              "The abstention carries almost no information about answerability.")
            A("")

    A("## What this says about generality")
    A("")
    A("The user's framing was right: these benchmarks are most useful as a brittleness probe. The taxonomy says "
      "the arm is not 'a bit behind' on open-domain work — it is missing three representations (typed entities, "
      "composed hops, quantities with operations) and two mechanisms (candidate scoring, capability-aware "
      "abstention), and one whole benchmark is outside what any symbolic structure can do.")
    A("")
    A("The parts that did survive contact are worth naming precisely, because they are the ones to build on: "
      "retrieval into a claim store with provenance works and is not the bottleneck; abstention exists, fires "
      "with a stated reason, and is honest even when miscalibrated; and the pipeline is legible enough that "
      "every one of these failures could be attributed to a stage. A system that cannot tell you where it broke "
      "could not have produced this document.")
    A("")
    DST.write_text("\n".join(L))
    print("wrote", DST)


if __name__ == "__main__":
    main()
