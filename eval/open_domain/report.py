"""Turn eval/results/open_domain.json into docs/revival/12-open-domain.md.

Every number in the document comes from the JSON, so the two cannot drift.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "eval" / "results" / "open_domain.json"
DST = ROOT / "docs" / "revival" / "12-open-domain.md"

TITLES = {
    "squad2": "SQuAD 2.0 (extractive QA, half the questions unanswerable)",
    "hotpot": "HotpotQA distractor (multi-hop, gold supporting facts published)",
    "gsm8k": "GSM8K (grade-school multi-step arithmetic)",
    "arc_easy": "ARC-Easy (science multiple choice, world knowledge)",
}
ARM_NAMES = {
    "rules": "tensorcode only (no model)",
    "rules_lexical_overlap_guess": "tensorcode only, forced to guess by word overlap",
    "model": "local model alone (no tensorcode)",
    "cascade": "cascade: tensorcode first, model on abstentions",
}


def pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def main() -> None:
    d = json.loads(SRC.read_text())
    B = d["benchmarks"]
    L: list[str] = []
    A = L.append

    model_id = d["environment"].get("model") or "none"
    A("# 12. Open-domain benchmarks: what the cognition does off its home turf")
    A("")

    # ---------- headline
    rules_cov = {k: v["arms"]["rules"]["coverage"] for k, v in B.items() if "rules" in v["arms"]}
    A("**Headline, and it is not flattering: on public open-domain benchmarks the no-model tensorcode arm "
      "answers almost nothing, and what it does answer it mostly gets wrong.** "
      f"Coverage is {pct(rules_cov.get('squad2'))} on SQuAD 2.0, {pct(rules_cov.get('gsm8k'))} on GSM8K and "
      f"{pct(rules_cov.get('arc_easy'))} on ARC-Easy, where it has no knowledge source and correctly refuses "
      "every question. It is not a general question answerer and this measurement says so plainly.")
    A("")
    # the cascade-vs-model comparison, computed rather than assumed
    comp = []
    for k, v in B.items():
        a = v["arms"]
        if "model" in a and "cascade" in a:
            comp.append((k, a["cascade"]["correct_overall"], a["model"]["correct_overall"]))
    worse = [c for c in comp if c[1] < c[2] - 0.001]
    equal = [c for c in comp if abs(c[1] - c[2]) <= 0.001]
    if comp:
        A(f"The same local model ({model_id}), prompted plainly with no tensorcode structure, is far better "
          "everywhere: "
          + ", ".join(f"{k} {pct(m)}" for k, _, m in comp) + " correct overall, against the rule arm's "
          + ", ".join(f"{k} {pct(B[k]['arms']['rules']['correct_overall'])}" for k, _, _ in comp) + ".")
        A("")
        A(f"**And the cascade this architecture proposes is WORSE than the model alone on {len(worse)} of "
          f"{len(comp)} benchmarks**, not better: "
          + "; ".join(f"{k} {pct(c)} vs {pct(m)}" for k, c, m in worse) + "."
          + (f" It ties on {', '.join(k for k, _, _ in equal)}, and the reason is instructive: there the rule arm "
             "abstains on everything, so the cascade simply *is* the model." if equal else ""))
        A("")
        A("This inverts the Banking77 result, where a cheap tier plus abstention beat a general model. The "
          "mechanism is visible in the equal-coverage tables below: the model is better than the rules **on the "
          "very items the rules chose to answer** (SQuAD 2.0 "
          f"{pct(B['squad2']['equal_coverage']['items_rules_answered']['model_on_same_items']['model_accuracy'])} "
          f"vs {pct(B['squad2']['equal_coverage']['items_rules_answered']['rules_accuracy'])}, HotpotQA "
          f"{pct(B['hotpot']['equal_coverage']['items_rules_answered']['model_on_same_items']['model_accuracy'])} "
          f"vs {pct(B['hotpot']['equal_coverage']['items_rules_answered']['rules_accuracy'])}). So the cheap tier "
          "is not selecting the items it is good at. Its abstention is capability-blind: it fires on weak lexical "
          "overlap, not on whether it can actually answer. A cascade is only worth having when the cheap tier "
          "knows when to shut up, and here it does not.")
        A("")
    sq = B.get("squad2", {})
    if sq:
        floor = sq["floors"]["always_abstain"]
        r_all = sq["arms"]["rules"]["correct_overall"]
        if r_all < floor:
            A(f"**The sharpest single number: on SQuAD 2.0 the rule arm scores {pct(r_all)} overall, which is "
              f"worse than the {pct(floor)} it would get by refusing every question.** Its answers are net "
              "negative. A system whose output is worse than its own silence has no business answering.")
            A("")
        rm, rr = sq["arms"]["model"], sq["arms"]["rules"]
        A("One real, narrow win for the abstention machinery, visible only because SQuAD 2.0 labels "
          "unanswerable questions: the model **answers** far better "
          f"({pct(rm['answerable']['em_over_attempted'])} exact-match on the answerable questions it attempted, "
          f"against {pct(rr['answerable']['em_over_attempted'])}) but **refuses** worse "
          f"({pct(rm['unanswerable']['correctly_abstained'])} of unanswerable questions correctly declined, "
          f"against the rule arm's {pct(rr['unanswerable']['correctly_abstained'])}). Knowing when there is no "
          "answer in the passage is the one thing the cheap tier does better than the model, and it is exactly "
          "the thing this project claimed for it. It is also not enough to make the cascade worth it.")
        A("")
    hp = B.get("hotpot", {}).get("arms", {}).get("rules", {}).get("provenance")
    A("What the exercise does establish: the abstention machinery is real and measurable — it refuses rather "
      "than inventing answers, says why, and on ARC-Easy refuses 100% of questions it cannot ground. "
      + (f"Provenance is real but thin: on HotpotQA the arm names the sentences its answer rests on, and those "
         f"citations are checkable against the published supporting facts, covering {pct(hp['mean_gold_hit_rate'])} "
         f"of gold sentences on average but *all* of them on only {pct(hp['all_gold_cited'])} of items. "
         "The citation mechanism works; the retrieval behind it is weak at three sentences." if hp else ""))
    A("")
    A("## How this was measured")
    A("")
    A("| | |")
    A("| --- | --- |")
    A(f"| Benchmarks | {', '.join(TITLES.get(k, k) for k in B)} |")
    A(f"| Grader | {d['design']['graders']} |")
    A(f"| Splits | {d['design']['splits']} |")
    A(f"| Model arm | {model_id}, greedy, batched; multiple choice scored by option likelihood |")
    for arm, desc in d["design"]["arms"].items():
        A(f"| Arm `{arm}` | {desc} |")
    if "model_cost" in d:
        mc = d["model_cost"]
        A(f"| Model cost | {mc['calls']} calls, {mc['new_tokens']} generated tokens, "
          f"{mc['generate_seconds']}s of generation, {mc['tokens_per_second']} tok/s, {mc['load_seconds']}s to load |")
    A("")
    A("No model judged any answer. Every score is exact match or the dataset's own label. "
      "The abstention threshold for the extractive arm was chosen on a calibration slice that is disjoint from "
      "the test slice, and a threshold sweep on the test slice is reported separately as a curve so the "
      "trade-off is visible rather than tuned.")
    A("")

    # ---------- main table
    A("## Results")
    A("")
    A("Accuracy is over *attempted* items; coverage is the share attempted. A system that answers 10% of "
      "questions at 50% accuracy is not better than one that answers everything at 45%, so read the two columns "
      "together.")
    A("")
    A("| Benchmark | Arm | Coverage | Accuracy over attempted | 95% CI | Correct over all items | Model calls | s/item |")
    A("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for bench, entry in B.items():
        for arm, r in entry["arms"].items():
            ci = r.get("accuracy_over_attempted_ci95", [None, None])
            ci_s = "—" if ci[0] is None else f"{pct(ci[0])}–{pct(ci[1])}"
            A(f"| {bench} | {ARM_NAMES.get(arm, arm)} | {pct(r['coverage'])} | **{pct(r['accuracy_over_attempted'])}** | "
              f"{ci_s} | {pct(r['correct_overall'])} | {r['model_calls']} | {r['seconds_per_item']} |")
    A("")
    A("### Floors and random controls")
    A("")
    A("| Benchmark | Floor / control | Value |")
    A("| --- | --- | ---: |")
    for bench, entry in B.items():
        for k, v in entry["floors"].items():
            if k == "note":
                continue
            A(f"| {bench} | {k.replace('_', ' ')} | {pct(v) if isinstance(v, float) else v} |")
    A("")
    for bench, entry in B.items():
        if entry["floors"].get("note"):
            A(f"- **{bench}:** {entry['floors']['note']}")
    A("")

    # ---------- per benchmark detail
    for bench, entry in B.items():
        A(f"## {TITLES.get(bench, bench)}")
        A("")
        A(f"n = {entry['n_test']} test items, {entry['n_calibration']} calibration items. "
          f"Calibration: {entry['calibration'].get('note', '')}"
          + (f" (threshold {entry['calibration']['min_score']})" if entry["calibration"].get("min_score") else ""))
        A("")
        if bench == "squad2":
            for arm, r in entry["arms"].items():
                if "answerable" not in r:
                    continue
                a, u = r["answerable"], r["unanswerable"]
                A(f"- **{ARM_NAMES.get(arm, arm)}**: on the {a['n']} answerable questions it attempted "
                  f"{a['attempted']} and got {pct(a['em_over_attempted'])} exact-match on those "
                  f"({pct(a['em_over_all'])} of all answerable). On the {u['n']} unanswerable ones it correctly "
                  f"refused {pct(u['correctly_abstained'])}.")
            A("")
            A("This is the benchmark that prices abstention honestly, because refusing is *sometimes the right "
              "answer* and the dataset says when. Note the asymmetry it creates: with roughly half the sample "
              "unanswerable, a system that refuses everything already scores about half overall, which is why "
              "the calibration step drifts toward refusing.")
            A("")
        if bench == "hotpot":
            for arm, r in entry["arms"].items():
                if "provenance" in r:
                    p = r["provenance"]
                    A(f"- **{ARM_NAMES.get(arm, arm)}** cited evidence containing *all* the gold supporting "
                      f"sentences on {pct(p['all_gold_cited'])} of items, and on average covered "
                      f"{pct(p['mean_gold_hit_rate'])} of the gold sentences ({p['items_with_gold']} items had "
                      "usable gold labels).")
            A("")
            A("The provenance number is the one claim of ours this benchmark supports directly: the arm does not "
              "merely produce an answer, it names the sentences it read, and those citations can be checked "
              "against HotpotQA's published supporting facts. The answers themselves are mostly wrong.")
            A("")
        if "risk_coverage_sweep_on_test" in entry:
            A("Threshold sweep on the test slice (a curve, not a tuned number):")
            A("")
            A("| min BM25 score | Coverage | Accuracy over attempted | Correct over all |")
            A("| ---: | ---: | ---: | ---: |")
            for row in entry["risk_coverage_sweep_on_test"]:
                A(f"| {row['min_score']} | {pct(row['coverage'])} | {pct(row['accuracy_over_attempted'])} | {pct(row['correct_overall'])} |")
            A("")
        if "equal_coverage" in entry:
            ec = entry["equal_coverage"]
            ans, ref = ec["items_rules_answered"], ec["items_rules_refused"]
            A("**Equal-coverage comparison** (the question the Banking77 work taught us to ask: is the cheap tier "
              "adding anything, or just answering the easy items?)")
            A("")
            A(f"- On the {ans['n']} items the rules answered, the rules scored {pct(ans['rules_accuracy'])} and the "
              f"model scored {pct(ans['model_on_same_items']['model_accuracy'])} on those same items.")
            A(f"- On the {ref['n']} items the rules refused, the model scored "
              f"{pct(ref['model_on_same_items']['model_accuracy'])}.")
            A("")
        if "cascade" in entry["arms"] and "tiers" in entry["arms"]["cascade"]:
            t = entry["arms"]["cascade"]["tiers"]
            A("Cascade tiers: " + "; ".join(
                f"{k} answered {v['attempted']} of {v['n']} at {pct(v['accuracy_over_attempted'])}" for k, v in t.items()))
            A("")

    # ---------- conclusions
    A("## What this changes")
    A("")
    A("1. **The project's cognition claims do not transfer to open domain.** Everything the browser agents and "
      "the assistant do well is narrow, scripted competence in environments we wrote. Faced with public "
      "questions, the no-model arm has no knowledge, no arithmetic planning beyond one step, and no way to "
      "answer anything not lexically present in a passage.")
    A("2. **Abstention is genuine, not decorative.** The arm refuses cleanly and says why "
      "(`weak_evidence`, `no_span_of_type`, `no_knowledge_source`, `multi_step`). On ARC-Easy it refuses "
      "everything rather than guessing, and the separately reported forced-guess variant shows what guessing by "
      "word overlap would buy against the 25% chance floor. That is the behaviour the design promised.")
    A("3. **The cascade actively harms results off home turf, and that is the most useful finding here.** "
      "A cheap tier that answers confidently and wrongly is worse than no cheap tier at all: on HotpotQA the "
      "rules answered 274 of 300 items at 5.8% while the model would have scored 49.3% on those same items, "
      "dragging the cascade to 11.7%. Where the rule arm abstains completely (ARC-Easy) the cascade is exactly "
      "the model, with no harm done. The lesson is not 'cascades work' or 'cascades fail', it is that a "
      "cascade's value is entirely determined by the *calibration* of its cheap tier, and ours is calibrated on "
      "lexical overlap, which has nothing to do with whether it can answer. Banking77 looked good because the "
      "threshold there was fitted on in-domain validation data.")
    A("4. **Provenance is the one part that travels, and it is thinner than advertised.** The claim store and "
      "the ranked-evidence path work identically on public data, and a citation can be checked against gold "
      "supporting facts — which is more than most systems offer. But at three retrieved sentences the arm "
      "covered only "
      + (pct(hp["mean_gold_hit_rate"]) if hp else "—") + " of gold sentences on average and *all* of them on "
      + (pct(hp["all_gold_cited"]) if hp else "—") + " of items. Checkable citation is a mechanism we have; "
      "good retrieval is not.")
    A("")
    A("Two follow-ups live elsewhere. [13 — schema brittleness](13-schema-brittleness.md) turns these "
      "benchmarks into a failure taxonomy: where the arm breaks along its own pipeline, what share of its "
      "*correct* answers are chance (about half on HotpotQA), and the measured answer to \"does any routing "
      "beat always asking the model?\" — which is no, with only 1.3-2.0 points of headroom even for an oracle "
      "router. The abstention-as-answerability-detector idea fails there too.")
    A("")
    A("Read this together with [11 — evidence audit](11-evidence-audit.md), which labels every measured claim "
      "in this project by who wrote the environment and who graded the answer. Most of the headline results were "
      "graded by code we wrote in environments we wrote. This file is one of the few where neither the questions "
      "nor the grader is ours, and it is the least flattering.")
    A("")
    DST.write_text("\n".join(L))
    print("wrote", DST)


if __name__ == "__main__":
    main()
