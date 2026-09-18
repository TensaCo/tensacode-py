"""Compute a cognitive profile: many axes, each labelled with how much it is worth.

    python -m eval.profile.profile [--probes DIR]

There is deliberately no single score. A single number is what invited the problems the
evidence audit found (docs/revival/11-evidence-audit.md), so every row carries its own
floor, its control, who wrote the environment, who wrote the grader, and whether the test
was fixed before tuning. An axis that cannot be grounded is reported as ungrounded with
the reason, never as a zero and never omitted.
"""

from __future__ import annotations

import os

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "eval" / "results"


# ----------------------------------------------------------------- statistics


def wilson(correct: float, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% interval for a proportion. None when there is nothing to bound."""
    if n <= 0:
        return None
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def luck_corrected(correct: float, n: int, expected_by_chance: float) -> float | None:
    """Accuracy with the credit chance would have got removed. Standard here, not a footnote."""
    if n <= 0:
        return None
    return round(max(0.0, (correct - expected_by_chance)) / n, 4)


# --------------------------------------------------------------------- rows


@dataclass
class Measure:
    """One number, and everything needed to judge how much it is worth."""

    name: str
    value: str
    n: int | None = None
    ci95: tuple[float, float] | None = None
    floor: str = "none run"
    control: str = "none run"
    chance_corrected: str | None = None
    env_author: str = "us"
    grader: str = "our own code"
    agent_can_influence_grader: str = "no"
    heldout: str = "not stated"
    source: str = ""
    note: str = ""


@dataclass
class Axis:
    key: str
    question: str
    grounded: bool
    verdict: str  # strong | moderate | weak | ungrounded
    measures: list[Measure] = field(default_factory=list)
    ungrounded_reason: str = ""
    reading: str = ""


def load(name: str) -> dict | list | None:
    path = RESULTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------- axes


def axis_transfer() -> Axis:
    vis, role, rec, cwe = load("vision_perception.json"), load("learned_role_model.report.json"), load("recognizer_comparison.json"), load("cw_pixel_e2e.json")
    ms: list[Measure] = []
    if vis:
        configs = vis["results"]
        full = next((v for k, v in configs.items() if k.startswith("full")), None)
        r = full or {}
        ms.append(Measure(
            "visible text read, hand-written perceiver: tuned apps -> unseen apps -> unseen OS",
            f"{r['tune']['words']['recall_exact']:.3f} -> {r['test_app']['words']['recall_exact']:.3f} -> {r['test_os']['words']['recall_exact']:.3f}",
            n=r["test_os"]["words"]["gt"], floor="n/a (recall of a known word list)",
            control="two ablations run (no icon memory, no text candidates)",
            heldout="apps and OS split before the rules were written; some rule edits came after held-out inspection (audit: perception-fusion)",
            source="eval/results/vision_perception.json", note="degrades monotonically as the environment moves away from where it was tuned"))
        t = {sp: r[sp]["targeting_by_name"] for sp in ("tune", "test_app", "test_os")}
        ms.append(Measure(
            "clicking the right control by name, hand-written perceiver: tuned -> unseen apps -> unseen OS",
            "text " + " -> ".join(f"{t[sp]['text']['clicks_right_control']:.3f}" for sp in t)
            + " | icon " + " -> ".join(f"{t[sp]['icon']['clicks_right_control']:.3f}" for sp in t)
            + " | field " + " -> ".join(f"{t[sp]['field']['clicks_right_control']:.3f}" for sp in t),
            n=sum(t["test_os"][k]["queries"] for k in ("text", "icon", "field")),
            floor="wrong-click rate reported alongside (text 0.106 on unseen apps)",
            control="three ablations; DOM perceiver as the reference arm",
            heldout="apps and OS split before the rules were written",
            source="eval/results/vision_perception.json",
            note="text-labelled controls transfer (0.74 -> 0.72); icons and outside-the-box field labels do not (0.24 and 0.09 on unseen OS)"))
    if role:
        s = role["splits"]
        ms.append(Measure(
            "learned role model, textbox recall: trained apps -> unseen apps -> unseen OS",
            f"{s['tune']['per_class']['textbox']['recall']:.3f} -> {s['test_app']['per_class']['textbox']['recall']:.3f} -> {s['test_os']['per_class']['textbox']['recall']:.3f}",
            n=s["test_os"]["n"], floor=f"majority class {s['test_os']['majority_baseline']:.3f} (which the model is BELOW overall on unseen OS: {s['test_os']['accuracy']:.3f})",
            control="majority baseline per split", heldout="apps/OS held out before training",
            source="eval/results/learned_role_model.report.json",
            note="the clearest training win in the repo, and it states its own cost: better fields, worse overall accuracy than guessing the majority on unseen OS"))
    if rec:
        base, wide = rec["baseline"], rec["apps+rendered fine-tune"]
        narrow = rec["apps-only fine-tune"]
        ms.append(Measure(
            "OCR fine-tune, exact word accuracy on unseen FONTS (narrow corpus vs broad)",
            f"baseline {base['held-out fonts']['exact']:.3f} | apps-only {narrow['held-out fonts']['exact']:.3f} | apps+rendered {wide['held-out fonts']['exact']:.3f}",
            n=base["held-out fonts"]["n"], floor="n/a", control="three corpora compared on one held-out split",
            heldout="2 fonts and 8 apps never trained on",
            source="eval/results/recognizer_comparison.json",
            note="training on the narrow corpus went BELOW baseline on unseen fonts; only the broad corpus transferred"))
    if cwe:
        arms = cwe["arms"]
        rows = []
        for arm in arms.values():
            a, b = arm["splits"]["A"], arm["splits"]["B"]
            rows.append(f"{arm['label']}: A {a['verified']}/{a['episodes']} B {b['verified']}/{b['episodes']}")
        ms.append(Measure(
            "the same OCR fix moved to a different engine (episodes verified, debugged seeds vs held out)",
            " | ".join(rows), n=20, floor="n/a",
            control="two splits, one never inspected while fixing; three recognizers compared",
            heldout="split B never looked at while writing the fixes",
            env_author="computerworld (third party) + our task",
            source="eval/results/cw_pixel_e2e.json",
            note="the 2->0 false-success gain from the fine-tune did NOT travel: all three recognizers make the same three errors here, because the failure mode changed (lost hyphens, not digits)"))
    return Axis("transfer", "Does a capability survive moving to material it was not built on?", bool(ms), "moderate" if ms else "ungrounded",
                ms, "" if ms else "no transfer measurements on disk",
                reading="Hand-written perception degrades gradually; a learned component transferred best when its training data was widened deliberately; and a fix validated in one environment did not survive a change of engine. Transfer is measured, and it is the axis where our claims have most often been too strong.")


def axis_horizon() -> Axis:
    rec, ranker, lh = load("browser_agents_recovered.json"), load("learned_intention_ranker.json"), load("longhorizon.json")
    ms: list[Measure] = []
    if rec:
        runs = rec["runs"]
        biggest = max(runs.items(), key=lambda kv: kv[1]["episodes"])
        name, big = biggest
        others = " | ".join(f"{k} {v['item_accuracy']:.4f} over {v['episodes']} episodes" for k, v in runs.items() if k != name)
        correct, items = big["items_correct"], big["items"]
        ms.append(Measure(
            "hand-written objective, items correct over long continuous running",
            f"{correct}/{items} = {correct / items:.4f} over {big['episodes']} episodes ({name}); other runs: {others}",
            n=items, ci95=wilson(correct, items), floor="none run (no random-policy control on these apps)",
            control="none run", heldout="live-wall seeds differ from bench seeds; apps randomize themselves",
            source="eval/results/browser_agents_recovered.json",
            note="strongest capability number we have, and both environment and grader are ours (audit verdict: weak evidence)"))
    if ranker:
        tasks = ranker["tasks"]
        parts = []
        for name, arms in tasks.items():
            hw, lr = arms["hand_written_objective"], arms["learned_ranker"]
            parts.append(f"{name}: hand {hw['item_accuracy']:.2f} vs learned {lr['item_accuracy']:.2f} (mean actions {hw['mean_actions']:.0f})")
        ms.append(Measure(
            "learned ranker vs hand-written objective in closed loop",
            " | ".join(parts), n=sum(a["hand_written_objective"]["items"] for a in tasks.values()),
            floor="hand-written objective as the reference arm", control="same seeds both arms",
            heldout="seeds 9001+ not used in training",
            source="eval/results/learned_intention_ranker.json",
            note="0.79 per-decision accuracy over 20-45 sequential decisions leaves ~1% of episodes intact: the horizon multiplies errors, so per-decision accuracy near 0.999 is what these tasks demand"))
    if lh:
        rows = [t for run in lh["runs"].values() for t in run["tasks"]]
        passed = sum(1 for t in rows if t.get("passed"))
        ms.append(Measure(
            "long-horizon tasks with hidden graders (CAD / EEG / coding)",
            f"{passed}/{len(rows)} passed", n=len(rows), floor="0 expected from an agent with no such skills",
            control="hidden test sets the agent cannot read", heldout="tasks authored before the runs; one held-out split",
            grader="hidden tests (independent of the agent)", agent_can_influence_grader="no",
            source="eval/results/longhorizon.json",
            note="runs were throughput-bound: both arms hit a 30-minute cap after ~9 model calls at ~200 s per call, so this bounds nothing about the architecture yet"))
    return Axis("horizon", "How does success scale with the number of sequential decisions?", bool(ms), "strong" if len(ms) >= 3 else "moderate",
                ms, reading="The hand-written objective holds ~0.986 across ~103k episodes of 20-45 decisions; a cloned policy at 0.79 per decision collapses to ~1% of episodes. Nothing we have completes a genuinely long, open task: the long-horizon suite scored 0, and its runs were throughput-bound rather than capability-bound.")


def axis_sample_efficiency() -> Axis:
    scratch = Path(os.environ.get("TENSORCODE_SCRATCH", os.path.expanduser("~/.cache/tensorcode")))
    libs = sorted(scratch.glob("skills*.json"))
    skills = []
    for p in libs:
        try:
            skills += [(p.name, s) for s in json.loads(p.read_text())]
        except (json.JSONDecodeError, OSError):
            continue
    if not skills:
        return Axis("sample_efficiency", "How many examples does a new skill take?", False, "ungrounded",
                    [], "no skill library on disk")
    adopted = [s for _, s in skills if s.get("status") == "adopted"]
    uses = sum((s.get("stats") or {}).get("uses", 0) for _, s in skills)
    ms = [Measure(
        "skills learned from one demonstration, and whether they were ever reused",
        f"{len(skills)} learned ({len(adopted)} adopted, {len(skills) - len(adopted)} still on trial); {uses} total reuses",
        n=len(skills), floor="n/a", control="trial->adopted requires success on a request it was not learned from",
        heldout="adoption requires different slot values", source=", ".join(p.name for p in libs),
        note="one demonstration per skill is the design; the sample is far too small to quote a rate")]
    return Axis("sample_efficiency", "How many examples does a new skill take?", False, "ungrounded", ms,
                f"n={len(skills)} skills in total: enough to show the mechanism runs, far too few for a rate. The teacher model is not running, so no new learning could be measured now.",
                reading="The one-demonstration mechanism works and has a real adoption gate, but with two skills ever learned this axis is a demonstration, not a measurement.")


def axis_calibration() -> Axis:
    od, br, cwe, inv = load("open_domain.json"), load("schema_brittleness.json"), load("cw_pixel_e2e.json"), load("invariant_eval.json")
    ms: list[Measure] = []
    if od and br:
        sq = od["benchmarks"]["squad2"]
        luck = br["benchmarks"]["squad2"]["luck_audit"]
        rules = sq["arms"]["rules"]
        corrected = luck_corrected(luck["correct_items"], rules["n"] * rules["coverage"], luck["expected_correct_by_chance_among_them"])
        ms.append(Measure(
            "SQuAD 2.0, rules arm: accuracy over attempted, and what is left after chance",
            f"{rules['accuracy_over_attempted']:.4f} raw; {corrected:.4f} after removing chance credit",
            n=int(round(rules["n"] * rules["coverage"])), ci95=wilson(luck["correct_items"], int(round(rules["n"] * rules["coverage"]))),
            floor=f"always-abstain scores {sq['floors']['always_abstain']:.2f} overall; the arm scores {rules['correct_overall']:.2f}",
            control="random-span floor 0.0133; model arm on the same items",
            chance_corrected=f"{luck['expected_correct_by_chance_among_them']} of {luck['correct_items']} correct answers were expected by chance",
            env_author="public dataset (SQuAD 2.0)", grader="public labels", agent_can_influence_grader="no",
            heldout="disjoint 150-item calibration and 300-item test slices",
            source="eval/results/open_domain.json, eval/results/schema_brittleness.json",
            note="the arm's answers are net negative: refusing everything scores higher than answering"))
        stages = br["benchmarks"]["squad2"]["stages"]
        ms.append(Measure(
            "the two error types kept apart, on SQuAD 2.0",
            f"{stages.get('abstained_though_answerable', 0)} wrongful refusals vs {stages.get('answered_an_unanswerable_question', 0)} answers to unanswerable questions",
            n=300, floor="n/a", control="n/a", env_author="public dataset", grader="public labels",
            agent_can_influence_grader="no", heldout="test slice",
            source="eval/results/schema_brittleness.json",
            note="miscalibrated in both directions, so abstention is not simply conservative"))
    if od:
        hp = od["benchmarks"]["hotpot"]["equal_coverage"]["items_rules_answered"]
        ms.append(Measure(
            "HotpotQA: does the cheap tier beat the model on the items it chose to answer?",
            f"rules {hp['rules_accuracy']:.4f} vs model {hp['model_on_same_items']['model_accuracy']:.4f} on the same {hp['n']} items",
            n=hp["n"], ci95=wilson(hp["rules_accuracy"] * hp["n"], hp["n"]), floor="chance ~0.007 for a free-form span",
            control="model arm restricted to exactly the items the rules attempted",
            env_author="public dataset (HotpotQA)", grader="public labels", agent_can_influence_grader="no",
            heldout="300-item test slice", source="eval/results/open_domain.json",
            note="the abstention is capability-blind: where it commits, it is far worse than the tier it was meant to protect"))
    if cwe:
        arms = cwe["arms"]
        fs = {a["label"]: (len(a["splits"]["A"]["false_successes"]), len(a["splits"]["B"]["false_successes"]), len(a["splits"]["A"].get("refused", [])), len(a["splits"]["B"].get("refused", []))) for a in arms.values()}
        best = [k for k, v in fs.items() if v[0] == 0 and v[1] == 0 and "invar" in k]
        ms.append(Measure(
            "false successes (claiming a result that did not happen), pixels only",
            " | ".join(f"{k}: {v[0]}+{v[1]} false, {v[2]}+{v[3]} refusals" for k, v in fs.items()),
            n=20, floor="the structured-scene arm has 0 false successes on both splits",
            control="four arms, two splits, one never inspected while fixing",
            env_author="computerworld (third party) + our task", heldout="split B held out",
            source="eval/results/cw_pixel_e2e.json",
            note=f"task-declared invariants remove them ({best[0] if best else 'n/a'}); the recognizer fine-tunes did not"))
    if isinstance(inv, dict) and inv:
        parts = []
        for arm, rows in inv.items():
            if not isinstance(rows, list):
                continue
            refuse = sum(1 for r in rows if r.get("would_refuse"))
            ok = sum(1 for r in rows if r.get("corroborated_correctly"))
            wrong = sum(1 for r in rows if r.get("corroborated") and not r.get("corroborated_correctly"))
            parts.append(f"{arm}: {refuse} refused, {ok} corroborated correctly, {wrong} corroborated wrongly, of {len(rows)}")
        if parts:
            ms.append(Measure(
                "task-declared invariants on the recorded failure frames",
                " | ".join(parts), n=11, floor="n/a",
                control="checked on the frames that had already failed, including the false-success frame",
                env_author="computerworld (third party) + our task", heldout="frames recorded before the invariants were written",
                source="eval/results/invariant_eval.json",
                note="it refuses the frame that produced a false success and corroborates nothing wrongly; n=11, so this is a demonstration rather than a rate"))
    return Axis("calibration_and_abstention", "When it does not know, does it know that?", bool(ms), "strong" if len(ms) >= 3 else "moderate", ms,
                reading="Abstention exists, is measurable, and refuses rather than inventing. But it is keyed to evidence strength rather than to capability: on public data the cheap tier is worse than the model precisely where it commits, it refuses 54 answerable questions, and its answers are net negative on SQuAD 2.0. False successes are the one error type we have reduced to zero, and structure (declared invariants) did that, not training.")


def axis_compositionality(probes: dict) -> Axis:
    rows = probes.get("compositionality") or []
    if not rows:
        return Axis("compositionality", "How far does accuracy hold as clauses are chained in one message?", False, "ungrounded", [], "probe did not run")
    ms = []
    total_clauses = sum(len(r["checks"]) for r in rows)
    ok_clauses = sum(sum(1 for v in r["checks"].values() if v) for r in rows)
    ladder = " | ".join(f"depth {i + 1}: {sum(1 for v in r['checks'].values() if v)}/{len(r['checks'])}" for i, r in enumerate(rows))
    ms.append(Measure(
        "clauses carried out correctly, by chained-clause depth in ONE message",
        ladder, n=total_clauses, ci95=wilson(ok_clauses, total_clauses),
        floor="n/a (each clause is checked against the machine's own state)",
        control="the same five clauses, measured at every depth",
        grader="the simulator's shell, read by a separate process",
        agent_can_influence_grader="no (it cannot see or change the checks)",
        heldout="fresh names per run; the ladder was written before it was run",
        source="live probes against the previous assistant (probe harness since removed)",
        note=f"{ok_clauses}/{total_clauses} clauses; the only failure is the 5th clause, which is a broken `copy` act rather than a composition failure"))
    ms.append(Measure(
        "does a single broken clause end the conversation?",
        "yes: after the failing clause, every later turn in that conversation also fails",
        n=1, floor="n/a", control="the same later turns succeed in a fresh conversation",
        grader="observed replies", heldout="n/a",
        source="live probes against the previous assistant, reproduced three times (probe harness since removed)",
        note="the crashed request stays selected forever, so there is no recovery short of a restart"))
    return Axis("compositionality", "How far does accuracy hold as clauses are chained in one message?", True, "moderate", ms,
                reading="Composition itself holds: four chained clauses in one message were carried out correctly, each verified against the machine. The 5-clause case fails on a broken act, not on depth. The more serious finding is recovery: one failing clause poisons the rest of the conversation permanently.")


def axis_world_modeling() -> Axis:
    exp, cau = load("structures_expectation.json"), load("structures_causal.json")
    ms: list[Measure] = []
    if exp:
        best = max(exp["arms"].items(), key=lambda kv: kv[1]["accuracy_all_aspects"])
        worst = min(exp["arms"].items(), key=lambda kv: kv[1]["accuracy_all_aspects"])
        name, arm = best
        ms.append(Measure(
            "predicting the screen BEFORE acting, then scoring the prediction",
            f"best arm ({name}) {arm['accuracy_all_aspects']:.4f}; first half {arm['accuracy_first_half']:.4f} -> second half {arm['accuracy_second_half']:.4f}",
            n=arm["attempted"], ci95=wilson(arm["accuracy_all_aspects"] * arm["attempted"], arm["attempted"]),
            floor=f"the absolute-value framing scores {worst[1]['accuracy_all_aspects']:.4f} on the same steps",
            control="four framings compared (absolute vs change, with and without state)",
            env_author="computerworld (third party)", grader="the engine's own next state",
            agent_can_influence_grader="no", heldout="predictions are made before the action is taken",
            source="eval/results/structures_expectation.json",
            note="ground truth is free and exact here; accuracy rises with experience, and it refuses to predict where it has too few trials"))
    if cau:
        clock = cau["clock_case"]
        ms.append(Measure(
            "telling cause from correlation using interventions",
            f"{cau['arms']['active']['causal_links']} causal links from active controls vs "
            f"{cau['correlational']['links']} from correlation alone; every aspect tested came back "
            f"'{sorted(set(cau['verdicts_vs_correlation'].values()))[0]}'",
            n=cau["arms"]["active"]["contrasts"], floor="correlational baseline finds "
            f"{cau['correlational']['links']} links from {cau['correlational']['observations']} observations",
            control=f"label shuffle: {cau['shuffle_control']['links_shuffled']} links, {cau['shuffle_control']['misattributed']} misattributed",
            env_author="computerworld (third party)", grader="the engine's own state under fork/restore",
            agent_can_influence_grader="no", heldout="held-out interventions",
            source="eval/results/structures_causal.json",
            note="the same state run with and without an action is a real controlled experiment, which the engine's fork/restore makes possible. "
                 f"The clock case, which was meant to show a co-occurring-but-uncaused signal being rejected, did NOT demonstrate it: "
                 f"the clock never entered the correlational links (clock_in_correlational_links="
                 f"{clock['clock_in_correlational_links']}) and BOTH framings measured an effect of "
                 f"{clock['clock_effect_passive_control']}, so there is no non-zero contrast here to speak of"))
    return Axis("world_modeling", "Can it predict what its own actions will do?", bool(ms), "moderate", ms,
                "" if ms else "the expectation work had not landed when this ran",
                reading="Prediction is the best-graded thing in the whole profile: the environment scores it, we do not, and it improves with experience "
                        "(0.745 -> 0.841 within one run). The framing matters more than the mechanism: predicting CHANGE works (~0.79) while predicting "
                        "absolute values fails (~0.06), which is a statement about representation, not about effort. The causal half is NOT yet demonstrated: "
                        "intervention found 14 links against correlation's 13, every aspect was judged 'caused', and the case designed to show a co-occurring "
                        "signal being rejected produced a zero effect in both framings. Prediction: strong. Causal discrimination: unproven.")


def axis_belief_revision(probes: dict) -> Axis:
    rows = probes.get("belief_revision") or []
    if not rows:
        return Axis("belief_revision", "Does a new fact replace an old one, and does the world override memory?", False, "ungrounded", [], "probe did not run")
    passed = sum(1 for r in rows if r["passed"])
    ms = [Measure(
        "told, replaced, forgotten, and re-perceived after a change made behind its back",
        f"{passed}/{len(rows)} cases pass: " + ", ".join(f"{r['name']}={'ok' if r['passed'] else 'FAIL'}" for r in rows),
        n=sum(len(r["checks"]) for r in rows), floor="n/a",
        control="the stale-belief case changes the world outside the agent and re-asks",
        grader="the simulator's shell, read by a separate process",
        agent_can_influence_grader="no", heldout="written before it was run; fresh conversation",
        source="live probes against the previous assistant (probe harness since removed)",
        note="the strongest case is the last: a folder it created was deleted behind its back, and it did not report it afterwards")]
    return Axis("belief_revision", "Does a new fact replace an old one, and does the world override memory?", True, "moderate", ms,
                reading="All four cases pass, including the one with independent ground truth: it re-perceives rather than trusting what it did earlier. This is what snapshot scopes were for, and it is the axis where the architecture most visibly earns its keep.")


def axis_grounding(probes: dict) -> Axis:
    rows = probes.get("grounding") or []
    ms: list[Measure] = []
    od = load("open_domain.json")
    if rows:
        passed = sum(1 for r in rows if r["passed"])
        ms.append(Measure(
            "does the source an answer cites actually support it?",
            f"{passed}/{len(rows)} cases pass: " + ", ".join(f"{r['name']}={'ok' if r['passed'] else 'FAIL'}" for r in rows),
            n=sum(len(r["checks"]) for r in rows), floor="n/a",
            control="the dock count is checked against our own DOM read, on a different code path from the agent's perceiver",
            grader="our own DOM read plus facts we told it",
            agent_can_influence_grader="no", heldout="written before it was run",
            source="live probes against the previous assistant (probe harness since removed)",
            note="'how many icons' has two defensible answers (13 launchers, or 14 including the app-grid button); it gave 13 and named all 13, so the row scores either reading and says so"))
    ms.append(Measure(
        "citation coverage on public data (HotpotQA supporting facts)",
        "31.6% of gold supporting sentences covered on average; ALL of them on 3.3% of items",
        n=300, floor="n/a", control="citations checked against published supporting facts",
        env_author="public dataset (HotpotQA)", grader="public labels", agent_can_influence_grader="no",
        heldout="300-item test slice", source="docs/revival/12-open-domain.md, eval/results/open_domain.json",
        note="the citation mechanism is real and checkable; the retrieval behind it is thin"))
    return Axis("grounding", "Is an answer traceable to something that actually supports it?", True, "moderate", ms,
                reading="Every answer the assistant gives now names its source, and where we can check the source independently it holds. On public multi-hop data the same mechanism covers under a third of the gold evidence, so checkable citation is a mechanism we have and good retrieval is not.")


def axis_robustness() -> Axis:
    rec, chart, ties = load("browser_agents_recovered.json"), load("chart_environment_sweep400.json"), load("chart_environment_tie_seeds.json")
    ms: list[Measure] = []
    if rec:
        runs = rec["runs"]
        per_task: dict[str, tuple[int, int]] = {}
        for r in runs.values():
            for name, t in r["tasks"].items():
                c, i = per_task.get(name, (0, 0))
                per_task[name] = (c + t["correct"], i + t["items"])
        ms.append(Measure(
            "accuracy per task under each app's own randomization (label variants, injected 503s, dialogs, pagination)",
            " | ".join(f"{k} {c / i:.4f}" for k, (c, i) in sorted(per_task.items())),
            n=sum(i for _, i in per_task.values()), floor="none run",
            control="none run", heldout="the apps randomize every episode; live seeds differ from bench seeds",
            source="eval/results/browser_agents_recovered.json",
            note="inbox is the weakest at ~0.944 and its misses are abstentions; the perturbations are ones we built, so this measures robustness to anticipated variation"))
    if chart and ties:
        allow, avoid = chart["configs"]["allow"], chart["configs"]["avoid"]
        tallow = ties["configs"]["allow"]
        ms.append(Measure(
            "the one environment we changed after a failure, restored and re-measured",
            f"original generator {allow['item_accuracy']:.4f} with {allow['abstained_near_tie']} abstentions and {allow['answered_and_wrong']} wrong; changed generator {avoid['item_accuracy']:.4f}",
            n=allow["items"], ci95=wilson(allow["items_correct"], allow["items"]),
            floor="n/a", control=f"on the 5 known tie seeds the original scores {tallow['item_accuracy']:.2f} with {tallow['abstained_near_tie']} abstentions and {tallow['answered_and_wrong']} wrong",
            heldout="400 fresh seeds never used for tuning",
            source="eval/results/chart_environment_sweep400.json, chart_environment_tie_seeds.json",
            note="the post-hoc change was worth 0.5 points; where it bites, the agent abstains every time instead of guessing"))
    return Axis("robustness", "Does it hold up under perturbation?", bool(ms), "moderate", ms,
                reading="It holds up well against the variations the apps were built to throw (0.944-1.000 per task across ~103k episodes), and it abstains rather than guessing on the genuinely unanswerable case. But the perturbations are ones we anticipated, so this is robustness to known unknowns.")


def axis_cost() -> Axis:
    audit, od, cw = load("evidence_audit.json"), load("open_domain.json"), load("computerworld.json")
    ms: list[Measure] = []
    if audit:
        s = audit["summary"]
        ms.append(Measure(
            "share of our own headline results graded by someone other than us",
            f"{s['headline_public_or_hidden_grader']} headline rows have a public or hidden grader; {s['headline_both_ours']} have both environment and grader ours",
            n=s["headline_rows"], floor="n/a", control="n/a",
            grader="the audit reads the result files themselves", agent_can_influence_grader="no",
            heldout="n/a", source="eval/results/evidence_audit.json",
            note=f"verdicts across all rows: {json.dumps(s['verdicts'])}; post-hoc environment changes: {s['post_hoc_environment_changes']}"))
    if od:
        mc = od["model_cost"]
        rules = od["benchmarks"]["squad2"]["arms"]["rules"]
        model = od["benchmarks"]["squad2"]["arms"]["model"]
        ms.append(Measure(
            "cost per item: cheap tier vs model tier",
            f"rules {rules['seconds_per_item']:.4f} s and 0 model calls; model {model['seconds_per_item']:.3f} s and {model['model_calls']} calls "
            f"({mc['tokens_per_second']:.0f} tok/s, {mc['new_tokens']} new tokens over {mc['calls']} calls)",
            n=300, floor="n/a", control="same items both arms", env_author="public dataset",
            grader="public labels", agent_can_influence_grader="no", heldout="test slice",
            source="eval/results/open_domain.json",
            note="the cheap tier is ~3,600x cheaper per item and far less accurate; cost is the one axis where it wins outright"))
    if cw:
        ms.append(Measure(
            "environment cost after moving to a third-party engine",
            json.dumps({k: v for k, v in cw.items() if isinstance(v, (int, float, str))})[:200],
            n=None, env_author="computerworld (third party)", source="eval/results/computerworld.json",
            note="determinism by state hash makes repeated measurement cheap, which is what let the prediction axis exist at all"))
    return Axis("cost_and_evidence_integrity", "What does it cost, and who graded it?", bool(ms), "strong", ms,
                reading="Zero model calls at ~0.986 items correct is real and cheap. The integrity picture is the uncomfortable half: most headline rows are graded by code we wrote, in environments we wrote, and one environment was changed after a failure (since restored and re-measured).")


# ------------------------------------------------------------------- assembly


def build(probes: dict) -> dict:
    axes = [
        axis_transfer(), axis_horizon(), axis_sample_efficiency(), axis_calibration(),
        axis_compositionality(probes), axis_world_modeling(), axis_belief_revision(probes),
        axis_grounding(probes), axis_robustness(), axis_cost(),
    ]
    grounded = [a for a in axes if a.grounded]
    return {
        "method": "A profile, not a score. Each row carries its floor, its control, who wrote the environment, "
                  "who graded it, and whether the test was fixed before tuning. Accuracies are luck-corrected "
                  "where the chance credit is known. An axis that cannot be grounded says so.",
        "axes_total": len(axes),
        "axes_grounded": len(grounded),
        "axes_ungrounded": [a.key for a in axes if not a.grounded],
        "verdicts": {a.key: a.verdict for a in axes},
        "axes": [asdict(a) for a in axes],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", default=None, help="directory holding probe_*.json from a previous probe run")
    ap.add_argument("--out", default=str(RESULTS / "cognitive_profile.json"))
    args = ap.parse_args()
    probes: dict = {}
    if args.probes:
        for key, name in (("compositionality", "probe_comp.json"), ("belief_revision", "probe_belief.json"), ("grounding", "probe_ground.json")):
            p = Path(args.probes) / name
            if p.exists():
                probes[key] = json.loads(p.read_text())
    profile = build(probes)
    Path(args.out).write_text(json.dumps(profile, indent=1))
    print(f"axes grounded: {profile['axes_grounded']}/{profile['axes_total']}")
    for a in profile["axes"]:
        mark = a["verdict"] if a["grounded"] else f"UNGROUNDED ({a['ungrounded_reason'][:50]})"
        print(f"  {a['key']:28s} {mark}")
        for m in a["measures"]:
            print(f"      {m['name'][:72]}")
            print(f"        {m['value'][:150]}")


if __name__ == "__main__":
    main()
