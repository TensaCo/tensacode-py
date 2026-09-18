"""Cluster the failure corpus by the DISTINCTION the system cannot make.

    python eval/schema/cluster.py

The unit is an ITEM, not a row: for each input, which tiers failed it. That is what separates
the two kinds of problem, and the distinction decides whether structure is the answer at all:

    every tier fails        a SCHEMA problem — no representation exists to get it right
    only the learned tier   a DATA problem — the representation exists, the model has not seen it
    only hand-written tiers a COVERAGE problem — rules were never written for this phrasing,
                            and writing more is a treadmill the learned tier already beat

Open-domain rows arrive as stage attributions from `schema_brittleness.json`, so they are
weighted by the item counts recorded there rather than counted once each.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

CORPUS = Path(__file__).parent / "corpus" / "rows.jsonl"
OUT = Path(__file__).resolve().parents[2] / "eval" / "results" / "schema_clusters.json"

TIERS = ("regex", "grammar", "learned")
COMMON_NOUNS = {"file", "folder", "directory", "app", "application", "window", "one", "thing"}
CANON = {"~": "home", "~/documents": "documents", "~/desktop": "desktop", "~/downloads": "downloads",
         "~/pictures": "pictures", "~/music": "music", "~/videos": "videos", "~/projects": "projects", "/tmp": "tmp"}
MEMORY_ACTS = {"tell", "ask_memory", "forget", "ask_self"}
PERCEPTION_ACTS = {"ask_screen", "ask_pixels"}
DIALOGUE_ACTS = {"greet", "thanks", "help", "choose", "confirm", "cancel"}

STAGE_CLUSTER = {
    "span_not_produced": "evidence_to_answer",
    "selection_chose_wrong_candidate": "evidence_to_answer",
    "retrieval_missed_evidence": "retrieval_semantics",
    "abstained_though_answerable": "capability_calibration",
    "answered_an_unanswerable_question": "capability_calibration",
    "arithmetic_composition": "quantity_composition",
    "no_quantities_parsed": "quantity_composition",
    "world_knowledge": "knowledge_gap",
    "boolean_or_other_type": "answer_type_coverage",
    "answer_not_in_passage_but_labelled_answerable": "our_wiring",
}

MEANING = {
    "reference_kind": "A slot value is a bare string that silently means one of four things: a literal path, a place word to canonicalise, a name to resolve against the world, or a discourse reference. The '@' prefix is a stringly-typed stand-in and authors apply it inconsistently.",
    "speech_act": "No distinction between a statement, a question and a request, so a declarative sentence is forced into the act vocabulary and acted upon.",
    "domain_membership": "No test for whether an input is a task this agent could perform at all, so an open-domain question is read as a file operation.",
    "self_and_memory": "No representation of the conversation or of the agent itself, so nothing can be told to it, recalled from it, or asked about what it did.",
    "perception_query": "No route from a question to perceived state, although the state is already in the claim graph.",
    "multi_value": "A slot holds one value, so a request naming several things cannot be represented.",
    "content_span_boundary": "A quoted or content span is taken with its introducing words ('the text ...'): the span is delimited by position rather than by what introduces it.",
    "functional_description": "A thing described by what it is for ('the app used for writing code') has no path from function to identity to location.",
    "evidence_to_answer": "Evidence is retrieved and then cannot be turned into an answer: no typed entity recogniser, and candidate choice is by position.",
    "retrieval_semantics": "Retrieval matches surface strings, so evidence sharing no words with the question is unreachable.",
    "quantity_composition": "Quantities parse but nothing decides which of them combine, or in what order.",
    "capability_calibration": "Confidence is measured on evidence strength rather than on whether this tier can answer, so abstention is wrong in both directions; and a tier that reports no confidence at all is gated into escalation.",
    "answer_type_coverage": "No projection for yes/no and comparison answers.",
    "knowledge_gap": "The fact is not in the input, the graph or the grammar. No symbolic structure supplies it.",
    "our_wiring": "Our own harness or label error, not a failure of the system under test.",
    "phrasing_coverage": "The act and its slots are representable; this phrasing was simply never matched. Writing more rules is the treadmill the learned tier exists to end.",
    "act_confusion": "One known act read as another, with the representation present for both.",
}


def canon(value: object) -> str:
    return CANON.get(str(value).strip().strip("'\"“”").lower(), str(value).strip().strip("'\"“”").lower()).lstrip("@")


def only_reference_convention(want: dict, got: dict) -> bool:
    if not want or set(want) - set(got):
        return False
    diffs = [(str(want[k]), str(got.get(k))) for k in want if str(want[k]) != str(got.get(k))]
    return bool(diffs) and all(canon(a) == canon(b) for a, b in diffs)


def classify_row(row: dict) -> str:
    """The distinction this single (input, tier) failure reveals."""
    want_act, got_act = row["want_act"], row["got_act"]
    want, got = row["want_slots"] or {}, row["got_slots"] or {}
    text = row["text"]

    if row["source"] == "open_domain_stages":
        return STAGE_CLUSTER.get((row["got_slots"] or {}).get("stage", ""), "evidence_to_answer")
    if row["source"] == "decisions_banking77":
        return "capability_calibration"
    if row["source"] == "civ_193":
        return "speech_act"
    if row["source"] == "squad_open_150":
        return "domain_membership"

    if want_act in MEMORY_ACTS and got_act not in MEMORY_ACTS:
        return "self_and_memory"
    if want_act in PERCEPTION_ACTS and got_act not in PERCEPTION_ACTS:
        return "perception_query"
    if want_act == "open_function" and got_act != "open_function":
        return "functional_description"
    if only_reference_convention(want, got):
        return "reference_kind"
    if any(str(v).strip().lower() in COMMON_NOUNS for v in got.values()):
        return "reference_kind"
    for key in ("text", "needle", "value", "message", "command"):
        if key in want and key in got:
            a, b = str(want[key]).lower(), str(got[key]).lower()
            if a != b and (a in b or b in a):
                return "content_span_boundary"
    if re.search(r"\b(?:folders|files)\b[^.]*\band\b", text, re.I) or re.search(r",\s*\w+\s+and\s+\w+", text):
        return "multi_value"
    if got_act == "unknown":
        return "phrasing_coverage" if want_act not in DIALOGUE_ACTS else "phrasing_coverage"
    if got_act != want_act:
        return "act_confusion"
    return "phrasing_coverage"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    args = ap.parse_args()
    rows = [json.loads(l) for l in args.corpus.read_text().splitlines()]

    # --- group by item: (source, text) -> what each tier did
    items: dict[tuple[str, str], dict] = defaultdict(lambda: {"tiers": {}, "want": None, "provenance": {}, "note": ""})
    weights: dict[tuple[str, str], int] = {}
    for r in rows:
        key = (r["source"], r["text"])
        it = items[key]
        it["want"] = f"{r['want_act']}({r['want_slots']})"
        it["provenance"] = r["provenance"]
        it["note"] = r["note"]
        it["tiers"][r["tier"]] = r
        weights[key] = int((r["got_slots"] or {}).get("count") or 1) if r["source"] == "open_domain_stages" else 1

    clusters: dict[str, dict] = defaultdict(lambda: {"items": 0, "weighted": 0, "failing_tiers": Counter(), "kinds": Counter(),
                                                     "sources": Counter(), "wrong_action_items": 0, "examples": [],
                                                     "failing_sets": Counter()})
    for key, it in items.items():
        failed = {t: r for t, r in it["tiers"].items() if not r["ok"]}
        if not failed:
            continue
        # the cluster of an item is the distinction agreed on by the tiers that failed it
        votes = Counter(classify_row(r) for r in failed.values())
        name = votes.most_common(1)[0][0]
        c = clusters[name]
        c["items"] += 1
        c["weighted"] += weights[key]
        for t in failed:
            c["failing_tiers"][t] += 1
        for r in failed.values():
            c["kinds"][r["kind"] or "?"] += 1
        c["sources"][key[0]] += 1
        c["failing_sets"]["+".join(t for t in TIERS if t in failed)] += 1
        if any(r["kind"] == "wrong_action" for r in failed.values()):
            c["wrong_action_items"] += 1
        if len(c["examples"]) < 5:
            worst = next((r for r in failed.values() if r["kind"] == "wrong_action"), next(iter(failed.values())))
            c["examples"].append({"source": key[0], "text": key[1][:95], "want": it["want"],
                                  "got": f"{worst['got_act']}({worst['got_slots']})", "by": worst["tier"],
                                  "failed_tiers": sorted(failed)})

    total_items = sum(1 for it in items.values() if any(not r["ok"] for r in it["tiers"].values()))
    report = {
        "what": "language-eval failures clustered by the distinction the system cannot make; unit = item, not row",
        "corpus_rows": len(rows), "failing_items": total_items,
        "tier_coverage": {t: sum(1 for it in items.values() if t in it["tiers"]) for t in TIERS},
        "clusters": {},
    }
    for name, c in sorted(clusters.items(), key=lambda kv: -(kv[1]["wrong_action_items"] * 100 + kv[1]["weighted"])):
        tested_by = [t for t in TIERS if any(t in it["tiers"] for k, it in items.items())]
        defeated = [t for t in TIERS if c["failing_tiers"].get(t, 0) > 0]
        # only tiers that actually saw these items count toward the verdict
        saw = {t: sum(1 for k, it in items.items() if t in it["tiers"] and any(not r["ok"] for r in it["tiers"].values()) and classify_row(it["tiers"][t]) == name)
               for t in TIERS}
        # The verdict is a property of ITEMS, not of the cluster: a cluster where every tier
        # appears among the failures may still contain no single item that defeats all three.
        # Reporting the modal failing set, and the share that defeat all three, keeps that visible.
        sets = c["failing_sets"]
        all_three = sets.get("regex+grammar+learned", 0)
        if not defeated:
            verdict = "attributed from recorded stages (no tier run here)"
        else:
            modal, modal_n = sets.most_common(1)[0]
            share = all_three / c["items"] if c["items"] else 0.0
            label = ("SCHEMA" if share >= 0.5 else
                     "DATA" if modal == "learned" else
                     "COVERAGE" if "learned" not in modal else "MIXED")
            verdict = (f"{label}: {all_three}/{c['items']} items defeat all three tiers; "
                       f"most common failing set is {modal} ({modal_n})")
        report["clusters"][name] = {
            "items": c["items"], "weighted_items": c["weighted"], "wrong_action_items": c["wrong_action_items"],
            "failing_tiers": dict(c["failing_tiers"]), "failing_sets": dict(c["failing_sets"].most_common()),
            "items_defeating_all_three": c["failing_sets"].get("regex+grammar+learned", 0),
            "kinds": dict(c["kinds"]), "sources": dict(c["sources"]),
            "verdict": verdict, "distinction_missing": MEANING.get(name, ""), "examples": c["examples"],
        }
        _ = (tested_by, saw)
    OUT.write_text(json.dumps(report, indent=1))

    print(f"{total_items} failing items in {len(clusters)} clusters, ranked by wrong actions then weight\n")
    print(f"{'cluster':24} {'items':>6} {'weighted':>9} {'wrong-act':>10}  {'regex':>5} {'gram':>5} {'learn':>5}  verdict")
    for name, c in report["clusters"].items():
        t = c["failing_tiers"]
        print(f"{name:24} {c['items']:6} {c['weighted_items']:9} {c['wrong_action_items']:10}  "
              f"{t.get('regex',0):5} {t.get('grammar',0):5} {t.get('learned',0):5}  {c['verdict']}")


if __name__ == "__main__":
    main()
