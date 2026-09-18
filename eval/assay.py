"""The assay: one scorecard for the agent, across everything it is measured on.

    python -m eval.assay            # read the latest recorded result of each eval
    python -m eval.assay --list     # show what exists, including what is NOT measured

Computer use is one row here, not the point. The agent is meant to be general, so the
scorecard covers language, knowledge, memory, arithmetic, vision, tool use, safety and
learning, and says plainly which of them have never been measured.

Each row reports: the metric, who grades it (a dataset's own answers, world state, or a
frozen annotation), the number, and how it was obtained. Rows marked "not measured" are
the honest gaps; they are listed rather than left out so the scorecard cannot flatter by
omission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"


def latest(name: str, key: str | None = None) -> dict | None:
    """The last row of a .jsonl result file, or the contents of a .json one."""
    p = RESULTS / name
    if not p.exists():
        return None
    if p.suffix == ".jsonl":
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        if key:
            rows = [r for r in rows if r.get(key)]
        return rows[-1] if rows else None
    return json.loads(p.read_text())


def fmt(value: object) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def rows() -> list[dict]:
    out: list[dict] = []

    def row(area, what, metric, value, grader, source, note=""):
        out.append({"area": area, "what": what, "metric": metric, "value": value, "grader": grader,
                    "source": source, "note": note})

    parsing = latest("parsing_ud.jsonl")
    dev = (parsing or {}).get("dev", {})
    row("language", "syntax: dependency parsing", "LAS (labelled attachment)", fmt(dev.get("las")),
        "UD English-EWT gold trees", "eval/parsing/train_ud.py", "dev split; UAS " + fmt(dev.get("uas")))
    row("language", "part of speech", "accuracy", fmt(dev.get("tagging_accuracy")), "UD gold tags",
        "eval/parsing/train_ud.py")

    v4 = latest("grammar_induction_formal_v4.json")
    if v4:
        verdict = v4.get("verdict", {})
        row("learning", "grammar induction (formal languages)", "languages learned exactly",
            f"{verdict.get('dev_meeting_bar')} dev, {verdict.get('fresh_meeting_bar')} fresh",
            "mathematical membership", "eval/grammar_induction/formal_v4.py",
            "counting languages fail; pre-registered")
    row("learning", "learning in use (new words, new rules from conversation)", "constructions recovered",
        "not measured", "—", "—", "not built yet")

    answers = latest("general_agent_answers.jsonl")
    if answers:
        for cat, v in (answers.get("by_category") or {}).items():
            metric = "accuracy" if v.get("accuracy") is not None else "answered"
            value = fmt(v.get("accuracy")) if v.get("accuracy") is not None else f"{v.get('answered')}/{v.get('n')}"
            area = {"general_knowledge": "knowledge", "arithmetic": "reasoning", "image_questions": "vision",
                    "screen_questions": "vision", "conversation_facts": "memory", "ambiguous": "pragmatics",
                    "open_ended": "conversation"}.get(cat, "other")
            note = f"answered {v.get('answered')}/{v.get('n')}, wrong {v.get('wrong')}"
            if cat == "ambiguous":
                note += f", asked back {v.get('asked_back')}"
            row(area, f"held-out prompts: {cat}", metric, value, "the dataset's own answers",
                "eval/general_agent/grade_answers.py", note)

    split = latest("general_agent_runs.jsonl")
    if split:
        row("safety", "does it change things it was not asked to", "false actions",
            split.get("false_action_rate"), "the world's state, from the event log",
            "eval/general_agent/run_split.py", f"reader: {split.get('reader')}")
        tasks = sum(v["n"] for k, v in (split.get("by_category") or {}).items()
                    if k in ("desktop_gui", "shell_files", "multi_step"))
        solved = sum(v["items_with_a_write"] for k, v in (split.get("by_category") or {}).items()
                     if k in ("desktop_gui", "shell_files", "multi_step"))
        row("computer use", "held-out tasks (OSWorld, NL2Bash, Mind2Web)", "tasks completed",
            f"{solved}/{tasks}", "world state (graders not written yet)", "eval/general_agent/run_split.py",
            "about 5 of these are attemptable on computerworld's shell")

    vision = latest("vision_concepts.jsonl")
    if vision:
        row("vision", "object recognition (CIFAR-10)", "accuracy", fmt(vision.get("test_accuracy_all")),
            "dataset labels", "eval/vision_hierarchy/train_concepts.py",
            f"precision {fmt(vision.get('test_precision_when_answering'))} at coverage {fmt(vision.get('test_coverage'))}")
    hier = latest("vision_hierarchy_cifar.json")
    if hier:
        arms = hier.get("arms", {})
        row("vision", "is a learned hierarchy better than one flat layer", "hierarchy − flat (accuracy)",
            fmt(round(arms.get("hier2", {}).get("accuracy", 0) - arms.get("flat", {}).get("accuracy", 0), 3)),
            "dataset labels", "eval/vision_hierarchy/cifar.py", "negative: the hierarchy lost; pre-registered")

    row("tool use", "picking the right capability and arguments (non-desktop)", "accuracy", "not measured",
        "—", "—", "no general tool-calling eval yet")
    row("conversation", "multi-turn coherence and usefulness", "—", "not measured", "—", "—",
        "no grader that isn't a person")
    row("language", "generation: can it say what it means", "round-trip parse of its own reply",
        "not measured", "—", "—", "planned: re-read the reply, recover the frame")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show the gaps too (default shows everything)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    data = rows()
    if args.json:
        print(json.dumps(data, indent=1))
        return
    width = max(len(f"{r['area']}/{r['what']}") for r in data)
    print(f"{'area / what':{width}}  {'metric':34} {'value':>10}  grader")
    print("-" * (width + 60))
    for r in data:
        print(f"{r['area'] + ' / ' + r['what']:{width}}  {r['metric']:34} {str(r['value']):>10}  {r['grader']}")
        if r["note"]:
            print(f"{'':{width}}  {'':34} {'':>10}  {r['note']}")
    missing = [r for r in data if r["value"] == "not measured"]
    print(f"\n{len(missing)} of {len(data)} rows are not measured yet:")
    for r in missing:
        print(f"  - {r['area']}: {r['what']} ({r['note']})")


if __name__ == "__main__":
    main()
