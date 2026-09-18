"""Harvest every language-eval failure we have into one corpus, with provenance.

    python eval/schema/corpus.py                 # writes eval/schema/corpus/rows.jsonl

One row per (input, tier) pair. A row records what the tier did, what it should have done,
and — the field that matters most — which KIND of failure it is:

    wrong_action    the tier committed to an act the input did not ask for (the only kind
                    that damages anything: it deletes, writes, or answers a question the
                    user did not ask)
    wrong_answer    right act, wrong content
    missed_answer   refused something it should have handled
    miscalibration  answered when it should have abstained, or abstained when capable,
                    with the content itself not at issue

Provenance per source, in the scheme of docs/revival/11-evidence-audit.md: who authored the
inputs, who wrote the gold labels, and whether the tier could have been tuned against them.
"""

from __future__ import annotations

import os

import argparse
import importlib.util
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

SP = Path(os.environ.get("TENSORCODE_SCRATCH", os.path.expanduser("~/.cache/tensorcode")))
OUT = Path(__file__).parent / "corpus"

#: acts that change the machine: reading one of these into a statement is the worst failure we have
EFFECTFUL = frozenset({"delete", "write", "create_file", "create_folder", "move", "copy", "rename", "run", "install", "cd", "open_app", "open_function", "setup_project", "git_init", "git_commit"})


@dataclass
class Row:
    source: str
    text: str
    tier: str
    got_act: str
    got_slots: dict
    want_act: str | None
    want_slots: dict | None
    ok: bool
    kind: str | None  # wrong_action | wrong_answer | missed_answer | miscalibration
    provenance: dict = field(default_factory=dict)
    note: str = ""


# --------------------------------------------------------------------- sources


PROV = {
    "user_prompts": {"inputs": "the user, from a live session", "gold": "the user's own report", "tunable_against": False},
    "benchmark_152": {"inputs": "assistant fork, written beside the regexes", "gold": "same author", "tunable_against": True},
    "civ_193": {"inputs": "civilization fork, a running world's speech", "gold": "structural (none is a request)", "tunable_against": False},
    "paraphrase_928": {"inputs": "local model, rewriting held-out templates", "gold": "carried from the template", "tunable_against": False},
    "squad_open_150": {"inputs": "SQuAD 2.0 (public)", "gold": "structural (none is a machine task)", "tunable_against": False},
    "open_domain_stages": {"inputs": "SQuAD 2.0 / HotpotQA / GSM8K / ARC (public)", "gold": "public labels", "tunable_against": False},
    "decisions_banking77": {"inputs": "Banking77 (public)", "gold": "public labels", "tunable_against": False},
}

USER_PROMPTS = [
    ("hello", "greet", {}),
    ("my name is Jacob", "tell", {"topic": "name", "value": "Jacob"}),
    ("what is my name?", "ask_memory", {"topic": "name"}),
    ("what color is the display", "ask_pixels", {"region": "display"}),
    ("how many icons are in the sidebar", "ask_screen", {"aspect": "icons"}),
    ("open the app that is used for writing code", "open_function", {"function": "writing code"}),
    # later live-session failures, same provenance: the user's own use
    ("delete the report.txt on my desktop", "delete", {"target": "report.txt"}),
    ("what do you know", "ask_memory", {"topic": "everything"}),
    ("do you know who i am", "ask_memory", {"topic": "name"}),
    ("am i Jacob", "ask_memory", {"topic": "name"}),
    ("my name isn't Jacob anymore", "forget", {"topic": "name"}),
    ("what font is the terminal using", "ask_pixels", {"region": "terminal"}),
    ("how big is the Files window", "ask_screen", {"aspect": "window_size"}),
    ("what changed since my last message", "ask_screen", {"aspect": "changed"}),
    ("my name and my favourite colour", "ask_memory", {"topic": "name"}),
]


def load_152() -> list[tuple[str, str, dict]]:
    spec = importlib.util.spec_from_file_location("bench152", ROOT / "tests" / "test_assistant_language.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = []
    for text, frames in mod.CASES:
        act, slots = frames[0]
        out.append((text, act, dict(slots)))
    return out


def load_civ(limit: int = 200) -> list[str]:
    import numpy as np

    from research.civ_sim import language as L

    lex = L.base_lexicon()
    claims = [("village:Coralin", "has_amount", "food:much"), ("village:Aldmere", "has_amount", "food:none"),
              ("village:Brenholt", "has_amount", "wood:little"), ("person:Nise", "is", "hungry"),
              ("resource:wood", "is", "cheap"), ("resource:wood", "is", "expensive"),
              ("person:Kasa", "is", "trustworthy"), ("weather:snow", "comes", True)]
    said: list[str] = []
    for index in range(4):
        dialect = L.dialect(lex, index, np.random.default_rng(index))
        for subject, predicate, obj in claims:
            for kw in ({}, {"negated": True}, {"secondhand": "Anem"}):
                try:
                    text = L.say(subject, predicate, obj, dialect, **kw)
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(text, str) and text.strip():
                    said.append(text)
    return list(dict.fromkeys(said))[:limit]


def load_paraphrases() -> list[tuple[str, str, dict]]:
    """Gold labels in the ASSISTANT's slot vocabulary.

    The training rows carry internal closed-head labels (``place_kind``, ``info_topic``,
    ``target_kind``, ``name_canon``). Those are not slots any tier emits, so they are decoded
    here exactly as ``NeuralRequestParser`` decodes them — otherwise every hand-written tier is
    graded against a vocabulary it was never asked to produce, which is a measurement error and
    not a failure of the tier.
    """
    path = SP / "training" / "paraphrase_eval.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        spans = {k: r["text"][a:b] for k, (a, b) in r.get("spans", {}).items()}
        closed = r.get("closed", {})
        act = r["act"]
        slots = {k: v for k, v in spans.items() if k != "place"}
        kind = closed.get("place_kind", "none")
        if kind == "span" and "place" in spans:
            slots["place"] = spans["place"]
        elif kind not in ("none", "span"):
            slots["place"] = kind
        if closed.get("target_kind", "none") == "@it":
            slots["target"] = "@it"
        if closed.get("name_canon", "none") != "none" and "name" not in slots:
            slots["name"] = closed["name_canon"]
        if act == "info" and closed.get("info_topic", "none") != "none":
            slots["topic"] = closed["info_topic"]
        if closed.get("unit", "none") != "none":
            slots["unit"] = closed["unit"]
        if closed.get("aspect", "none") not in (None, "none"):
            slots["aspect"] = closed["aspect"]
        slots.update({k: True for k, v in r.get("flags", {}).items() if v})
        if closed.get("speech_act") == "world_statement":
            act, slots = "unknown", {}
        out.append((r["text"], act, slots))
    return out


def load_squad_questions(n: int = 150, seed: int = 11) -> list[str]:
    path = SP / "open_domain" / "squad2_val.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    random.Random(seed).shuffle(rows)
    return [r["question"].strip() for r in rows[:n]]


# ----------------------------------------------------------------------- tiers


def arm_regex(texts: list[str]) -> list[tuple[str, dict]]:
    from examples.browser_agents.assistant.language import parse_message

    out = []
    for t in texts:
        frames = parse_message(t)
        out.append((frames[0].act, dict(frames[0].slots)) if frames else ("unknown", {}))
    return out


def arm_grammar(texts: list[str]) -> list[tuple[str, dict]]:
    from tensorcode.language.domains.desktop import read_request

    out = []
    for t in texts:
        try:
            acts, _ = read_request(t)
        except Exception:  # noqa: BLE001
            acts = []
        out.append((acts[0].act, dict(acts[0].slots)) if acts else ("unknown", {}))
    return out


def arm_learned(texts: list[str], artifact: Path) -> list[tuple[str, dict]]:
    from tensorcode.backends.neural import NeuralRequestParser

    parser = NeuralRequestParser(artifact)
    return [(p.act, dict(p.slots)) for p in parser.parse(texts)]


# --------------------------------------------------------------------- scoring


def norm(value: object) -> str:
    return str(value).strip().strip("'\"“”").lower()


def judge(got: tuple[str, dict], want_act: str, want_slots: dict) -> tuple[bool, str | None]:
    """Right or wrong, and if wrong, which kind of wrong."""
    act, slots = got
    if want_act == "unknown":
        if act == "unknown":
            return True, None
        return False, "wrong_action" if act in EFFECTFUL else "miscalibration"
    if act == "unknown":
        return False, "missed_answer"
    if act != want_act:
        return False, "wrong_action" if act in EFFECTFUL else "wrong_answer"
    needed = {k: v for k, v in want_slots.items() if k not in ("append", "count", "ask_only")}
    flags = {k: v for k, v in want_slots.items() if k in ("append", "count", "ask_only")}
    if all(norm(slots.get(k)) == norm(v) for k, v in needed.items()) and all(bool(slots.get(k)) == bool(v) for k, v in flags.items()):
        return True, None
    return False, "wrong_answer"


# ------------------------------------------------------------------ harvesting


def harvest(artifact: Path, with_learned: bool = True) -> list[Row]:
    labelled: list[tuple[str, list[tuple[str, str, dict]]]] = [
        ("user_prompts", USER_PROMPTS),
        ("benchmark_152", load_152()),
        ("paraphrase_928", load_paraphrases()),
    ]
    refusals: list[tuple[str, list[str]]] = [
        ("civ_193", load_civ()),
        ("squad_open_150", load_squad_questions()),
    ]

    tiers = {"regex": arm_regex, "grammar": arm_grammar}
    if with_learned:
        tiers["learned"] = lambda texts: arm_learned(texts, artifact)

    rows: list[Row] = []
    for source, cases in labelled:
        if not cases:
            continue
        texts = [c[0] for c in cases]
        for tier, fn in tiers.items():
            for (text, want_act, want_slots), got in zip(cases, fn(texts)):
                ok, kind = judge(got, want_act, want_slots)
                rows.append(Row(source, text, tier, got[0], got[1], want_act, want_slots, ok, kind, PROV[source]))
    for source, texts in refusals:
        if not texts:
            continue
        for tier, fn in tiers.items():
            for text, got in zip(texts, fn(texts)):
                ok, kind = judge(got, "unknown", {})
                rows.append(Row(source, text, tier, got[0], got[1], "unknown", {}, ok, kind, PROV[source],
                                note="a statement, not a request" if source == "civ_193" else "open-domain question, not a machine task"))
    rows += from_open_domain()
    rows += from_decisions()
    return rows


def from_open_domain() -> list[Row]:
    """The 1,058 already-diagnosed open-domain failures, kept as stage-attributed rows."""
    path = ROOT / "eval" / "results" / "schema_brittleness.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    rows: list[Row] = []
    kinds = {"abstained_though_answerable": "miscalibration", "answered_an_unanswerable_question": "miscalibration",
             "selection_chose_wrong_candidate": "wrong_answer", "span_not_produced": "missed_answer",
             "retrieval_missed_evidence": "missed_answer", "arithmetic_composition": "missed_answer",
             "no_quantities_parsed": "missed_answer", "world_knowledge": "missed_answer",
             "boolean_or_other_type": "missed_answer", "answer_not_in_passage_but_labelled_answerable": "wrong_answer"}
    for bench, info in data.get("benchmarks", {}).items():
        meaning = info.get("stage_meaning", {})
        for stage, count in info.get("stages", {}).items():
            rows.append(Row(
                "open_domain_stages", f"{bench}:{stage}", "rules", "stage_failure",
                {"stage": stage, "count": count, "benchmark": bench},
                "answer", {}, False, kinds.get(stage, "wrong_answer"), PROV["open_domain_stages"],
                note=f"class {meaning.get(stage, {}).get('class', '?')}: {meaning.get(stage, {}).get('would_have_needed', '')}"))
    return rows


def from_decisions() -> list[Row]:
    """Where a decision tier escalated or answered wrongly — one row per arm, with its gate counts."""
    path = ROOT / "eval" / "results" / "decisions_measure.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    rows: list[Row] = []
    for info in data.get("arms", []):
        arm = info.get("arm", "?")
        rows.append(Row("decisions_banking77", f"arm:{arm}", arm, "decision",
                        {"coverage": info.get("intent_coverage"), "accuracy": info.get("intent_accuracy_over_attempted"),
                         "auto": info.get("gate_auto"), "escalate": info.get("gate_escalate")},
                        "decision", {}, False, "miscalibration", PROV["decisions_banking77"],
                        note="a tier answering accurately but reporting no confidence is gated into escalation"))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, default=SP / "artifacts" / "request-parser")
    ap.add_argument("--no-learned", action="store_true")
    args = ap.parse_args()

    rows = harvest(args.artifact, with_learned=not args.no_learned)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rows.jsonl").write_text("".join(json.dumps(asdict(r)) + "\n" for r in rows))

    failures = [r for r in rows if not r.ok]
    by_kind: dict[str, int] = {}
    by_tier: dict[str, int] = {}
    for r in failures:
        by_kind[r.kind or "?"] = by_kind.get(r.kind or "?", 0) + 1
        by_tier[r.tier] = by_tier.get(r.tier, 0) + 1
    print(f"{len(rows)} rows, {len(failures)} failures")
    print("by kind:", json.dumps(by_kind, indent=1))
    print("by tier:", json.dumps(by_tier, indent=1))


if __name__ == "__main__":
    main()
