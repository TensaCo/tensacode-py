"""Does confidence per commitment convert flat refusals into answerable questions?

One scalar cannot say "I am sure what you asked for but unsure which file you meant"
(``docs/revival/15`` §15.8). Here confidence is carried per commitment — speech act, act
choice, each slot — and the signal is *external*: agreement between two independently built
readers (the hand-written regex tier and the unification grammar), not one reader's own
posterior, which ``docs/revival/15`` measured as nearly useless (+0.9 points of selective
accuracy for 2.8% refusals, 446 of 464 items in the top bin).

Prediction registered before running (docs/revival/23): a decomposed gate converts a
measurable fraction of the flat gate's refusals into questions, and a majority of those
questions name the commitment that was actually wrong or missing. Falsified if the
conversions are mostly about the *right* slots (a question nobody needed to be asked) or if
the named commitment is usually the wrong one.

Provenance is mixed and labelled per corpus: the 152-case set was authored beside the regex
tier, so it flatters it; the user's own failing prompts and the civilization's declarative
statements have independent provenance. Numbers are reported per corpus, never pooled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.browser_agents.assistant.agent import _read_with_grammar  # noqa: E402
from examples.browser_agents.assistant.language import parse_message  # noqa: E402
from tensorcode.metacognition import Belief, Confidences, agreement  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "eval" / "results" / "metacognition_decomposed.json"

#: the user's own failing transcript, plus the two prompts found after it. Independent provenance.
USER_PROMPTS = [
    ("hello", "greet", {}),
    ("my name is Jacob. what is my name?", "tell", {}),
    ("what color is the display", "ask_pixels", {"region": "display"}),
    ("how many icons are in the sidebar", "ask_screen", {}),
    ("open the app that is used for writing code", "open_function", {}),
    ("delete the report.txt", "delete", {"target": "report.txt"}),
]
#: statements the world says; every one of these must be refused rather than acted on
CIV_STATEMENTS = [
    "Nise is hungry.", "Coralin holds much food.", "Wood is cheap.", "Anem died.",
    "Ibrase said Wud is dear.", "Coralin holds heaps of bread.", "the north field failed",
]


def read_both(text: str) -> tuple[str | None, dict, str | None, dict]:
    """Two independent readings of one utterance."""
    frames = parse_message(text)
    regex_act = frames[0].act if frames else None
    regex_slots = dict(frames[0].slots) if frames else {}
    grammar = _read_with_grammar(text)
    return (regex_act if regex_act != "unknown" else None), regex_slots, (grammar.act if grammar else None), (dict(grammar.slots) if grammar else {})


def confidences(text: str) -> Confidences:
    """Per-commitment confidence from reader agreement."""
    r_act, r_slots, g_act, g_slots = read_both(text)
    parts = [Belief("act", r_act or g_act or "?", r_act or g_act, agreement([r_act, g_act], basis="regex+grammar"))]
    for slot in sorted(set(r_slots) | set(g_slots)):
        readings = [r_slots.get(slot), g_slots.get(slot)]
        parts.append(Belief("slot", slot, r_slots.get(slot, g_slots.get(slot)), agreement(readings, basis="regex+grammar")))
    return Confidences(tuple(parts))


def flat_gate(text: str, *, act_at: float) -> str:
    """Today's shape: one number for the whole reading, so any weakness refuses everything."""
    conf = confidences(text)
    return "act" if conf.weakest() and conf.weakest().strength >= act_at else "refuse"


def run(act_at: float, ask_below: float) -> dict:
    corpora: dict[str, list[tuple[str, str | None, dict]]] = {
        "user_prompts (independent provenance)": USER_PROMPTS,
        "civ_statements (independent provenance; all must be refused)": [(s, None, {}) for s in CIV_STATEMENTS],
    }
    # the 152-case set, read from the test file it lives in rather than copied
    try:
        from tests.test_assistant_language import CASES
        corpora["assistant_152 (authored beside the regex tier)"] = [
            (text, expected[0][0] if expected else None, expected[0][1] if expected else {}) for text, expected in CASES
        ]
    except Exception as exc:  # noqa: BLE001
        corpora["assistant_152"] = []
        print(f"(could not load the 152-case set: {exc})")

    report: dict = {"note": __doc__.strip().splitlines()[0], "thresholds": {"act_at": act_at, "ask_below": ask_below}, "corpora": {}}
    for name, cases in corpora.items():
        rows = []
        for text, want_act, want_slots in cases:
            conf = confidences(text)
            gate = conf.gate(act_at=act_at, ask_below=ask_below)
            flat = flat_gate(text, act_at=act_at)
            weak = conf.weakest()
            # was the commitment it singled out actually the broken one?
            named_right = None
            if gate.decision == "ask":
                if want_act is None:
                    named_right = False  # nothing should have been asked: this utterance is not a request
                else:
                    got = conf.of("slot")
                    asked = gate.about
                    named_right = bool(want_slots) and (asked not in want_slots or want_slots.get(asked) != next(
                        (b.value for b in got if b.about == asked), None))
            rows.append({"text": text, "want_act": want_act, "flat": flat, "decomposed": gate.decision,
                         "about": gate.about, "weakest": weak.describe() if weak else None,
                         "named_the_broken_part": named_right})
        refused_flat = [r for r in rows if r["flat"] == "refuse"]
        converted = [r for r in refused_flat if r["decomposed"] == "ask"]
        report["corpora"][name] = {
            "n": len(rows),
            "flat_refusals": len(refused_flat),
            "converted_to_a_question": len(converted),
            "conversion_rate": round(len(converted) / len(refused_flat), 4) if refused_flat else None,
            "question_named_the_broken_part": sum(1 for r in converted if r["named_the_broken_part"] is True),
            "question_was_unnecessary": sum(1 for r in converted if r["named_the_broken_part"] is False),
            "acted_when_flat_would_refuse": sum(1 for r in refused_flat if r["decomposed"] == "act"),
            "rows": rows,
        }
    conv = [b for b in report["corpora"].values() if b.get("flat_refusals")]
    total_ref = sum(b["flat_refusals"] for b in conv)
    total_conv = sum(b["converted_to_a_question"] for b in conv)
    right = sum(b["question_named_the_broken_part"] for b in conv)
    report["prediction"] = {
        "refusals_converted": f"{total_conv}/{total_ref}",
        "conversion_rate": round(total_conv / total_ref, 4) if total_ref else None,
        "questions_that_named_the_broken_part": f"{right}/{total_conv}" if total_conv else "0/0",
        "holds": bool(total_conv) and right > total_conv / 2,
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--act-at", type=float, default=1.0, help="every reader must agree to act")
    ap.add_argument("--ask-below", type=float, default=0.4)
    args = ap.parse_args()
    report = run(args.act_at, args.ask_below)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(f"wrote {OUT}")
    for name, block in report["corpora"].items():
        print(f"  {name}: n={block['n']} flat refusals={block['flat_refusals']} "
              f"-> asks={block['converted_to_a_question']} (named the broken part {block['question_named_the_broken_part']}, "
              f"unnecessary {block['question_was_unnecessary']})")
    print("prediction:", json.dumps(report["prediction"]))


if __name__ == "__main__":
    main()
