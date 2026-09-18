"""Measure the learned parser against the two hand-written parsers it is meant to replace.

    python eval/training/eval_parser.py

Evaluation sets, by provenance (the first three were written by other people or other
forks before this model existed, which is the only reason they are worth anything):

  held_out_user   the five prompts the user reported failing, verbatim
  benchmark_152   tests/test_assistant_language.py, written beside the regexes
  civ_statements  utterances a running civ_sim world produces: a running world's speech,
                  none of it a request, so the honest answer is to refuse every one
  squad_open      public SQuAD 2.0 questions: open-domain, also nothing this agent can do
  paraphrase      model-written rewrites of held-out templates (labelled model-written)
"""

from __future__ import annotations

import os

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

SP = Path(os.environ.get("TENSORCODE_SCRATCH", os.path.expanduser("~/.cache/tensorcode")))

#: exactly as the user pasted them, with what a competent assistant should do
USER_PROMPTS = [
    ("hello", "greet", {}),
    ("my name is Jacob", "tell", {"topic": "name", "value": "Jacob"}),
    ("what is my name?", "ask_memory", {"topic": "name"}),
    ("what color is the display", "ask_pixels", {"region": "display"}),
    ("how many icons are in the sidebar", "ask_screen", {"aspect": "icons"}),
    ("open the app that is used for writing code", "open_function", {"function": "writing code"}),
]


def _load_cases() -> list[tuple[str, list[tuple[str, dict]]]]:
    spec = importlib.util.spec_from_file_location("bench152", ROOT / "tests" / "test_assistant_language.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.CASES)


def _civ_utterances(limit: int = 200) -> list[str]:
    """What villagers say: statements, not requests. Nothing here is an act the assistant can do."""
    import numpy as np

    from research.civ_sim import language as L  # noqa: PLC0415

    lex = L.base_lexicon()
    claims = [("village:Coralin", "has_amount", "food:much"), ("village:Aldmere", "has_amount", "food:none"),
              ("village:Brenholt", "has_amount", "wood:little"), ("person:Nise", "is", "hungry"),
              ("resource:wood", "is", "cheap"), ("resource:wood", "is", "expensive"),
              ("person:Kasa", "is", "trustworthy"), ("weather:snow", "comes", True)]
    said = []
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


def _squad_questions(n: int, seed: int = 11) -> list[str]:
    path = SP / "open_domain" / "squad2_val.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    random.Random(seed).shuffle(rows)
    return [r["question"].strip() for r in rows[:n]]


# ------------------------------------------------------------------- the arms


def arm_regex(texts: list[str]) -> list[tuple[str, dict]]:
    from examples.browser_agents.assistant.language import parse_message  # noqa: PLC0415

    out = []
    for t in texts:
        frames = parse_message(t)
        out.append((frames[0].act, dict(frames[0].slots)) if frames else ("unknown", {}))
    return out


def arm_grammar(texts: list[str]) -> list[tuple[str, dict]]:
    from tensorcode.language.domains.desktop import read_request  # noqa: PLC0415

    out = []
    for t in texts:
        try:
            acts, _ = read_request(t)
        except Exception:  # noqa: BLE001
            acts = []
        out.append((acts[0].act, dict(acts[0].slots)) if acts else ("unknown", {}))
    return out


def arm_learned(parser, texts: list[str]) -> list[tuple[str, dict]]:
    return [(p.act, dict(p.slots)) for p in parser.parse(texts)]


# ------------------------------------------------------------------- scoring


def _norm(value: object) -> str:
    return str(value).strip().strip("'\"“”").lower()


def score(got: tuple[str, dict], want_act: str, want_slots: dict) -> dict:
    act_ok = got[0] == want_act
    slots = got[1]
    needed = {k: v for k, v in want_slots.items() if k not in ("append", "count", "ask_only", "aspect")}
    slot_ok = all(_norm(slots.get(k)) == _norm(v) for k, v in needed.items())
    flags_ok = all(bool(slots.get(k)) == bool(v) for k, v in want_slots.items() if k in ("append", "count", "ask_only"))
    aspect_ok = all(_norm(slots.get(k)) == _norm(v) for k, v in want_slots.items() if k == "aspect")
    return {"act": act_ok, "slots": act_ok and slot_ok and flags_ok and aspect_ok}


def evaluate_labelled(name: str, cases: list[tuple[str, str, dict]], arms: dict) -> dict:
    texts = [c[0] for c in cases]
    out = {"n": len(cases), "arms": {}}
    for arm, fn in arms.items():
        t0 = time.perf_counter()
        got = fn(texts)
        seconds = time.perf_counter() - t0
        rows = [score(g, want_act, want_slots) for g, (_, want_act, want_slots) in zip(got, cases)]
        out["arms"][arm] = {
            "act_accuracy": round(sum(r["act"] for r in rows) / len(rows), 4),
            "exact_accuracy": round(sum(r["slots"] for r in rows) / len(rows), 4),
            "ms_per_utterance": round(seconds / len(texts) * 1e3, 3),
            "wrong": [{"text": t, "want": f"{a}({b})", "got": f"{g[0]}({g[1]})"}
                      for t, (_, a, b), g, r in zip(texts, cases, got, rows) if not r["slots"]][:12],
        }
    return out


def evaluate_refusals(name: str, texts: list[str], arms: dict) -> dict:
    """Every one of these should come back unknown; anything else is a confident wrong answer."""
    out = {"n": len(texts), "arms": {}}
    for arm, fn in arms.items():
        got = fn(texts)
        refused = [g[0] == "unknown" for g in got]
        acted = [{"text": t, "got": f"{g[0]}({g[1]})"} for t, g, r in zip(texts, got, refused) if not r]
        out["arms"][arm] = {"refused": round(sum(refused) / len(texts), 4), "acted_anyway": acted[:12]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, default=SP / "artifacts" / "request-parser")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "learned_parser.json")
    ap.add_argument("--paraphrases", type=Path, default=SP / "training" / "paraphrase_eval.jsonl")
    ap.add_argument("--squad", type=int, default=150)
    args = ap.parse_args()

    from tensorcode.backends.neural import NeuralRequestParser  # noqa: PLC0415

    parser = NeuralRequestParser(args.artifact)
    arms = {"regex": arm_regex, "grammar": arm_grammar, "learned": lambda t: arm_learned(parser, t)}

    report: dict = {"artifact": str(args.artifact), "parameters": parser.config["parameters"],
                    "train_size": parser.config["train_size"], "train_seconds": parser.config["train_seconds"],
                    "sets": {}}

    report["sets"]["held_out_user"] = evaluate_labelled("held_out_user", USER_PROMPTS, arms)

    cases = _load_cases()
    flat = [(text, want[0][0], want[0][1]) for text, want in cases if len(want) == 1]
    report["sets"]["benchmark_152"] = evaluate_labelled("benchmark_152", flat, arms)
    report["sets"]["benchmark_152"]["note"] = "written beside the regexes, so the regex arm is at its ceiling by construction"

    if args.paraphrases.exists():
        rows = [json.loads(l) for l in args.paraphrases.read_text().splitlines()]
        para = [(r["text"], r["act"], r.get("want_slots", {})) for r in rows]
        report["sets"]["paraphrase"] = evaluate_labelled("paraphrase", para, arms)
        report["sets"]["paraphrase"]["note"] = ("model-written rewrites of held-out templates; labels carried over and "
                                                "value-checked. want_slots is empty here, so exact == act: this row "
                                                "measures the act only.")
        # the same rewrites, scored on the spans too: the row above never checked a value
        with_slots = [(r["text"], r["act"], {k: r["text"][a:b] for k, (a, b) in r.get("spans", {}).items()})
                      for r in rows]
        report["sets"]["paraphrase_with_slots"] = evaluate_labelled("paraphrase_with_slots", with_slots, arms)
        report["sets"]["paraphrase_with_slots"]["note"] = "as above but every recorded span value must come back exactly"

        # the copy/move distinction specifically: the label swap the audit found lived here
        mc = [c for c in with_slots if c[1] in ("move", "copy")]
        if mc:
            report["sets"]["move_copy"] = evaluate_labelled("move_copy", mc, arms)
            report["sets"]["move_copy"]["note"] = ("the corrected copy/move subset of the rewrites: gold is the verb "
                                                   "the utterance actually uses")

    civ = _civ_utterances()
    if civ:
        report["sets"]["civ_statements"] = evaluate_refusals("civ_statements", civ, arms)
        report["sets"]["civ_statements"]["note"] = ("distinct utterances a running world produces (4 dialects x 8 claims x 3 moods, deduplicated -- NOT the 193 cases in tests/test_civ_language_demands.py). None is a request, so refusing all of them is correct.")
    report["sets"]["squad_open"] = evaluate_refusals("squad_open", _squad_questions(args.squad), arms)
    report["sets"]["squad_open"]["note"] = "public open-domain questions: this agent cannot answer any of them from a machine"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    for name, block in report["sets"].items():
        print(f"\n== {name} (n={block['n']})")
        for arm, scores in block["arms"].items():
            if "act_accuracy" in scores:
                print(f"   {arm:<8} act {scores['act_accuracy']:.3f}  exact {scores['exact_accuracy']:.3f}  {scores['ms_per_utterance']:.2f} ms")
            else:
                print(f"   {arm:<8} refused {scores['refused']:.3f}")


if __name__ == "__main__":
    main()
