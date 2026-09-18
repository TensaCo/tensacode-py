"""Does each induced schema change pay? Measured per rule, per tier, on two disjoint halves.

    python eval/schema/measure.py

Honest split, stated because it limits what the numbers mean:

  design    the 51 items that defeat every tier. I read all of them to adjudicate their
            labels, so they are NOT held out. Items whose gold label I judged wrong are
            excluded entirely rather than counted as gaps.
  heldout   every other failing item in the corpus, plus every item that already passed.
            I never inspected these individually. The second group is the one that matters:
            a schema change that helps the design set and breaks passing items is a loss.

Each rule is measured alone and in combination, so a rule that only pays inside a stack is
visible as such.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from adjudicate import verdict  # noqa: E402
from repair import NAMES, PLACE_WORDS, repair  # noqa: E402

CORPUS = HERE / "corpus" / "rows.jsonl"
OUT = Path(__file__).resolve().parents[2] / "eval" / "results" / "schema_repair.json"
TIERS = ("regex", "grammar", "learned")


def norm(value: object) -> str:
    return str(value).strip().strip("'\"“”").lower()


#: The corpus does not agree with itself about the canonical form of a place, a topic or an app
#: name: the same eval wants `music` on one row and `~/Music` on another, `phone number` here and
#: `phone` there, `the system monitor` with its article and `terminal` without. Under strict
#: comparison a canonicalising rule is therefore penalised for the corpus's inconsistency rather
#: than for being wrong. `agree()` collapses exactly those three conventions on BOTH sides, so a
#: rule can be scored on whether it identifies the right thing, separately from whether it spells
#: it the way a particular row happened to spell it. Both scores are reported; neither is the
#: headline on its own.
_ARTICLE = re.compile(r"^(?:a|an|the|my|our|your)\s+")


def agree(value: object) -> str:
    word = _ARTICLE.sub("", norm(value)).lstrip("@")
    word = re.sub(r"\s+(?:folder|directory|dir|window|app|application|number|address)$", "", word)
    return PLACE_WORDS.get(word, word).lower()


def correct(got_act: str, got_slots: dict, want_act: str, want_slots: dict, *, lenient: bool = False) -> bool:
    if got_act != want_act:
        return False
    needed = {k: v for k, v in (want_slots or {}).items() if k not in ("append", "count", "ask_only")}
    flags = {k: v for k, v in (want_slots or {}).items() if k in ("append", "count", "ask_only")}
    same = agree if lenient else norm
    return (all(same(got_slots.get(k)) == same(v) for k, v in needed.items())
            and all(bool(got_slots.get(k)) == bool(v) for k, v in flags.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--lenient", action="store_true",
                    help="collapse the corpus's own inconsistent place/topic/app conventions on both sides")
    args = ap.parse_args()
    rows = [json.loads(l) for l in args.corpus.read_text().splitlines()
            if json.loads(l)["tier"] in TIERS and json.loads(l)["source"] != "open_domain_stages"]

    # which items defeat every tier (the design set), and which of those have a usable label
    by_item: dict[tuple[str, str], dict] = {}
    for r in rows:
        by_item.setdefault((r["source"], r["text"]), {})[r["tier"]] = r
    all_fail = {k for k, t in by_item.items() if len(t) == len(TIERS) and all(not r["ok"] for r in t.values())}
    excluded = {k for k in all_fail if verdict(k[1])[0] == "gold_wrong"}

    def split_of(row: dict) -> str | None:
        key = (row["source"], row["text"])
        if key in excluded:
            return None
        return "design" if key in all_fail else "heldout"

    configs: dict[str, set[str] | None] = {"baseline": set()}
    for name in NAMES:
        configs[name] = {name}
    configs["all_rules"] = None

    report = {
        "what": "each induced schema change, measured per tier on the design set and on everything I never inspected",
        "split": {
            "design": {"items": len(all_fail - excluded), "held_out": False,
                       "note": "items that defeat every tier, minus those whose gold label I judged wrong"},
            "excluded_bad_labels": {"items": len(excluded), "note": "recorded label does not describe the sentence; counted as our wiring error, not a gap"},
        },
        "rules": {},
    }
    for config, enabled in configs.items():
        result = {}
        for tier in TIERS:
            tier_rows = [r for r in rows if r["tier"] == tier and split_of(r) is not None]
            counts = {s: Counter() for s in ("design", "heldout")}
            moved = {"fixed": [], "broke": []}
            for r in tier_rows:
                split = split_of(r)
                was = correct(r["got_act"], r["got_slots"] or {}, r["want_act"], r["want_slots"] or {}, lenient=args.lenient)
                act, slots = repair(r["text"], r["got_act"], dict(r["got_slots"] or {}), enabled)
                now = correct(act, slots, r["want_act"], r["want_slots"] or {}, lenient=args.lenient)
                counts[split]["n"] += 1
                counts[split]["was"] += was
                counts[split]["now"] += now
                if now and not was:
                    counts[split]["fixed"] += 1
                    if len(moved["fixed"]) < 4:
                        moved["fixed"].append({"split": split, "text": r["text"][:80], "want": f"{r['want_act']}({r['want_slots']})", "now": f"{act}({slots})"})
                if was and not now:
                    counts[split]["broke"] += 1
                    if len(moved["broke"]) < 4:
                        moved["broke"].append({"split": split, "text": r["text"][:80], "want": f"{r['want_act']}({r['want_slots']})", "was_ok_now": f"{act}({slots})"})
            result[tier] = {
                s: {"n": counts[s]["n"], "correct_before": counts[s]["was"], "correct_after": counts[s]["now"],
                    "fixed": counts[s]["fixed"], "broke": counts[s]["broke"],
                    "accuracy_before": round(counts[s]["was"] / counts[s]["n"], 4) if counts[s]["n"] else None,
                    "accuracy_after": round(counts[s]["now"] / counts[s]["n"], 4) if counts[s]["n"] else None}
                for s in ("design", "heldout")
            }
            result[tier]["examples"] = moved
        report["rules"][config] = result
    report["scoring"] = ("lenient: place/topic/app conventions collapsed on both sides, so a rule is judged on "
                         "what it identifies, not on which spelling a row happened to use") if args.lenient else "strict: exact slot match"
    out = OUT if not args.lenient else OUT.with_name("schema_repair_lenient.json")
    out.write_text(json.dumps(report, indent=1))

    print(f"[{'lenient' if args.lenient else 'strict'} scoring] design set {len(all_fail - excluded)} items (seen; labels adjudicated), "
          f"{len(excluded)} excluded as bad labels, heldout = everything else\n")
    header = f"{'rule':22} {'tier':8} {'design fixed/broke':>19} {'heldout fixed/broke':>21} {'heldout acc':>22}"
    print(header)
    for config, result in report["rules"].items():
        if config == "baseline":
            continue
        for tier in TIERS:
            d, h = result[tier]["design"], result[tier]["heldout"]
            print(f"{config:22} {tier:8} {d['fixed']:8}/{d['broke']:<10} {h['fixed']:10}/{h['broke']:<10} "
                  f"{h['accuracy_before']:.4f} -> {h['accuracy_after']:.4f}")
        print()


if __name__ == "__main__":
    main()
