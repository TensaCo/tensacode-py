"""Where does the oracle arm lose? Four mechanisms, counted, on the same 300 items.

    PYTHONPATH=src:. venv-eval/bin/python eval/schema/multihop_failures.py

`oracle_gold_only` in `eval/schema/multihop.py` hands the answerer exactly the gold supporting
sentences and still scores 0.433 EM. No retrieval or routing change can touch that 0.567, so it
is the whole of the remaining multi-hop gap and worth attributing precisely. The mechanisms are
mutually exclusive and assigned in this order:

  answer_type_unavailable  the gold answer is yes/no and the tier emits spans. It cannot be right.
  bridge_entity_returned   the prediction is the intermediate entity the question travels THROUGH
                           -- the title of a gold passage, or a phrase already in the question --
                           rather than the thing asked for. This is the failure to distinguish a
                           subgoal's value from the goal's value.
  span_boundary            the prediction overlaps the gold but is not equal to it: right referent,
                           wrong extent ('Alachua County' for 'Alachua', '28,776' for
                           '28,776 at the 2010 census').
  wrong_span               everything else: it picked an unrelated span.

Counted separately: whether the tier was confident while wrong, because a mechanism that comes
with high confidence cannot be handled by abstention.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from eval.open_domain.data import hotpot  # noqa: E402
from eval.open_domain.score import f1  # noqa: E402

ITEMS = Path(__file__).parent / "corpus" / "multihop_items.jsonl"
OUT = ROOT / "eval" / "results" / "schema_multihop_failures.json"


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", (text or "").lower()))


def mechanism(row: dict, item, titles: set[str]) -> str:
    gold = (row["gold"][0] or "").strip()
    pred = (row["pred"] or "").strip()
    if gold.lower() in ("yes", "no"):
        return "answer_type_unavailable"
    if not pred:
        return "abstained"
    if row["f1"] > 0:
        return "span_boundary"
    # the intermediate entity: a gold passage title, or a phrase the question already contains
    pw, qw = words(pred), words(item.question)
    if any(words(t) and words(t) <= pw or pw and pw <= words(t) for t in titles):
        return "bridge_entity_returned"
    if pw and pw <= qw:
        return "bridge_entity_returned"
    return "wrong_span"


def main() -> None:
    rows = [json.loads(l) for l in ITEMS.read_text().splitlines()]
    items = {it.id: it for it in hotpot(300, seed=0)}

    report = {
        "what": "attribution of the oracle arm's failures, where retrieval is perfect by construction",
        "source": "eval/schema/corpus/multihop_items.jsonl, arm=oracle_gold_only",
        "provenance": "inputs: HotpotQA; gold: dataset authors; mechanism labels: assigned by rule, "
                      "not by hand, so they are reproducible and can be wrong in bulk rather than selectively",
        "arms": {},
    }
    for arm in ("oracle_gold_only", "two_hop_reseeded"):
        sel = [r for r in rows if r["arm"] == arm]
        by_mech: Counter = Counter()
        confident_wrong: Counter = Counter()
        examples: dict[str, list] = {}
        splits: dict[str, Counter] = {"design": Counter(), "heldout": Counter()}
        for r in sel:
            it = items[r["id"]]
            titles = {t for t, _ in it.passages if t in {tt for tt, _ in it.supporting}}
            if r["em"]:
                by_mech["correct"] += 1
                splits[r["split"]]["correct"] += 1
                continue
            m = mechanism(r, it, titles)
            by_mech[m] += 1
            splits[r["split"]][m] += 1
            if (r["confidence"] or 0) >= 0.5:
                confident_wrong[m] += 1
            examples.setdefault(m, [])
            if len(examples[m]) < 4 and r["split"] == "design":
                examples[m].append({"q": it.question[:110], "gold": r["gold"], "pred": r["pred"],
                                    "confidence": round(r["confidence"] or 0, 3)})
        n = len(sel)
        report["arms"][arm] = {
            "n": n,
            "share": {m: round(c / n, 4) for m, c in by_mech.most_common()},
            "counts": dict(by_mech.most_common()),
            "confident_while_wrong": dict(confident_wrong.most_common()),
            "by_split": {s: dict(c.most_common()) for s, c in splits.items()},
            "design_examples": examples,
        }
    OUT.write_text(json.dumps(report, indent=1))

    for arm, a in report["arms"].items():
        print(f"\n{arm}  (n={a['n']})")
        print(f"  {'mechanism':26} {'items':>6} {'share':>7} {'confident while wrong':>22}")
        for m, c in a["counts"].items():
            print(f"  {m:26} {c:6} {a['share'][m]:7.3f} {a['confident_while_wrong'].get(m, 0):22}")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
