"""Does a relation operation close the comparison gap doc 26 measured, and at what cost?

    PYTHONPATH=src:. venv-eval/bin/python eval/relations/eval_relations.py

Doc 26 found that comparison questions score 0.1935 EM with the gold evidence in hand and 0.1935
through a trained selector — identical, so the gap is not selection. This measures
:mod:`tensorcode.relation`, which reads the relation and the two candidates off the question,
reads one value per candidate out of the evidence, applies the relation and returns what was
asked for; and which refuses, with a named reason, when a value is missing or two values are
incomparable.

Three predictions per arm are kept apart, because conflating them is how a faculty gets credit it
has not earned:

``span``      the extractive answerer alone, which is doc 26's number and the baseline.
``relation``  the operation alone, on the items where it resolves, against the answerer on those
              same items — the only comparison that attributes the change to the operation.
``combined``  the operation where it resolves, the answerer everywhere else. This is the arm the
              pre-registered bar is about, because a refusal still has to be answered by someone.

Bridge questions are reported in every arm. An operation that buys comparison accuracy by
damaging the other 79% of the dataset has not helped, and the only way to know is to look.
"""

from __future__ import annotations

import os

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

SP = Path(os.environ.get("TENSORCODE_SCRATCH", os.path.expanduser("~/.cache/tensorcode")))
OUT = ROOT / "eval" / "results" / "relation_hotpot.json"

#: doc 26, table "arms", for the same 300 items
BASELINE = {
    "oracle_gold_only": {"all": 0.4333, "bridge": 0.4958, "comparison": 0.1935},
    "selector_k8": {"all": 0.3600, "bridge": 0.4034, "comparison": 0.1935},
}


def summarise(rows: list[dict], key: str = "em") -> dict:
    from eval.open_domain.score import wilson
    n = len(rows)
    if not n:
        return {"n": 0}
    correct = sum(bool(r[key]) for r in rows)
    lo, hi = wilson(correct, n)
    return {"n": n, "em": round(correct / n, 4), "em_ci": [round(lo, 4), round(hi, 4)],
            "f1": round(sum(r["f1"] for r in rows) / n, 4),
            "contained": round(sum(r["contained"] for r in rows) / n, 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=8, help="selector arm size; doc 26's best")
    ap.add_argument("--selector", type=Path, default=SP / "artifacts" / "sentence-selector")
    ap.add_argument("--artifact", type=Path, default=SP / "artifacts" / "span-answerer")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    from eval.open_domain.data import hotpot
    from eval.open_domain.score import em, f1
    from eval.schema.multihop import gold_sentences, question_types
    from eval.selection.eval_selection import contains
    from eval.selection.selector import Selector, truncation_of
    from tensorcode.answer_type import Shape, shape
    from tensorcode.backends.neural import NeuralAnswerer, QuestionOverPassages
    from tensorcode.outcomes import Unknown
    from tensorcode.relation import Resolved, read, resolve

    items = hotpot(args.n, seed=0)
    kinds = question_types()
    answerer = NeuralAnswerer(args.artifact)
    selector = Selector(args.selector)

    t0 = time.time()
    scores = [selector.score(it.question, it.passages) for it in items]
    print(f"scored {sum(len(s) for s in scores)} sentences in {time.time() - t0:.0f}s", flush=True)

    def top(i: int, k: int) -> list[tuple[str, str]]:
        keep = {j for j, _ in sorted(enumerate(scores[i]), key=lambda p: -p[1])[:k]}
        return [s for j, s in enumerate(items[i].passages) if j in keep]

    arms = {"oracle_gold_only": [gold_sentences(it) for it in items],
            f"selector_k{args.k}": [top(i, args.k) for i in range(len(items))]}

    report: dict[str, dict] = {}
    detection = Counter()
    refusals: Counter = Counter()
    families: Counter = Counter()
    adjudicate: list[dict] = []

    for name, evidence in arms.items():
        pairs = [(it.question, " ".join(s for _, s in ev)) for it, ev in zip(items, evidence)]
        trunc = truncation_of(answerer.tokenizer, pairs, answerer.max_length)
        spans = [a.text for a in answerer.answer([QuestionOverPassages(it.question, tuple(ev))
                                                  for it, ev in zip(items, evidence)])]

        rows: list[dict] = []
        for it, ev, span in zip(items, evidence, spans):
            out = resolve(it.question, tuple(ev))
            resolved = isinstance(out, Resolved)
            answer = out.text if resolved else span
            row = {
                "id": it.id, "kind": kinds.get(it.id, "?"),
                "yes_no_gold": it.gold[0].strip().lower() in ("yes", "no"),
                "resolved": resolved,
                "relation": out.relation.value if resolved else None,
                "reason": None if resolved else out.reason,
                "span": span, "answer": answer,
                "em": em(answer, it.gold), "f1": f1(answer, it.gold),
                "contained": contains(answer, it.gold),
                "span_em": em(span, it.gold) if span else False,
                "span_f1": f1(span, it.gold) if span else 0.0,
                "span_contained": contains(span, it.gold) if span else False,
            }
            rows.append(row)
            if name == "oracle_gold_only":
                got = read(it.question, tuple(dict.fromkeys(t for t, _ in ev)))
                fired = not isinstance(got, Unknown)
                detection[(kinds.get(it.id, "?"), "read" if fired else "not_read")] += 1
                if fired:
                    families[got.family] += 1
                if not resolved:
                    refusals[out.reason] += 1
                if kinds.get(it.id) == "comparison":
                    adjudicate.append({
                        "question": it.question, "gold": it.gold[0],
                        "relation_answer": out.text if resolved else None,
                        "refused": None if resolved else f"{out.reason}: {out.detail}",
                        "steps": list(out.steps) if resolved else [],
                        "span_answer": span, "em": row["em"], "span_em": row["span_em"],
                        "shape": shape(it.question).value,
                    })

        comparison = [r for r in rows if r["kind"] == "comparison"]
        bridge = [r for r in rows if r["kind"] == "bridge"]
        fired = [r for r in comparison if r["resolved"]]
        report[name] = {
            "truncation": trunc.report(),
            "combined": {"all": summarise(rows), "bridge": summarise(bridge),
                         "comparison": summarise(comparison),
                         "yes_no_gold": summarise([r for r in rows if r["yes_no_gold"]])},
            "span_only": {"all": summarise(rows, "span_em"), "bridge": summarise(bridge, "span_em"),
                          "comparison": summarise(comparison, "span_em"),
                          "yes_no_gold": summarise([r for r in rows if r["yes_no_gold"]], "span_em")},
            "on_items_the_relation_resolved": {
                "n": len(fired),
                "share_of_comparisons": round(len(fired) / max(1, len(comparison)), 4),
                "relation_em": round(sum(r["em"] for r in fired) / max(1, len(fired)), 4),
                "answerer_em_same_items": round(sum(r["span_em"] for r in fired) / max(1, len(fired)), 4),
                "relation_f1": round(sum(r["f1"] for r in fired) / max(1, len(fired)), 4),
                "answerer_f1_same_items": round(sum(r["span_f1"] for r in fired) / max(1, len(fired)), 4),
                "relation_contained": round(sum(r["contained"] for r in fired) / max(1, len(fired)), 4),
            },
            "bridge_items_the_relation_touched": sum(1 for r in bridge if r["resolved"]),
            "bridge_em_change": round(summarise(bridge)["em"] - summarise(bridge, "span_em")["em"], 4),
            "refusal_share_of_comparisons": round(
                sum(1 for r in comparison if not r["resolved"]) / max(1, len(comparison)), 4),
            "by_relation": {
                rel: {"n": sum(1 for r in rows if r["relation"] == rel),
                      "em": round(sum(r["em"] for r in rows if r["relation"] == rel)
                                  / max(1, sum(1 for r in rows if r["relation"] == rel)), 4)}
                for rel in sorted({r["relation"] for r in rows if r["relation"]})},
            "doc26_baseline": BASELINE.get(name, {}),
        }
        r = report[name]
        print(f"\n== {name} (truncated {r['truncation']['share_truncated']:.2f})")
        for arm in ("span_only", "combined"):
            a = r[arm]
            print(f"  {arm:10s} all {a['all']['em']:.4f}  bridge {a['bridge']['em']:.4f}  "
                  f"comparison {a['comparison']['em']:.4f}  yes/no {a['yes_no_gold']['em']:.4f} "
                  f"(n={a['yes_no_gold']['n']})")
        f = r["on_items_the_relation_resolved"]
        print(f"  relation fired on {f['n']}/{report[name]['combined']['comparison']['n']} comparisons"
              f" ({f['share_of_comparisons']:.2f}): EM {f['relation_em']:.4f} vs answerer "
              f"{f['answerer_em_same_items']:.4f} on the same items", flush=True)

    tp = detection[("comparison", "read")]
    fp = detection[("bridge", "read")]
    fn = detection[("comparison", "not_read")]
    body = {
        "measured_on": f"HotpotQA distractor validation, hotpot({args.n}, seed=0) — the same items as doc 26",
        "designed_on": "hotpot_distractor_train_shard0.parquet, 600 comparison questions, seed 0",
        "answerer": answerer.config.get("encoder"), "answerer_max_length": answerer.max_length,
        "selector": selector.config,
        "arms": report,
        "detection": {"comparison_read": tp, "bridge_read": fp, "comparison_not_read": fn,
                      "precision": round(tp / max(1, tp + fp), 4),
                      "recall": round(tp / max(1, tp + fn), 4),
                      "note": "detection is surface rules over the question, not HotpotQA's type field"},
        "families_read": dict(families),
        "refusal_reasons": dict(refusals),
        "adjudication_sample": adjudicate,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=1))
    print(f"\ndetection precision {body['detection']['precision']:.4f} recall {body['detection']['recall']:.4f}")
    print(f"families {dict(families)}\nrefusals {dict(refusals)}\nwrote {args.out}")


if __name__ == "__main__":
    main()
