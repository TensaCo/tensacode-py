"""Is the multi-hop gap a missing composition faculty, or just too little evidence?

    PYTHONPATH=src:. venv-eval/bin/python eval/schema/multihop.py --n 300

The claim under test is narrow and mechanical: that retrieval is nucleated by the words of
the utterance, so evidence reachable only *through* an intermediate result is unreachable at
all. If that is the whole story, then re-seeding a second retrieval with the entity found in
the first should recover it, using nothing but the single-hop answerer we already have.

Five arms, because the interesting confound is evidence volume rather than evidence routing:

  no_retrieval        every sentence, truncated by the answerer's 384-token window. This is
                      how `eval/training/eval_span.py` measured 0.093, and it performs no
                      retrieval at all, so that number cannot by itself indict composition.
  single_hop_k        BM25 top-k by the question. One query, k sentences.
  single_hop_2k       BM25 top-2k by the question. One query, 2k sentences — the same volume
                      of evidence the two-hop arms get, so any gain they show over THIS arm
                      is routing rather than quantity.
  two_hop_reseeded    hop 1 = top-k by the question -> answerer -> span; hop 2 = top-k by
                      that span, excluding what hop 1 already had; answer over the union.
  oracle_gold_only    exactly the gold supporting sentences, nothing else. No retrieval change
                      can ever beat this arm, so it separates "retrieval does not reach the
                      evidence" from "the evidence is in front of it and it cannot compose".
  oracle_selection    retrieve 2k by the question, then keep exactly the gold sentences that are
                      IN that pool and nothing else. Retrieval stays realistic; selection becomes
                      perfect. The distance from this arm to chain_selected is what a better
                      selector could buy; the distance from here to oracle_gold_only is what only
                      better retrieval could buy.
  chain_selected      retrieve 2k by the question, then keep only the <=3 sentences that form a
                      CHAIN: a pair that between them covers the most question terms AND shares a
                      capitalised entity with each other. This is the composition claim moved from
                      retrieval to selection -- evidence must connect, not merely score.
  precision_control   retrieve 2k, then keep the top 3 by BM25 alone. Same number of sentences as
                      chain_selected, chosen without any chaining requirement, so a gain for
                      chain_selected over THIS arm is the chain criterion and not just brevity.
  two_hop_control     identical in every respect, except hop 2 is seeded by the ORIGINAL
                      question. BM25 is deterministic, so that returns ranks k+1..2k: the
                      same two-stage loop, the same 2k sentences, only the seed differs.

Prediction being tested (stated before running, so it can fail): two_hop_reseeded >= 0.30
while two_hop_control stays < 0.15. If the control moves as much, the routing claim is wrong
and the gain is only that two retrievals beat one.

Reported alongside accuracy, and more sensitive than it: gold supporting-fact recall over the
evidence each arm actually hands to its final answering call. Accuracy can fail to move
because retrieval is still broken, or because retrieval was fixed and the answerer cannot
compose. Those are different diagnoses and only the recall number separates them.

Design/heldout split: items are partitioned by a hash of their id, 1/3 design and 2/3 heldout.
Both are reported. Only the design third may be inspected while iterating.
"""

from __future__ import annotations

import os

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from eval.open_domain.data import hotpot  # noqa: E402
from eval.open_domain.score import em, f1, wilson  # noqa: E402
from tensacode.backends.builtin import BM25Ranker  # noqa: E402

OUT = ROOT / "eval" / "results" / "schema_multihop.json"
RANKER = BM25Ranker(text_of=lambda s: s[1])
STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is", "are", "was", "were", "be",
        "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "many", "much", "did", "does",
        "do", "this", "that", "these", "those", "it", "its", "his", "her", "their", "there", "as", "by", "with",
        "from", "first", "name", "also", "known", "between", "during", "after", "before", "than", "then"}


def rank(query: str, candidates: list[tuple[str, str]], limit: int) -> list[tuple[str, str]]:
    return [c for c, _ in RANKER._rank(query, candidates)[:limit]]


def gold_recall(item, evidence: list[tuple[str, str]]) -> tuple[float, bool]:
    """What share of the gold supporting sentences is in the evidence actually used."""
    by_title: dict[str, list[str]] = {}
    for t, s in item.passages:
        by_title.setdefault(t, []).append(s)
    wanted = {(t, by_title[t][i]) for t, i in item.supporting if t in by_title and i < len(by_title[t])}
    if not wanted:
        return (0.0, False)
    have = set(evidence)
    hit = len(wanted & have)
    return (hit / len(wanted), hit == len(wanted))


def question_types() -> dict[str, str]:
    """id -> bridge|comparison, read from the parquet rather than added to the shared loader."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    from eval.open_domain.data import DATA  # noqa: PLC0415

    table = pq.read_table(DATA / "hotpot_distractor_validation.parquet", columns=["id", "type"])
    return dict(zip(table.column("id").to_pylist(), table.column("type").to_pylist()))


ENTITY = re.compile(r"\b[A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+)*")


def entities(text: str) -> set[str]:
    """Capitalised runs, minus the sentence-initial word, which is capitalised by position."""
    found = {m.group(0) for m in ENTITY.finditer(text)}
    return {e for e in found if len(e) > 2 and e.lower() not in STOP}


def content(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP and len(w) > 2}


def chain_select(question: str, pool: list[tuple[str, str]], limit: int = 3) -> list[tuple[str, str]]:
    """The sentences that form a chain from the question, rather than the ones that score highest.

    A multi-hop question is answered by evidence that is *connected*: one sentence anchors on the
    question's own terms, the next shares an entity with the first and supplies what the question
    still lacks. Scoring each sentence independently against the question cannot express that, so
    top-k keeps near-duplicates of hop 1 and drops hop 2 entirely. This scores PAIRS: question
    coverage of the union, plus a bonus for sharing an entity, minus a penalty for redundancy.
    """
    if len(pool) <= limit:
        return pool
    q = content(question)
    best, best_score = None, -1e9
    for i, a in enumerate(pool):
        wa, ea = content(a[1]), entities(a[1]) | {a[0]}
        for b in pool[i + 1:]:
            wb, eb = content(b[1]), entities(b[1]) | {b[0]}
            covered = len((wa | wb) & q) / (len(q) or 1)
            linked = 1.0 if (ea & eb) else 0.0
            redundant = len(wa & wb) / (len(wa | wb) or 1)
            score = 2.0 * covered + 0.6 * linked - 0.5 * redundant
            if score > best_score:
                best, best_score = (a, b), score
    chosen = list(best or pool[:2])
    if limit > 2:  # one spare sentence, for the question terms the pair still does not cover
        missing = q - content(" ".join(s for _, s in chosen))
        rest = [p for p in pool if p not in chosen]
        if missing and rest:
            chosen.append(max(rest, key=lambda p: len(content(p[1]) & missing)))
    return [p for p in pool if p in chosen]  # keep the passage order the answerer saw


def gold_sentences(item) -> list[tuple[str, str]]:
    by_title: dict[str, list[str]] = {}
    for t, sent in item.passages:
        by_title.setdefault(t, []).append(sent)
    return [(t, by_title[t][i]) for t, i in item.supporting if t in by_title and i < len(by_title[t])]


def split_of(item_id: str) -> str:
    return "design" if int(hashlib.sha1(item_id.encode()).hexdigest(), 16) % 3 == 0 else "heldout"


def seed_from_span(span: str, question: str) -> str:
    """The second retrieval's query: the entity hop 1 found, plus the question's content words.

    The span alone is a poor BM25 query (one or two terms, often a common name). The question's
    own terms are what the utterance already nucleated on, and hop 1 already used them; keeping
    them here is what makes the control a fair comparison rather than a weaker query.
    """
    return span.strip() or question


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--artifact", type=Path,
                    default=Path(os.environ.get("TENSACODE_SCRATCH", os.path.expanduser("~/.cache/tensacode")) + "/artifacts/span-answerer"))
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    from tensacode.backends.neural import NeuralAnswerer, QuestionOverPassages  # noqa: PLC0415

    answerer = NeuralAnswerer(args.artifact)
    items = hotpot(args.n, seed=0)
    kinds = question_types()

    def ask(pairs: list[tuple[str, list[tuple[str, str]]]]) -> list:
        """One batched answering call: (question, evidence) -> Answer."""
        qs = [QuestionOverPassages(q, tuple(ev)) for q, ev in pairs]
        return answerer.answer(qs)

    k = args.k
    arms: dict[str, list] = {}
    evidence_used: dict[str, list[list[tuple[str, str]]]] = {}
    timing: dict[str, float] = {}

    # --- the three single-query arms
    for name, chooser in (("no_retrieval", lambda it: list(it.passages)),
                          ("oracle_gold_only", gold_sentences),
                          ("single_hop_k", lambda it: rank(it.question, list(it.passages), k)),
                          ("single_hop_2k", lambda it: rank(it.question, list(it.passages), 2 * k))):
        t0 = time.time()
        ev = [chooser(it) for it in items]
        arms[name] = ask([(it.question, e) for it, e in zip(items, ev)])
        evidence_used[name] = ev
        timing[name] = round(time.time() - t0, 2)

    # --- selection arms: same retrieval, different criterion for what to keep
    gold_of = {it.id: set(gold_sentences(it)) for it in items}
    for name, select in (("chain_selected", lambda it, pool: chain_select(it.question, pool, 3)),
                         ("precision_control", lambda it, pool: pool[:3]),
                         ("oracle_selection", lambda it, pool: [p for p in pool if p in gold_of[it.id]] or pool[:2])):
        t0 = time.time()
        ev = []
        for it in items:
            pool = rank(it.question, list(it.passages), 2 * k)
            ev.append(select(it, pool))
        arms[name] = ask([(it.question, e) for it, e in zip(items, ev)])
        evidence_used[name] = ev
        timing[name] = round(time.time() - t0, 2)

    # --- hop 1, shared by both two-hop arms so they differ in exactly one thing
    t0 = time.time()
    hop1_ev = [rank(it.question, list(it.passages), k) for it in items]
    hop1 = ask([(it.question, e) for it, e in zip(items, hop1_ev)])
    hop1_seconds = time.time() - t0

    # --- hop 2: re-seeded by what hop 1 found, versus re-seeded by the original question
    for name, query_of in (("two_hop_reseeded", lambda it, got: seed_from_span(got.text or "", it.question)),
                           ("two_hop_control", lambda it, got: it.question)):
        t0 = time.time()
        ev = []
        for it, got, seen in zip(items, hop1, hop1_ev):
            rest = [p for p in it.passages if p not in set(seen)]
            ev.append(seen + rank(query_of(it, got), rest, k))
        arms[name] = ask([(it.question, e) for it, e in zip(items, ev)])
        evidence_used[name] = ev
        timing[name] = round(time.time() - t0 + hop1_seconds, 2)

    # --- score every arm on both halves, and by question type
    report = {
        "what": "does re-seeding a second retrieval with hop 1's result recover multi-hop answers, "
                "or does any second retrieval do as well",
        "prediction_stated_before_running": {
            "reseeding": "two_hop_reseeded >= 0.30 EM while two_hop_control < 0.15",
            "chaining": "chain_selected beats precision_control by >= 0.03 EM and reaches >= 0.33; "
                        "if precision_control matches it, the gain is brevity and not chaining",
        },
        "setup": {"benchmark": "HotpotQA distractor validation", "n": len(items), "k_sentences_per_hop": k,
                  "answerer": str(args.artifact), "answerer_trained_on": "SQuAD 2.0 train (single paragraph)",
                  "retrieval": "the repo's BM25Ranker over the item's own 40-ish sentences",
                  "grader": "dataset labels, exact match and token F1; no model judges anything",
                  "provenance": "inputs: public dataset; gold: dataset authors; tunable_against: none "
                                "(k was not swept on this data)"},
        "reference_points": {"published_no_retrieval_learned_answerer": 0.0933,
                             "published_rules_arm": 0.0584,
                             "local_8b_model_prompted_plainly": 0.5133,
                             "source": "eval/results/learned_answerer.json and eval/results/open_domain.json"},
        "arms": {},
    }
    for name, answers in arms.items():
        per_split: dict[str, dict] = {}
        for split in ("design", "heldout", "all"):
            sel = [(it, a, e) for it, a, e in zip(items, answers, evidence_used[name])
                   if split == "all" or split_of(it.id) == split]
            if not sel:
                continue
            attempted = [(it, a, e) for it, a, e in sel if (a.text or "").strip()]
            correct = sum(em(a.text, it.gold) for it, a, _ in attempted)
            lo, hi = wilson(correct, len(attempted))
            recall = [gold_recall(it, e) for it, _, e in sel]
            per_split[split] = {
                "n": len(sel), "coverage": round(len(attempted) / len(sel), 4),
                "em_over_attempted": round(correct / len(attempted), 4) if attempted else None,
                "em_over_attempted_ci95": [round(lo, 4), round(hi, 4)],
                "em_over_all": round(correct / len(sel), 4),
                "f1_over_attempted": round(sum(f1(a.text, it.gold) for it, a, _ in attempted) / len(attempted), 4) if attempted else None,
                "gold_fact_recall": round(sum(r for r, _ in recall) / len(recall), 4),
                "all_gold_facts_present": round(sum(c for _, c in recall) / len(recall), 4),
                "mean_sentences_given": round(sum(len(e) for _, _, e in sel) / len(sel), 2),
            }
        by_type: dict[str, dict] = {}
        for kind in ("bridge", "comparison"):
            sel = [(it, a) for it, a in zip(items, answers) if kinds.get(it.id) == kind]
            if sel:
                att = [(it, a) for it, a in sel if (a.text or "").strip()]
                by_type[kind] = {"n": len(sel),
                                 "em_over_all": round(sum(em(a.text, it.gold) for it, a in att) / len(sel), 4)}
        report["arms"][name] = {"splits": per_split, "by_question_type": by_type, "seconds": timing[name]}

    rows = []
    for name, answers in arms.items():
        for it, a, e in zip(items, answers, evidence_used[name]):
            recall, complete = gold_recall(it, e)
            rows.append({"arm": name, "id": it.id, "type": kinds.get(it.id), "split": split_of(it.id),
                         "question": it.question, "gold": it.gold, "pred": a.text, "confidence": a.confidence,
                         "em": bool(em(a.text, it.gold)), "f1": round(f1(a.text, it.gold), 4),
                         "gold_recall": round(recall, 4), "all_gold_present": complete, "n_sentences": len(e)})
    (Path(__file__).parent / "corpus" / "multihop_items.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows))

    # --- the conditional that decides the question: accuracy when the evidence IS in the window
    conditional = {}
    for name, answers in arms.items():
        sel = [(it, a) for it, a, e in zip(items, answers, evidence_used[name]) if gold_recall(it, e)[1]]
        att = [(it, a) for it, a in sel if (a.text or "").strip()]
        if sel:
            conditional[name] = {
                "items_with_all_gold_facts_present": len(sel),
                "coverage": round(len(att) / len(sel), 4),
                "em_over_all": round(sum(em(a.text, it.gold) for it, a in att) / len(sel), 4),
                "em_over_attempted": round(sum(em(a.text, it.gold) for it, a in att) / len(att), 4) if att else None,
            }
    report["accuracy_when_all_gold_evidence_is_present"] = {
        "why": "if this is low, no retrieval change can help: the evidence is already in front of the answerer",
        "arms": conditional,
    }
    args.out.write_text(json.dumps(report, indent=1))

    print(f"HotpotQA distractor, n={len(items)}, k={k} sentences per hop\n")
    print(f"{'arm':20} {'sents':>6} {'EM all':>8} {'EM att':>8} {'cover':>7} {'gold recall':>12} {'all gold':>9}")
    for name, a in report["arms"].items():
        s = a["splits"]["all"]
        print(f"{name:20} {s['mean_sentences_given']:6.1f} {s['em_over_all']:8.4f} "
              f"{(s['em_over_attempted'] or 0):8.4f} {s['coverage']:7.4f} {s['gold_fact_recall']:12.4f} "
              f"{s['all_gold_facts_present']:9.4f}")
    print(f"\n{'arm':20} {'when all gold present: n':>26} {'EM':>7} {'cover':>7}")
    for name, c in report["accuracy_when_all_gold_evidence_is_present"]["arms"].items():
        print(f"{name:20} {c['items_with_all_gold_facts_present']:26} {c['em_over_all']:7.4f} {c['coverage']:7.4f}")
    print(f"\n{'arm':20} {'design EM':>10} {'heldout EM':>11}")
    for name, a in report["arms"].items():
        d, h = a["splits"].get("design", {}), a["splits"].get("heldout", {})
        print(f"{name:20} {d.get('em_over_all', 0):10.4f} {h.get('em_over_all', 0):11.4f}")
    def em_of(name: str) -> float:
        return report["arms"][name]["splits"]["all"]["em_over_all"]

    t, c = em_of("two_hop_reseeded"), em_of("two_hop_control")
    print(f"\nreseeding prediction (reseeded >= 0.30, control < 0.15) -> "
          f"reseeded {t:.4f}, control {c:.4f}: {'HOLDS' if t >= 0.30 and c < 0.15 else 'FAILS'}")
    ch, pc = em_of("chain_selected"), em_of("precision_control")
    print(f"chaining prediction (chain >= 0.33 and beats control by >= 0.03) -> "
          f"chain {ch:.4f}, control {pc:.4f}, delta {ch - pc:+.4f}: "
          f"{'HOLDS' if ch >= 0.33 and ch - pc >= 0.03 else 'FAILS'}")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
