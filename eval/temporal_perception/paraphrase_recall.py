"""Can a cue find a memory when the cue is not worded the way the memory was?

Three recallers answer the same questions over the same live store:

    exact     the cue must contain the claim's predicate verbatim (no tolerance at all)
    overlap   word n-gram overlap against the rendered claim — what ``Memory.recall`` does today
    structure ``tensacode.cues``: role-wise lemma overlap, widened by the store's own links

The store is real: the facts are told to the live assistant, so the claims have whatever shape the
assistant gives them. The cues are authored (there is no corpus of people paraphrasing questions
about a simulated desktop), so half of them are held out: the role weights were chosen looking only
at the training half, and the held-out numbers are reported separately and are the ones that count.

    python -m eval.temporal_perception.paraphrase_recall
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from tensacode.context import shingle_similarity
from tensacode.cues import Cues, RoleWeights, exact_find
from tensacode.memory import _describe

from .live_harness import Session

OUT = Path(__file__).resolve().parents[2] / "eval" / "results" / "paraphrase_recall.json"

# (what we tell it, predicate the assistant files it under, the value, cues that should find it)
# Cue wordings alternate train/held-out by position, so neither half is the easy one.
FACTS = [
    ("my cat is Mackerel", "cat", "Mackerel",
     ["what is my cat", "what did I say about cats", "the name of my cat", "my cats name"]),
    ("my notes folder is ~/notes", "notes folder", "~/notes",
     ["what is my notes folder", "where are my notes kept", "the folder with my notes", "my note folder"]),
    ("my project is Halverson", "project", "Halverson",
     ["what is my project", "which project am I on", "the project I mentioned", "my projects name"]),
    ("my deadline is the 30th", "deadline", "the 30th",
     ["what is my deadline", "when is the deadline", "the date I have to finish by", "my deadlines"]),
    ("my office is room 412", "office", "room 412",
     ["what is my office", "which room is my office", "where do I sit", "my offices number"]),
    ("my bike is a Brompton", "bike", "a Brompton",
     ["what is my bike", "what bike do I ride", "the make of my bicycle", "my bikes brand"]),
    ("my editor is neovim", "editor", "neovim",
     ["what is my editor", "which editor do I use", "the text editor I prefer", "my editors name"]),
    ("my timezone is UTC+1", "timezone", "UTC+1",
     ["what is my timezone", "which timezone am I in", "my time zone", "the zone I am in"]),
]


def rank_of(hits, predicate: str, value: str) -> int | None:
    for i, hit in enumerate(hits):
        rec = hit.record if hasattr(hit, "record") else hit[0]
        if rec.claim.predicate == predicate and value.lower() in str(rec.claim.object).lower():
            return i + 1
    return None


def overlap_find(mind, cue: str, k: int = 3, *, n: int = 2):
    """Token overlap against the rendered claim. n=2 is what ``Memory.recall`` does today; n=1 is
    the stronger baseline — plain word overlap, which is what beat embeddings elsewhere in this
    repository and is therefore the number worth being compared against."""
    scored = []
    for rec in mind.claims():
        if rec.retracted:
            continue
        score = shingle_similarity(cue, _describe(rec), n=n)
        if score.value > 0:
            scored.append((rec, score))
    scored.sort(key=lambda pair: (-pair[1].value, pair[0].id))
    return scored[:k]


def main() -> None:
    rows: list[dict] = []
    # Mechanism check, labelled as such: the link path is only worth having if a fact the agent has
    # been told actually bridges a cue it could not bridge before. One linking claim is written
    # directly (by the grader, not by the assistant's language layer) and the cues that failed for
    # want of it are asked again.
    link_demo: list[dict] = []
    with Session(hostname="recall-eval") as s:
        for told, *_ in FACTS:
            s.send(told)
        s.send("hello")  # a turn of ordinary traffic between telling and asking

        # weights chosen on the training half only (the defaults in cues.RoleWeights)
        index = Cues(s.mind, weights=RoleWeights(), skip_predicates=("words", "goal:what", "text", "said", "summary"))
        unbridged = [("the make of my bicycle", "bike", "Brompton"), ("where do I sit", "office", "412")]
        for told, predicate, value, cues in FACTS:
            for i, cue in enumerate(cues):
                split = "train" if i % 2 == 0 else "held_out"
                ranks = {
                    "exact": rank_of(exact_find(s.mind, cue, k=5), predicate, value),
                    "overlap_bigram": rank_of(overlap_find(s.mind, cue, k=5, n=2), predicate, value),
                    "overlap_word": rank_of(overlap_find(s.mind, cue, k=5, n=1), predicate, value),
                    "structure": rank_of(index.find(cue, k=5), predicate, value),
                }
                top = index.find(cue, k=1)
                rows.append({"cue": cue, "split": split, "target": f"{predicate}={value}", "ranks": ranks,
                             "structure_top": _describe(top[0].record) if top else "",
                             "structure_why": top[0].why() if top else ""})
                mark = "".join("." if ranks[r] == 1 else ("+" if ranks[r] else "x")
                               for r in ("exact", "overlap_bigram", "overlap_word", "structure"))
                print(f"  [{split:<8}] {mark}  {cue!r} -> {rows[-1]['structure_top'][:56]!r}")

        import tensacode as tc
        for cue, predicate, value in unbridged:
            before = rank_of(index.find(cue, k=5), predicate, value)
            word = {"bike": "bicycle", "office": "desk"}[predicate]
            s.mind.tell(tc.Claim(tc.Ref(f"word:{word}"), "same_as", tc.Ref(f"word:{predicate}")),
                        tc.Evidence(tc.Ref("grader:link-demo"), datetime.now(timezone.utc),
                                    method="authored-for-the-demonstration"))
            after = rank_of(Cues(s.mind, skip_predicates=("words", "goal:what", "text", "said", "summary")).find(cue, k=5),
                            predicate, value)
            link_demo.append({"cue": cue, "link_added": f"word:{word} same_as word:{predicate}",
                              "rank_before": before, "rank_after": after})
            print(f"  link demo: {cue!r} rank {before} -> {after} after telling it {word} = {predicate}")

    def metrics(split: str | None, recaller: str) -> dict:
        subset = [r for r in rows if split is None or r["split"] == split]
        ranks = [r["ranks"][recaller] for r in subset]
        return {"n": len(ranks),
                "recall@1": round(sum(1 for x in ranks if x == 1) / len(ranks), 3),
                "recall@3": round(sum(1 for x in ranks if x and x <= 3) / len(ranks), 3),
                "mrr": round(statistics.mean([1 / x if x else 0.0 for x in ranks]), 3)}

    RECALLERS = ("exact", "overlap_bigram", "overlap_word", "structure")
    report = {
        "what": "cue-based recall surviving paraphrase, by structure rather than learned vectors",
        "provenance": {"environment": "facts told to the live assistant; claims have the shape it gives them",
                       "grader": "this script — it knows the target because it planted the fact",
                       "held_out": "half the cue wordings (every second one); role weights were set on the training half only",
                       "author": "the cue paraphrases are authored by the same model that wrote the recaller — the honest weakness of this measurement"},
        "rows": rows,
        "overall": {r: metrics(None, r) for r in RECALLERS},
        "train": {r: metrics("train", r) for r in RECALLERS},
        "held_out": {r: metrics("held_out", r) for r in RECALLERS},
        "link_demonstration": link_demo,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    for split in ("train", "held_out"):
        line = " · ".join(f"{r}: @1={report[split][r]['recall@1']} mrr={report[split][r]['mrr']}"
                          for r in RECALLERS)
        print(f"{split:<9} {line}")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
