"""Does the agent need a whole correct parse, or only a correct clause?

    python -m eval.parsing.whole_vs_clause [--split dev]

At LAS 0.77, a long sentence is almost never *entirely* right, and the agent's converter
asks for the whole tree: it builds one frame from the root and asserts what it finds. That
is the wrong unit if most sentences have some correct clauses inside an incorrect tree.

So this measures three things on the same sentences:

* **whole-sentence accuracy** — every non-punctuation arc correct. What the current
  converter effectively needs;
* **clause accuracy** — for each verb, whether *its own* core arguments (subject, object,
  indirect object, obliques, clausal complements) are all attached and labelled correctly.
  What a converter that asserted one proposition per clause would need;
* the same two, by sentence length, because that is where the loss was found to be.

If clause accuracy is much higher than whole-sentence accuracy, reading clause by clause is
worth more than any amount of extra treebank data — and it is a change to the converter, not
to the parser.

A row is appended to ``eval/results/parsing_whole_vs_clause.jsonl``.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parents[2] / "src")]

from tensorcode.language.learned_parser import load_model  # noqa: E402
from tensorcode.language.treebank import load_with_genres  # noqa: E402

OUT = Path(__file__).parents[1] / "results" / "parsing_whole_vs_clause.jsonl"
MODEL = Path.home() / ".cache" / "tensorcode" / "models" / "ud_ewt_parser.pickle"

#: The dependents that make a clause mean what it means. A wrong `det` or `punct` does not
#: change which proposition the clause states; a wrong `obj` does.
CORE = ("nsubj", "obj", "iobj", "obl", "ccomp", "xcomp", "nsubj:pass", "obl:agent")
BANDS = ((1, 10), (11, 20), (21, 35), (36, 10_000))


def band(n: int) -> str:
    for low, high in BANDS:
        if low <= n <= high:
            return f"{low}-{high if high < 10_000 else '+'}"
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--i-am-running-the-test-split", action="store_true")
    args = ap.parse_args()
    if args.split == "test" and not args.i_am_running_the_test_split:
        sys.exit("test is for headline numbers: pass --i-am-running-the-test-split")
    got = load_model(MODEL)
    if got is None:
        sys.exit(f"no trained parser at {MODEL}: run eval/parsing/train_ud.py")
    tagger, parser = got
    tagged = load_with_genres(args.split)
    if not tagged:
        sys.exit("no treebank found")

    whole = collections.Counter()
    clauses = collections.Counter()
    for _, s in tagged:
        words = [t.form for t in s]
        heads, labels = parser.parse(words, tagger.tag(words))
        b = band(len(s))
        arcs = [t for t in s if t.upos != "PUNCT"]
        ok = all(heads.get(t.id) == t.head and labels.get(t.id) == t.deprel for t in arcs)
        whole[b, "n"] += 1
        whole[b, "right"] += int(ok)
        # one clause per verb, judged on its own core arguments
        for verb in (t for t in s if t.upos in ("VERB", "AUX")):
            core = [t for t in s if t.head == verb.id and t.deprel.split(":")[0] in
                    {c.split(":")[0] for c in CORE}]
            if not core:
                continue
            attached = heads.get(verb.id) == verb.head
            right = attached and all(heads.get(t.id) == t.head and labels.get(t.id) == t.deprel for t in core)
            clauses[b, "n"] += 1
            clauses[b, "right"] += int(right)

    def rate(counter, b):
        n = counter[b, "n"]
        return round(counter[b, "right"] / n, 4) if n else None

    bands = [band(n) for n in (5, 15, 25, 40)]
    row = {"eval": "parsing_whole_vs_clause", "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "split": args.split,
           "whole_sentence": {b: {"n": whole[b, "n"], "accuracy": rate(whole, b)} for b in bands},
           "clause": {b: {"n": clauses[b, "n"], "accuracy": rate(clauses, b)} for b in bands},
           "whole_sentence_overall": round(sum(whole[b, "right"] for b in bands) / max(1, sum(whole[b, "n"] for b in bands)), 4),
           "clause_overall": round(sum(clauses[b, "right"] for b in bands) / max(1, sum(clauses[b, "n"] for b in bands)), 4)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")

    print(f"{'length':10}{'sentences':>10}{'whole right':>13}{'clauses':>10}{'clause right':>14}")
    for b in bands:
        print(f"{b:10}{whole[b, 'n']:>10}{rate(whole, b) or 0:>13.3f}{clauses[b, 'n']:>10}{rate(clauses, b) or 0:>14.3f}")
    print(f"\nwhole-sentence accuracy overall: {row['whole_sentence_overall']:.3f}")
    print(f"clause accuracy overall:         {row['clause_overall']:.3f}")


if __name__ == "__main__":
    main()
