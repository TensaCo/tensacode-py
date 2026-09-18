"""Train the tagger and parser on a UD treebank and score them on its held-out splits.

    python -m eval.parsing.train_ud [--test]

Dev is for development. ``--test`` reports the treebank's test split, which is the
headline number for "does it parse English": tagging accuracy, unlabelled attachment
(UAS) and labelled attachment (LAS), punctuation excluded as usual.

The model is written to ``~/.cache/tensorcode/models/ud_ewt_parser.pickle`` and a row is
appended to ``eval/results/parsing_ud.jsonl``. Nothing about the held-out prompt sets is
involved: this measures syntax against linguists' annotations.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parents[2] / "src")]

from tensorcode.language.learned_parser import save, scores, train  # noqa: E402
from tensorcode.language.treebank import find_treebank, load  # noqa: E402

OUT = Path(__file__).parents[1] / "results" / "parsing_ud.jsonl"
MODEL = Path.home() / ".cache" / "tensorcode" / "models" / "ud_ewt_parser.pickle"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-epochs", type=int, default=6)
    ap.add_argument("--parse-epochs", type=int, default=10)
    ap.add_argument("--test", action="store_true", help="also score the test split")
    args = ap.parse_args()
    root = find_treebank()
    if root is None:
        sys.exit("no treebank: set $TENSORCODE_TREEBANK or clone UD_English-EWT into ~/.cache/tensorcode/seeds")
    t0 = time.perf_counter()
    tagger, parser = train(load("train"), tag_epochs=args.tag_epochs, parse_epochs=args.parse_epochs)
    row = {"eval": "parsing_ud", "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "treebank": root.name,
           "tag_epochs": args.tag_epochs, "parse_epochs": args.parse_epochs,
           "seconds": round(time.perf_counter() - t0, 1), "dev": scores(tagger, parser, load("dev"))}
    if args.test:
        row["test"] = scores(tagger, parser, load("test"))
    save(MODEL, tagger, parser)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
