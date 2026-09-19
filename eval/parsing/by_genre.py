"""Where the parser is weak, by the kind of writing it is reading.

    python -m eval.parsing.by_genre [--split dev] [--length]

The aggregate LAS is one number over five genres, and the recall failures that motivated
this were all on chat-like text. UD English-EWT is *entirely* web English — reviews, email,
answers, newsgroup, weblog — so "train it on informal text" is not available as a fix: it
already is. What is available is knowing which of those five it handles worst, and whether
the damage tracks genre at all or simply tracks sentence length.

That distinction decides what to do next. If accuracy is flat across genres and falls with
length, the problem is the parser's capacity on long sentences and more in-domain data will
not help. If one genre is far worse, its constructions are the thing to work on.

A row is appended to ``eval/results/parsing_by_genre.jsonl``.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parents[2] / "src")]

from tensorcode.language.learned_parser import load_model, scores  # noqa: E402
from tensorcode.language.treebank import load_with_genres  # noqa: E402

OUT = Path(__file__).parents[1] / "results" / "parsing_by_genre.jsonl"
MODEL = Path.home() / ".cache" / "tensorcode" / "models" / "ud_ewt_parser.pickle"
BANDS = ((1, 10), (11, 20), (21, 35), (36, 10_000))


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

    by_genre = collections.defaultdict(list)
    for genre, sentence in tagged:
        by_genre[genre].append(sentence)
    per_genre = {g: scores(tagger, parser, ss) for g, ss in sorted(by_genre.items())}

    by_length = collections.defaultdict(list)
    for _, sentence in tagged:
        for low, high in BANDS:
            if low <= len(sentence) <= high:
                by_length[f"{low}-{high if high < 10_000 else '+'}"].append(sentence)
                break
    per_length = {b: scores(tagger, parser, ss) for b, ss in by_length.items()}

    overall = scores(tagger, parser, [s for _, s in tagged])
    row = {"eval": "parsing_by_genre", "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "split": args.split,
           "overall": overall, "by_genre": per_genre, "by_length": per_length}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")

    print(f"overall  las={overall['las']:.3f} uas={overall['uas']:.3f} tags={overall['tagging_accuracy']:.3f} "
          f"n={overall['sentences']}")
    print("\nby genre")
    for g, s in sorted(per_genre.items(), key=lambda kv: kv[1]["las"]):
        print(f"  {g:12} las={s['las']:.3f} uas={s['uas']:.3f} tags={s['tagging_accuracy']:.3f} n={s['sentences']:5}")
    print("\nby sentence length (tokens)")
    for b, s in per_length.items():
        print(f"  {b:8} las={s['las']:.3f} uas={s['uas']:.3f} tags={s['tagging_accuracy']:.3f} n={s['sentences']:5}")
    spread = max(s["las"] for s in per_genre.values()) - min(s["las"] for s in per_genre.values())
    lengths = [per_length[k]["las"] for k in sorted(per_length, key=lambda k: int(k.split("-")[0]))]
    print(f"\ngenre spread in LAS: {spread:.3f}; LAS from shortest to longest band: "
          f"{' -> '.join(f'{x:.3f}' for x in lengths)}")


if __name__ == "__main__":
    main()
