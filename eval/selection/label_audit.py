"""Sweep the authored corpora for the label faults doc 21 found by hand.

Doc 21 hand-adjudicated 51 items of a corpus this repo trains and reports on, and found 21 of
them wrong: acts swapped, a template placeholder (`@it`) leaked into a closed-class value,
paraphrases corrupted into non-words ("backupson"), and utterances whose act contradicts their
plain reading. That was a tail sample; nobody had checked the other 877 rows, and several headline
numbers are averages over them.

Only *mechanical* faults are corrected here — ones provable from the row itself or from a
contradiction between two rows, with no judgement about what the label ought to be:

  offset      a span whose character offsets do not point at a plausible substring
  swapped     an utterance whose leading imperative verb is `copy` labelled `move`, or vice
              versa — two acts that both exist in the label space, so the verb settles it
  corrupt     a word that exists nowhere else in the corpus and is two corpus words run together
  conflict    two rows with identical text and different acts (at least one must be wrong)

Two detectors were written, run, and retracted, which is the part of this sweep worth recording.
The first flagged every `@it` as a template leak on the strength of doc 21 reporting one: wrong,
because `@it` is a deliberate anaphora encoding that sits beside `place_kind` = `span` and
appears on utterances that really do say "inside it". The second kept `@it` only where the text
also named an explicit path: also wrong, because in "move that to ~/Downloads" the `@it` is the
*source* and the path is the destination. Between them they would have rewritten 1,279 correct
rows. A detector that reports its own guess as a fault is the failure mode this sweep exists to
catch, so both counts are reported as zero and the reasoning is kept here.

The third detector found the real fault, and found it inside the second one's false positives:
"copy it to ~/Downloads" was labelled `move` and "hey move it to ~/Downloads?" was labelled
`copy`, in adjacent rows.

Statistical suspicions (an act rare for its own leading verb) are reported but NOT corrected:
they need a reading, and a detector that rewrites labels toward the corpus majority would launder
its own prior into the data.
"""

from __future__ import annotations

import os

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SP = Path(os.environ.get("TENSORCODE_SCRATCH", os.path.expanduser("~/.cache/tensorcode")))
WORD = re.compile(r"[a-z]+")
PLACEHOLDER = re.compile(r"^@\w+$")
SWAPPABLE = {"copy": "copy", "cp": "copy", "duplicate": "copy", "move": "move", "mv": "move"}
DESTINATION_IS_TRASH = re.compile(r"\b(trash|bin|recycle)\b", re.I)
GREETING = frozenset(("hi", "hey", "hello", "ok", "okay", "yo", "please", "could", "would", "can",
                      "you", "the", "a", "i", "to", "for", "me", "will", "d", "like", "quick", "so",
                      "now", "just", "pls", "plz", "thanks", "sorry", "um", "uh", "and", "also"))


def leading_verb(text: str) -> str:
    """The first word that is not a greeting or a politeness marker."""
    return next((w for w in WORD.findall(text.lower()) if w not in GREETING), "")


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def vocabulary(rows: list[dict]) -> Counter:
    return Counter(w for r in rows for w in WORD.findall(r["text"].lower()))


def faults(rows: list[dict], vocab: Counter) -> list[dict]:
    """One record per provable fault, with the evidence that makes it provable."""
    out: list[dict] = []
    by_text: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_text[r["text"].strip().lower()].add(r.get("act", ""))

    for i, r in enumerate(rows):
        text = r["text"]
        for key, span in (r.get("spans") or {}).items():
            if not (isinstance(span, list) and len(span) == 2):
                out.append({"row": i, "kind": "offset", "field": key,
                            "evidence": f"span {span!r} is not a pair of offsets"})
                continue
            a, b = span
            if not (0 <= a < b <= len(text)):
                out.append({"row": i, "kind": "offset", "field": key,
                            "evidence": f"offsets {a},{b} fall outside a {len(text)}-character text"})
            elif not text[a:b].strip():
                out.append({"row": i, "kind": "offset", "field": key,
                            "evidence": f"offsets {a},{b} select whitespace {text[a:b]!r}"})
            elif (a and text[a - 1].isalnum() and text[a].isalnum()):
                out.append({"row": i, "kind": "offset", "field": key,
                            "evidence": f"span starts mid-word: {text[max(0, a - 6):b]!r}"})

        verb = leading_verb(text)
        act = r.get("act", "")
        if verb in SWAPPABLE and act in SWAPPABLE.values() and SWAPPABLE[verb] != act \
                and not DESTINATION_IS_TRASH.search(text):
            out.append({"row": i, "kind": "swapped", "field": "act",
                        "evidence": f"the utterance says {verb!r} and the label says {act!r}"})

        for w in WORD.findall(text.lower()):
            if len(w) < 7 or vocab[w] != 1:
                continue
            for cut in range(4, len(w) - 1):
                if vocab.get(w[:cut], 0) > 1 and vocab.get(w[cut:], 0) > 1:
                    out.append({"row": i, "kind": "corrupt", "field": "text",
                                "evidence": f"{w!r} occurs once and is {w[:cut]!r}+{w[cut:]!r} run together"})
                    break

        if len(by_text[text.strip().lower()]) > 1:
            out.append({"row": i, "kind": "conflict", "field": "act",
                        "evidence": f"identical text is labelled {sorted(by_text[text.strip().lower()])}"})
    return out


def suspicions(rows: list[dict]) -> list[dict]:
    """An act that is rare for its own leading verb. Reported, never corrected.

    Read the output before believing it. The first version took the first non-stopword of each
    utterance as its verb, which on this corpus is the greeting: it reported that "hi" takes act
    `create_file` in 6 of 61 rows and flagged every other "Hi, ..." row as suspect. All 15
    adjudicated hits were correctly labelled. Greetings are skipped now, but this remains a
    statistical signal about a corpus that deliberately varies its phrasing, so its hits are
    counted separately from the provable faults and none of them is corrected.
    """
    verb_acts: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        verb = leading_verb(r["text"])
        if verb:
            verb_acts[verb][r.get("act", "")] += 1
    out = []
    for i, r in enumerate(rows):
        verb = leading_verb(r["text"])
        if not verb:
            continue
        acts = verb_acts[verb]
        if sum(acts.values()) >= 10 and acts[r.get("act", "")] / sum(acts.values()) < 0.05:
            out.append({"row": i, "text": r["text"], "act": r.get("act"),
                        "evidence": f"{verb!r} takes act {acts.most_common(1)[0][0]!r} in "
                                    f"{acts.most_common(1)[0][1]}/{sum(acts.values())} rows; this one says "
                                    f"{r.get('act')!r} in {acts[r.get('act', '')]}"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", type=Path, nargs="+",
                    default=[SP / "training" / "paraphrase_eval.jsonl",
                             SP / "training" / "paraphrase_train.jsonl",
                             SP / "training" / "parser_dev.jsonl"])
    ap.add_argument("--out", type=Path, default=Path("eval/results/selection_label_audit.json"))
    args = ap.parse_args()

    body = {}
    for path in args.corpora:
        rows = load(path)
        found = faults(rows, vocabulary(rows))
        susp = suspicions(rows)
        affected = {f["row"] for f in found}
        body[path.name] = {
            "rows": len(rows),
            "provable_faults": len(found), "rows_affected": len(affected),
            "share_of_rows": round(len(affected) / max(1, len(rows)), 4),
            "by_kind": dict(Counter(f["kind"] for f in found)),
            "suspect_not_corrected": len(susp),
            "examples": [dict(f, text=rows[f["row"]]["text"], act=rows[f["row"]].get("act")) for f in found[:25]],
            "suspect_examples": susp[:15],
        }
        r = body[path.name]
        print(f"{path.name}: {r['rows']} rows, {r['provable_faults']} provable faults on "
              f"{r['rows_affected']} rows ({r['share_of_rows']:.4f}) {r['by_kind']}, "
              f"{r['suspect_not_corrected']} suspect")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
