"""Correct the provable label faults, recording every change and its justification.

Run `label_audit.py` first; this applies only what that sweep proves, plus a short list of
hand-adjudicated rows named individually below. Every change lands in a diff file next to the
results, and the corpora are backed up before they are touched, so any number in this repo can be
recomputed against either version.

Two rules this follows and one it refuses:
  * a mangled utterance is dropped, not repaired — "backupson" had an intent, and writing my guess
    at it into the gold label would be inventing data rather than correcting it;
  * an act is only rewritten where the utterance's own verb contradicts it, or where the row is
    named in ADJUDICATED with a reason;
  * a row whose right label is genuinely unclear is left alone and listed as unresolved. "Can you
    do it again?" is labelled `choose`, which it is not, but the label space has no act for
    repeating a request, so there is nothing to correct it to.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from eval.selection.label_audit import SP, SWAPPABLE, faults, leading_verb, load, vocabulary

#: text (exact) -> (from, to, why). Hand-read, one line of justification each.
ADJUDICATED: dict[str, tuple[str, str, str]] = {
    "quick question, what's your take? thanks":
        ("ask_screen", "ask_self", "asks the assistant's own view; nothing in it refers to the screen"),
    "You're welcome, thanks.":
        ("confirm", "thanks", "a closing courtesy, not an instruction to proceed"),
    "You're welcome.":
        ("cancel", "thanks", "a closing courtesy; the identical text is labelled `thanks` elsewhere "
                             "in the same file, so one of the two must be wrong"),
    "I was just wondering, do you remember my favourite colour now?":
        ("forget", "ask_memory", "asks whether the memory is held; `forget` is the opposite instruction"),
    "quick question: what's the purpose of the sidebar?":
        ("ask_pixels", "ask_screen", "asks what a UI element is for; ask_pixels is for colour and "
                                     "pixel queries, which this is not"),
    "just curious, what's the purpose of the top bar for me?":
        ("ask_pixels", "ask_screen", "asks what a UI element is for, as above"),
}

#: acts that are wrong and have no right answer in the label space, so they stay wrong on purpose
UNRESOLVED = {
    "Can you do it again?": "labelled `choose`, which it is not; no act in the space means 'repeat'",
    "Hey, can I get that now?": "labelled `confirm`; plausibly `read`, but 'that' has no referent here",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", type=Path, nargs="+",
                    default=[SP / "training" / "paraphrase_eval.jsonl",
                             SP / "training" / "paraphrase_train.jsonl",
                             SP / "training" / "parser_dev.jsonl"])
    ap.add_argument("--diff", type=Path, default=Path("eval/results/selection_label_diff.json"))
    ap.add_argument("--apply", action="store_true", help="write the corpora; otherwise dry-run")
    args = ap.parse_args()

    diff: dict[str, dict] = {}
    for path in args.corpora:
        rows = load(path)
        found = faults(rows, vocabulary(rows))
        changes: list[dict] = []
        drop: set[int] = set()

        for f in found:
            row = rows[f["row"]]
            if f["kind"] == "swapped":
                want = SWAPPABLE[leading_verb(row["text"])]
                changes.append({"row": f["row"], "text": row["text"], "field": "act",
                                "from": row["act"], "to": want, "kind": "swapped",
                                "why": f"the utterance's imperative verb is "
                                       f"{leading_verb(row['text'])!r}, so the act is {want!r}"})
                row["act"] = want
            elif f["kind"] == "corrupt":
                drop.add(f["row"])
                changes.append({"row": f["row"], "text": row["text"], "field": "row",
                                "from": row.get("act"), "to": None, "kind": "dropped",
                                "why": f["evidence"] + "; the utterance is mangled, so its intent "
                                                       "cannot be recovered without inventing it"})

        for i, row in enumerate(rows):
            entry = ADJUDICATED.get(row["text"].strip())
            if entry and row.get("act") == entry[0]:
                changes.append({"row": i, "text": row["text"], "field": "act", "from": entry[0],
                                "to": entry[1], "kind": "adjudicated", "why": entry[2]})
                row["act"] = entry[1]

        kept = [r for i, r in enumerate(rows) if i not in drop]
        diff[path.name] = {
            "rows_before": len(rows), "rows_after": len(kept),
            "changes": len(changes), "dropped": len(drop),
            "share_of_rows_changed": round(len(changes) / max(1, len(rows)), 4),
            "unresolved": {t: w for t, w in UNRESOLVED.items() if any(r["text"].strip() == t for r in rows)},
            "diff": changes,
        }
        print(f"{path.name}: {len(changes)} changes ({len(drop)} dropped) on {len(rows)} rows "
              f"= {len(changes) / max(1, len(rows)):.4f}")
        if args.apply:
            backup = path.with_suffix(".jsonl.before_label_fix")
            if not backup.exists():
                shutil.copy2(path, backup)
            path.write_text("".join(json.dumps(r) + "\n" for r in kept))
            print(f"  wrote {path} (backup {backup.name})")

    args.diff.parent.mkdir(parents=True, exist_ok=True)
    args.diff.write_text(json.dumps(diff, indent=1))
    print(f"wrote {args.diff}" + ("" if args.apply else "  [dry run: pass --apply to write corpora]"))


if __name__ == "__main__":
    main()
