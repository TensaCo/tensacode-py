"""FACULTY 4 — automatization (chunking).

Prediction (stated before the measurement): chunking cuts the number of deliberated steps and
the time sharply, with identical outcomes, and the graph shows one act rather than seven.
Falsified if error recovery degrades — so a failure is injected deliberately, mid-chunk, and the
fallback to the expanded form is checked.

What is counted:

* *deliberated steps* — the per-step decision records the interpreter writes (``index`` claims).
  Under a chunk these are not written, so this is the direct measure of deliberation, not a proxy.
* *claims* in the store at the end, which is what the mind has to carry and search.
* *seconds per replay*, on a loaded machine, so read the ratio and not the absolute.
* *the replies*, which must be identical in both arms. That is the acceptance criterion: an
  automatized skill that answers differently is not the same skill.
"""

from __future__ import annotations

import json
import statistics
import sys

from tensorcode.chunking import Chunks

from eval.control.harness import TREE, Conversation

REPLAYS = 20
SKILL = "how many words are in ~/Desktop/notes.txt"


def deliberated(conv: Conversation) -> int:
    return len([r for r in conv.mind.claims(predicate="index")])


def replay(*, chunks: Chunks | None, replays: int = REPLAYS, fail_at: int | None = None,
           fail: dict[str, int] | None = None) -> dict:
    conv = Conversation(TREE, chunks=chunks)
    rows = []
    before = 0
    for i in range(replays):
        if fail_at is not None and i == fail_at:
            conv.shell.fail.update(fail or {})
        turn = conv.say(SKILL)
        now = deliberated(conv)
        rows.append({"replay": i, "deliberated_steps": now - before, "commands": len(turn.commands),
                     "seconds": round(turn.seconds, 4), "reply": turn.replies[0] if turn.replies else None,
                     "cycles": turn.cycles})
        before = now
    settled = rows[5:]  # after the chunk has had a chance to compile
    return {
        "chunking": chunks is not None,
        "replays": replays,
        "deliberated_steps_total": sum(r["deliberated_steps"] for r in rows),
        "deliberated_steps_first": rows[0]["deliberated_steps"],
        "deliberated_steps_settled_mean": round(statistics.mean(r["deliberated_steps"] for r in settled), 2),
        "seconds_settled_mean": round(statistics.mean(r["seconds"] for r in settled), 4),
        "claims_at_end": conv.claims(),
        "commands_total": sum(r["commands"] for r in rows),
        "replies_distinct": sorted({r["reply"] for r in rows}),
        "chunk_stats": chunks.stats() if chunks is not None else None,
        "chunk_history": list(chunks.history) if chunks is not None else [],
        "rows": rows,
    }


def main() -> None:
    off = replay(chunks=None)
    on_chunks = Chunks(repeats=3)
    on = replay(chunks=on_chunks)
    # the deliberate cost of automatization: one step of the skill goes wrong halfway through the
    # run of replays, after the chunk is live. Held out in the sense that matters — it was written
    # to break the mechanism, not to show it working.
    broken_chunks = Chunks(repeats=3)
    broken = replay(chunks=broken_chunks, fail_at=10, fail={"wc -w": 1})
    # the same failure without chunking, so "did recovery degrade?" is answered by comparing
    # what the two arms *said*, not by inspecting the mechanism
    broken_off = replay(chunks=None, fail_at=10, fail={"wc -w": 1})

    out = {"faculty": "automatization / chunking",
           "prediction": "fewer deliberated steps and less time, identical replies; falsified if recovery degrades",
           "provenance": {"environment": "ours (fake shell, real interpreter and procedures)",
                          "grader": "ours (claims counted in the store; replies compared verbatim)",
                          "note": "wall time on a machine running other work; the ratio is the number to read"},
           "skill": SKILL,
           "off": off, "on": on, "with_injected_failure": broken,
           "with_injected_failure_no_chunking": broken_off,
           "recovery_identical": broken["replies_distinct"] == broken_off["replies_distinct"]}
    same = off["replies_distinct"] == on["replies_distinct"]
    out["identical_replies"] = same
    out["deliberated_steps_ratio"] = round(on["deliberated_steps_settled_mean"] / off["deliberated_steps_settled_mean"], 3) if off["deliberated_steps_settled_mean"] else None
    out["seconds_ratio"] = round(on["seconds_settled_mean"] / off["seconds_settled_mean"], 3) if off["seconds_settled_mean"] else None

    for label, run in (("chunking off", off), ("chunking on", on),
                       ("chunking on, one step fails at replay 10", broken),
                       ("chunking off, the same step fails", broken_off)):
        print(f"\n=== {label}")
        print(f"  deliberated steps: first={run['deliberated_steps_first']} "
              f"settled mean={run['deliberated_steps_settled_mean']} total={run['deliberated_steps_total']}")
        print(f"  seconds/replay (settled): {run['seconds_settled_mean']}   claims at end: {run['claims_at_end']}")
        print(f"  commands run: {run['commands_total']}   distinct replies: {run['replies_distinct']}")
        if run["chunk_history"]:
            for h in run["chunk_history"]:
                print(f"    · {h}")
        if run in (broken, broken_off):
            for r in run["rows"][8:14]:
                print(f"    replay {r['replay']:2}: {r['deliberated_steps']:3} steps  {r['reply']}")
    print(f"\nidentical replies with and without chunking: {same}")
    print(f"identical replies under the injected failure: {out['recovery_identical']}")
    print(f"deliberated steps ratio (on/off): {out['deliberated_steps_ratio']}   seconds ratio: {out['seconds_ratio']}")
    path = "eval/results/control_chunking.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
