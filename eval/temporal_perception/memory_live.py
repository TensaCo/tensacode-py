"""Memory dynamics in a long live conversation, not in a microbenchmark.

The episodic/semantic/consolidation/forgetting machinery has only ever been exercised on made-up
stores. Here it runs inside the assistant for a few hundred turns while the desktop is perceived
every turn, which is where its costs actually live: ~500 perceptual claims arrive per turn, and
something has to decide what not to keep.

Three predictions, all falsifiable:

    plateau   the claim count stops growing once forgetting keeps up with perception
    flat      per-turn latency does not grow with conversation length
    recall    a fact told at turn 5 is still answerable at turn 200

The third is the one worth running: the informative outcome is forgetting dropping something the
user later needs. Facts are planted early, with different salience (said once, or said twice), and
asked at the end; the recall curve is reported by age and by salience.

The clock is scaled, and this matters for reading the numbers: the policy's half-life is set to
seconds instead of half an hour, so a conversation that takes a minute of wall time exercises the
same dynamics an hour-long one would. Nothing else is scaled.

    python -m eval.temporal_perception.memory_live [--turns 240]
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import timedelta
from pathlib import Path

from tensacode.memory import Memory, MemoryPolicy

from .live_harness import Session

OUT = Path(__file__).resolve().parents[2] / "eval" / "results" / "memory_live.json"

# (what we tell it, what we ask later, what the answer must contain, how many times we say it)
# Phrased in the forms the assistant's language layer actually accepts ("my X is Y" / "what is my
# X"), so the measurement is of memory and not of its front-end. The two are separated in the
# report anyway: a fact can be in memory and still not be sayable.
FACTS = [
    ("my name is Dana", "what is my name", "Dana", "name", 2),
    ("my cat is Mackerel", "what is my cat", "Mackerel", "cat", 1),
    ("my notes folder is ~/notes", "what is my notes folder", "notes", "notes folder", 1),
    ("my project is Halverson", "what is my project", "Halverson", "project", 2),
    ("my deadline is the 30th", "what is my deadline", "30th", "deadline", 1),
    ("my indent style is tabs", "what is my indent style", "tabs", "indent style", 1),
    ("my office is room 412", "what is my office", "412", "office", 1),
    ("my bike is a Brompton", "what is my bike", "Brompton", "bike", 1),
]

REFUSALS = ("didn't understand", "don't know yet", "i have no record", "not sure what")


def answered(reply: str, needle: str) -> bool:
    """The reply quotes the question back when it fails, so an echo must not count as recall."""
    low = reply.lower()
    if any(r in low for r in REFUSALS):
        return False
    return needle.lower() in low

FILLER = ["list my desktop", "what time is it", "run `uname -s`", "hello", "what is on my screen",
          "run `echo tick`", "how are you", "what can you do"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=240)
    ap.add_argument("--half-life", type=float, default=25.0, help="seconds; scaled down from 30 minutes")
    ap.add_argument("--protection", choices=("predicates", "provenance"), default="provenance",
                    help="what forgetting is told to spare: a list of predicate spellings, or anything a person said")
    args = ap.parse_args()

    samples: list[dict] = []
    planted: dict[str, int] = {}

    with Session(hostname="memory-eval") as s:
        # the live policy, with the clock scaled so a short conversation exercises the dynamics
        s.agent.BODY.memory = Memory(s.mind, MemoryPolicy(
            protect_predicates=frozenset({"said", "told", "name", "act", "words", "status"}),
            # the arm under test: protecting told facts by predicate name cannot work, because the
            # predicate of a told fact is whichever word the user chose
            protect_sources=() if args.protection == "predicates" else ("utterance:", "person:", "user:"),
            half_life=timedelta(seconds=args.half_life)))

        print(f"planting {len(FACTS)} facts in the first turns")
        for text, _, _, _, times in FACTS:
            for _ in range(times):
                s.send(text)
            planted[text] = len(s.turns)

        print(f"{args.turns} turns of ordinary conversation")
        for i in range(args.turns):
            turn = s.send(FILLER[i % len(FILLER)])
            if i % 10 == 0:
                memory = s.agent.BODY.memory
                report = s.agent.BODY.last_memory
                samples.append({
                    "turn": len(s.turns), "claims": turn.claims, "live_claims": turn.live,
                    "seconds": round(turn.seconds, 4),
                    "episodes": len(memory.episodes()), "semantic": len(memory.semantic()),
                    "consolidated": getattr(report, "consolidated", 0),
                    "forgotten": getattr(getattr(report, "forgotten", None), "claims_forgotten", None),
                    "episodes_dropped": getattr(getattr(report, "forgotten", None), "episodes_dropped", None),
                    "memory_ms": round(getattr(report, "ms", 0.0), 2),
                })
                print(f"  turn {len(s.turns):>4}: claims={turn.claims:>6} episodes={samples[-1]['episodes']:>4} "
                      f"semantic={samples[-1]['semantic']:>4} {turn.seconds*1000:.0f}ms "
                      f"(memory {samples[-1]['memory_ms']:.1f}ms, forgot {samples[-1]['forgotten']})")

        print("asking for every planted fact, at the end")
        recalls = []
        for text, question, needle, topic, times in FACTS:
            turn = s.send(question)
            got = answered(turn.reply, needle)
            in_store = any(needle.lower() in str(r.claim.object).lower()
                           for r in s.mind.claims(predicate=topic) if not r.retracted)
            cued = needle.lower() in " ".join(
                w for w in getattr(s.agent.BODY.memory.recall(topic, k=5), "why", lambda: [])()).lower()
            recalls.append({"told": text, "asked": question, "expected": needle, "recalled": got,
                            "still_in_store": in_store, "found_by_cue": cued,
                            "told_at_turn": planted[text], "age_turns": len(s.turns) - planted[text],
                            "times_told": times, "reply": turn.reply[:160]})
            print(f"  {'OK ' if got else 'LOST'} (in store: {in_store}, by cue: {cued}) {question!r} -> {turn.reply[:70]!r}")

        final_claims = len(s.mind._claims)
        latencies = [t.seconds for t in s.turns]

    early = [x["live_claims"] for x in samples[: max(1, len(samples) // 3)]]
    late = [x["live_claims"] for x in samples[-max(1, len(samples) // 3):]]
    held_early = [x["claims"] for x in samples[: max(1, len(samples) // 3)]]
    held_late = [x["claims"] for x in samples[-max(1, len(samples) // 3):]]
    first_half = [t for t in latencies[: len(latencies) // 2]]
    second_half = [t for t in latencies[len(latencies) // 2:]]
    report = {
        "what": "memory dynamics in a live assistant over a long conversation",
        "arm": args.protection,
        "provenance": {"environment": "Seed simulator (third-party)", "grader": "this script, from facts it authored and planted itself",
                       "held_out": "n/a — the planted facts are the ground truth, fixed before the run",
                       "scaled": f"MemoryPolicy.half_life = {args.half_life}s instead of 30min"},
        "turns": len(latencies), "final_claims": final_claims, "samples": samples, "recall": recalls,
        "held_records_growth": {"early_mean": round(statistics.mean(held_early), 1),
                                "late_mean": round(statistics.mean(held_late), 1),
                                "ratio": round(statistics.mean(held_late) / max(1.0, statistics.mean(held_early)), 3)},
        "claims_growth": {"early_mean": round(statistics.mean(early), 1), "late_mean": round(statistics.mean(late), 1),
                          "ratio": round(statistics.mean(late) / max(1.0, statistics.mean(early)), 3)},
        "latency_ms": {"first_half_median": round(1e3 * statistics.median(first_half), 2),
                       "second_half_median": round(1e3 * statistics.median(second_half), 2),
                       "ratio": round(statistics.median(second_half) / max(1e-6, statistics.median(first_half)), 3),
                       "p95": round(1e3 * sorted(latencies)[int(0.95 * len(latencies))], 2)},
        "recall_rate": round(sum(r["recalled"] for r in recalls) / len(recalls), 3),
        "retained_in_store_rate": round(sum(r["still_in_store"] for r in recalls) / len(recalls), 3),
        "found_by_cue_rate": round(sum(r["found_by_cue"] for r in recalls) / len(recalls), 3),
        "recall_by_salience": {str(times): round(
            sum(r["recalled"] for r in recalls if r["times_told"] == times) /
            max(1, len([r for r in recalls if r["times_told"] == times])), 3) for times in {f[4] for f in FACTS}},
        "lost": [r["asked"] for r in recalls if not r["recalled"]],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged = json.loads(OUT.read_text()) if OUT.exists() and "arms" in OUT.read_text() else {"arms": {}}
    merged["arms"][args.protection] = report
    merged["what"] = "memory dynamics in a live assistant over a long conversation, two forgetting policies"
    OUT.write_text(json.dumps(merged, indent=1))
    print(f"\nheld {report['held_records_growth']}\nlive {report['claims_growth']} · latency {report['latency_ms']} · recall {report['recall_rate']}")
    print(f"lost: {report['lost'] or 'nothing'}")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
