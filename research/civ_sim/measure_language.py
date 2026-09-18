"""Numbers for the language layer: parse success, ambiguity, cost per utterance, rumour mutation
over N hops, and mutual intelligibility between dialects after generations of drift.

    PYTHONPATH=src:. python -m research.civ_sim.measure_language [--days 60] [--hops 10] [--generations 20]

Everything here goes through `language.py`, the only seam between the simulation and the grammar.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import language as lang
from .sim import Simulation


def in_situ(days: int, people: int, focal: int, seed: int) -> dict:
    """What the language layer does inside a running world, where minds choose what to say."""
    sim = Simulation(seed=seed, people=people, focal=focal, size=96, villages=4, minds=True)
    t0 = time.perf_counter()
    for _ in range(days):
        sim.step()
    wall = time.perf_counter() - t0
    c = sim.counters
    u = max(1, c["utterances"])
    return {
        "days": days, "people": people, "minds": focal, "wall_s": round(wall, 1),
        "utterances": c["utterances"],
        "parsed_share": round(1 - c["not_understood"] / u, 4),
        "not_understood": c["not_understood"],
        "ambiguous_share": round(c["ambiguous"] / u, 4),
        "misheard": c["misheard"],
        "claims_transmitted": c["claims_transmitted"],
        "conversations": sim.minds.talks,
        "ms_per_utterance": round(sim.minds.talk_ms / u, 3),
        "ms_per_conversation": round(sim.minds.talk_ms / max(1, sim.minds.talks), 3),
        "words_for_food": {v.name: sim.dialects[v.id].say("food") for v in sim.villages},
    }


def divergence(generations: int, rate: float, seed: int) -> dict:
    """Let four dialects drift, with occasional borrowing, and measure who can still follow whom."""
    rng = np.random.default_rng(seed)
    base = lang.base_lexicon()
    d = [lang.dialect(base, i, rng) for i in range(4)]
    grid = lambda: [[round(lang.intelligibility(a, b), 2) for b in d] for a in d]
    before = grid()
    changes = 0
    for gen in range(generations):
        for k in range(4):
            borrow = d[(k + 1) % 4] if gen % 7 == 0 else None
            changes += lang.drift(d[k], rng, borrow_from=borrow, rate=rate)
    after = grid()
    off = [after[i][j] for i in range(4) for j in range(4) if i != j]
    return {
        "generations": generations, "rate": rate, "changes_made": changes,
        "intelligibility_before": before, "intelligibility_after": after,
        "cross_dialect_mean": round(float(np.mean(off)), 3),
        "cross_dialect_min": min(off), "cross_dialect_max": max(off),
        "asymmetric_pairs": sum(1 for i in range(4) for j in range(i + 1, 4) if after[i][j] != after[j][i]),
        "words_for_food": [x.say("food") for x in d],
    }, d


def rumour(d: list, hops: int) -> dict:
    """One claim passed hop to hop between dialects: what survives, and what the sentence becomes."""
    places = ["Aldmere", "Brenholt", "Coralin", "Dunmarsh"]
    claim, who, trail = ("village:Aldmere", "has_amount", "food:much"), 0, []
    for hop in range(hops):
        nxt = (who + 1) % len(d)
        sentence = lang.say(claim[0], claim[1], claim[2], d[who])
        heard = lang.hear(sentence, d[nxt], names=places, context={}, speaker="X", settlements=places)
        got = heard.claims[0] if heard.claims else None
        trail.append({
            "hop": hop + 1, "from": who, "to": nxt, "sentence": sentence,
            "heard": None if not got else f"{got['subject']} {got['predicate']} {got['object']}",
            "readings": heard.readings, "unknown": list(heard.unknown),
        })
        if not got or got["subject"] is None:
            break
        claim, who = (got["subject"], got["predicate"], str(got["object"])), nxt
    first, last = trail[0], trail[-1]
    return {
        "hops_survived": len(trail), "start": "village:Aldmere has_amount food:much",
        "end": last["heard"], "meaning_held": last["heard"] == first["heard"],
        "surface_forms": sorted({t["sentence"] for t in trail}), "trail": trail,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--people", type=int, default=700)
    ap.add_argument("--focal", type=int, default=60)
    ap.add_argument("--hops", type=int, default=10)
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--rate", type=float, default=0.22)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    live = in_situ(args.days, args.people, args.focal, args.seed)
    print(f"in a running world: {live['utterances']} utterances · parsed {live['parsed_share']:.1%} "
          f"· ambiguous {live['ambiguous_share']:.1%} · misheard {live['misheard']} "
          f"· {live['ms_per_utterance']} ms/utterance · {live['ms_per_conversation']} ms/conversation")
    print("  each settlement's word for food:", live["words_for_food"])

    div, dialects = divergence(args.generations, args.rate, args.seed)
    print(f"\nafter {div['generations']} generations of drift ({div['changes_made']} changes):")
    for row in div["intelligibility_after"]:
        print("   ", row)
    print(f"  cross-dialect intelligibility {div['cross_dialect_min']}–{div['cross_dialect_max']} "
          f"(mean {div['cross_dialect_mean']}), {div['asymmetric_pairs']}/6 pairs asymmetric")
    print("  words for food:", div["words_for_food"])

    rum = rumour(dialects, args.hops)
    print(f"\na rumour over {rum['hops_survived']} hops — meaning held: {rum['meaning_held']}")
    for t in rum["trail"]:
        note = f"  [{t['readings']} readings]" if t["readings"] > 1 else ""
        note += f"  [unknown: {', '.join(t['unknown'])}]" if t["unknown"] else ""
        print(f"  hop {t['hop']} ({t['from']}->{t['to']}): “{t['sentence']}” -> {t['heard'] or 'NOT UNDERSTOOD'}{note}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"in_situ": live, "divergence": div, "rumour": rum}, indent=1, default=str))


if __name__ == "__main__":
    main()
