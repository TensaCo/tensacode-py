"""Headless runs: dynamics over decades, per-tier cost, memory, degenerate outcomes.

    PYTHONPATH=src:. python -m research.civ_sim.measure --years 50 --seeds 1,2,3 [--no-minds] [--out eval/results/civ_slice.json]
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import platform
import resource
import time
from pathlib import Path

import numpy as np

from .sim import MOTIFS, Simulation
from .world import DAYS_PER_YEAR


def run(seed: int, years: int, minds: bool, people: int, focal: int) -> dict:
    t0 = time.perf_counter()
    sim = Simulation(seed=seed, people=people, minds=minds)
    init_s = time.perf_counter() - t0
    start = sim.totals()
    yearly = []
    last = dict(sim.counters)
    worst_conservation = 0.0
    for day in range(years * DAYS_PER_YEAR):
        sim.step()
        if sim.day % DAYS_PER_YEAR == 0:
            now = sim.totals()
            L, W = sim.ledger, sim.world.ledger
            err = max(abs(start["food"] + W["regrowth_food"] - L["eaten"] - L["spoiled"] - L["frost_loss"] - W["build_food"] - now["food"]) / max(1.0, now["food"]),
                      abs(start["wood"] + W["regrowth_wood"] - L["wood_burned"] - W["build_wood"] - L["craft_wood"] - L["wood_lost"] - now["wood"]) / max(1.0, now["wood"]),
                      abs(start["stone"] - L["stone_used"] - L["craft_stone"] - L["stone_lost"] - now["stone"]) / max(1.0, now["stone"]),
                      abs(start["tools"] + L["tools_made"] - L["tools_lost"] - now["tools"]) / max(1.0, now["tools"]))
            worst_conservation = max(worst_conservation, err)
            idx = sim.living()
            p = sim.p
            vil = p.village[idx]
            pops = np.bincount(vil, minlength=len(sim.villages)).tolist()
            motifs = np.bincount(p.motif[idx], minlength=len(MOTIFS))
            yearly.append({
                "year": sim.year, "pop": pops, "total": int(len(idx)),
                **{k: (round(sim.counters[k] - last.get(k, 0), 1)) for k in ("births", "deaths", "raids", "trade_volume", "migrants", "sanctions", "bonds", "monuments", "laws")},
                "food_stores": [round(v.food) for v in sim.villages], "wood_stores": [round(v.wood) for v in sim.villages], "laws_in_force": sum(v.law_share for v in sim.villages),
                "martial_share": round(float((p.martial[idx] > 0.5).mean()), 3) if len(idx) else 0,
                "communal_share": round(float((p.share[idx] > 0.5).mean()), 3) if len(idx) else 0,
                "ideology_std": [round(float(p.share[idx].std()), 3), round(float(p.martial[idx].std()), 3)] if len(idx) else [0, 0],
                "mean_valence": round(float(p.valence[idx].mean()), 3) if len(idx) else 0,
                "mean_alpha_out": round(float(np.mean([p.alpha_p[idx[vil == a], b].mean() for a in range(len(sim.villages)) for b in range(len(sim.villages)) if a != b and (vil == a).any()])), 3) if len(idx) else 0,
                "genome_std": round(float(p.genome[idx].astype(float).std(0).mean()), 2) if len(idx) else 0,
                "top_motifs": {MOTIFS[i]: int(motifs[i]) for i in np.argsort(-motifs)[:4]},
                "focal_claims": sim.minds.claims_live if sim.minds else 0,
                "conversations": sim.counters["conversations"], "claims_transmitted": sim.counters["claims_transmitted"],
                "buildings": sim.counters["buildings"], "festivals": sim.counters["festivals"], "omens": sim.counters["omens"],
                "weather": {"rain": round(float(sim.weather.rain.mean()), 4), "events": list(sim.weather.events)},
                "economy": _economy_row(sim),
                "settlements": _settlement_row(sim),
            })
            last = dict(sim.counters)
    # cost and behaviour of the mind tier, plus how much a claim survives being passed along
    mind_stats = {}
    if sim.minds is not None:
        import tracemalloc

        ms = sim.minds
        claims = sum(len(m.store._claims) for m in ms.minds.values())
        episodes = sum(len(m.episodes) for m in ms.minds.values())
        gc.collect()
        tracemalloc.start()
        snap0 = tracemalloc.take_snapshot()
        copies = [dict(m.store._claims) for m in list(ms.minds.values())[:20]]  # rough per-mind claim footprint
        n_copies = len(copies)
        snap1 = tracemalloc.take_snapshot()
        tracemalloc.stop()
        per_copy = sum(st.size_diff for st in snap1.compare_to(snap0, "filename"))
        del copies
        _ = n_copies
        # what minds believe about each village's stores, against the truth: rumour divergence
        truth, belief = {}, {}
        counts_v = np.bincount(sim.p.village[sim.living()], minlength=len(sim.villages)).clip(1)
        for v in sim.villages:
            pc = v.food / counts_v[v.id]
            truth[v.name] = "much" if pc > 8 else "some" if pc > 3 else "little" if pc > 1 else "none"
        for m in ms.minds.values():
            for rec in m.store._claims.values():
                if rec.claim.predicate == "has_amount" and rec.claim.subject.id.startswith("village:"):
                    name = rec.claim.subject.id.split(":", 1)[1]
                    amount = str(rec.claim.object).split(":")[-1]  # "food:much" -> "much"
                    belief.setdefault(name, []).append(amount)
        divergence = {}
        for name, held in belief.items():
            right = sum(1 for h in held if h == truth.get(name))
            divergence[name] = {"truth": truth.get(name), "minds_holding_a_belief": len(held), "share_correct": round(right / len(held), 3),
                                "spread": {k: held.count(k) for k in sorted(set(held))}}
        mind_stats = {
            "belief_about_stores_vs_truth": divergence,
            "minds": len(ms.minds), "claims_live": claims, "claims_per_mind": round(claims / max(1, len(ms.minds)), 1),
            "episodes_per_mind": round(episodes / max(1, len(ms.minds)), 1),
            "thinks": ms.thinks, "ms_per_think": round(sim.timing["minds_ms"] / max(1, ms.thinks), 3),
            "conversations": ms.talks, "ms_per_conversation": round(ms.talk_ms / max(1, ms.talks), 3),
            "claims_transmitted": sim.counters["claims_transmitted"],
            "claim_dict_bytes_per_mind_estimate": round(per_copy / 20),
            "relations_per_mind": round(sum(len(m.relations) for m in ms.minds.values()) / max(1, len(ms.minds)), 1),
            "theory_of_mind_entries_per_mind": round(sum(len(m.tom) for m in ms.minds.values()) / max(1, len(ms.minds)), 1),
        }
    wall = time.perf_counter() - t0
    ticks = sim.timing["ticks"]
    totals = [y["total"] for y in yearly]
    pops = np.array([y["pop"] for y in yearly])
    degenerate = {
        "extinct": bool(totals[-1] == 0),
        "villages_emptied": [sim.villages[i].name for i in range(len(sim.villages)) if pops[-1, i] == 0],
        "runaway_growth": bool(totals[-1] > 3 * people),
        "collapse_below_20pct": bool(min(totals) < 0.2 * people),
        "ideology_homogenized": bool(yearly[-1]["ideology_std"][0] < 0.03 and yearly[-1]["ideology_std"][1] < 0.03),
        "genome_diversity_lost": bool(yearly[-1]["genome_std"] < 10),
        "one_settlement_left": bool((yearly[-1].get("settlements") or {}).get("count", 2) < 2),
        "prices_never_diverged": bool(max((y.get("economy") or {}).get("price_spread_wood", 0) for y in yearly) < 0.02),
        "no_trade": bool((yearly[-1].get("economy") or {}).get("trades", 0) == 0),
    }
    return {
        "seed": seed, "years": years, "minds": minds, "people_start": people, "focal": focal if minds else 0,
        "init_s": round(init_s, 2), "wall_s": round(wall, 1), "days_per_s": round(ticks / wall, 1),
        "ms_per_day": {k: round(sim.timing[k] / ticks, 2) for k in ("background_ms", "minds_ms", "society_ms", "weather_ms", "economy_ms")},
        "mind_stats": mind_stats,
        "mind_thinks": sim.minds.thinks if sim.minds else 0,
        "ms_per_think": round(sim.timing["minds_ms"] / max(1, sim.minds.thinks), 3) if sim.minds else None,
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "worst_relative_conservation_error": worst_conservation,
        "totals": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in sim.counters.items()},
        "degenerate": degenerate,
        "yearly": yearly,
        "chronicle_sample": [e for e in sim.events if e["kind"] in ("raid", "law", "monument", "migration", "leader", "raid_start", "festival", "build", "trade")][:400],
        "transcript_sample": sim.transcript[-12:],
    }


def _economy_row(sim) -> dict:
    """What the economy looked like this year: prices, what money is, inequality, credit, trade."""
    from .economy import GOODS, TOOLS, WOOD

    e = sim.economy
    idx = sim.living()
    if e is None or len(idx) == 0:
        return {}
    total = float(e.settled.sum()) or 1.0
    worth = e.net_worth(idx)
    return {
        "money": e.money,
        "settled_share": {GOODS[g]: round(float(e.settled[g] / total), 3) for g in range(4)},
        "acceptability": {GOODS[g]: round(float(e.accept[idx, g].mean()), 3) for g in range(4)},
        "price_wood": [e.price(v.id, WOOD) for v in sim.villages],
        "price_tools": [e.price(v.id, TOOLS) for v in sim.villages],
        "price_spread_wood": round(max(e.price(v.id, WOOD) for v in sim.villages) - min(e.price(v.id, WOOD) for v in sim.villages), 3),
        "gini_net_worth": round(float(e.gini(worth)), 3),
        "median_net_worth": round(float(np.median(worth)), 2),
        "deprived_share": round(float((sim.p.energy[idx] < 0.35).mean()), 4),
        "tools_per_person": round(float(e.stock[idx, TOOLS].mean()), 3),
        "trades": e.counters["trades"], "attempts": e.counters["attempts"],
        "cleared_share": round(e.counters["trades"] / max(1, e.counters["attempts"]), 3),
        "taken_to_pass_on_share": round(e.counters["to_pass_on"] / max(1, e.counters["trades"]), 3),
        "caravans": e.counters["caravans"], "loans": e.counters["loans"], "defaults": e.counters["defaults"],
        "debt_outstanding": round(sum(d[2] for d in e.debts if not d[4]), 1),
        "tax_collected": round(e.counters["tax"], 1),
        "route_flows": {f"{sim.villages[a].name}->{sim.villages[b].name}": round(q, 1) for (a, b), q in sorted(e.route_flows.items())},
    }


def _settlement_row(sim) -> dict:
    """The coarse-graining: how many settlements there are, of what kind, and how they differ."""
    places = sim.settlements.current
    if not places:
        return {}
    return {
        "count": len(places),
        "kinds": {k: sum(1 for p in places if p.kind == k) for k in ("hamlet", "village", "town", "city") if any(p.kind == k for p in places)},
        "largest": places[0].pop, "smallest": places[-1].pop,
        "names": [p.name for p in places],
        "specialization": [p.specialization for p in places],
        "gini": [p.gini for p in places],
        "cohesion": [p.cohesion for p in places],
        "institutions": [p.institutions for p in places],
        "hinterland": [p.hinterland for p in places],
        "bands_per_settlement": [len(p.bands) for p in places],
        "events_this_year": [e for e in sim.settlements.events if e["day"] > sim.day - DAYS_PER_YEAR],
    }


def _child(args):
    return run(*args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=50)
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--no-minds", action="store_true")
    ap.add_argument("--people", type=int, default=240, help="how many people, every one of them a full mind")
    ap.add_argument("--focal", type=int, default=0, help="ignored: every person is a mind now")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    jobs = [(s, args.years, not args.no_minds, args.people, args.focal) for s in seeds]
    with mp.get_context("spawn").Pool(min(len(jobs), 6)) as pool:
        results = pool.map(_child, jobs)
    for r in results:
        y = r["yearly"]
        print(f"seed {r['seed']} minds={r['minds']} wall {r['wall_s']}s {r['days_per_s']} days/s ms/day {r['ms_per_day']} rss {r['peak_rss_mb']}MB cons {r['worst_relative_conservation_error']:.2e}")
        for row in y[:: max(1, len(y) // 10)] + [y[-1]]:
            print(f"  y{row['year']:>3} pop {row['pop']} births {row['births']} deaths {row['deaths']} raids {row['raids']} trade {row['trade_volume']} mig {row['migrants']} laws {row['laws_in_force']} wood {row['wood_stores']} mon {row['monuments']} martial {row['martial_share']} communal {row['communal_share']} val {row['mean_valence']} αP_out {row['mean_alpha_out']} gstd {row['genome_std']}")
        print("  degenerate:", r["degenerate"], "totals:", r["totals"])
        e0, eN = (y[0].get("economy") or {}), (y[-1].get("economy") or {})
        print(f"  economy: money {e0.get('money')} -> {eN.get('money')} · settled {eN.get('settled_share')} · gini {e0.get('gini_net_worth')} -> {eN.get('gini_net_worth')}"
              f" · price spread (wood) {eN.get('price_spread_wood')} · cleared {eN.get('cleared_share')} of {eN.get('attempts')} · deprived {eN.get('deprived_share')}"
              f" · loans {eN.get('loans')} defaults {eN.get('defaults')} · caravans {eN.get('caravans')}")
        sN = (y[-1].get("settlements") or {})
        print(f"  settlements: {sN.get('count')} {sN.get('kinds')} largest {sN.get('largest')} smallest {sN.get('smallest')}"
              f" · specialization {sN.get('specialization')} · gini {sN.get('gini')} · names {sN.get('names')}")
    if args.out:
        out = {"machine": {"platform": platform.platform(), "python": platform.python_version()}, "runs": results}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
