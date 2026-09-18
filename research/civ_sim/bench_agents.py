"""Measure what simulated agents cost in memory and time, per representation tier.

    PYTHONPATH=src:. python -m research.civ_sim.bench_agents [--quick]

Tiers measured (numbers go to research/civ_sim/results.json):

1. focal agents as tensorcode Stores (claims with evidence and provenance), at several
   memory sizes K, with and without interned Refs/strings; bytes per agent and per claim
   (tracemalloc), and the cost of one agent-tick: integrate a perception snapshot, then
   think() with appraisal rules, then choose() among intentions.
2. background agents as a NumPy structure of arrays (fixed-size human-shaped state:
   affect geometry, perceptual axes, needs, genome, kin, sparse relations, salient
   episodes); bytes per agent from nbytes and one vectorized tick over 1M agents,
   including a spatial social-coupling step (per-tile mean affect).
3. cohorts (statistical populations): bytes per cohort.

Peak memory is kept small (a few GB at most); the process RSS peak is reported.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import statistics
import time
import tracemalloc
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import tensorcode as tc
from tensorcode.backends.builtin import UtilityChooser
from tensorcode.cognition import Fragment, Rule, Thought, integrate, think

OUT = Path(__file__).parent / "results.json"
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
V = tc.Var


# ---------------------------------------------------------------- tier 1


class Interner:
    """Share Ref objects and predicate strings across agents (a cheap, obvious optimization)."""

    def __init__(self, on: bool) -> None:
        self.on, self.refs = on, {}

    def ref(self, s: str) -> tc.Ref:
        if not self.on:
            return tc.Ref(s)
        r = self.refs.get(s)
        if r is None:
            r = self.refs[s] = tc.Ref(s)
        return r


def fill_memory(mind: tc.Store, agent: int, k: int, rng: np.random.Generator, names: Interner) -> None:
    """K representative long-term claims: kin, relations (trust/affinity), episodes, beliefs, possessions."""
    me = names.ref(f"person:{agent}")
    src = names.ref(f"obs:agent{agent}")
    edits = []
    for i in range(k):
        kind = i % 5
        other = names.ref(f"person:{int(rng.integers(0, 5000))}")
        if kind == 0:
            claim = tc.Claim(me, "kin_of", other)
        elif kind == 1:
            claim = tc.Claim(me, "trust", (other.id, round(float(rng.random()), 2)))
        elif kind == 2:
            ev = names.ref(f"event:{agent}-{i}")
            claim = tc.Claim(ev, "involved", other)
        elif kind == 3:
            claim = tc.Claim(names.ref(f"event:{agent}-{i - 1}"), "felt", (round(float(rng.normal()), 2), round(float(rng.random()), 2)))  # (valence, arousal)
        else:
            claim = tc.Claim(names.ref(f"idea:{int(rng.integers(0, 64))}"), "believed_by", me)
        edits.append(tc.Tell(claim, (tc.Evidence(src, T0 + timedelta(minutes=i)),)))
    mind.apply(tc.Patch(tuple(edits), mind.revision))


def appraisal_rules(me: tc.Ref) -> list[Rule]:
    """A small, representative appraisal set (structure, not content, is what is being timed)."""

    def fear(b, mind):
        yield tc.Claim(me, "appraises", ("threat", b["x"].id)), tc.Score(0.8, "appraisal")

    def attach(b, mind):
        yield tc.Claim(me, "appraises", ("attachment", b["x"].id))

    def desire(b, mind):
        yield tc.Claim(me, "wants", ("eat", b["x"].id))

    def grief(b, mind):
        yield tc.Claim(me, "appraises", ("loss", b["x"].id))

    def motif(b, mind):
        # affect motif as structure: many appraisals collapse effective rank, threat raises counterfactual weight
        kinds = [r.claim.object[0] for r in mind.claims(me, "appraises")]
        yield tc.Claim(me, "motif", ("threat" in kinds, "attachment" in kinds, "loss" in kinds))

    P = tc.Ref("scope:percept")
    return [
        Rule("fear_from_armed_stranger", ((V("x"), "near", me), (V("x"), "armed", True)), fear),
        Rule("attachment_to_kin_nearby", ((V("x"), "near", me), (me, "kin_of", V("x"))), attach),
        Rule("desire_food", ((V("x"), "is_a", "food"), (me, "hungry", True)), desire),
        Rule("grief_on_death", ((V("x"), "dead", True), (me, "kin_of", V("x"))), grief),
        Rule("affect_motif", ((me, "appraises", V("a")),), motif),
    ], P


def bench_stores(ks: list[int], n: int, intern: bool) -> list[dict]:
    rows = []
    for k in ks:
        rng = np.random.default_rng(0)
        names = Interner(intern)
        gc.collect()
        tracemalloc.start()
        before = tracemalloc.take_snapshot()
        minds = []
        for a in range(n):
            m = tc.Store()
            fill_memory(m, a, k, rng, names)
            minds.append(m)
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        used = sum(s.size_diff for s in after.compare_to(before, "filename"))
        live = sum(len(m._claims) for m in minds)
        rows.append({"claims_per_agent": k, "agents": n, "interned": intern, "live_claims": live,
                     "bytes_per_agent": round(used / n), "bytes_per_claim": round(used / max(live, 1))})
        del minds
        gc.collect()
    return rows


def bench_tick(k: int, ticks: int, agents: int, broad: bool = False) -> dict:
    """One agent-tick = integrate a perception snapshot (~12 claims) + think(appraisal rules) + choose among 6 intentions."""
    runtime = tc.Runtime([UtilityChooser()])
    rng = np.random.default_rng(1)
    names = Interner(True)
    integ_ms, think_ms, choose_ms, derived = [], [], [], 0
    objective = tc.Objective("priority", "highest urgency", lambda option, mind: option[1])
    with tc.use(runtime):
        for a in range(agents):
            mind = tc.Store()
            fill_memory(mind, a, k, rng, names)
            me = names.ref(f"person:{a}")
            kin = [r.claim.object for r in mind.claims(me, "kin_of")][:3]
            rules, scope = appraisal_rules(me)
            if broad:  # a rule that joins a fresh appraisal against every trust relation: cost grows with memory size
                rules = rules + [Rule("reassess_trust_on_threat", ((me, "appraises", V("a")), (me, "trust", V("t"))), lambda b, m: iter(()))]
            for t in range(ticks):
                near = [names.ref(f"person:{int(x)}") for x in rng.integers(0, 5000, 6)] + ([kin[t % len(kin)]] if kin and t % 3 == 0 else [])
                claims = [(tc.Claim(p, "near", me, scope=scope), None) for p in near]
                claims += [(tc.Claim(near[0], "armed", True, scope=scope), None)] if t % 7 == 0 else []
                claims += [(tc.Claim(names.ref(f"thing:berries{t % 4}"), "is_a", "food", scope=scope), None)]
                claims += [(tc.Claim(me, "hungry", True, scope=scope), None)] if t % 2 == 0 else []
                frag = Fragment(names.ref(f"obs:tick{t}"), tuple(claims), snapshot_of=scope, observed_at=T0 + timedelta(hours=t))
                t1 = time.perf_counter()
                thought = integrate(mind, frag)
                t2 = time.perf_counter()
                d = think(mind, rules, since=thought)
                t3 = time.perf_counter()
                options = [("flee", 0.9 if t % 7 == 0 else 0.1), ("eat", 0.5 + 0.01 * t % 3), ("greet", 0.3), ("work", 0.4), ("rest", 0.2), ("talk", 0.35)]
                tc.choose(options, objective=objective, given=mind)
                t4 = time.perf_counter()
                integ_ms.append((t2 - t1) * 1e3)
                think_ms.append((t3 - t2) * 1e3)
                choose_ms.append((t4 - t3) * 1e3)
                derived += len(d.added)
            runtime.trace.spans.clear() if hasattr(runtime.trace, "spans") else None
    total = [a + b + c for a, b, c in zip(integ_ms, think_ms, choose_ms)]
    return {
        "claims_per_agent": k, "agents": agents, "ticks_per_agent": ticks, "broad_join_rule": broad, "derived_claims": derived,
        "integrate_ms_mean": round(statistics.mean(integ_ms), 3), "think_ms_mean": round(statistics.mean(think_ms), 3),
        "choose_ms_mean": round(statistics.mean(choose_ms), 3), "tick_ms_mean": round(statistics.mean(total), 3),
        "tick_ms_p95": round(sorted(total)[int(0.95 * len(total))], 3), "agent_ticks_per_second_one_core": round(1000 / statistics.mean(total), 1),
    }


# ---------------------------------------------------------------- tier 2

BACKGROUND = np.dtype([
    ("pos", np.float32, 2), ("elev", np.float16),
    ("affect", np.float16, 7),     # valence, arousal, integration, effective rank, counterfactual weight, self-salience (attention, causal)
    ("axes", np.float16, 4),       # default agency ascription, default phenomenality ascription, coupling, gain
    ("needs", np.float16, 6),      # food, water, safety, belonging, status, meaning
    ("genome", np.uint8, 16),      # heritable trait loci (temperament, fertility, health, ...)
    ("age_days", np.uint16), ("life", np.uint8),  # life stage / sex / alive flags packed
    ("parents", np.int32, 2), ("partner", np.int32), ("household", np.int32), ("faction", np.int32),
    ("ideology", np.float16, 8), ("wealth", np.float32),
    ("rel_id", np.int32, 8), ("rel_w", np.float16, 8),              # top-8 relations: who and signed weight
    ("ep_id", np.int32, 8), ("ep_val", np.float16, 8), ("ep_age", np.uint16, 8),  # 8 salient episodes (ids into a shared event log)
])

COHORT = np.dtype([
    ("tile", np.int32), ("faction", np.int32), ("count", np.uint32), ("age_hist", np.uint32, 16), ("affect_mean", np.float16, 7),
    ("affect_cov_diag", np.float16, 7), ("ideology_mean", np.float16, 8), ("wealth_quantiles", np.float32, 5), ("genome_freq", np.float16, 16),
])


def bench_soa(n: int, ticks: int, grid: int) -> dict:
    rng = np.random.default_rng(2)
    t = time.perf_counter()
    a = np.zeros(n, dtype=BACKGROUND)
    a["pos"] = rng.random((n, 2), dtype=np.float32) * grid
    a["affect"] = rng.normal(0, 0.3, (n, 7)).astype(np.float16)
    a["axes"] = rng.random((n, 4)).astype(np.float16)
    a["needs"] = rng.random((n, 6)).astype(np.float16)
    a["rel_id"] = rng.integers(0, n, (n, 8))
    alloc_s = time.perf_counter() - t
    tick_s = []
    for _ in range(ticks):
        t = time.perf_counter()
        pos = a["pos"]
        pos += rng.normal(0, 0.5, pos.shape).astype(np.float32)
        np.clip(pos, 0, grid - 1, out=pos)
        a["pos"] = pos
        tile = pos[:, 0].astype(np.int32) * grid + pos[:, 1].astype(np.int32)
        needs = a["needs"].astype(np.float32)
        needs[:, :2] += 0.01                                     # hunger, thirst accumulate
        aff = a["affect"].astype(np.float32)
        axes = a["axes"].astype(np.float32)
        # appraisal: valence follows need satisfaction; arousal follows need change; gain scales bottom-up input
        gain = axes[:, 3]
        aff[:, 0] += 0.1 * gain * (0.5 - needs[:, :2].mean(1)) - 0.05 * aff[:, 0]
        aff[:, 1] = 0.9 * aff[:, 1] + 0.1 * np.abs(needs[:, 0] - 0.5)
        # social coupling: each agent drifts toward its tile's mean valence, weighted by coupling
        counts = np.bincount(tile, minlength=grid * grid)
        mean_val = np.bincount(tile, weights=aff[:, 0], minlength=grid * grid) / np.maximum(counts, 1)
        aff[:, 0] += 0.05 * axes[:, 2] * (mean_val[tile] - aff[:, 0])
        # relations: weight toward partners with similar valence (gather over sparse top-8)
        rel_val = aff[a["rel_id"], 0]
        a["rel_w"] = (0.95 * a["rel_w"].astype(np.float32) + 0.05 * (1 - np.abs(rel_val - aff[:, :1]))).astype(np.float16)
        a["affect"] = aff.astype(np.float16)
        a["needs"] = np.clip(needs, 0, 1).astype(np.float16)
        tick_s.append(time.perf_counter() - t)
    return {"agents": n, "bytes_per_agent": BACKGROUND.itemsize, "total_mb": round(a.nbytes / 2**20, 1), "alloc_s": round(alloc_s, 3),
            "tick_s_mean": round(statistics.mean(tick_s), 4), "tick_s_min": round(min(tick_s), 4), "grid": f"{grid}x{grid}", "threads": "numpy default (mostly single-threaded ops)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    q = args.quick
    ks = [100, 500, 2000] if not q else [100, 500]
    results = {
        "date": time.strftime("%Y-%m-%d"),
        "machine": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__},
        "tier1_store_memory": bench_stores(ks, 20 if not q else 5, intern=False) + bench_stores(ks, 20 if not q else 5, intern=True),
        "tier1_tick": [bench_tick(k, 60 if not q else 20, 5 if not q else 2, broad) for broad in (False, True) for k in ([100, 500, 2000, 8000] if not q else [100, 500])],
        "tier2_soa": bench_soa(1_000_000 if not q else 100_000, 10 if not q else 3, 1024),
        "tier3_cohort_bytes": COHORT.itemsize,
    }
    results["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    OUT.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
