# Civilization-scale simulation with tensacode agents

**Question.** Can we simulate a massive multi-agent 2.5D world, like a civilization builder, whose agents have cognition, affect, identity, social life, kinship and reproduction, mortality, war, ideology, economics and institutions? Can we do it on this machine (NVIDIA GB10, aarch64, ~121 GB unified memory), with human-shaped cognitive and affective structure?

## It runs: see it

A working vertical slice is in `research/civ_sim/`. Start it and watch:

    PYTHONPATH=src:. python -m research.civ_sim.server        # http://127.0.0.1:8780/

A toroidal planet (it wraps on both axes), four founding bands, ~1,600 people, **120 of them
carrying a full tensacode mind** with a thought stream, decaying episodic memory, theory of mind,
and conversations that pass claims between minds as English sentences (one grammar used in both
directions, a lossy parser, so rumours drift and dialects diverge). Day and night sweep around the
world, two moons and four planets move overhead, weather drifts across the seams, people build and
the ground changes under their feet — paths wear in, soil wears out, fields ripen.

Two things are derived rather than declared, and they are the point:

- **The economy** (`economy.py`): production from labour, skill, land and tools; household
  inventories; a call auction per settlement that nobody sets a price in; and **money that emerges**
  — whichever good people turn out to accept in payment becomes the numéraire, and it has changed
  hands mid-history in a 30-year run.
- **Settlements** (`settlements.py`): there are no city objects. Every 12 days the settlements are
  read off the world by coarse-graining wrapped spatial density, the social graph of bonds and
  kinship, and economic structure — which yields their population, area, specialization, inequality,
  institutions, cohesion, hinterland and hamlet/village/town/city class, and tracks their identity
  as they found, split, merge, move and empty.

Zoom in (mouse wheel) and the map changes level of detail: buildings gain roofs, doors, lit windows
and ripening crop rows, then people gain bodies, clothing, carried tools and **speech bubbles with
the sentence they actually said**.

Measured numbers and what is emergent versus scripted: [slice-results.md](slice-results.md).

## Short answer

**Yes, under three conditions. Not by giving every agent a tensacode `Store` as it exists today.**

1. **Levels of detail are mandatory.** Only a few thousand to ~10k *focal* agents can carry a full claim graph with provenance. A population in the millions has to live as compact numeric state (background agents), and beyond that as statistical cohorts. Agents move between tiers as attention and story salience change.
2. **The claim store has to change before 10k rich agents are comfortable.** Today a claim costs **~1.2 KB** (measured) because every claim is a set of Python objects. A columnar claim store with interned ids is **estimated** at ~60–70 bytes per claim, about 18× smaller. That's an estimate from the row layout; it isn't built.
3. **Rules must be bounded and index-friendly.** A single rule that joins against an agent's whole memory makes an agent-tick 30× slower at 8k claims (measured). The appraisal rules below are written to join on percepts and indexed predicates only.

Language models do **not** belong in the tick loop at any scale. They can author and compile rule packs offline, write chronicles for the viewer, and occasionally reflect for a handful of story-salient agents, with the result compiled back into claims.

## Measured numbers

From `research/civ_sim/bench_agents.py`; raw output in `research/civ_sim/results.json`. Single process, Python 3.12, NumPy, on the GB10. Claim content is synthetic but representative (kin, trust, episodes with felt valence/arousal, beliefs). Peak RSS of the whole run: 540 MB.

| What | Measured |
|---|---|
| tensacode `Store` memory, 100 / 500 / 2,000 claims per agent | 0.12 / 0.53 / 1.98 MB per agent: **~1.15–1.25 KB per claim**, linear |
| Same, with Refs interned across agents | ~1.12–1.25 KB per claim: **interning alone does not help** (records, evidence, hash ids and index sets dominate) |
| One focal agent-tick (integrate a ~12-claim percept snapshot, then `think()` with 5 appraisal rules, then `choose()` among 6 intentions), memory 100 → 8,000 claims | **0.17 → 0.25 ms mean** (≈ 4,000–5,900 agent-ticks/s per core); p95 ≤ 0.31 ms |
| Same, plus one rule that joins a fresh appraisal against *all* trust relations | 0.18 → **7.85 ms** mean at 8,000 claims (p95 21 ms): cost now grows with memory size |
| Background agent as a NumPy struct (affect geometry, perceptual axes, needs, genome, kin, top-8 relations, 8 salient episodes, ideology, wealth) | **215 bytes per agent**; 1M agents = 205 MB |
| Vectorized background tick for 1M agents (movement, needs, gain-weighted appraisal, per-tile social coupling of valence, relation update) | **0.29 s** (single-threaded NumPy) |
| Cohort row (tile × faction: counts, age histogram, affect mean/variance, ideology mean, wealth quantiles, allele frequencies) | 172 bytes |

Why the rule cost grows: `cognition.think` evaluates each rule's full `Store.match` every round, then discards matches that involve no new claim. Semi-naive evaluation saves firings, not matching. A delta-driven matcher (start the join from the new claims) is the core change that makes broad rules affordable.

## What fits in memory (estimates built on the measured unit costs)

The simulation gets about **60 GB**. The remaining ~60 GB covers a 17 GB local LLM, a vision model, the OS and page cache, and headroom.

| Tier | Unit cost | Population | Memory |
|---|---|---|---|
| Focal, today's Store, 2,000 claims each | 1.98 MB (measured) | 10,000 | ~20 GB |
| Focal, today's Store, 20,000 claims each (years of life) | ~24 MB (linear extrapolation) | 2,000 | ~48 GB |
| Focal, columnar store (estimate ~65 B/claim), 20,000 claims each | ~1.3 MB | 10,000 | ~13 GB |
| Background, NumPy struct | 215 B (measured) | 10M | ~2 GB |
| Cohorts | 172 B (measured) | 1M | ~0.16 GB |
| World: 4096×4096 hex tiles at ~16 B/tile, plus 100M-event log at ~32 B/event | computed | | ~0.25 GB + ~3.2 GB |

Throughput (projections from the micro-benchmarks, not a running simulation):
- **10k focal agents** at ~0.2 ms per agent-tick is ~2 s per tick on one core. Sharded by region across ~16 processes it would be roughly 0.15–0.25 s per tick. Cross-shard interaction and IPC aren't measured.
- **1M background agents**: 0.29 s per daily tick single-threaded. 10M would take ~3 s single-threaded, parallelizable by chunk.

## Documents

- [architecture.md](architecture.md): tiers, memory and consolidation, perception, social cognition, kinship and reproduction, mortality, ideology, economy, institutions, war, the 2.5D world, scheduling, the viewer, where LLMs fit, and the phased plan with the smallest vertical slice.
- [slice-results.md](slice-results.md): the running slice, its measured costs, and what is emergent versus scripted.
- [affect-and-selfhood.md](affect-and-selfhood.md): what the three *Shape of Experience* pages actually claim, and how the affect geometry, the perceptual axes (ascription, coupling, gain) and responses to inescapable selfhood become concrete agent structure and update rules.

## Smallest vertical slice

**One village:**
- 200 focal agents (tensacode Stores capped at ~2,000 claims each, ≈ 0.4 GB) and 5,000 background agents.
- Affect geometry, perceptual axes, kinship with reproduction and death, grief, and a household economy.
- An in-browser 2.5D hex view with an agent inspector that shows affect, axes, and the provenance behind each appraisal.

**Projected speed:** ~40 ms per focal tick on one core (≈ 25 ticks/s), with the background tier's cost negligible at this size.

**Test:** do grief, attachment and dehumanization-enabled conflict emerge from the stated rules, and can each be traced back through `explain()`?

## Biggest risks

1. **Python object overhead:** ~1.2 KB per claim, which caps rich agents until the claim store is columnar.
2. **Unbounded rule joins:** measured 30× slowdown; needs a delta-driven matcher plus authoring discipline.
3. **Validity:** the affect measures are an open research framework (the site says the dimensions are a toolkit, not exhaustive). Our proxies (e.g. for integration Φ) aren't the book's quantities, and nothing here shows agents *experience* anything.
4. **Level-of-detail consistency:** demotion loses information, and promotion must not invent history.
5. **Authoring burden:** everything above is still hand-written rules, the same hardcoding problem as the browser agents. It needs a skill-compilation loop to scale.
