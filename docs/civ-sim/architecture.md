# Architecture: a 2.5D civilization of tensacode agents

Measured unit costs are in [README.md](README.md). Items marked *(estimate)* are arithmetic, not measurements.

## 1. Three tiers, one world

| Tier | Representation | Who | Tick | Unit cost |
|---|---|---|---|---|
| **T0 focal** | tensacode `Store` per agent: claims with evidence, scopes and provenance, rules, `choose` with constraints, `authorize` for institutional acts, trace | near the camera, story-salient, leaders, anyone in a dense event | discrete-event: wakes on percept change or timers, ~hourly sim-time when active | 1.2 KB/claim measured; ~0.2 ms per agent-tick measured with bounded rules |
| **T1 background** | one row in a NumPy structured array (215 B) | everyone else who is individually alive | fixed daily tick, vectorized per chunk | 0.29 s per 1M agents per tick, measured single-threaded |
| **T2 cohort** | one row per (tile-cluster × faction × age band) (172 B) | far regions, the deep past, off-map peoples | seasonal tick | trivial |

**Shared, not per-agent:**
- the world grid;
- the append-only **event log**, columnar (event id, tick, place, kind, participants offset, magnitude);
- the genome table;
- the ideology registry (doctrines as shared claim sets plus an 8-D embedding);
- institutions (roles, rules, legitimacy);
- markets (order books per settlement);
- the culture tables (motif prototypes, the transition-cost matrix C, viability boundaries).

Agents hold *beliefs about* shared state as their own claims, in their own scopes. The world holds the truth. Divergence between the two is where rumor, ideology and mistakes live.

### Promotion and demotion

- **Promote T1→T0** when the camera approaches, story salience rises (kin of a focal agent died, a leader is chosen, a battle starts), or event density crosses a threshold. The new Store is seeded from the row:
  - top-8 relations become `trust` and `kin_of` claims;
  - the 8 salient episodes are re-read from the event log;
  - axes, needs, ideology and wealth carry over.

  Nothing is invented: absent history stays absent (`Unknown`), and later rules can only fill it from the event log.
- **Demote T0→T1** after consolidation (see [affect-and-selfhood.md §2.6](affect-and-selfhood.md)) and log what was dropped.
- **Aggregate T1→T2** when a region has no focal agents for a season. Disaggregating samples individuals from the cohort's distributions and allele frequencies, and is marked as sampled.

## 2. A focal agent's cycle

This is the same loop as the browser agents, with a world instead of a screen.

1. **Perceive.** Visible entities within radius (terrain line of sight) are ranked by salience S(x)·novelty·γ, and the top N (≈12–30) are written as a snapshot Fragment in `scope:percept`.
2. **Integrate.** Claims that are no longer perceived retract, and derived appraisals fall with them. Belief updates are weighted by gain γ.
3. **Think.** Appraisal rules produce claims; module tags and κ gate the cross-module rules. Theory-of-mind rules run in `scope:model:x` for targets with high αA(x). Affect readings and the motif are computed from this tick's provenance.
4. **Deliberate.** Intention generators propose options from needs, motif policy biases, roles and B (existential burden). `tc.choose` then picks one:
   - under **constraints**: norms and laws, the αP-weighted harm constraint, physical feasibility;
   - with an **objective**: expected valence plus role duty plus ideology.
   - `Unknown` means an honest non-decision: the agent waits or seeks information.
5. **Act.** World actions are `@tc.action` values. Institutional actions (taxing, arresting, declaring war) go through `tc.authorize` against the agent's role authority, and receipts land in the event log.

**Rule discipline** (from the benchmark: one broad join made ticks 30× slower at 8k claims):
- every rule's first pattern must bind to the percept scope or a small indexed predicate;
- joins against long-term memory must be keyed (`(me, kin_of, x)` with x already bound);
- no rule may enumerate all trust, belief or episode claims;
- aggregates like "average trust in my faction" are maintained incrementally as entity payloads, not recomputed by rules.

## 3. Domains

### Perception and attention
- Hex line-of-sight with elevation (from a cached horizon map per chunk).
- Perception radius scales with elevation, light, weather and role (scouts).
- Attention is the entity-indexed salience field S(x); S(me) is attentional self-salience.

### Social cognition
- **Relations:** `trust`, `affinity`, `debt`, `rivalry` are claims with provenance from episodes.
- **Theory of mind:** depth 1 by default. A `model:x` scope holds what x wants, fears and believes about me, for high-αA(x) targets only; depth 2 is reserved for leaders and negotiations.
- **Reputation** is a shared, noisy aggregate per settlement, sampled by agents via gossip claims weighted by trust.

### Kinship, pair bonding, reproduction (population-level biology, nothing explicit)
- **Pair bonding:** a bond claim forms when mutual S(x) and attachment appraisals persist and compatibility is above threshold. Compatibility comes from ideology distance, status and genome-coded temperament. Bonds co-regulate arousal.
- **Fertility** is a function of age, health, nutrition and bond status. **Heredity:** 16 loci (u8) with recombination per locus and a small mutation rate. Loci shape temperament priors (baseline γ and κ ranges), health, fertility and size. Kin distance from parent links prevents close-kin pairing.
- **Population genetics** is tracked as allele frequencies per cohort. Drift and selection emerge from mortality and fertility.

### Mortality
- A hazard function of age, injury, disease, famine and violence.
- **Death** is an event with inheritance (household wealth, roles vacated, institutional succession rules) and **grief appraisals** in everyone with S(dead) above threshold, kin first.
- Grief follows the asymmetric transition costs (one-way back toward prior coupling) and raises mortality salience, and with it the selfhood responses.

### Ideology and belief transmission
- An ideology is a shared doctrine: a claim set in the registry plus an 8-D vector, for example cosmology, in-group scope, authority, hierarchy, purity, reciprocity, afterlife promise, and violence licence.
- **Transmission:** vertical (parents), then oblique (teachers, priests), then horizontal (peers).
  - Weights: trust·αA(source)·κ, with prestige bias and conformity bias (a frequency-dependent term).
  - Gain γ sets resistance to contrary evidence.
- **Mortality salience raises adherence** (terror management). Doctrines also carry axis defaults: animist cosmologies raise αA for nature; exclusionary doctrines lower αP for named out-groups.

### Economics
- Tiles hold resources (fertility, timber, stone, ore, fish, water). Households produce, consume, store and trade.
- **Early:** gift and reciprocity obligations as `debt` claims.
- **Later:** settlement markets clear daily by call auction, vectorized over background agents, with prices as shared state.
- **Wealth, inequality and taxation** feed the status need and grievance appraisals.

### Institutions, norms, law
- **Norms emerge** as frequently fulfilled expectations: a claim `(norm:n, expected, behavior)` with an adherence aggregate.
- **Codification** turns a norm into law: an institution adds it as a `tc.Constraint` for members, with sanctions.
- **An institution** is a shared entity: roles, authority grants (a `tc.Authorization` per role), rules, treasury, and legitimacy (the aggregate belief of members). Temples, schools and courts are also *affect infrastructure* (see [affect-and-selfhood.md §2.5](affect-and-selfhood.md)).

### Conflict and war
- **Grievances** are claims derived from losses, injustice appraisals and scarcity.
- **Mobilization** probability per faction rises with grievance, ideology violence licence, low out-group αP, economic stress, and leader intentions chosen through `choose`.
- **Battles** resolve at army level (Lanchester-style attrition on tiles, terrain and elevation modifiers), vectorized. Focal soldiers experience episodes: fear motifs, deaths, trauma.
- **Aftermath:** grief and mortality salience cascade, dehumanization persists in the memory of affected agents, and peace rises through trade, intermarriage and ritual that raise αP.

## 4. The 2.5D world

- **Grid:** axial hex, 64×64 chunks. Per tile: elevation i16, terrain u8, water u8, fertility u8, resources u8×4, owner i32, plus a road/structure byte. About 16 B/tile *(estimate)*, so 4096² tiles ≈ 256 MB.
- **Pathing:**
  - flow fields per settlement or target, cached and recomputed on terrain change;
  - hierarchical A* over chunk portals for long trips;
  - background agents step along flow fields, vectorized.
- **Visibility:** a per-chunk horizon map from elevation gives cheap line of sight for focal agents; background agents use radius only.
- **Scheduling:**
  - a fixed day tick for T1, a season tick for T2;
  - a discrete-event priority queue for T0 (wake on percept change, timers, messages). Idle focal agents cost nothing.
  - Cross-tier events (a T1 agent strikes a T0 agent) are delivered as percepts.
- **Chunks** near focal agents or the camera run at full detail. Far chunks run as cohorts.
- **Sharding:** one process per region band, each owning its chunks' agents. Cross-shard interaction goes through a message queue with a one-tick delay. The 20 cores make ~16 shards plausible *(estimate)*.
- **Viewer** (same style as the Agent Wall: dark, IBM Plex, SSE):
  - canvas isometric hex rendering with elevation shading;
  - focal agents as sprites, background agents as density heatmaps, cohorts as region tints;
  - an **inspector** for a clicked agent: affect readings and motif trajectory, axes, S(x) top list, current intention, and `explain()` chains behind appraisals ("afraid of Oda ← saw Oda armed at the ford ← …");
  - a **chronicle** panel of salient events.

## 5. Where language models fit

- **Never** in the tick loop. 10k agents × one call per tick is out of reach, and the result wouldn't be inspectable.
- **Offline authoring:** draft rule packs, motif prototypes, doctrine texts and dialogue templates. They are compiled into rules and claims, reviewed, and tested like code.
- **Narrative:** on-demand chronicle summaries for the viewer from the event log, rate-limited and cached.
- **Rare reflection:** a few story-salient focal agents per sim-day may get a budgeted model call (`tc.Budget`). The result is parsed back into claims with `method="model-reflection"` provenance, so it can be audited and turned off.
- Local model throughput on this box for these uses has not been measured here.

## 6. Core tensacode changes needed for scale

1. **Columnar claim store.** Interned ids (subject u32, predicate u16), a tagged 8-byte object, scope u32, interval as two u32 ticks, source u32, observed u32, confidence f16+kind u8, and a derived-from offset u32, about 44 B per row. Sorted index arrays add ~12 B and derived-from lists ~8 B, for **~64 B/claim** *(estimate)*, versus 1.15–1.25 KB measured today. `Ref` and `Claim` objects would be materialized views, not storage.
2. **Content ids without SHA per claim.** A 64-bit hash of interned fields is enough in-simulation; SHA-256 stays for export.
3. **Delta-driven matching in `think`.** Start each rule's join from the new claims (true semi-naive evaluation), instead of full-match-then-filter. This also removes the broad-rule cliff.
4. **Compiled rules.** Patterns become index lookups plus vectorized filters over the columnar store, optionally batched across agents that share a rule pack.
5. **Evidence compaction.** Keep first and last evidence plus a count per claim; move full history to the shared event log.
6. **Shared read-only memory** for culture tables and the world grid across shard processes (numpy memmaps or shared memory).

## 7. Phased plan

| Phase | Deliverable | Exit test |
|---|---|---|
| **P0** | Hex world with elevation, resources and flow fields; T1 background agents with needs, movement, fertility and mortality; viewer heatmaps | 1M background agents, daily tick < 0.5 s; population stable over 100 sim-years |
| **P1 (vertical slice)** | One village: **200 focal agents** (Stores capped at ~2k claims, ≈0.4 GB), 5k background agents. Affect readings and motifs, axes (γ weighting, κ gating, α per target), kinship and bonds, reproduction and heredity, death and grief, household economy, inspector with `explain()` | runs at ≥ 20 focal ticks/s on one core (projection: ~40 ms per tick for 200 agents); scripted scenario checks that a kin death produces grief motifs in kin (not strangers), traceable through provenance |
| **P2** | Ideology transmission, norms→law, temples and rituals as affect infrastructure, existential burden and the four response families, pathology detectors; T0↔T1 promotion and demotion with consolidation | demotion/promotion round-trip keeps top relations and episodes; no invented history |
| **P3** | Core changes: columnar claims, delta matcher, compiled rules | re-run `research/civ_sim/bench_agents.py`: ≤ 100 B/claim and flat tick cost with broad rules at 8k claims |
| **P4** | Multiple settlements, markets, grievances and war, dehumanization and peace mechanisms, sharded processes | a conflict emerges and resolves under stated rules; cross-shard consistency checks |
| **P5** | Scale run: 10k focal, 1M+ background, cohorts beyond; LLM chronicle | memory < 60 GB, ≥ 1 sim-hour per wall-second for focal regions |
