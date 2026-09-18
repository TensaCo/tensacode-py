# A toroidal planet at full fidelity

A world that wraps on both axes, three founding bands on a 72² torus, ~240 people, and **every one of them
carrying a full tensacode mind**: claims with provenance, a thought stream, episodic memory that
decays and consolidates, affect read off their own processing, theory of mind, and conversations that
move claims between minds as English sentences. Watch it at **http://127.0.0.1:8780/**
(`PYTHONPATH=src:. python -m research.civ_sim.server --people 240`).

> **There is no longer a background tier.** An earlier build ran ~120 rich minds among 1,600 people
> kept as rows of NumPy, promoting and demoting them toward the camera. That level-of-detail scheme
> is gone: `Minds.refill` gives a mind to anyone alive who lacks one, a mind dies with its person,
> and nobody is a statistic. The population is whatever the machine can carry at that fidelity, and
> `--people` is the knob. **Smaller and complete beats larger and thin.**
>
> Removing it was not cosmetic. The heuristic tier had been *masking two real defects*: the minds
> were myopic about the common store (well-fed people all gathered firewood until the granary emptied
> and then starved, 130 starvation deaths in 200 days), and the commons rationed food even when the
> store was overflowing. Both were invisible while a NumPy heuristic was choosing most people's work.
> Full fidelity found them in an afternoon. That is the argument for it.

Numbers below are measured on this machine (NVIDIA GB10, aarch64) unless marked *estimate*. Three
runs back them, all from `research/civ_sim/measure.py` and `measure_language.py`:

Tests: `tests/test_civ_world.py`, `tests/test_civ_sim.py`, `tests/test_civ_talk.py`,
`tests/test_civ_economy.py`, `tests/test_civ_settlements.py` — 47 tests, all passing.

## What full fidelity costs, and how many people it buys

Measured on this machine (CPU only; a sibling agent has the GPU), 12 days after a 4-day warm-up,
every person a full mind:

| People (all minds) | Torus | Sim days/s | ms/day | ms per mind per day | Peak RSS | Claims/mind |
|---|---|---|---|---|---|---|
| 60 | 48² | **10.4** | 94 | 1.54 | 50 MB | 30 |
| 120 | 56² | **2.70** | 367 | 3.16 | 62 MB | 35 |
| **240** (default) | **72²** | **1.65** | 603 | 2.65 | 81 MB | 36 |
| 360 | 80² | 0.83 | 1,197 | 3.28 | 103 MB | 37 |
| 480 | 96² | 0.85 | 1,175 | 2.44 | 123 MB | 36 |

Per-mind cost is flat at **2.4–3.3 ms per simulated day**, so this scales linearly and the ceiling is
wall-clock, not memory: 480 full minds fit in 123 MB of the 94 GB free. **240 people on a 72² torus
with three bands is the shipped default** — a simulated day every 0.6 s, which is watchable — and
`--people 480 --size 96` still runs at a day per 1.2 s if you would rather have the larger world.
Nothing about the agents changes when you raise it.

Two defaults were chosen by measurement, not taste. At **two** bands on 64² the same 240 people
collapse to 129 with **104 starvation deaths**, one band reduced to two survivors; at **three** bands
on 72² the population grows to 283 with **zero** starvation and the bands stay balanced (110 / 96 /
77). The failure looked like war at first — one band nearly wiped out — but the death ledger said
starvation: the world was simply too small for the people in it. Land per person is the binding
constraint at this fidelity, and it is worth checking before blaming the agents.

**Nothing here is evidence that anyone in it experiences anything.** The affect readings are
structural statistics over each mind's own processing (see
[affect-and-selfhood.md](affect-and-selfhood.md) §2.7), and all language is a hand-written grammar
plus a deliberately lossy parser. No language model runs at any point: the viewer's "model calls"
counter reads 0 by construction.

## What is simulated

| Layer | How |
|---|---|
| **Planet** | 160×160 tiles, wrapping on both axes (a torus). Terrain from periodic noise, so continents cross the seams. Elevation, water, grass, forest, stone, mountain, per-tile soil. |
| **Sky** | The planet turns, so local time runs with longitude and dawn sweeps around the world. Two moons (periods 9.4 and 27.3 days) with phases and moonlight, four planets with their own periods, solar and lunar eclipses, planetary conjunctions. All analytic functions of the clock: deterministic and periodic. |
| **Weather** | A coarse field (40×40) carrying pressure, moisture and a temperature anomaly. Wind comes from the pressure gradient, moisture is advected along it, rain falls on orographic lift and convergence, snow below freezing. Drought, heatwave, frost, storm and flood are named when thresholds are crossed. Everything wraps. |
| **Ground** | Paths wear in where people walk, fields hold worked soil, raids leave burn scars, and **soil fertility falls under continuous cropping and recovers when land is left fallow**; all four decay or recover on their own. Visible on the map. |
| **Buildings** | Houses, granary, well, shrine, workshop, fields, palisade, watchtower, hearth, monuments. Bands queue what they need, pay food and wood, and build over days; progress is visible. Granaries cut spoilage, wells cut illness, palisades and towers make raiding costlier, hearths and houses keep people warm, workshops make tools, shrines and monuments ease existential burden. |
| **People (background tier)** | One row of NumPy per person: needs, warmth, illness, affect, perceptual axes, genome (16 loci incl. appearance), ideology, holdings, skills, kin, sleep. They walk (wrapped), forage, farm, cut wood, quarry, craft, build, eat, sleep by local darkness, bond, give birth, sicken, age and die. |
| **People (focal tier)** | 120 of those same people also have a `tc.Store`. Each waking phase: perceive → integrate → think (appraisal rules) → read affect → recall a memory → `tc.choose` an intention under constraints → act; in the evening, talk. |
| **Economy** | Four goods with a production function, household inventories, a call auction per settlement, money that emerges rather than being assigned, caravans, credit, tax. See below. |
| **Settlements** | Not objects in the simulation. Read off it every 12 days by coarse-graining space, the social graph and economic structure. See below. |
| **Language** | A unification grammar with an Earley chart parser used in both directions, per-settlement dialects that drift, and measured mutual intelligibility. Behind one narrow seam (`language.py`) so it can be replaced. |

## The economy

`research/civ_sim/economy.py`. Goods: food, wood, stone, tools.

- **Production** is multiplicative in labour, skill, land and capital:
  `yield = rate · (0.5 + skill) · health · weather · light · (1 + tools)^0.25`, capped by what the
  tile actually holds — so nine people on one patch share one patch's worth.
- **Capital** is tools, made in workshops out of wood and stone by people whose comparative
  advantage is building, and it wears out. Tool stocks rise and fall on their own.
- **Prices** are written nowhere. Each settlement runs a call auction per good: households post what
  a unit is worth to them (rising with need, falling with what they already hold), the book is
  crossed, each pair meets in the middle, and realized ratios move the local price. A good nobody
  will buy gets cheaper even on a day with no trade.
- **Money is discovered.** Each household holds an *acceptability* estimate per good. A trade clears
  only if the buyer holds something the seller will take — because the seller wants it, or expects to
  pass it on. Every success raises both sides' acceptability for that good and lowers its rivals, and
  neighbours imitate whatever is working. Nothing designates a numéraire.
- **Routes**: caravans carry goods when the price gap beats the cost of the journey — 0.4% of the
  load lost per tile, capped at 35%. (At 1%/tile the cap bound on *every* route between four
  settlements on a 160-tile torus, so distance stopped mattering; the chronicle showing "45% lost on
  the road" on every single caravan is what exposed it.) Prices converge only as far as transport
  allows.
- **Credit**: the hungry borrow from the comfortable, at 10% over 72 days; anyone already in arrears
  is refused, and a mind will not lend to someone it holds a grudge against. Default is remembered —
  the creditor gains a grudge and a `trustworthy = False` belief it can then say out loud. Over 400
  days at 700 people: **434 loans, 101 defaults**, nothing scheduled.
- **The commons is reciprocal**, and this turned out to be load-bearing. Children, the old and the
  sick are fed from the common store first whatever they brought in; an able adult who brought
  nothing home gets as little as 30% of their shortfall. Before this, everyone ate to satiety every
  day, so nobody ever borrowed and population grew without limit. With it, a bad day is a hungry
  evening, credit has a reason to exist, and population goes stationary (see below).
- **Tax**: a settlement with a leader and a sharing law takes 4% of household food into the store.
- **Two exchange layers, on purpose.** Households trade with each other in the auction and along
  caravan routes (this module). Separately, the *common stores* barter food against wood between
  bands at the geometric mean of their marginal rates (`sim._trade`), which is the older and cruder
  of the two. Both are real transfers and both are ledgered; the household layer is the one with
  prices, money and credit in it.
- **Conservation**: every removal is named (`eaten`, `spoiled`, `frost_loss`, `wood_burned`,
  `build_*`, `craft_*`, `*_lost`). Measured error over 1,440 days: ≤2e-8 absolute, ~1e-13 relative,
  for all four goods.

## Settlements, read off the world

`research/civ_sim/settlements.py`. Nothing in the tick loop knows what a town is.

1. **Spatial**: a density field over wrapped 4×4 cells, thresholded at 0.6× the mean occupied cell,
   then connected components *with wrapping* — so a settlement straddling the seam is one settlement.
2. **Social**: two adjacent lumps are merged into one place only if the bonds and parent–child links
   crossing between them are dense relative to the smaller lump. Adjacency gives them the chance;
   the social graph decides.
3. **Economic**: labour mix, holdings, prices, worth.

Derived from that, with nothing assigned: population, area, density, specialization (entropy of the
labour mix), median net worth, Gini, institutional capacity, cohesion, faction count, hinterland
(tiles nearer here than anywhere else), and a **hamlet / village / town / city** class a settlement
crosses on its own. Identity is matched to the previous period by membership overlap, so a place
keeps its name while its people turn over, and *founded / split / merged / moved / grew / emptied*
are logged with the evidence behind them.

The one thing still primitive is the **common store**: a founding band keeps a shared granary. A
settlement may hold several bands and a band may live in two settlements, which is what makes the
coarse-graining do real work — but the granary is not derived, and this is the place that says so.

Over 40 years (seed 7, 1,600 starting people, no minds) the settlement hierarchy **pulls apart on
its own**:

| Year | Settlements |
|---|---|
| 5 | Vannhaven *town* 398 · Eskmarsh *town* 396 · Gellmere *town* 327 · Aldcombe *village* 168 |
| 20 | Eskmarsh 588 · Vannhaven 581 · Gellmere 407 · Aldcombe *village* **62** |
| 35 | Eskmarsh **city** 803 · Vannhaven *town* 768 · Gellmere 434 · Aldcombe 68 |
| 40 | Vannhaven **city** 928 · Eskmarsh **city** 900 · Gellmere *town* 467 · Aldcombe 78 |

Two settlements crossed into *city* on their own while a fourth fell to a third of its original
size, and the class labels followed rather than caused it. Smaller lumps come and go alongside them:
a hamlet of 18 appeared near the towns in seed 1, and in a 128-tile run a 69-person lump lived one
season under its own name (Tormere) before being reabsorbed into Fargard — logged as `merged`.

## Zoom levels

The map has three levels of detail, and the viewer reports what each costs.

| Zoom | What is drawn |
|---|---|
| **far** (< 1.7×) | The prerendered base: terrain, elevation shading, ground state, village rings, weather (cloud, rain, snow), the night gradient, the torus inset globe. |
| **mid** (1.7–3.6×) | Visible tiles redrawn crisply: walls lit on one side, thatch roofs in the band's colour, doorways, windows that glow after dark, staddles under granaries, workshop benches, scaffolding on anything still going up, **crop rows whose colour and height follow how full the tile is**, hearth fire with drifting smoke, carts moving along routes that carried real caravan volume, and the coarse-graining cells on request. |
| **near** (> 3.6×) | Full bodies: striding legs, tunic or robe (longer on women and elders, heavier in autumn and winter), a gold hem on people with a mind, arms that swing or work depending on the task, hair that thins with age, eyes and a mouth that follow the affect motif, brows that furrow on anger/fear/grief, and the tool or load they are actually carrying (basket, axe, pick, hoe, timber, spear). **Speech bubbles carry the sentence the person really uttered**, and the inspected mind gets a thought bubble with its latest inner speech. |

Frame cost, JavaScript side only — measured by driving the real viewer script in node against the
live server with a stubbed 2D context, so it excludes GPU rasterization — over two runs:
**far 4.4–8.5 ms, mid 1.7–2.8 ms, near 0.9–1.8 ms** per frame. Far costs the *most*, which is the
opposite of the usual expectation: it blits the base bitmap nine times for the wrap and iterates
every person on the planet, while the detail passes are bounded to a 48-cell window with the base
bitmap showing through behind them and only draw the few dozen people on screen.

## Language: the swap, and what it did to the bias

`research/civ_sim/language.py` is still the whole seam — six functions (`base_lexicon`, `dialect`,
`drift`, `intelligibility`, `say`, `hear`). What changed is what sits behind it.

**Both directions now go through `tensacode.language`** (the general package: unification features,
a chart parser, semantic frames, open-vocabulary words). Understanding moved first; speech followed
once the five copula failures this world found were fixed. `Heard.via` records which grammar
recovered each claim, and the local grammar in `research/civ_sim/grammar.py` remains the fallback
for anything their grammar cannot say, so a new kind of claim degrades to the old wording rather
than striking the speaker mute.

All **17 sentence shapes this world speaks now round-trip with the right predicate**, including the
copula ("Nise is hungry"), its negation ("Kasa is not trustworthy") and past tense. That last one
fixed a fault of *ours*: claims carry no tense, so the local generator said *"Anem dies"*; villagers
now say *"Anem died"*.

What the world sounds like, live from the Overheard feed:

```
Anibse  [greet]    You look tired, Antuse.
Anibse  [lie]      Coralin holds no bread.
Antuse  [tell]     Coralin holds plenty of gren.
Kaem    [tell]     Coralin holds heaps of bread.
Liemse  [tell]     Emnise said Wood is cheap.
Ibtuse  [confused] I don't take your meaning.
```

`gren` is a drifted form of *grain*, the partitive is correct, the reported speech carries its
source, and the last line is the lossy channel doing its job — 18.5% of utterances are not
understood across diverged dialects.

### Two memos, and a bug the swap exposed

**Both directions are memoized, and both upstream causes have since been fixed.** The general
parser was once ~100× slower than the local one (cProfile: `build_chart` at 112 ms per utterance,
because it keyed its chart on `repr()` of dataclasses — 3.9 M `repr` calls), and generation arrived
with the same shape of problem: swapping `say` onto `realize` took the civ test suite from 35 s to
**359 s**. Both are now fixed upstream — saying one of this world's sentences went 91.15 → 2.84 ms,
and this tree's 193-case suite with the caches **off** went 19.73 s → **1.09 s**.

Saying and hearing are pure functions, so both are still memoized, but they are now an optimization
rather than a crutch, and that is a *proven* property rather than an assumption: memo-on and memo-off
produce **byte-identical** state after 30 simulated days — claims hash `69376d73abc061ef`, utterance
hash `a8c1106d7924a848`, both ways. `test_the_memos_are_transparent_not_load_bearing` holds every
sentence shape to that, and **every verification in this tree is run with the caches off**, because
a warm cache can make an A/B look clean while the change under test has broken something.

**A new bad-English case, found by widening our own test rather than by reading output.** The
well-formedness check ran only against the shared vocabulary, and dialects use mass-quantifier
synonyms for *much* (`plenty`, `heaps`) — so the viewer was showing *"Coralin holds heaps bread"*.
Their grammar cannot say the partitive at all (`realize` returns `None` for "heaps of bread"), so
generation falls back to the local grammar whenever a dialect's quantifier is not one of
`much / little / no / some / many / few / full`, and the sentence comes out as *"Coralin holds heaps
of bread"*. The test is now parametrized over every dialect, which is what would have caught it.

### What the world demands of the grammar, as a suite

`tests/test_civ_language_demands.py` (193 cases) is every claim shape `minds.py` actually speaks,
crossed with four dialects, asserting two things: that `say → hear` recovers the predicate and the
object, and that what comes out is **well-formed English** — no doubled copula, no bare *am* with a
third-person subject, no partitive without *of*, no raw `True`/`False` on the surface, one full stop.

The language fork's own conclusion is worth carrying here: an authored grammar suite is best read as
*a floor on obvious breakage, not as evidence of coverage*, and the sets that actually found bugs
were the ones with independent provenance. These 21 shapes were not written to exercise a grammar —
they are what a simulated village needs to say — and between them they have turned up five copula
failures, a lost quantifier, a mis-stemmed verb, a tense that could not be expressed, and the
partitive gap above.

### The answer: the shared understatement was an artifact, and it is gone

The previous build measured every mind holding `"some"` when the truth was `"much"` — 0% correct, all
wrong in the same direction — and this document concluded the lossy channel *collapses* belief rather
than scattering it. **That conclusion was wrong, and it was wrong because of the thin local lexicon,
not because of anything about rumour.** Re-measured at 240 full minds after 120 days, with the
general parser:

Re-measured with the caches off after the generation swap and the partitive fix:

| Settlement | Truth | Minds holding a belief | Share correct | What they hold |
|---|---|---|---|---|
| Aldmere | much | 229 | **60.7%** | much ×139 · none ×90 |
| Brenholt | **none** | 472 | **27.3%** | none ×129 · little ×128 · much ×125 · some ×90 |

Brenholt is the interesting row: its store is genuinely empty, and its people are spread across all
four amounts — 125 of them believe it is full. That is a four-way disagreement about a fact of the
world, held by people who each heard it from someone. The earlier measurement of this table (43.1%
correct, and a `much`/`some`/`unsaid` split) was taken before speech was well-formed English; the
numbers above supersede it.

Real divergence: a village splits roughly evenly between people who think the granary is full and
people who think it is middling, plus a minority who heard an amount they could not pin down. The
old parser threw the quantifier away and every mind landed on the same fallback; a grammar that can
represent `much food` keeps it, and a drifted word (`"Coralin holds much fud"`) now costs the
*word*, not the *quantity*.

The channel is still lossy and now more honestly so: **14.6% of utterances are not understood at
all** in a live run (up from 0.1%), because a listener whose dialect lacks a content word often loses
the sentence. Hearsay provenance names the intermediate speaker — `heard via Dorase`, 12,505 `heard`
citations in one 120-day run.

### How much of the lossy channel was ours, and how much was theirs

A trailing-dot bug in the general tokenizer made `food.` a single token with no lexicon entry, so
the **last word of every sentence** entered as a guessed unknown name — the same mechanism as the
`am`/`dies` problem above, but applying to every sentence rather than four shapes. It was fair to ask
how much of this world's "misunderstanding" was that bug rather than dialect drift.

**None of it.** `_canonicalize` tokenizes with its own pattern and drops punctuation before the
general tokenizer sees anything: of the 15 sentence shapes this world speaks, **0 hand a
dot-terminated token across the seam** — they receive `Coralin holds much food`, never
`Coralin holds much food.`. `test_our_canonicaliser_never_hands_the_parser_a_dot_terminated_word` is
the evidence and the guard. So the not-understood rate is dialect drift, which is what the channel
is supposed to model.

What *did* move our comprehension numbers was our own bad grammar (the `am`/`dies` spurious unknowns,
above) and the partitive fix. Measured now, with the caches off, 240 minds, 120 days:

| | value |
|---|---|
| Utterances | 29,995 |
| Not understood | 5,048 (**16.8%**) — all of it dialect drift |
| Ambiguous | 151 |
| Hearsay citations | 15,354, naming the intermediate speaker (`heard via Kose`, …) |

### Determinism, after the parser's tie-break fix

The language fork also fixed a nondeterminism — `grammar.categories()` returned a set, so
equal-scoring readings could vary run to run, and 14 of their 171 parses moved as ties broke the
other way. Since this world's beliefs are built out of those parses, that is worth re-checking
rather than assuming. Two runs of seed 7, 200 people, 30 days, compared on twelve fields including
SHA-256 hashes over every mind's claims, every utterance and every decision:

```
seed 7 run A vs run B — identical fields: 12 of 12
   every field identical, including claim/utterance/decision hashes
   8619 claims, 3810 utterances, claims hash 386d75b6d9ca04d6, said hash 6b7b1991e148679d
seed 7 vs seed 8 differ: True
```

Determinism by seed holds through the new parser, at the grain of individual beliefs and sentences,
not just aggregate counts. It still holds after speech moved to their grammar too (12 of 12 fields
identical across two runs of seed 7).

**The hashes did move between builds, though, and the reason is worth recording.** Swapping
generation changed the claim and decision hashes, not just the utterance hash — which looks like
something crossing over from speech into meaning. It is not. The surface form feeds back into the
random stream: `hear` reports words the listener's dialect lacks, and an unfamiliar word triggers a
comprehension roll (`min(0.8, 0.45 × unknown words)`) that consumes a draw. The *old* wording was
producing spurious unfamiliar words because it was ungrammatical —

| old wording | unfamiliar to the listener | new wording | unfamiliar |
|---|---|---|---|
| "Nise am hungry." | `am` | "Nise is hungry." | — |
| "Anem dies." | `dies` | "Anem died." | — |
| "Wood am dear." | `am` | "Wood is dear." | — |

— so four of the most common sentence types were carrying a **45% chance of being misunderstood for
no reason other than our own bad grammar**, and each of those rolls moved the random stream. Fixing
the English removed a spurious comprehension failure, which is a behavioural fix and not a cosmetic
one, and the changed hashes are its expected consequence. Claim and utterance *counts* are identical
across the two builds (8,619 and 3,810), which is what one would expect if the meaning layer is
behaving equivalently and only the trajectory moved.

### Still true about the language

Each settlement keeps its own word for food, dialects drift by sound change, borrowing and
forgetting, and after 20 generations at rate 0.22 mutual intelligibility spans **0.4–1.0** (mean
0.65) with **3 of 6 pairs asymmetric**. Reproduce with `python -m research.civ_sim.measure_language`.
Reported speech is now generated as *"Nise said Coralin holds much food"* rather than *"I heard from
Nise that ..."*, because the general parser keeps the content clause of the first form and drops it
from the second.

## Measured

**40 years × seeds 1–3, 1,600 starting people, no minds** (`civ_slice_dynamics.json`). Every
degeneracy check passes in all three seeds: not extinct, no village emptied, no runaway growth, no
collapse below 20%, ideology not homogenized, genome diversity intact, more than one settlement,
prices diverged, trade happened.

| | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| Settlement classes at year 40 | 1 city, 1 town, 2 villages | **3 cities**, 1 village | **3 cities**, 1 village |
| Largest / smallest | 1,005 / 128 | 965 / 86 | 1,020 / 105 |
| Money | **wood** (63% of settled trades) | none (tools 47%, wood 43%) | none (wood 54%, tools 42%) |
| Wealth Gini | 0.38 → 0.70 | 0.47 → 0.74 | 0.47 → 0.72 |
| Per-settlement Gini | 0.63 – 0.72 | 0.62 – 0.77 | 0.65 – 0.74 |
| Specialization per settlement | 0.0 – 0.96 | 0.79 – 1.0 | 0.0 – 0.94 |
| Deprived | 2.3% | 1.0% | 2.8% |
| Loans / defaults | 2,586 / 872 | 3,155 / 1,014 | 3,274 / 1,034 |
| Trades cleared | 25% of 835k attempts | 33% of 755k | 28% of 821k |
| Caravans | 620 | 590 | 590 |
| Price spread, wood | 0.32 | 0.30 | 0.30 |
| Speed | 57 days/s | 50 days/s | 53 days/s |
| Peak memory | 62.6 MB | 63.2 MB | 63.2 MB |
| Conservation error | 2.2e-14 | 5.3e-14 | 5.8e-14 |

A settlement with **specialization 0.0** appears in seeds 1 and 3: the band whose woodland ran out
has everyone doing the same single job. Inequality also differs *between* settlements inside one
world (0.62 to 0.77), which is the kind of thing the coarse-graining exists to surface.

The urban hierarchy that forms is **different in each seed** — one primate city dominating three
stunted villages in seed 1, three comparable cities in seed 3 — from identical rules.

**Why some settlements stall** is the best result in the sweep, and it was not designed: the wood
column shows local forest exhaustion. In seed 1 the first band's store runs 2,522 → 1,146 → 535 →
**0** by year 17 and never recovers, and its population sits at 209 after 40 years while the band
that kept its woodland grows to 1,120. In seed 2 two of four bands hit zero wood. A settlement can
outgrow its hinterland's timber, and then it stops growing — deforestation as a cap on urbanization,
falling out of per-tile regrowth and travel costs alone.

**10 years × seeds 1–2, 1,200 people, 100 minds** (`civ_slice.json`):

| What | Measured |
|---|---|
| Speed | 2.8–2.9 simulated days/s |
| Cost per simulated day | background ~9.5 ms · **minds 321–334 ms** · society ~7 ms · weather ~2.4 ms · economy ~4.3 ms |
| Cost per mind-think | 1.61 ms · per conversation 3.49 ms |
| Peak memory | 158.7 and 160.6 MB |
| Conservation error | 6.0e-15 and 9.2e-15 relative |
| Claims held per mind | 75 · episodes 32.9 · relationships 26.8 · theory-of-mind entries 21.7 |
| Utterances | 29,363 and 33,414 |
| Not understood | 28 (0.10%) and 84 (0.25%) — dialect drift really does lose sentences |
| Ambiguous | 39% and 43% |
| Claims moved between minds | 29,335 and 33,330 |
| Conversations | 13,563 and 15,297 |
| Omens observed | 1,126 and 1,035 |
| Settlements at year 10 | 2 towns, 2 villages **and a hamlet of 14**; 3 towns + 1 village |
| Per-settlement Gini | 0.28 – 0.79 |

### What the minds believe about the granaries, against the truth

Measured at year 10 (`mind_stats.belief_about_stores_vs_truth`), and it corrects something this
document previously claimed:

| Settlement | Truth | Minds holding a belief | Share correct | What they hold |
|---|---|---|---|---|
| Coralin | much | 26 | **0%** | "some" ×26 |
| Dunmarsh | much | 26 | **0%** | "some" ×26 |
| Brenholt | much | 26 | **0%** | "some" ×26 |
| Aldmere | much | 14 | **0%** | "some" ×14 |

Every mind is wrong, and **every mind is wrong in the same direction**. The lossy channel does not
scatter belief — it *collapses* it. "much" degrades to "some" on the first hop through a dialect that
has drifted, and "some" is then stable because every dialect can still parse it, so the population
converges on a shared systematic understatement of how much food there is. Earlier drafts of this
document claimed the opposite (that minds end up holding "much"/"little"/"none" about the same
granary); the measurement says that is not what happens, and the measurement wins.

### Bytes per agent, by tier

| Tier | Per agent | What it is |
|---|---|---|
| **background** | **182 B** | 39 scalar fields (needs, affect, skills, kin, sleep, ideology…) + genome[16] + ascription[6], all structure-of-arrays |
| + economy | **+64 B** → 246 B | holdings[4] and acceptability[4] as float64 |
| **focal** | **~23 KB** | a `tc.Store` holding ~47 live claims with provenance, 26 episodes, 10 relationships, a thought stream. Pickled: ~473 bytes/claim, which is an upper bound on the in-memory cost |

So a focal agent costs about **95× a background agent** in memory, and the 1.2 KB/claim figure from
[the bench](../../research/civ_sim/bench_agents.py) is the thing the columnar store in
[architecture.md](architecture.md) §6 would attack.

### With minds and without

| | no minds | 100–120 minds |
|---|---|---|
| Speed | 55–63 simulated days/s | 3.1 days/s |
| ms per simulated day | 20 ms total | 323 ms total (302 of it minds) |
| Peak RSS, ~1,200–1,600 people | 62–63 MB | 157–159 MB |
| Conservation error | ~5e-14 | ~4e-15 |

Dropping the mind tier makes the world **~19× faster** and leaves the dynamics intact — which is why
the decade-scale sweeps above are run without it and the cost and belief measurements with it.

The **memory ceiling is not a constraint**: 159 MB against a 6 GB budget. What binds is *time* — the
mind tier is ~15× the rest of the simulation put together.

### The settlement lifecycle

30 years, seed 2, 128 tiles: **10 founded, 6 merged, 6 emptied, 1 shrank a class**. Verbatim:

```
day    0 Ossford      founded   266 people, town
day  551 Ossford      merged    Rhunmarsh (9) is now part of Ossford
day  551 Rhunmarsh    emptied   Rhunmarsh has no one left
day  587 Vannmarsh    founded   11 people, hamlet
day  599 Ossford      merged    Vannmarsh (11) is now part of Ossford
day 1067 Norgard      founded   8 people, hamlet
day 1079 Ithmere      merged    Norgard (8) is now part of Ithmere
```

The pattern is real and repeats: small groups drift out of a town, hold together long enough to be
recognised as their own hamlet with their own name, and are then reabsorbed when the kinship ties
back to the parent outweigh the distance. Nothing schedules any of it — it falls out of where people
walk and who they are related to.

Language layer, measured separately (`measure_language.py`, `civ_language.json`): 1.23 ms per
utterance, 2.47 ms per conversation.

### Conflict, across seeds

Raids are rare and that is a finding, not an omission. Across 40 years × 3 seeds without minds:
**0, 0 and 1 raid**. The reason is the harm constraint: a raid needs a leader whose ascription
toward the target (αP) has fallen below the threshold, and in these runs αP *rises* over time —
0.64 → 0.90 in seed 1, 0.65 → 0.81 in seed 2, 0.67 → 0.84 in seed 3 — because trade keeps happening
and nothing starves badly enough to push it down. Martial temper falls with it (0.33 → 0.04 in seed
1) under the hawk–dove frequency dependence. A world that trades does not raid, and the model says
so without being told to.

The sanctions counter tells the opposite story: **57, 130 and 101 hoarders fined**. Coercion in these
runs is internal and legal, not external and violent.

### Degenerate outcomes we actually found

The degeneracy checks in `measure.py` all pass now, but two real degeneracies were found and fixed,
and one remains:

1. **Deprivation pinned at 0.000, and credit as dead code.** With an unconditional commons everyone
   ate to satiety every day: no hunger, no borrowing (0 loans in 400 days), and population growing
   without limit. Found by asking why a whole subsystem never fired. Fixed by making the commons
   reciprocal — dependents first, able adults in proportion to what they brought in. Deprivation is
   now 0.3–2.3% and the credit market runs.
2. **Runaway growth.** 1,700 → 3,700 over 30 years with the `runaway_growth` flag tripped. Soil
   exhaustion was added (and is real) but was *not* what fixed it; distribution was.
3. **Still degenerate: ideology converges toward communal.** Communal share rises to 0.90–0.93 in
   all three seeds while martial falls to 0.04–0.15. The `ideology_homogenized` check passes only
   because it tests the standard deviation, which stays above the threshold. The interior equilibrium
   the frequency dependence is supposed to produce is not holding over 40 years, and it is not
   because the law forces it. This is unresolved.

## The communal drift, explained

This document previously listed "ideology drifts communal and we do not know why" as an unresolved
attractor. It is now resolved, by decomposing the daily change in mean `share` into the four things
that can move it: survivors changing their own minds (copying), who is born, who dies, and residual.

Measured over 960 days at the old scale (1,600 people, 160² torus, seed 1):

| Day | Mean share | Copying | Births | Deaths | Unexplained |
|---|---|---|---|---|---|
| 240 | 0.559 | **+0.012** | −0.001 | −0.001 | 0.000 |
| 480 | 0.582 | **+0.032** | +0.001 | −0.001 | 0.000 |
| 960 | 0.613 | **+0.061** | +0.002 | +0.001 | 0.000 |

**Copying accounts for essentially all of it**, with births and deaths contributing ~2% between them
and zero residual. So it is not selection (communal people surviving better) and not inheritance
(children being born more communal) — it is payoff-biased imitation, exactly the mechanism that is
written down, running in the direction the payoff term points.

And the direction is not universal: **at full fidelity the drift is not there.** At 240 people on a
64² torus, mean share sits at 0.50–0.53 over 400 days and the decomposition attributes the movement
to copying in the *downward* direction (−0.023, −0.036, −0.016 at days 100/200/300), with the people
who die being *more* communal than the living (0.540 against 0.512). The hawk–dove frequency
dependence is doing its job; which way it points depends on land pressure and granary state, which
depend on scale. An attractor at one scale, an interior equilibrium at another — and now measured
rather than asserted.

## Band fission: a settlement can now hold more than one

The blocker was that `p.village` meant two things at once — *which common store feeds me* and *which
band I am* — so marrying into another settlement transferred a person wholly and no place could ever
contain two bands. Those are now separate:

- **`p.village`** is the commons whose granary you eat from. Marrying out changes it; a fission
  changes it.
- **`p.lineage`** is the founding line you descend from, inherited from your mother and never
  changed by anything. **Dialect follows lineage**, so language change is now change in a line of
  descent rather than in a postcode, and `_language_change` borrows between lines whose speakers
  live among each other.

And a splinter group can now open its own store (`sim._fission`). The conditions are read off the
world, not scheduled: a commons of 40+ people, at least 12 of them living more than 7 tiles out and
knotted together within 6 tiles of each other, and either land crowding above 1.0 or a mean grudge
above 0.3 against whoever leads the place. They take a share of the food, wood, stone and herd in
proportion to their numbers, keep their lineage and dialect, and start building a granary.

Two things had to be fixed to get there, and both were found by instrumenting rather than guessing:

1. **Nothing dispersed anyone.** People lived a mean 3.5 tiles from their store (p90 6.5), so no
   group of twelve was ever far out. `working_range` now gives each person a fixed preferred distance
   from their store, scaled by how crowded it is, and `_retarget` biases their work toward that ring.
   A crowded settlement pushes its working radius outward and some people end up living out there.
2. **Fission keyed on the coarse-graining could never fire**, because a dispersed population is one
   *connected* lump — spreading out makes the lump bigger, not two lumps. It is now keyed on the
   commons' own members, which is also what produces the thing we wanted: one settlement, two stores.
3. **"Crowded" was measuring the wrong thing** — people per unit of *stored food*, so a rich village
   read as uncrowded and the gate never opened. It is now people per unit of workable *land* in
   reach, one shared definition (`land_crowding`) used by both the dispersal and the split, with the
   threshold taken from the observed distribution (0.8–0.9 while comfortable; 1.12 and 1.49 for the
   two largest commons at day 1,500 of seed 1).

### What happens then

Seed 1, 320 people, 1,800 days: **5 fissions, 3 founding bands becoming 8 commons, and 11 sightings
of a multi-commons settlement.** The largest:

| Day | Settlement | Class | Pop | Commons living there | Lineages | Inequality *between* its commons |
|---|---|---|---|---|---|---|
| 1,200 | Norstead | village | 169 | {2: 161, 3: 8} | all three | 0.765 |
| 1,440 | Norstead | **town** | 216 | {2: 202, 3: 14} | all three | 1.000 |
| 1,560 | Norstead | town | 227 | {2: 201, 3: 14, 4: 12} | all three | 1.414 |
| 1,680 | Farmarsh | town | 256 | {1: 211, 5: 17, 6: 28} | all three | 0.774 |
| 1,680 | Torgard | town | 219 | {0: 198, 7: 21} | all three | 1.000 |

Norstead holds **three separate household economies** and has done for 480+ days and counting. So a
town here is no longer a renamed founding band: it is a place several economies share, and the
derived properties say so — `factions` rises to 2 when a settlement holds more than one commons, and
`between_band_inequality` (the spread of median net worth across the commons inside one settlement)
ranges from 0.02 to 1.41, so some of these towns are shared evenly and others are not at all.

Marriage across settlements does the rest of the mixing: **every settlement holds people of all
three founding lines** within a few hundred days (201 marriages out by day 1,680), so lineage and
residence are thoroughly decoupled even where there is only one store.

### The ceiling is gone

There used to be a hard `MAX_COMMONS = 12`, because each person's ascription `alpha_p` was a
fixed-width array — and a long run *reached* it at day 2,160, after which fission simply stopped and
nothing in the world said why. A cap that quietly changes the physics is the same class of defect as
the dead credit subsystem and the always-binding transport loss recorded above, so it is fixed the
honest way: `People.widen` adds a column when a store is founded. A new column starts at 0.5 —
neither kin nor stranger — because that is what a settlement you were not born knowing is to you,
and the founding then sets the real values.

The only bound left is the `int8` that holds the id (126 stores). If it is ever reached the world
*says so* — a chronicle line ("no further bands can form") and a memory in the minds of the people
who wanted the split — rather than fission ending in silence. Tested both ways.

Seed 1, 320 people, **3,600 days**, straight past the old ceiling:

| Day | Population | Commons | `alpha_p` width | Fissions | Refused | Multi-commons settlements | Food error | RSS |
|---|---|---|---|---|---|---|---|---|
| 1,200 | 626 | 5 | 8 | 2 | 0 | 1 | −1.8e-10 | 49 MB |
| 1,800 | 892 | 8 | 8 | 5 | 0 | 3 | −4.2e-10 | 52 MB |
| 2,400 | 1,283 | **18** | 18 | 15 | 0 | 3 | +5.8e-11 | 62 MB |
| 3,000 | 1,949 | **33** | 33 | 30 | 0 | 3 | +3.5e-09 | 91 MB |
| 3,600 | 2,775 | **48** | 48 | 45 | 0 | 3 | +1.2e-09 | 136 MB |

Commons keep forming — 48 of them from 3 founding bands, none refused — and **conservation holds
unchanged** with four times the old ceiling (food and wood errors stay at 1e-9 on totals in the
hundreds of thousands). Norstead ends as a **city of 1,074 people containing twenty separate
commons**, which is the clearest form of the point: a city here is a place many household economies
share, not a founding band with a bigger number after it.

The cost of the honest fix is linear and small. Measured at 240 full minds with 3, 12 and 24 bands:
`alpha_p` grows 32 → 56 → 104 bytes per person, RSS 77 → 91 → 94 MB, and **per-mind cost does not
rise** (2.02 → 1.56 → 1.48 ms per simulated day; it falls slightly because more bands means smaller
settlements and fewer neighbours to scan).

### What is still load-bearing

- **Nothing amalgamates.** Fission is one-way: across 3,600 days the number of commons only ever
  goes up, and the coarse-graining still reports three settlements the whole time (one of which is
  the twenty-commons city). Real villages merge their granaries; this world cannot.
- **Fissions are capped at one per season** to keep each split legible, so a very crowded world
  splits more slowly than it should.
- **One other fixed width was found by this audit and fixed**: `VILLAGE_NAMES` is a six-name tuple
  that `_place_villages` indexed directly, so founding a seventh band crashed with `IndexError`.
  Fission had its own name generator and so never hit it. Founding names are now generated when the
  written list runs out. Everything else keyed by commons was already dynamic
  (`minlength=len(self.villages)` and list comprehensions over `self.villages`); `village` and
  `lineage` are `int8`, which is where the 126 bound comes from.

## Does a poor commons recover, or empty?

Neither. It stays poor. Seed 1, 320 people, 2,400 days, tracking the worst-off store each season:

| Day | Population | Commons | Poorest | Richest |
|---|---|---|---|---|
| 1,200 | 588 | 4 | Coralin, 181 people, 9.8 food/head, **median worth 1.43**, 32 head | Corstead, 16 people, worth 9.61 |
| 1,680 | 780 | 8 | Corstead, 23 people, 8.4 food/head, **worth 0.00**, 2 head | Brencombe, 28 people, worth 5.75 |
| 2,040 | 1,027 | 11 | Aldmere, 235 people, 8.9 food/head, **worth 0.00**, no herd | Brencombe, 37 people, worth 12.81 |
| 2,400 | 1,326 | 12 | Aldstead, 58 people, 8.7 food/head, **worth 0.00**, no herd | Aldcombe, 24 people, worth 11.10 |

Two things come out of this that nobody wrote down:

1. **Poverty is stable, not fatal.** The poorest commons keeps feeding its people at subsistence
   (8–10 days of food per head, all of it in the common store) while private wealth goes to zero and
   the herd is eaten down to nothing. It does not empty and it does not recover. The commons is what
   makes that possible: it is exactly the institution that converts destitution into survival.
2. **Big is poor and small is rich, per head.** The poorest is nearly always one of the large old
   stores (Coralin 10 of 20 periods, Aldmere 6) and the richest is nearly always a small young
   splinter (Brencombe, 28–37 people, median worth 11–13 against Coralin's 0.4). Fission hands a
   splinter a share of the stores proportional to its *numbers* but it leaves with fewer mouths and
   unworked land, so it starts rich and stays rich. Inequality here is between settlements and it is
   produced by the way settlements are born.

The ceiling bites in this run: the world reaches `MAX_COMMONS = 12` at day 2,160 and fission stops.

## Leadership: a threshold derived instead of tuned

Leaders used to change five times in two years, which this document listed as implausible. The cause
was not the prestige formula but the absence of incumbency: `argmax` was recomputed every season over
a prestige score that includes a luck term of `0.2 · U(0,1)`, and on a flat distribution the luck
alone decided it.

The fix takes the threshold from the noise rather than from taste. Displacement is now judged on
*standing* with the luck term removed, and a challenger must lead the incumbent by more than
**2σ of the luck term** — `2 · 0.2/√12 ≈ 0.115` — i.e. by more than a lucky draw could have
explained. Luck still settles a near-tie when there is no incumbent.

Measured over 20 years and three commons: **leader changes fell from an every-few-seasons churn to
22, an average tenure of about 175 days (3.6 years)**, with leaders still replaced when someone is
genuinely more prestigious or when one dies.

## Livestock, stalls and interiors

These were asked for twice and were absent, on the rule that nothing is drawn without state behind
it. They now have state:

- **Livestock** is a real store: surplus grain above 10 days per head is fattened into animals at 12
  food per head of cattle, and when the store falls below 5 per head the herd is slaughtered to get
  through the season. Animals also die on their own. Every animal is food that was really put in, so
  they are inside the conservation ledger (`to_herd`, `from_herd`, `herd_lost`) — measured error with
  herds running: 1.5e-11. The viewer draws one beast per ten head the settlement actually holds. In
  the live run this produced a stark divergence: **Aldmere holds 382 head and 1,230 food while
  Brenholt holds 0.6 head and none** — the poor settlement has eaten its savings.
- **Market stalls** are drawn only in settlements where the auction actually cleared trades that day.
- **Interiors**: past the nearest zoom, a finished house is cut away and you see who is inside. The
  occupancy is the people really asleep within a tile of it, and an empty house shows a cold hearth.

## What is emergent and what is scripted

**Emergent** (not written down anywhere as an outcome):
- **Which good becomes money**, and that it can change hands mid-history when the stock of the
  incumbent money collapses.
- **Local prices.** Wood at 0.35 in one settlement and 0.81 in another on the same day, converging
  only as far as caravans and transport loss allow. Stone falls to ~0.2 because nobody wants it.
- **Specialization.** People with a building advantage drift into wood, stone and crafting; the
  labour entropy per settlement ranges 0.54–0.95 across places in the same run.
- **Where the settlements are**, how many, what class, and that hamlets appear and are reabsorbed.
- **The urban hierarchy.** Two settlements grow into cities while a neighbour withers to a third of
  its size, from identical rules and one shared world.
- **Towns made of several economies.** A crowded commons throws off a splinter that opens its own
  granary in the same place, so a town ends up holding two or three household economies with
  different median wealth — and which of them stays poor is not assigned.
- **Which settlement becomes the trade hub.** In seed 1 Coralin ends up shipping 942 / 733 / 790
  units out along three routes while taking 285–363 back — an export position nobody assigned.
- **Population equilibrium and famine/recovery** against the land: an early die-back (deaths outrun
  births for the first five years), then slow growth.
- **Ideology mix.** Payoff-biased copying with frequency dependence settles at an interior
  equilibrium that differs by seed and by band.
- **Law**, appearing when enough of a band already shares and repealed when adherence falls.
- **War, and its precondition**: a raid needs a leader whose ascription toward the target has fallen
  below the harm threshold. Scarcity and past raids lower it; trade and time raise it.
- **Which word a settlement uses for food** (*food* / *grain* / *bread* / *forage*), and where the
  dialect boundaries fall, out of who borrows from whom.
- **A shared misconception.** Not divergence, as we first assumed — the lossy channel makes the whole
  population understate the granaries in the same direction (see the table above). Emergent, and not
  the emergent thing we expected.
- **Grudges and gratitude**, including from unpaid debt, which then change how someone is spoken to.
- **Local deforestation as a cap on growth**: a band that outgrows its timber stalls at ~150 people
  while its neighbour reaches 1,000.
- **Whether money appears at all.** Wood took the role in one of three seeds; in the other two, wood
  and tools split it and no good passed the 55% test after 40 years.

**Scripted** (authored rules; the interesting behaviour is their interaction): the appraisal rules
and the intention set; the affect prototypes and transition costs; the grammar and lexicon; building
types, costs and effects; that a festival falls at a full moon and that an eclipse is read as an
omen at all; the shape of the harm constraint, the fine and the prestige formula; the goods, their
decay rates, durability and portability; the auction form; the size-class thresholds.

## Honest limits

1. **Full fidelity costs population, and that is the deliberate trade.** 2.4–3.3 ms per mind per
   simulated day means ~240 people at a watchable rate and ~480 at the edge of one. Thousands of
   full minds would need the columnar claim store and delta-driven matcher from
   [architecture.md](architecture.md) §6 (*estimate*: ~18× smaller claims, and matching that does
   not rescan). Until that exists, the world is a valley, not a continent.
2. **Malthus arrived late, and by accident.** Until the commons was made reciprocal, every person
   ate to satiety every day, deprivation read 0.000, population grew without limit (1,700 → 3,700
   over 30 years), and the credit market never fired once. Soil exhaustion was not the binding
   check — fields are a small share of the food supply. What bound it was *distribution*: once an
   able adult who forages badly eats less, population goes stationary (710 → 784 over 400 days,
   births 285 / deaths 201) and borrowing appears. That was a modelling correction found by asking
   why a whole subsystem was dead code, and it is reported here rather than quietly tuned in.
3. **Inequality is high and partly an artefact.** Gini on net worth reaches 0.6–0.65, but much of
   that is because the sharing norm moves household food into the common store, so most households
   hold near zero and the measure sees concentration that the commons is in fact smoothing.
4. **Inequality between settlements can be brutal, and one side may be dying.** In the live
   full-fidelity run Aldmere holds 382 head of cattle and 1,230 food while Brenholt holds 0.6 head
   and nothing at all. That is the livestock mechanism working as designed (savings eaten in a lean
   season), but whether Brenholt recovers or empties has not been run out, and a settlement dying of
   poverty next to a rich one is a result that deserves its own measurement rather than a shrug.
5. **Leadership churns implausibly fast.** In the chronicle Coralin changes leader five times in two
   years, because prestige is recomputed seasonally and the top of a flat distribution flips easily.
6. **Theory of mind is depth 1** and only about what someone said or seemed, not nested belief.
7. **Fission is one-way and capped.** Two commons in one town can never amalgamate their granaries
   again, and `MAX_COMMONS = 12` is a hard ceiling from the fixed-width ascription array — seed 1
   reaches 8 in 1,800 days, so it is reachable, and hitting it stops fission with no reason given
   inside the world.
8. **Ambiguity is high** (46% of utterances have more than one reading) and the ranking is a simple
   score, so the "best" reading is sometimes just the first plausible one.
9. **`intelligibility(d, d)` can read 0.8** — a dialect can fail its own older forms. Known wart in
   the strict scoring, not root-caused.
10. **The lossy channel drops whole sentences.** 14.6% of utterances are not understood at all once
    dialects have drifted, because a listener missing a content word usually loses the clause rather
    than recovering part of it. Partial understanding exists (`food:unsaid` when a quantifier is
    lost) but it is the minority outcome.
11. **Affect proxies are not the book's quantities.** The "integration" reading is a provenance
   statistic over cross-module derivations, not integrated information.
12. **Weather is coherent but crude**: one layer, no fronts, no seasonal wind reversal.
13. **Frame costs above are JavaScript only.** No browser was available in this environment, so the
    viewer was driven headlessly with a stubbed canvas; real rasterization cost is not included.
14. **The parse memo is still a crutch, by the rule written before the result was known.** The rule
    was: re-measure with `CIV_PARSE_MEMO=0` once the general parser's `repr`-keyed chart was fixed,
    and if per-mind cost lands within ~1.5× of the memoized number, keep the memo as an optimization
    and drop it from this list; otherwise it stays a crutch and the docs say so.

    The fix landed (their chart no longer keys on `repr()`; 520k repr calls → 0). Re-measured at 240
    full minds:

    | | ms per mind per day | simulated days/s |
    |---|---|---|
    | memo off, before their fix | 32.58 | 0.13 |
    | **memo off, after their fix** | **14.38** | **0.29** |
    | memo on | 3.33 | 1.23 |

    Their fix was worth a genuine **2.3×**, leaving a **4.3×** gap — outside 1.5×, so the memo
    stayed a crutch. Re-applied again once *generation* also moved onto their grammar, and this time
    it went the wrong way:

    | | ms per mind per day | simulated days/s |
    |---|---|---|
    | both memos off | **36.65** | **0.11** |
    | both memos on | 3.00 | 1.36 |

    Speaking and understanding both go through their grammar now, so with the memos off the cost
    was briefly *higher* than before the generation swap (36.65 against 14.38). Then the generation
    fix landed (32× on a single sentence) and the picture changed again:

    | | ms per mind per day | simulated days/s |
    |---|---|---|
    | memos off, before their generation fix | 36.65 | 0.11 |
    | **memos off, now** | **6.91** | **0.59** |
    | memos on | 2.00 | 2.04 |

    The letter of the rule and the thing it was protecting against have come apart, so both are
    reported. The measured gap is **3.45×**, still outside the 1.5× threshold, so by the letter the
    memo stays on this list. But what the rule was guarding against — "this world only runs at a
    watchable speed because a village repeats itself" — is **no longer true**: without either cache
    the world runs at 0.59 simulated days per second, which is watchable, and the caches are proven
    to change nothing but the clock. The memo is an optimization now. Keeping the entry, reworded,
    rather than deleting it on a threshold it does not quite meet.
15. **Our own generator realizes `died` in the present tense** ("Anem dies") because the claim
    carries no tense. The general package gets this one right; we do not.
16. **Dialects do not vary by seed.** `dialect_from` keys its synonym choice on the settlement index
    more than on the generator, so seeds 7 and 8 both give food / grain / bread to their three lines.
    Drift over generations *is* seeded; the starting vocabularies are not.
