# Cognitive structures: quantity, contingency, cause, probability, time

> **Direction update, 2026-09-19:** [36 — The structured cognitive workspace](36-structured-cognitive-workspace.md)
> places these structures inside a broader requirement: preserve and revise the meaning of
> language and visual input before and during reasoning. A quantity, causal, or temporal
> record is useful only when its semantics survive interpretation, inference, and action.
> The measurements below are historical mechanism results, not evidence of that full loop.

**The question this answers:** "what are we doing to improve the language performance and
ability to align parsed language with general cognitive structures like causal modeling,
probability, contingency awareness, math, etc?"

**The short answer.** Five structures now exist, each with claims, provenance and a refusal
path. Three earn their place by measurement, one is scaffolding that has not yet paid, and
one is a clean negative result that says the missing capability is not the one I built.

| Structure | Measurement | Verdict |
| --- | --- | --- |
| Quantity + unit arithmetic | GSM8K, 300 items: **0.0% correct**, 2.7% coverage | **Necessary, nowhere near sufficient.** Earns its place only on refusal discipline |
| Contingency / expectation | Screen prediction, 240 steps: **0.793** predicting *changes* vs **0.069** predicting *values* | **Earns its place**, and the encoding matters more than the mechanism |
| Causal structure | Intervention vs co-occurrence | **Earns its place in a world with confounds** (toy: correlation gets it wrong in both directions). **Buys nothing in computerworld**, which has none |
| Probability | Calibration ECE **0.079** on a real task; pooling **0.918** vs last-writer-wins **0.570** | **Earns its place** |
| Temporal | 364 queries: **357 right, 0 wrong**, 7 refused | **Works, but it is a mechanism test**, not a hard benchmark |

Everything is optional and additive: 906 tests pass. The one failure in the suite is
`test_assistant_confirmation.py`, another fork's file, which asserts that procedures ask
before changing things — a check the repo-wide removal of safety gates invalidated. It does
not touch any module here.

---

## 0. What was actually wrong (verified, not assumed)

The brief said the parser produces propositional content and drops the rest. Checking it
found the diagnosis was in the wrong place, which changed the design:

* **The number is in the parse.** "Anem has 12 sheep" parses with
  `count=Entity(number, '12')` on the noun's features. It is `to_claims` that discards it,
  emitting `Anem have sheep`. The alignment gap is in the **projection**, not the grammar.
* **Conditionals parse as two unlinked readings.** "if I press Send an announcement appears"
  returns `press(...)` and `appear(...)` as separate meanings with the connective gone. Both
  clauses survive; only the relation is lost.
* **Connectives fare worse than that.** "the field failed because it did not rain" gives one
  reading with `object=name:because` — "because" parsed as a thing — alongside a clean
  negated `rain` frame. "twice as much as" drops to 0.56 coverage. "probably" becomes the
  predicate.

So `semantics_bridge.py` recovers quantity from the parse (where it already is) and
connectives from the surface string (where the grammar leaves them). Every surface-string
workaround is listed in `needed_from_grammar`, so the gaps stay visible instead of becoming
permanent; the grammar is another fork's file and these are what it would need to close.

---

## 1. Quantities and math — a clean negative result

`quantity.py`. A `Quantity` is a value with a `Unit`; a unit is a product of symbols with
exponents; a dimension signature decides what may be added to what. Rates compose
(`12 dollar/hour × 3 hour = 36 dollar`), spellings of one dimension convert (`1 kg + 500 g`),
and a mismatch **refuses**:

```
add(12 sheep, 5 coin/sheep) → Unknown(dimension_mismatch,
    "cannot add 12 sheep and 5 coin/sheep: count:sheep vs count:sheep^-1·currency")
```

`derive` records the working, so `explain` shows the arithmetic and retracting a premise
withdraws the conclusion:

```
entity:Anem revenue '60 coin'
  ← reasoning:arithmetic via arithmetic:mul from:
    entity:Anem has '12 sheep'
    entity:market price '5 coin/sheep'
```

### GSM8K, 300 test items (`eval/results/structures_gsm8k.json`)

| | value |
| --- | --- |
| Accuracy | **0.000** (rule arm before: 0.000; local Qwen3-8B: 0.813) |
| Coverage (items it answered at all) | 0.027 |
| Answered and wrong | 8 |
| Refused at composition | 217 (72%) |
| Refused at reading the question's unit | 70 (23%) |
| Refused at reading any number | 5 (2%) |
| Items whose gold working needs a constant the question never states | 122 (41%) |

**Where it stops: composition, not reading.** 72% of items yield their quantities and then
fail to compose them. That is planning over quantities — deciding which two of five numbers
combine and in what order — and no amount of unit machinery supplies it.

Two honest notes on method:

* The solver is **cue-driven and refuses ambiguity** rather than searching all arithmetic and
  keeping whatever looks plausible. A search that picks by plausibility scores better and
  means nothing, because selection would be doing the work the derivation is supposed to do.
* A **negative count is refused**, not returned: "8 pieces − 14 pieces = −6 pieces" is a sign
  the composition was wrong. This cut wrong answers from 12 to 8 without inventing any
  correctness.
* The 41% figure had a **bug in its first version**: counting every number in the gold working
  as "external knowledge" made it 96%, because intermediate results are not in the question.
  Walking the working in order and counting only operands no earlier step produced gives 41%.

**Verdict: necessary, nowhere near sufficient.** The unit type prevents a class of silent
nonsense and makes arithmetic explainable. It does not make word problems tractable, and the
0.0% says so plainly.

---

## 2. Contingency and expectation — the one that clearly works

`expectation.py`. An `Expectation` is a cue, what should then hold, and how sure. `check`
compares it against what was observed and writes a `Violation` naming both sides with its
surprise in bits; violations are claims, so attention can be seeded from them. A `Predictor`
learns the probability from its own hits and misses and **refuses below three trials**.

### Screen prediction, 240 actions, deterministic engine (`structures_expectation.json`)

| Arm | Accuracy (all aspects) | First half → second half |
| --- | --- | --- |
| absolute values, action cue | 0.069 | 0.139 → **0.000** |
| absolute values, action+state cue | 0.056 | 0.113 → 0.000 |
| **changes, action cue** | 0.732 | 0.685 → 0.778 |
| **changes, action+state cue** | **0.793** | 0.745 → **0.841** |

Per aspect, the change encoding: title 1.00, terminal-open 1.00, lines-direction 0.99,
interactions-changed 0.84, nodes-direction 0.80. 24–27 early trials were **refused** for too
few observations.

**Two findings, one of them about my own measurement.** Predicting *what the screen is*
collapses to zero: a node count drifts with clock and panels, and a most-frequent-value
predictor cannot track it. Predicting *what the action did* reaches 0.79 and **improves with
experience** (0.745 → 0.841). Conditioning the cue on state helps in the change encoding
(+0.061) and is noise in the absolute one. The first version of this measurement used only
absolute aspects and read as a failure of the mechanism; it was a failure of the encoding.

A bug the tests caught in the same module: Laplace smoothing over *observed* outcomes only
meant 20 hits and no misses reported probability **1.000** — the certainty the docstring
claimed to prevent. Smoothing over at least one unseen outcome gives 0.954.

**Verdict: earns its place.** And the lesson generalizes: expectations should be about
changes.

---

## 3. Causal structure — right mechanism, wrong world to prove it in

`causal.py`. Causal claims are separate from evidential ones and carry `support`:
`observational` (seen together) or `interventional` (the same state run with and without the
act). `experiment` runs that controlled pair from a `prepare` callable — a fork, a restored
checkpoint — and `control` is what the untreated branch does *instead*, so that everything
following from merely acting cancels. `counterfactual` records the branch that did not happen
in its own scope, where it cannot answer a question about what is.

### In a world with a confound (toy, ground truth by construction)

A switch lights a bulb; a clock ticks on every step whether or not the switch moved.

| Reader | On the bulb (true cause) | On the clock (confound) |
| --- | --- | --- |
| Co-occurrence | **missed** (effect saturates after the first flip: 1/5) | **credited** (5/5) |
| Intervention, do-nothing control | caught (1.00) | **wrongly credited** (1.00) |
| **Intervention, active control** | **caught (1.00)** | **correctly rejected (0.00)** |

`distinguish` returns `{"lit": "caused", "tick": "merely_correlated"}`. Co-occurrence is
wrong in *both* directions — it credits the confound and misses the real cause — which is a
stronger result than I expected to find.

This also exposed two flaws in my own code. `experiment` originally had no way for the
control branch to act, so the treated branch ran one step ahead and "time passed" never
cancelled. And `correlations` keyed on exact values, so an aspect whose value differs every
time (a clock) never accumulated; co-occurrence has to be counted over **movements**, the
same lesson measurement 2 gave.

### In computerworld (`structures_causal.json`)

14 interventional links, deterministic and identical across runs, with effect 1.00 on title,
terminal-open, interactions and node count. But: **the passive and active controls agree
exactly, and co-occurrence gets the same answers** (13 links, same aspects). The reason is
that this world has no spontaneous dynamics — the scene revision never advances, there is no
clock node, nothing moves unless the agent moves it.

**Verdict: earns its place where there are confounds; buys nothing here.** Intervention is
worth its cost exactly when something else is moving, and I could not demonstrate that in
computerworld because nothing there does. The shuffle control is recorded but weak for the
same reason: permuting labels leaves effect sizes intact and only misattributes them (2 of
14 pairs), which a correlational reader could not detect either.

---

## 4. Probability — calibrated where it is learnable, refused where it is not

Added to `expectation.py`: `combine` pools probabilities in log-odds and **writes the
independence assumption into the basis** (`pooled(independent,n=2):source0+source1`);
`disagreement` reports the spread, because pooling hides whether a belief is settled or
contested (0.1 and 0.9 pool to the same 0.5 as 0.5 and 0.5); `calibration` bins stated
probabilities against observed frequencies.

### Calibration on the screen-prediction task (213 stated probabilities)

| Stated | Observed | n |
| --- | --- | --- |
| 0.54 | 0.51 | 70 |
| 0.63 | 0.73 | 11 |
| 0.75 | 1.00 | 20 |
| 0.85 | 0.93 | 103 |
| 0.93 | 1.00 | 9 |

**ECE 0.079**, mean stated 0.732 against an observed 0.793 — slightly *under*confident,
which is what Laplace smoothing should do. The number tracks being right.

### Updating against last-writer-wins (400 facts, 4 reports each)

| | accuracy |
| --- | --- |
| Pooled evidence | **0.918** |
| Last-writer-wins | 0.570 |

Pooled ECE 0.021; every fact was contested. Last-writer-wins is what a store without
probability does when a new claim arrives, and it ends up near the reliability of whichever
source happened to speak last (0.55–0.9). This reproduces the village simulation's pattern —
many minds, one fact, four-way disagreement — with ground truth known by construction rather
than by reading their tree.

**It refuses** to mix a similarity into a probability (`not_probabilities`), to combine
nothing (`no_evidence`), and to honour an assumption it does not implement
(`unsupported_assumption`).

**Verdict: earns its place.**

---

## 5. Temporal — works, and the test is easy

`temporal.py`. A parsed tense becomes an `Interval`; events carry their own time, kept apart
from when a source reported them; `before`/`after`/`since`/`during` answer ordering
questions; `changed_since` reads the reporting clock, which is the right one for "what
changed since my last message"; `tell_order` records an ordering asserted by language when
neither event has a clock ("the grain arrived before the snow came").

### 364 queries over a recorded 40-action session (`structures_temporal.json`)

| Query | Right / asked |
| --- | --- |
| before / after / since | 40/40 each |
| relate (pairwise ordering) | 228/228 |
| during (window) | 1/1 |
| changed_since | 1/1 |
| undated event's time | **6 refused**, 0 guessed |
| undated event kept out of an ordering | 6/6 |
| ordering asserted by language | 1/1 (plus 1 correct refusal) |

**357 right, 0 wrong, 7 refused.** Undated events are refused rather than ordered, which is
the behaviour that matters: an agent that orders them anyway is inventing history.

**Verdict: it works, and I am not claiming much for it.** Ground truth is the recording
itself, so this is a mechanism test, not a hard benchmark. Its value is that "what did you do
before that" is now answerable from the graph at all.

---

## What I would cut

**The causal module, if forced** — not because it is wrong, but because its value is
unproven in the environments we actually run. It is the most code for the least demonstrated
benefit here: computerworld has no confounds, so co-occurrence is as good, and the one place
it clearly wins is a toy world I wrote. It becomes the first thing worth keeping the moment
an environment has something moving on its own — a real desktop, a network, another agent.

**What I would not cut:** expectation (highest value per line, and the change-encoding lesson
applies everywhere), probability (calibrated, and pooling beats the alternative by 35 points),
quantity (only for its refusals, and it should stay small until something supplies planning).

## What this says about the original question

Aligning parsed language with cognitive structure was **two separate problems**, and I had
the weights wrong. The projection was throwing away content the parse already had — that part
was cheap to fix. Supplying the structures themselves is also largely done. But the GSM8K
result says the binding constraint is not representation at all: with quantities, units and
recorded derivations in hand, 72% of problems still die at "which of these numbers combine".
That is planning, and it is the next thing to build or to learn.

## Architectural addendum, 2026-09-19

The quantity result does not establish that representation is sufficiently complete and only
planning remains. Choosing which numbers combine depends on collection identity, overlap,
ownership, rates, and change over time. Projection can also discard distinctions still
present in the parse. Arithmetic and a derivation trace are mechanisms; their input semantics
and their path from an ordinary agent request require separate evidence.

These are revised design requirements, not new measurements or claims that the proposed
behavior is implemented. See [34 — Cognitive primitives and implementation](34-cognitive-primitives-and-implementation.md)
and the [reassessment of the ten fronts](33-the-cognitive-fronts.md#architectural-reassessment-2026-09-19).
