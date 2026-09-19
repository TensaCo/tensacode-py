# 14. A cognitive architecture over the claim store

**Visual scope correction, 2026-09-19:** a common claim store must accommodate whole-scene
organization and alternative relational interpretations, not only facts about identified
objects. Visual and linguistic hypotheses should share references and evidence contracts while
retaining their original sources. Scene graphs should use extensible relational semantics;
a fixed UI taxonomy does not define the cognitive core. The historical mechanisms below do
not establish learned scene formation or automatic revision of global visual hypotheses.

> **Direction update, 2026-09-19:** the measured mechanisms below remain historical.
> [36 — The structured cognitive workspace](36-structured-cognitive-workspace.md) sets the
> current objective: interpretation, evidence, alternatives, model formation, planning,
> and realization must participate in one revisable agent loop. A claim store and optional
> cognitive modules do not establish that integration or a complete intermediate core.

*Measured on 2026-09-17. Numbers in `eval/results/cognitive_layer.json`, produced by
`$SP/measure_cognitive.py`. Tests: `tests/test_awareness_nucleation.py`,
`tests/test_memory_stores.py`, `tests/test_awareness_wants.py` (35 tests). Suite: 803 passed,
4 skipped, against a 752-passed baseline — everything here is additive.*

## Why

Four chat messages, all of which failed:

| message | what happened |
| --- | --- |
| "my name is Jacob. what is my name?" | two × "I don't know how to do that" |
| "what color is the display" | listed the home folder |
| "how many icons are in the sidebar" | listed the home folder, with counts |
| "open the app that is used for writing code" | listed the home folder |

Every one is a *question*, and the assistant maps language to procedures that act on a
machine. It has no question-answering path at all. Worse, two of them produced a confident
wrong action rather than an abstention.

And the answers to three of them were already in the claim graph. It perceives the dock
every cycle, with names, roles and positions. It knew, and had no way to say.

That is not a missing feature; it is a missing shape. This layer adds five mechanisms, all
optional, all over the existing `Store`:

| module | mechanism |
| --- | --- |
| `awareness.py` | a bounded, salience-weighted active set nucleated from the graph |
| `wants.py` | an open question as an object, with satisfiers ranked by worth |
| `memory.py` | working / episodic / semantic / spatial / procedural, with real forgetting |
| `frames.py` | the same claim indexed semantically, spatially, and by modality |
| `priming.py` | procedures warmed by what is aware, with synfire-style chains |

Nothing is imported into the `tensacode` namespace root: `from tensacode.awareness import …`.
An agent that does not ask for them pays nothing, which is why the baseline suite is
unchanged.

## API surface

```python
from tensorcode.awareness import Awareness, AwarenessPolicy, nucleate
a = nucleate(mind, seeds, AwarenessPolicy(budget=64), extra_links=frames.links)
a.aware(); a.salience(claim); a.why(claim); a.fade(); a.view()        # a Store-shaped view

from tensorcode.wants import Want, Wants, Satisfier, Answer, want_from
wants.add(Want("what is my name?", subject=user, predicate="name", requires="observed",
               satisfiers=(Satisfier("memory", "what I was told", cost=.1),)))
wants.look_up(want, mind, frames=frames)       # Answer | Unknown, never a guess
wants.next_to_pursue()                          # (want, satisfier) by value/cost

from tensorcode.memory import Memory, MemoryPolicy
m.told(user, "name", "Jacob"); m.encode(thought, summary=…); m.recall("recipes folder")
m.consolidate(); m.forget_stale(); m.here(window="Dock"); m.skills(cue, procedures)

from tensorcode.frames import Frames, Place, modality_of
f.semantic(subject, predicate); f.spatial(window=…, near=(x, y)); f.modality("pixels")
f.binding(claim); f.disagreements(); f.links                      # feeds awareness

from tensorcode.priming import Cue, Priming, primeable_from, prime
p.observe(awareness); p.ready(); p.partial(); p.why(id); p.reset(id)
```

`cognition.py` was **not changed**. Awareness reaches into `Store`'s existing indexes
(`_by_subject`, `_by_object`, `_dependents`) and `think` accepts `Awareness.view()` because
it only needs `claims` and `match`.

## Measured

### Nucleation, and a real tension

Claims in the store, one cycle, budget 64, three seeds (p50 of 7 runs):

| claims | graph links only | + spatial frame | no group cap |
| --- | --- | --- | --- |
| 200 | 0.41 ms / 33 aware | 0.97 ms / 33 | 0.42 ms / 33 |
| 1,000 | 1.52 ms / 64 | 13.04 ms / 64 | 1.53 ms / 64 |
| 2,000 (assistant scale) | 0.01 ms / **3** | **2.57 ms / 64** | 2.20 ms / 64 |
| 5,000 | 0.01 ms / **3** | 2.76 ms / 64 | 3.15 ms / 64 |
| 20,000 | 0.01 ms / **3** | 6.54 ms / 64 | 7.09 ms / 64 |

The middle column is the configuration to use, and the outer two are the tension I found
while measuring, reported rather than tuned away:

* A claim read in the same observation as thousands of others is a link that connects
  everything to everything. Following it makes nucleation cost grow with memory (right
  column); `max_group` skips such groups, which keeps cost flat but can leave awareness with
  nothing to spread along (left column: **3 aware** claims, which is useless).
* The spatial frame is what resolves it: within one screen, *adjacency* is the informative
  link, and it is bounded by a grid. With `extra_links=frames.links`, awareness stays full at
  every size, at 2.6–6.5 ms.
* So the modality/spatial frame is not decoration on top of nucleation — at realistic sizes
  nucleation does not work without it. That was not obvious to me before measuring.

The 13 ms at 1,000 claims is an artifact of the synthetic layout (controls packed into few
grid cells); it is not monotone in size and I have not chased it further.

### Rule firing, bounded by awareness

A rule whose pattern matches any `label` claim, fired over the aware view against the whole
store (the ~30× pathology this was meant to fix):

| claims | over awareness | over the whole store | speed-up |
| --- | --- | --- | --- |
| 1,000 | 0.02 ms (21 firings) | 0.23 ms (330) | 10× |
| 5,000 | 0.00 ms (1) | 2.40 ms (1,662) | ~800× |
| 20,000 | 0.00 ms (1) | 28.05 ms (6,666) | ~9,000× |

The bounded cost is flat because `AwareView` walks the aware set and filters, instead of
asking the store and filtering the answer. **That distinction is the entire mechanism**: my
first version delegated to `Store.claims` and measured 1.1–2.1×, because it still paid for
the scan. If a future change makes the view delegate again, the number to watch is the
flatness of the middle column, not the ratio.

### Memory

| operation | cost |
| --- | --- |
| encode an episode (5 claims) | 0.074 ms |
| associative recall over 60 episodes + 2,000 claims | 3.9 ms |
| consolidate (60 episodes) | 0.19 ms |
| forget stale | 4.7 ms |

Forgetting on a 2,542-claim mind, four hours on: **1,978 claims forgotten, 564 left**, with
22 kept because something was derived from them and 542 kept because they were told facts or
generalizations. Unbounded growth was a real defect (~150 claims per request, nothing
forgetting); this is the measurement that it is gone.

### Priming and footprint

60 candidate procedures against a 64-claim aware set: **0.071 ms** per cycle. (My first
version cost 143 ms, because a cue match recomputed the aware set; the aware set is now
resolved once per `observe`.)

| footprint | bytes |
| --- | --- |
| layer indexes + episodes, assistant-sized mind (2,000 claims) | 1.60 MB |
| layer per civ-sim-sized mind (60 claims) | 11.0 KB |
| 240 civ minds, one nucleation each | 70 ms total, **0.29 ms/mind** |

0.29 ms per mind per cycle sits inside the civ sim's measured 2.0–3.3 ms per mind per day,
so awareness is affordable there; 1.6 MB per rich agent is not (240 × 1.6 MB ≈ 380 MB), so a
rich agent should share one `Frames` per world rather than own one.

## Falsifiable predictions

Each names the measurement that would refute it. None is implemented in the assistant yet —
that is the sibling fork's work, and these are what to check when it lands.

1. **Awareness makes questions answerable without exact patterns.** With `Memory.told` and
   `Wants.look_up`, "my name is Jacob / what is my name?" answers correctly, and "how many
   icons are in the sidebar" answers from `frames.spatial(window="Dock")`. *Refuted if* the
   assistant still needs a hand-written rule per question shape; the test for it is that
   three of the four failures above pass with no new act in the grammar.
2. **Bounding rules by awareness removes the 30× penalty.** A civ-sim rule that scans all of
   memory should cost within 2× of a bounded one. *Refuted if* the ratio stays above 5×.
3. **Wants convert abstention into pursuit without inventing answers.** On the open-domain
   set (docs 12–13), routing a want to its cheapest satisfier should not lower accuracy at
   equal coverage. *Refuted if* accuracy at equal coverage drops, which would mean wants are
   guessing where `Unknown` used to refuse.
4. **Forgetting bounds a long conversation.** Live claim count should plateau rather than
   grow linearly in turns, with no increase in wrong answers. *Refuted if* a 50-turn
   conversation still grows linearly, or if answers degrade because something needed was
   forgotten (the report's `kept_because_depended_on` should be the guard).
5. **Priming subsumes dispatch and catches sequences dispatch cannot.** Every currently
   dispatched request still selects the same procedure (priming on the `act` cue reproduces
   lookup exactly), and a terminal read that needs "open → typed → prompt returned" is
   expressible as a chain. *Refuted if* any request selects a different procedure, or if
   thresholds need per-request tuning to reproduce today's behaviour.

## What I need from the assistant's procedures

`primeable_from` already works on them unchanged: a `Procedure` with `act="list"` becomes a
primeable with the single cue `(?, "act", "list")`, which reproduces `BY_ACT` dispatch
exactly, because `hear` already writes `(request, "act", …)` claims. To go beyond dispatch I
need two optional fields on `Procedure`:

```python
cues: list[dict]    # claim patterns, e.g. {"predicate": "is_a", "object": "directory"}
chain: list[list[dict]]   # ordered stages for a sequence
prime_threshold: float    # default 0.6
```

They are data, so they serialize with the procedure and stay inspectable. Nothing else is
required: preconditions stay where they are, and the interpreter keeps deciding what a step
does. I have not touched `assistant/`.

## If I had to cut one

**Priming.** It is the most speculative and the least load-bearing: dispatch works today,
and the chain has no measured task behind it. Awareness and frames are the pair that cannot
be cut — the measurements above show they only work together — and `Wants` plus
`Memory.told` are what actually fix the failures that motivated this. I would keep priming
as a design in this document until a task needs a sequence detector.

## Honest limits

* **Awareness needs the spatial frame at scale**, as measured. A mind with no placed claims
  and one huge observation gets 3 aware claims, and that configuration is silently useless
  rather than loudly broken. There is no warning for it yet.
* **Link weights are hand-set** (`LINK_WEIGHTS`), never learned or tuned against a task.
  They are a guess with a plausible shape, and the whole aware set depends on them.
* **Recall is word-trigram similarity** (`context.shingle_similarity`) over episode summaries
  and rendered claims. It finds "recipes folder" but will miss any paraphrase, which is
  exactly the brittleness doc 13 measured elsewhere.
* **Consolidation is repetition counting**, not generalization: it promotes a claim seen in
  ≥ *k* episodes. It cannot invent a predicate, and the induction machinery in
  `src/tensorcode/learning/` is where that belongs.
* **1.6 MB per assistant-sized mind** is the layer's own indexes. Sharing `Frames` per world
  is untested.
* **The `max_group` cut-off is an information-content judgement**, not a measured threshold.
  256 is a guess.
* **Nothing here has run in the assistant or the civ sim yet.** Every number above is a
  microbenchmark on synthetic minds shaped like theirs. The predictions are the contract.
