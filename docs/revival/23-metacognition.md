# 23. Metacognition and the self-model

Four faculties a system needs about *itself*. Each was built because a measurement said the
thing it replaces does not work, each carries a prediction registered before the run, and
two of the four predictions were **falsified**.

Method: `claude-powered (examples in) -> (cognitive structures and calibration out)`.
No gradient descent anywhere in this section. The structures are in
`src/tensorcode/metacognition.py` (351 lines), tested in
`tests/test_metacognition_faculties.py` (15 tests), measured by `eval/metacognition/*`.

| Faculty | Prediction | Outcome |
|---|---|---|
| 23.1 Self-model of competence | competence gating beats input-based gating by ≥5 pts pooled; <2 pts within SQuAD | **both falsified** — it *loses* by 1.7 pts pooled and 2.7 pts on SQuAD |
| 23.2 Decomposed confidence | converts refusals into questions that name the broken part | **met on the letter, failed on the substance** — 55/69 converted, but 44/50 name the *act*, not a slot |
| 23.3 Error monitoring and repair | repairs solve more than report-and-stop, with no identical retries | **held** — 1/5 → 3/5 solved, 0 identical repeats |
| 23.4 Agency attribution | self vs world attributed correctly, with controls | **held** — 4/4 including both controls |

---

## 23.1 A self-model of competence — falsified

**The structure.** `SelfModel` records outcomes against caller-supplied kinds ordered
specific-to-general (`"squad2/how_many"`, `"squad2"`, `"extractive_qa"`), credits every
level at once, and answers with the most specific kind that has enough history — backing
off otherwise, so a never-attempted kind inherits its family's record. `Competence.score()`
reports the **Wilson lower bound**, not the raw rate: ten successes is not certainty.
`worth_attempting` returns a three-state `Verdict`, where `unknown` (no history) is
deliberately not `fails`, because a caller that conflates them never tries anything new.

**Why it should have worked.** `docs/revival/13` measured two *input-based* abstention
signals and both failed: a BM25 threshold (AUC 0.508 on SQuAD) and a logistic capability
router on item features (0.548 SQuAD, **0.457 — below chance —** on HotpotQA). Both ask
"will I get *this item* right?". A competence prior asks the cheaper question "how do I do
at *this kind*?", which needs no per-item signal at all.

**The measurement** (`eval/metacognition/competence.py`, 1,200 items across SQuAD 2.0,
HotpotQA, GSM8K and ARC-Easy; public labels; competence fitted on a random half, every
number from the other half; the rule tier runs with abstention disabled so every item has
an outcome rather than baking the arm's own threshold into the data):

| Gate, held-out half, at 50% coverage | Selective accuracy | AUC |
|---|---|---|
| competence prior (per kind, from history) | 0.080 | 0.597 |
| BM25 input signal (the thing it was to replace) | **0.097** | **0.687** |
| oracle | 0.127 | — |
| answer everything | 0.063 | — |

Per benchmark, held-out half: SQuAD 0.120 vs 0.147 (competence loses), HotpotQA 0.107 vs
0.107 (tie), ARC-Easy 0.080 vs 0.147 (loses badly), GSM8K 0.0 vs 0.0 (the arm gets none
right, so nothing to select).

**Both predictions fail.** Pooled gain −0.017 where ≥ +0.05 was predicted; SQuAD gain
−0.027 where |gain| < 0.02 was predicted. A per-kind prior is a *coarse* signal: it gives
every item of a kind the same strength, so at fixed coverage it selects whole kinds, and a
per-item signal beats it. My framing — that the headroom lay in self-knowledge rather than
input features — is wrong on these benchmarks.

**Two findings worth more than the gate.**

1. **The `spread` diagnostic answers "would a prior help?" before any gate is built**, and
   it inverted my assumption: variation *within* a dataset exceeds variation *across*
   datasets (SQuAD 0.30, HotpotQA 0.20, across-benchmark 0.08). Kinds are not alike inside a
   dataset; they are alike *between* datasets.
2. On the largest real history we have — **305,330 items from 102,899 episodes graded by
   simulator state** — per-task competence is `inbox 0.944, recon 0.997, desktop 1.000,
   chart 1.000, shop 1.000, access 1.000`, spread **0.056**. The diagnostic predicts a prior
   buys almost nothing there, without running a gate at all. A competence prior needs
   competence to *vary*, and in the place we have the most history it does not.

**Honest caveat on the comparison.** For GSM8K and ARC-Easy the "input signal" is the arm's
own answer margin rather than a pure property of the question, and pooled AUC is inflated
because the pooled score partly encodes which benchmark an item came from. Both favour the
baseline, which makes the negative result safer, not weaker.

## 23.2 Decomposed confidence — the structure works, the signal does not

**The structure.** `Belief` carries one commitment (speech act / act / slot / value /
evidence) with its own `Score`; `Confidences.gate()` acts only when the *weakest*
commitment clears the bar, and otherwise repairs that commitment: a weak **slot** becomes a
question about that slot, while a weak **value** or unreadable **speech act** refuses,
because those are not questions a user can answer. An unreported confidence counts as 0.0,
never as certainty.

**The measurement** (`eval/metacognition/decomposed.py`; confidence from agreement between
two independently built readers — the regex tier and the unification grammar — never from
one reader's posterior, which `docs/revival/15` measured at +0.9 points for 2.8% refusals):

| Corpus | n | flat refusals | → questions | named the broken part |
|---|---|---|---|---|
| the user's own failing prompts (independent) | 6 | 5 | 5 | 2 |
| civ declarative statements (independent; all must refuse) | 7 | 7 | **0** | — |
| the 152-case set (authored beside the regex tier) | 152 | 57 | 50 | 33 |

**Two results, one good and one bad.** The good one: the civilization's statements convert
**zero** refusals into questions. A decomposed gate does not weaken refusal of things that
are not requests — the failure mode `docs/revival/15` §15.1 worked hard to fix.

The bad one, found by breaking the conversions down: **44 of 50 questions are about the
act, not a slot**, and they fire because the grammar tier *read nothing* (agreement 0.5)
on utterances the regex tier already read correctly. That is a coverage disparity being
misread as doubt. The prediction is met on its letter (35/55 named the broken part) and
fails on its substance: the mechanism is mostly firing for the wrong reason.

**The fix is the signal, not the structure.** Agreement is only a valid confidence source
between readers of *comparable coverage*. The right slot-level source is already in the
system: the resolve step knows when it found two things called `notes`. Given that signal,
the structure does the right thing —

```
genuine slot ambiguity (resolve found 2 candidates) -> ask about place
coverage disparity      (1 of 2 readers silent)     -> ask about list   # unnecessary
```

## 23.3 Error monitoring and repair — held

**The structure.** `Monitor` records attempts, names failures from what the environment
said (`no_effect`, `error_output`, `target_missing`, `stale_reference`, `timeout`,
`wrong_result`, `crashed`), and chooses a repair (`retry`, `retry_differently`,
`reperceive`, `ask`, `give_up`), escalating as the same failure recurs. The one rule it
will not break: **never propose repeating an action that already failed the same way** —
the measured pathology was an identical failing command re-issued three times.

**The measurement** (`eval/metacognition/repair_and_agency.py`, in computerworld, using the
engine's **own documented gaps** rather than injected faults: `stat -c`, `du`, `which` and
`2>/dev/null` are genuinely absent):

| | solved | identical repeats |
|---|---|---|
| baseline: run, read the error, report, stop (today's behaviour) | 1/5 | — |
| with the repair repertoire | **3/5** | **0** |

**What this does and does not show.** The Monitor supplies the control structure — naming
the failure, refusing to repeat it, choosing the kind of repair. It does **not** invent the
alternative action; the harness supplies that. So the honest claim is "a repertoire plus a
caller who has an alternative available solves 3/5 where report-and-stop solves 1/5", and
the open problem is generating the alternative.

**An accident worth recording.** The one task the baseline "solved" was `find -name`, whose
engine bug returns *every* file — so the expected string was present and a naive success
check passed. The environment's silently-wrong answer defeats an output-matching check,
which is the same false-success failure mode `docs/revival/07` §7.6.8 found in pixels.

## 23.4 Agency attribution — held

**The structure.** `attribute()` tags each changed aspect as `self` (my action predicted
it), `world` (it changed and my action did not predict it), `both` (my action touched it
and the result is not what it predicted) or nothing at all when the aspect did not change.
`surprising()` returns what I did not cause — which is the only sound basis for treating
surprise as informative, since otherwise every consequence of one's own action looks like
news.

**The measurement.** `docs/revival/17` concluded causal discrimination was untestable in
computerworld because nothing moves unless the agent moves it. That is fixable: `CwWorld`
has an *owner* shell the agent does not, so the harness can change the world between the
agent's actions. All four cases attribute correctly:

| Case | Attribution | Correct |
|---|---|---|
| agent acts, world frozen | `desktop_entries: self` | yes |
| agent acts **and** world moves in the same interval | `desktop_entries: self`, `home_entries: world` | yes |
| control: world moves, agent idle | `home_entries: world` | yes |
| control: agent acts, nothing changes | `{}` — no attribution at all | yes |

The second row is the one that matters: two changes in one interval, one mine and one not,
separated by whether my own action predicted it. The fourth is the cheap failure mode this
rules out — crediting yourself for a change you did not cause.

## What I would cut

**23.1, the competence gate.** It is falsified as a gate, and I would not ship
`worth_attempting` as an abstention mechanism on this evidence. I would keep two small
parts of it: `SelfModel.spread`, which cheaply answers whether *any* per-kind prior could
help (and correctly predicted the 0.056-spread case without a gate), and the Wilson lower
bound with `unknown ≠ fails`, which is the right shape for any competence claim we make
later.

Ranked by what the measurements support: **keep 23.4** (correct, cheap, and the
precondition for causal learning), **keep 23.3** (real gain, with the alternative-generation
problem named), **keep 23.2's structure but not its signal**, **cut 23.1's gate**.

## Files

- `src/tensorcode/metacognition.py` — `SelfModel`/`Competence`, `Belief`/`Confidences`/`Gate`,
  `Monitor`/`Repair`, `attribute`/`surprising`, `agreement`.
- `tests/test_metacognition_faculties.py` — 15 tests, including that an untried kind is
  `unknown` rather than refused, that a known-bad action is never proposed again, and that
  crediting yourself for a change you did not act on fails.
- `eval/metacognition/{competence,decomposed,repair_and_agency}.py`, results in
  `eval/results/metacognition_{competence,decomposed,repair_agency}.json`.

## Provenance

| Measurement | Environment | Grader | Held out |
|---|---|---|---|
| 23.1 competence | public datasets | dataset labels | fitted on one random half, reported on the other |
| 23.1 spread on real history | ours (self-authored apps) | simulator state | n/a — descriptive, 305,330 items |
| 23.2 decomposed | mixed, labelled per corpus | expected acts/slots from the test files | no fitting; thresholds fixed a priori |
| 23.3 repair | the user's computerworld engine | the engine's filesystem and terminal | failures are the engine's documented gaps, not injected |
| 23.4 agency | the user's computerworld engine | the engine's filesystem | two controls |

## Architectural addendum, 2026-09-19

A useful clarification needs explicit alternatives, a decision that depends on them, and an
answer the user can supply. Low confidence alone supplies none of these. Compare asking with
inspection, and test whether the answer actually updates the pending task. Likewise, an
operation trace is not yet an explanation: the response must connect a decision to the request,
assumptions, and observations that justified it.

These are revised design requirements, not new measurements or claims that the proposed
behavior is implemented. See [34 — Cognitive primitives and implementation](34-cognitive-primitives-and-implementation.md)
and the [reassessment of the ten fronts](33-the-cognitive-fronts.md#architectural-reassessment-2026-09-19).
