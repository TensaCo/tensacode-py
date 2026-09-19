# 24 — Social cognition and pragmatics

Four faculties, built the way the owner asked: examples in, cognitive structures and calibration
out. No gradient touched anything. Each section names the organ that was missing, the smallest
honest version of it, the prediction it was given, and what the measurement said — including
where the measurement said the set was too easy.

Library: `src/tensorcode/social.py`. Wiring: `examples/browser_agents/assistant/*`.
Cases: `eval/social/cases.py`. Harness: `eval/social/measure.py` (offline, no display, no server).
Numbers: `eval/results/social_measure.json`. Tests: `tests/test_social_ground.py` (structures),
`tests/test_social_faculties.py` (behaviour).

## What was actually missing

The assistant already stored what it was told, resolved names, ran guarded procedures and cited
provenance. What it lacked was any representation of the *other mind*:

| it had | it lacked |
| --- | --- |
| a fact in the store | whether that fact is **shared**, and how it became shared |
| a grammar over imperative forms | the **act** behind a non-imperative form |
| a question when a *name* was ambiguous | a question when the **goal** is underdetermined |
| an absence report | the **false belief** that produced the request |

All four are representational gaps, not accuracy gaps, which is why examples-in/structures-out is
the right method for them: there is nothing to fit, there is something to represent.

## 1. Common ground

**Structure.** `CommonGround` over the claim store. Ground is a claim about a claim
(`grounds` / `shared_via` / `mentioned` / `turn`), so `explain` reaches it. Three ways a thing
becomes shared — `I_SAID`, `YOU_SAID`, `BOTH_SAW` — and `status` prefers the strongest
(jointly seen > you said > I said), because that ordering is what a reply can lean on.

The load-bearing detail: `again()` asks **my own** ground specifically. That you told me your
name is no licence for me to answer "as I mentioned"; only my having said it is. The first
version consulted `status()`, took `YOU_SAID` as the answer and never marked a repeat. Fixed by
querying `mentions(about, I_SAID)`.

Where it attaches: grounding happens in the interpreter's `remember` step, not at parse time.
This was the second real bug — `hear()` grounded the frame before the procedure had written the
claim, so nothing was ever grounded (the measurement showed `grounded_from_you: 0` while the
"as I mentioned" marking still worked off the reply-side record). Common ground points at a
claim, and the claim does not exist until the step that records it runs.

**Prediction.** *The second mention is marked or omitted, and "what did I tell you" is answered
completely and in order. Falsified if tracking it produces no behavioural difference — then it is
bookkeeping.*

**Outcome — held, and two behavioural differences, not one.**

| | |
| --- | --- |
| first mention unmarked | ✅ "Your name is Jacob — you told me that (…)." |
| second mention marked | ✅ "**As I mentioned,** your name is Jacob — …" |
| "what did I tell you" complete | ✅ both facts |
| in **telling order** | ✅ name (turn 1) before colour (turn 2) |
| grounded from you | 2 claims, each with `shared_via: you_said` and a turn |

The ordering is the part that earns the structure. `told_facts` sorted by storage timestamp and
answered newest-first, which is not "in order" — it is the order of the database. It now reads the
turn off common ground (`_telling_order`), falling back to timestamps when there is no ground.
So: not bookkeeping. Two observable changes (the marking, the order) that nothing else in the
system could produce.

## 2. Indirect requests and implicature

**Structure.** Two shapes, one gate.

- A **wish** (`WISH`) whose embedded proposition is an order in disguise: the participle is
  de-inflected through `AS_ORDER` (`deleted` → `delete`) and the result is re-parsed by the
  ordinary grammar. "It would be good if you deleted notes.txt" becomes exactly the frame that
  "delete notes.txt" produces — no second grammar.
- An **implication** (`Implication`, `indirect_reading`): a complaint or question whose form
  implies an act ("I can't find X" → find X; "X is a mess" → clarify the goal for X).

The gate is **affordance**, not syntax: `in_domain(object)`. This is the falsifiable cognitive
claim of the section — *form cannot separate a request from a remark; what separates them is
whether the object is mine to act on*. "I can't find my invoice" and "I can't find my keys" are
the same sentence shape; only the second is about something I cannot touch.

The gate had to move. Built inside `indirect_frame`, it was never consulted for these remarks,
because an earlier tier got there first: the symbolic grammar (`_read_with_grammar`) already
turned "I can't find my keys" into a search over the home folder, and `hear` tries it before the
indirect reader. So the gate is now `act_is_affordable(text, frame)` in `language.py`, applied to
**any** tier's reading — the principle being that an act *inferred* rather than *ordered* has to
be about something I can act on. Two rules, both general:

- a reading must be **about what the utterance names**: "where did my report.pdf go" was read as
  `cd ~`, a frame that throws away the one thing the sentence is about. Refused.
- for a non-imperative with no nameable object, the reading's own objects must be in my domain —
  and a bare `~` does not count, since that is the parser's default place rather than evidence.

An imperative is never second-guessed: "delete keys" is an order about a file called keys, and
the gate has no business overruling an order (asserted in the tests).

**Prediction.** *An intention-level reading raises correct action on indirect requests without
raising wrong actions on the non-requests. Report both rates — a gain on one bought with a loss on
the other is a failure.*

**Outcome — held on both rates, and the second rate improved rather than degraded.** The before
column is the reading chain as it stood before this work (rules, then the grammar tier, ungated);
both columns go through `hear`, every tier, as a message from the user does.

| half | indirect requests read correctly | near-miss non-requests left alone |
| --- | --- | --- |
| design (n=8 / 5) | **1.00** (before 0.50) | **1.00** (before 0.40) |
| held-out (n=7 / 5) | **1.00** (before 0.57) | **1.00** (before 0.60) |

The prediction asked whether a gain on requests was bought with a loss on non-requests. It was
not — and the interesting part is the other direction: the old chain **acted on five of the ten
near misses**, searching the home folder for keys, a wallet, glasses, a car and a new job. The
gain on the second rate (0.40 → 1.00, 0.60 → 1.00) is the affordance gate removing a
pre-existing over-reading, not the new reader avoiding one it introduced.

**This number is only here because the harness was wrong first.** The first version of
`measure_indirect` asked `parse_message` — the rule grammar alone — which reports "unknown" for
"I can't find my keys" and made the non-request rate look like a flat 1.00 before and after. The
live assistant does not take that path. The harness now calls `hear`, and the bug it had been
hiding (three of five design-half remarks read as searches) is what forced the gate to move out
of `indirect_frame`. Measuring anything but the real entry point measures a path no user takes;
that is the methodological lesson of this section, and it cost the faculty's headline number its
first, flattering shape.

Three genuine defects surfaced here, all fixed at the level of the representation:

- `it'd be nice if …` never matched the wish pattern (the regex wanted a space where the
  apostrophe was). A whole indirect form was invisible for want of one character.
- The gate was in the wrong place (above): one tier out of three consulted it.
- **"my desktop is a mess" was being stored as a fact about the user.** The `my X is Y` rule filed
  it under topic `desktop`. The fix is the same affordance gate: an evaluative complement
  (`mess|state|disaster|nightmare|shambles|tip|pigsty`) about something *in my domain* is a
  complaint to act on, not a fact to keep. "My mood is a mess" is still filed; "my desktop is a
  mess" now reaches the clarifier. One rule, both directions, no special case.

## 3. Clarification as a strategy

**Structure.** `Reading` / `uncertainty` / `ask_or_act`. Readings of the goal carry probabilities
and a `cost_if_wrong`; asking costs `ask_cost` (a turn of the other mind's patience). Asking wins
only when the expected cost it avoids exceeds that cost, and a reading with p ≥ 0.8 is acted on
without asking. Entropy is reported in bits so the decision is checkable.

Wired as the `clarify_goal` procedure: resolve the place, look at what is there, price three
readings — by kind 0.45, by date 0.35, just list 0.20, the last cheap to get wrong (0.2) because
showing a folder harms nothing — then ask one question with the options numbered, and act on
the answer. The arithmetic for "organize my
desktop": 1.51 bits undetermined, expected cost of guessing 0.39, ask cost 0.25, so the question
is worth 0.14 — asked. Raise the ask cost past 0.39 and the same structure stops asking.

**Prediction.** *A majority become answerable after one question, with a low false-clarification
rate; state both.*

**Outcome — held.**

| half | answerable after one question | false clarification on clear requests |
| --- | --- | --- |
| design (n=3 / 6) | **1.00** | **0.00** |
| held-out (n=3 / 4) | **1.00** | **0.00** |

"Answerable" is strict: the reply must ask exactly one question *and* the answer must produce the
`mkdir … && mv …` work and a report of what moved. The clear set (list, read, delete, find, copy,
create-folder, ask-memory, ask-screen, open-app, info) is never questioned.

One option is deliberately unexecutable. "Group by month" is offered because it is a real reading
of the goal, and choosing it produces an explicit `admit(cannot)` — *"I can only read names from
the folder listing, not modification dates"* — rather than a silent substitution. Offering only
what I can do would misrepresent the ambiguity; offering it and then guessing would be worse.

## 4. The user's beliefs and visibility

**Structure.** `OtherMind.presupposes(slots, rules)` turns a request's slots into
presuppositions (a `target` presupposes existence, a `place` presupposes location); `check`
turns one that fails into a `FalseBelief`; `FalseBelief.correction()` words it as a repair.
`near_names` decides whether there is anything to repair with — normalized edit distance ≥ 0.6,
with a stem-containment bonus (0.75) so "recipe" reaches "recipes".

In the assistant, `resolve` now ends: exact candidates → home-wide exact `find` → listing of the
folder you meant → **prefix-widened `find -iname 'rep*'` across the home folder** → `near_names`
over both pools → correction, or a bare absence if genuinely nothing is near.

The prefix net was the measurement's doing. The first version looked for near misses only in the
folder the request named, and scored **0.00** on the design half: `reports.txt` is on the Desktop
while "delete the report.txt" assumes the home folder, so the near miss was one directory out of
reach. Two-stage retrieval fixed it — the shell proposes candidates by prefix, `near_names` scores
them. It is honest about its reach: a name whose *first* characters are wrong is still not caught.

**Prediction.** *Belief-aware replies correct the presupposition rather than reporting a bare
absence.*

**Outcome — held. Ablation: the same cases with `near_names` blinded (which reproduces the older
mind exactly — it still looks, still fails, still reports an absence).**

| half | corrected the belief | ablated |
| --- | --- | --- |
| design (n=3) | **1.00** | 0.00 |
| held-out (n=5) | **1.00** | 0.60 |

The held-out ablation is 0.60 rather than 0.00 because two of its rows do not need `near_names`:
one where nothing is near (a bare absence *is* the honest reply) and one where a folder is
missing. That second row found another false statement: **"organise my videos" answered "~/Videos
is already empty — nothing to organize"** about a folder that does not exist. An empty folder and a
missing folder are different states of the world, and the presupposition is about the second.
`clarify_goal` now stats the place first and says "There's no ~/Videos".

Nothing destructive runs on a corrected request (asserted, not assumed).

## Negatives, and what I would cut

**Report first: the authored floor is saturated.** Every after-number above is 1.00. That is not
evidence that the faculties are finished; it is evidence that a set written beside the system
inherits the system's blind spots and stops being informative once the system passes it. The
before/after ablations are doing the real work in this document — the absolute rates are a floor on
obvious breakage, not coverage. The next honest move is not more authored rows; it is the owner's
own transcript.

**Second: the harness was wrong twice, and both times in the flattering direction.** It measured
`parse_message` instead of `hear` (§2), and its fake machine had no Downloads, Pictures or
Projects folder, so three held-out clarification cases scored 0.00 for a reason that had nothing
to do with the faculty — the assistant correctly said the folder was empty. A fixture that does
not contain what the cases talk about measures the fixture. Both are fixed; I am recording them
because a measurement that only ever agreed with me would not be worth reporting.

**What did not pay:**

- `OtherMind.can_see` / visibility. Built, and nothing in the assistant needs it: the
  presupposition rules that matter are existence and location, both checkable against the file
  system. It is untested by any real case and is the one thing here I would cut today.
- `Clarification.options` as structure. The procedure re-formats them into a numbered list
  anyway, so the dataclass field earns nothing beyond what `goal_options` already returns.
- `FalseBelief.correction()` is not the string the user sees. The procedure composes the reply
  from the same parts (`near_joined`) because the reply must fit the procedure's own voice. Two
  ways of wording the same repair is one too many; the library version survives only because the
  tests use it to check the structure in isolation.
- One case expectation of mine was wrong, not the behaviour: I had "read pasta.txt" (right name,
  wrong place) expecting a "did you mean" correction. The assistant finds the file recursively and
  names the path it read, which *is* the belief repair. The row is reclassified (`located`) with
  the reason in `eval/social/cases.py` rather than quietly dropped.
- Transcript archaeology. I tried twice to read live transcripts off :8770 over SSE and the reads
  blocked; I stopped and used the owner's prompts from the brief as the acceptance rows instead.
  Those rows (`USER` provenance) are the only ones with independent provenance, and there are four
  of them.

**Which faculty I would cut if forced: none of the four, but only three are load-bearing.**
Common ground, implicature and presupposition each change a reply that no other part of the system
could change. Clarification is the weakest of the four *as a faculty*: its structure is a
three-row probability table written by hand, so what it really demonstrates is that asking can be
*priced* rather than reflexive. If the prices ever came from data instead of from my judgement it
would be the strongest of the four; until then it is a well-behaved placeholder around one correct
idea (ask when the goal is open, never when the name is merely unfamiliar).

## Provenance

- `USER` rows (4): the owner's own prompts, from the failing transcript in the brief and the
  limits recorded in docs/revival/15. Nothing is special-cased to pass them.
- `FLOOR` rows (~40): written here, split design / held-out, both halves reported everywhere.
- Two sibling-authored test expectations changed, with the reason recorded in the test itself:
  `tests/test_assistant_language.py` had "organize my desktop" and "clean up my desktop" as
  `unknown`. They were `unknown` because no clarification faculty existed — a benchmark that had
  inherited the system's pre-faculty limit as its expectation.

## Interaction with the memory fork (measured after the restart)

The memory work landed provenance-based protection in `MemoryPolicy.protect_sources`
(`utterance:` / `person:` / `user:`), which is right and fixes a real thing — a told fact's
predicate is whatever word the teller used, so a list of protected spellings can never contain it.

It protects one direction of common ground and not the other. My grounding writes `YOU_SAID` with
source `utterance:{turn}` (protected) and `I_SAID` with source `reply:{n}` (not), so under a
forced sweep:

```
again(about) -> True            # I have said it; a repeat would be marked
forget_stale() -> forgotten 81, kept_because_testimony 6
again(about) -> False           # the I_SAID ground is gone
told_me()    -> still 1         # the fact and its YOU_SAID ground survive
```

In a long conversation the assistant therefore keeps the fact and its provenance ("you told me
that") but silently stops saying "As I mentioned" when it repeats itself, roughly one half-life in.
The asymmetry comes from a source-prefix spelling rather than from a principle, which is the same
shape as the bug the provenance fix retired — mirrored onto my side of the conversation. Common
ground records utterance events on *both* sides, and neither side is perception, so neither belongs
on a perceptual clock.

I did not change `src/tensorcode/memory.py`: the memory fork owns it and measured its default, and a
unilateral edit from me would invalidate that measurement. I reported the repro to them instead.

**Resolved.** They reproduced it and extended `protect_sources` to
`("utterance:", "person:", "user:", "reply:", "said:")`, keeping my source spelling, with a comment
saying the list is a convention about spellings and not the principle. Re-measured on their
218-turn live conversation: recall 8/8 unchanged, latency ratio 1.13, claim footprint 3.31× against
3.22× — protecting both sides of the exchange costs about 3% more held claims. Verified from this
side: after a sweep that forgets 77 claims, `again()` is still true and the reply is still
"As I mentioned, your name is Jacob — you told me that". Guarded by
`test_the_mark_survives_forgetting_because_ground_is_not_perception`, which asserts the sweep
actually ran before asserting what survived it.

The residue is worth more than the fix, and it is their observation: `protect_sources` is a better
mechanism than `protect_predicates` and is still a list of spellings — it took a measurement from
another fork to notice that a whole side of the conversation was missing from it. The principle it
approximates is that forgetting is for *perception*, which the store could say directly (a snapshot
scope, or a perceiver's method) instead of enumerating what is not perception. Nobody attempted
that inversion, because it changes what every existing policy forgets and would invalidate every
number in both documents. It should be done deliberately, by someone who re-measures both.

## Architectural addendum, 2026-09-19

Common-ground claims and persistent references are useful but do not identify the task or
commitment that “instead,” “again,” or “never mind” changes. Conversation state needs pending
questions, alternatives, negotiated interpretations, and explicit corrections. Communicative
acts should record what question or purpose they address and what would count as an adequate
response; producing information alone does not establish that achievement.

These are revised design requirements, not new measurements or claims that the proposed
behavior is implemented. See [34 — Cognitive primitives and implementation](34-cognitive-primitives-and-implementation.md)
and the [reassessment of the ten fronts](33-the-cognitive-fronts.md#architectural-reassessment-2026-09-19).
