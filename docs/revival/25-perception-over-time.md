# 25 — Perception over time and memory dynamics

Four faculties, built the way the owner asked: examples in, cognitive structures and calibration
out. No gradient touched anything. Each section names the organ that was missing, the smallest
honest version of it, the prediction it was given, and what the measurement said — with the
negatives first where there are negatives, because in three of the four cases the negative is the
result.

Libraries: `src/tensorcode/change.py` (change detection), `src/tensorcode/permanence.py` (object
files), `src/tensorcode/cues.py` (cue recall); `src/tensorcode/memory.py` gained turn-level dynamics
and one policy change. Wiring: `examples/browser_agents/assistant/{agent,interpreter,procedure,procedures}.py`.
Harness: `eval/temporal_perception/live_harness.py` (headless Chromium, a Seed computer it creates
for itself, the assistant in-process). Measurements: `eval/temporal_perception/{change_live,
permanence_live,memory_live,paraphrase_recall}.py` → `eval/results/*.json`. Tests:
`tests/test_change_over_time.py`, `tests/test_object_permanence.py`, `tests/test_cue_recall.py`.

## What was actually missing

The assistant perceived the screen every cycle into a snapshot scope that **retracts what is no
longer perceived**. That is exactly right for beliefs about now and it makes four things
unrepresentable:

| it had | it lacked |
| --- | --- |
| the current frame | the **difference** between this frame and the one before it |
| "I can see it" | "it exists, I just can't see it" — and "it is gone" as a third thing |
| episodic/semantic machinery in unit tests | those dynamics running in a **live** conversation |
| n-gram overlap recall | a cue finding a memory **worded differently** |

## 1. Change detection

**Structure.** Per-cycle `Snapshot` of typed `Item`s, and a `diff` producing typed `Change`s
(`appeared`, `disappeared`, `moved`, `value_changed`, `focus_changed`, `window_opened`,
`window_closed`, `occluded`, `replaced`) with provenance and timestamps. A `Watcher` keeps the
snapshots so *any* earlier moment can be asked about; `since_first("turn:N")` is the boundary that
makes "since my last message" mean the whole turn rather than the last frame.

Noise is handled by **learning volatility rather than declaring it**: a key that changes in ≥50% of
the last N comparisons is volatile, and its changes are *flagged*, not dropped, so signal and noise
stay separately countable. Four structural interpretations turn perceptual differences into news:

- **part-whole subsumption** — a window opening is one change carrying `brought=N`, not N appearances;
- **spatial containment** — chrome the DOM does not attribute to a window belongs to the window it sits in;
- **occlusion** — a box ≥60% covered by a newly opened window is `occluded`, not `disappeared`;
- **consequence** — a box that shifted because its own text changed has not also *moved*.

**Prediction.** "What changed since my last message" becomes answerable in the live assistant, and
the risk is noise, so the false-change rate is reported too.

**Measurement** (`eval/results/change_live.json`; ground truth is what the grader itself caused —
it typed the commands and wrote the files through the simulator's API):

| | result |
| --- | --- |
| interventions detected | **2/2** (terminal text; a window opening) |
| controls clean (nothing changed → nothing reported) | **3/3** |
| steady changes the screen made on its own | **0** |
| lines beyond the one naming the intervention | **5**, all six traceable to the intervention |
| distinct facts behind those 6 lines | **3** — so 2.0 reports per fact |

The first version of this reported **33 separate appearances** for one window opening. Subsumption
and containment took the live extras from 45 to 6. The last honest defect is *doubling*: a window
opening also reflows the window list, and the top-bar label swap is reported as an appearance plus
a disappearance rather than one replacement, because the slot's new occupant is a moved item rather
than the appeared one. A fifth interpretation (list reflow) would take that trial from 6 lines to 3;
I did not build it.

**A negative worth keeping.** One trial deliberately expects *nothing*: a file written to the
Desktop through the simulator's API while the open Files window is showing `~`. The screen never
displayed it, so a hit would have been a lie. My first version of this trial asserted a change that
was never visible and scored 0 — the test was wrong, not the organ.

## 2. Object permanence — and the stale-belief bug it hands back

**Structure.** `ObjectFile`s in `permanence.py`, keyed by **their own id and never by a screen key**
(scene-graph keys carry an occurrence index, so a list losing a row renames every row after it, and
a registry keyed by them hands one object's file to the next occupant of its slot). Identity is by
name before key, where the name of a control is its label and the name of a *line of read-only text
is its content* — a terminal line **is** its text, and its key is only its position in a scrolling
region.

Three states, not two: `in_view`, `out_of_view`, `gone`. Only a window **seen to close** makes an
object stop existing; absence never does, because absence is what occlusion looks like. Every
attribute is dated, and `assertable()` refuses to state an attribute of something not in view.

**Prediction.** Identity survives occlusion — *and* stale-belief bugs do not come back.

**Measurement** (`eval/results/permanence_live.json`; the grader created six files, deleted three of
them through the simulator's API while their only on-screen trace had scrolled out of the terminal,
and clicked Close itself). Three readers, same object files, same questions:

| condition | naive (state the last reading) | dated (`assertable()` first) | fresh (trust a reading <60s old) |
| --- | --- | --- | --- |
| in view (n=6) | answers 6/6 | answers **6/6** | answers 6/6 |
| out of view, still true (n=3) | right by luck 3/3 | **holds back 3/3** | answers 3/3 |
| out of view, no longer true (n=3) | **states a falsehood 3/3** | wrong **0/3** | **wrong 3/3** |

So: permanence without dates re-introduces the bug retraction was protecting against at a **100%
rate on the queries where the world changed unobserved**; dating removes it completely; and the
compromise everyone reaches for — trust a recent reading — buys back all three useful answers *and*
all three falsehoods. It is exactly as wrong as the naive reader on the cases that matter.

The cost is explicit and is the reason to report this rather than claim a win: the dated reader says
"I saw it at 12:04; I cannot see it now" on **100%** of absent queries, including every one where it
would have been right.

Identity, same run: 1029 rematches across 8 observations (key-based identity would have lost those
objects), 26 objects marked *gone* by one seen window close, and 1 correctly un-goned on reopen —
seeing something beats having written it off.

**Two honest limitations.** Occlusion is not a perceptual event in this environment at all: the DOM
scene graph reports a covered window's contents, so a window stacked over another produces no
absence. What produces absence here is destruction (close) and content scrolling away, which is
what the measurement uses. And unlabelled window chrome has no name, so its identity is positional
and does not survive a reopen; only 1 of the 26 gone objects was re-matched when the window came
back.

## 3. Memory dynamics in a live conversation — the falsification

**Structure.** `Memory.turn()`: encode the turn as an episode, and consolidate and forget *on a
stride* (laying down an episode costs what just happened; the sweeps cost everything held, so doing
them every turn is what would make latency grow). The assistant calls it once per turn, and
**leaves perception out of the episode on purpose**: ~500 perceptual claims per turn would bury the
handful that record what happened.

**Prediction.** Claim count plateaus, latency stays flat, a fact told at turn 5 is answerable at
turn 200 — and the informative outcome is forgetting dropping something the user later needs.

**It did.** 218 turns, facts planted in the first turns and asked at the end:

| forgetting protects… | recall of planted facts | latency 2nd half / 1st | live claims late/early |
| --- | --- | --- | --- |
| a list of predicate spellings (`said`, `told`, `name`, …) | **1/8** | 0.99 | 3.05 |
| anything a person said (provenance) | **8/8** | 1.12 | 3.22 |

The mechanism: a told fact is filed under **whichever word the teller used** — "my cat is Mackerel"
becomes `person:user cat 'Mackerel'` — so a list of protected predicate *spellings* can never
contain it. Only `name` survived, because `name` happened to be on the list. Repetition did not
save the others: the fact told twice (`project`) was dropped like the rest, so salience is no
defence either. The repair is one line of policy in `MemoryPolicy`: **testimony is not perception,
and the half-life is a perceptual one.** Claims whose evidence comes from an utterance are held on
those grounds, whatever their predicate is called.

**The same bug, one level down, found by the social-cognition fork.** My first version protected
only what the *user* said (`utterance:`/`person:`/`user:`), which left the agent's own utterances on
the perceptual clock. Their repro: `CommonGround` records a fact becoming shared from both sides,
`YOU_SAID` with source `utterance:{turn}` and `I_SAID` with source `reply:{n}`; after one half-life
`again()` goes False while `told_me()` survives — an assistant that still knows your name and has
forgotten it already told you so. I reproduced it (4 claims forgotten, the I_SAID ground among
them), and it is my own argument turned against my default: the record of having said something is
an utterance too, and an asymmetry produced by a string prefix is not a principle. `protect_sources`
now covers either party's utterances. Re-measured: recall still **8/8**, latency ratio 1.13, and the
claim footprint over 218 turns grows 3.31× instead of 3.22× — about 3% more held claims for it.

**Latency: flat, as predicted.** Median per-turn 279ms→313ms across the conversation (ratio 1.12),
and the memory dynamics themselves never exceeded **4.9ms** per turn.

**Plateau: falsified, twice.** First by my own organ: `Objects.remember()` appended an existence
claim per object per turn, which reached **3652 claims in the objects scope after 10 turns** and
outgrew perception itself. Mutable attributes are now *superseded* rather than appended, and the
scope plateaus flat at 821 while the screen scope sits at 458. What still grows is episodic memory
and the conversation record, ~63 claims/turn, and the provenance repair *guarantees* it: nothing a
person said is ever dropped. There is no policy here that gives both a plateau and reliable recall.
If I had to choose I would keep recall and bound the conversation record by consolidating old turns
into summaries, which is what `consolidate()` is for and is not wired to do.

## 4. Cue recall surviving paraphrase, without vectors

**Structure.** `cues.py`: role-wise lemma overlap. Three things token overlap throws away —
**roles** (a cue word matching the predicate is worth more than one matching a substring of the
object), **morphology** (suffix stripping and compound joining, so "my time zone" reaches
`timezone`), and **the mind's own links** (if the store holds `bicycle same_as bike`, a cue saying
bicycle reaches a claim about the bike in one discounted hop — synonymy from what the agent was
told, not from a hand-written list or a trained model). A first-person cue also constrains the
*subject*: "my time zone" is a question about the asker, and the stop-word list was throwing that
away, which is how a column header reading "Time" came back as the answer.

No learned vectors, deliberately: a sibling measured embeddings losing to plain token overlap on
this repository's own recall tasks (0.005 vs 0.106 same-cycle, 0.741 vs 0.944 same-subject).

**Prediction.** Beats exact-pattern and token-overlap baselines on paraphrased cues.

**Measurement** (`eval/results/paraphrase_recall.json`; facts told to the live assistant so the
claims have the shape it gives them; 32 authored cue wordings, **half held out**, role weights
chosen looking only at the training half):

| recaller | held-out recall@1 | held-out MRR | train recall@1 |
| --- | --- | --- | --- |
| exact (predicate verbatim in the cue) | 0.188 | 0.203 | 0.562 |
| token overlap, bigrams (what `Memory.recall` does today) | **0.000** | 0.000 | 0.000 |
| token overlap, words | 0.062 | 0.091 | 0.000 |
| structure (`cues.py`) | **0.875** | **0.906** | 0.812 |

Held-out scores higher than train because the split is by cue position, not by difficulty, and the
training half happens to hold the hardest paraphrases ("where do I sit", "the make of my bicycle").
The bigram number is the one to be uncomfortable about in the other direction: `Memory.recall`
compares word *bigrams* against a three-word claim render, so it scores 0 on every cue here. Word
overlap — the baseline that beat embeddings elsewhere — gets 0.062. Token overlap is not the ceiling
on this task; it is barely a floor.

**The ceiling that remains** is synonymy the agent has never been told: "the make of my bicycle",
"where do I sit", "the zone I am in". The link path is not decorative — telling the store
`bicycle same_as bike` moves "the make of my bicycle" from unfound to rank 1 — but `desk same_as
office` does not rescue "where do I sit", because the cue word is *sit*, and no link in the store
relates sitting to an office. That gap is knowledge, not machinery.

**Weakness of this measurement, stated plainly:** the cue paraphrases were authored by the same
model that wrote the recaller. The held-out half limits tuning, not imagination.

## What I would cut

- **The `replaced` interpretation.** It is verified synthetically and has never once fired on the
  live path; the case it was built for (top-bar label swap) is a list reflow, which it cannot see.
- **`Memory.recall`'s bigram similarity.** It scores 0.0 on every paraphrase measured here. Either
  make it word-level or route it through `cues.py`.
- **`MemoryPolicy.protect_predicates`.** Protection by predicate spelling cannot work for told
  facts and gave a false sense of safety for as long as `name` happened to be on the list. Keep it
  for genuinely fixed vocabularies; testimony should be protected by provenance.
- **And `protect_sources` next, eventually.** It is a better mechanism and still a list of
  spellings: it took a second fork's measurement to notice that one side of a conversation was
  missing from it. The principle it approximates is that *forgetting is for perception*, which the
  store could say directly — a snapshot scope or a perceiver's method is identifiable — instead of
  enumerating everything that is not perception. I did not attempt that inversion: it would change
  what every existing policy forgets, and the numbers above would all need re-running.
- **My own `remember()` as first written.** A claim per object per turn is a leak with a scope name.

## Wiring changes to the live assistant

Each is surgical and each is listed so it can be reverted: a `watch_screen` perceiver (snapshots,
volatility, object files, and `objects.closed()` on a seen window close); `Host.changes_since` and a
`changes` mental step, reached by the `ask_screen` "changed" branch; `Memory` constructed in
`new_mind()` and `Memory.turn()` called at the end of `respond()` over the turn's non-perceptual
claims. The turn's claims are collected by wrapping `on_cycle`, so nothing else changed in
`run_mind`.
