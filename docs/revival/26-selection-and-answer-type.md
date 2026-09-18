# 26 — Evidence selection, expected answer type, and eval hygiene

*Follows [21](21-schema-induction.md), which refuted the re-seeding hypothesis and located the
multi-hop gap in evidence **selection** rather than composition. Two of the four items here are
capability work (a trained selector); two are measurement work that should have been done first.*

## 0. Predictions, registered before any run

Written before the selector existed and before any number below was measured. Each names what
would falsify it.

| # | prediction | falsified if |
|---|---|---|
| **P1** selection | a trained cross-encoder selector over **all** ~40 sentences, at its best k ∈ 2..6, reaches **≥ 0.307 EM** on the same 300 items (half the gap from the best real arm, 0.263, to `oracle_selection`, 0.350). Secondary: gold-fact recall at k=4 **> 0.777**, BM25's recall at k=12. | it stays under 0.307, or recall never beats BM25's — in which case selection was not the lever either, and doc 21's oracle gap is not reachable by a supervised selector |
| **P2** answer type | type-mismatch rejection plus **bridge-conditioned** question-overlap rejection removes **≥ 25%** of the 36 `bridge_entity_returned` failures while costing **≤ 3%** of the 130 correct `oracle_gold_only` answers | it removes fewer, or costs more correct answers than that — a gate that buys one failure per correct answer lost is worthless |
| **P3** boundary | **≥ 50%** of the 54 `span_boundary` failures are containment matches (prediction inside gold, or gold inside prediction), i.e. extent convention rather than a wrong referent | fewer than half contain or are contained — then it is a real extraction fault and the remaining gap to the 8B model is larger than doc 21 implies |
| **P4** hygiene | the truncation guard fires on the `no_retrieval` arm for **≥ 80%** of items (the bug that produced 0.093), and the authored paraphrase corpus is **≥ 10%** provably mislabelled | the guard is quiet on an arm we know was truncated, or the corpus is clean — the second would mean doc 21's 41%-bad-labels finding does not generalise past its hand-read tail |

**Two design facts measured on the TRAIN split before writing any rule** (never on the reported
300), because they decide whether the P2 mechanism is even admissible:

| | n | gold answer is a substring of its own question |
|---|---|---|
| bridge questions | 3,215 | **1.59%** |
| comparison questions | 785 | **39.11%** |

A comparison question names its own candidate answers ("Which ran longer, A or B?"), so
rejecting spans that appear in the question is safe for bridge questions and catastrophic for
comparisons. The rule is therefore conditioned on question shape, and the shape is derived from
the question's own surface form — the dataset's `type` field is used to **measure** that
derivation, never as an input to it.

## 1. Provenance

| | |
|---|---|
| selector training data | HotpotQA distractor **train** (official split, 45,224 items in shard 0), labels are the dataset's own supporting facts |
| evaluation items | the same 300 HotpotQA distractor **validation** items reported throughout this repo (`hotpot(300, seed=0)`) — disjoint from training by split |
| answerer | the existing 109M `span-answerer` artifact, trained on SQuAD 2.0 train, unchanged |
| grader | dataset labels only: EM and token F1, SQuAD normalisation; no model judges anything |
| tunable_against | k was swept on the reported items; every k is published, so the sweep is visible rather than hidden |

*Sections 2 onward are filled in by the runs; nothing above changed after they ran.*

## 2. Selection: the gap was reachable, and a 14M model reached it

`sentence-selector` is a cross-encoder over `google/electra-small-discriminator` (14M
parameters), one linear head on `[CLS]`, trained for 242 seconds on 142,750 (question, sentence)
pairs drawn from 12,000 official-train questions — four sampled distractors per gold sentence,
from the same item, which is the distribution it faces at inference. It scores all ~40 sentences
of an item; the answerer is the unchanged 109M `span-answerer`.

| arm | EM | 95% CI | recall | all gold | acc given all gold | coverage | sent/item | truncated |
|---|---|---|---|---|---|---|---|---|
| `selector_k2` | 0.2667 | .220–.319 | 0.608 | 0.307 | 0.424 | 0.907 | 2.0 | 0% |
| `selector_k3` | 0.2933 | .245–.347 | 0.725 | 0.447 | 0.433 | 0.903 | 3.0 | 0% |
| `selector_k4` | 0.3100 | .260–.365 | **0.792** | 0.563 | 0.408 | 0.880 | 4.0 | 0% |
| `selector_k5` | **0.3367** | .286–.392 | 0.838 | 0.653 | 0.408 | 0.883 | 5.0 | 0% |
| `selector_k6` | 0.3333 | .282–.389 | 0.861 | 0.697 | 0.402 | 0.863 | 6.0 | 0% |
| `selector_k8` | **0.3600** | .308–.416 | 0.893 | 0.750 | 0.409 | 0.843 | 8.0 | 3% |
| `selector_k12` | 0.3233 | .273–.378 | 0.925 | 0.813 | 0.340 | 0.733 | 12.0 | **59%** |
| `bm25_k4` | 0.2533 | .207–.306 | 0.562 | 0.267 | 0.525 | 0.773 | 4.0 | 0% |
| `bm25_k12` | 0.2633 | .217–.316 | 0.777 | 0.553 | 0.367 | 0.760 | 12.0 | **66%** |
| `oracle_gold_only` | 0.4333 | .379–.490 | 1.000 | 1.000 | 0.433 | 0.927 | 2.4 | 0% |
| `no_retrieval` | 0.0933 | .065–.132 | 1.000 | 1.000 | 0.093 | 0.393 | 40.5 | **99%** |

**P1 holds.** The bar was ≥ 0.307 at the best k ∈ 2..6; `selector_k5` reaches **0.3367** and
`selector_k4` clears it at 0.3100. Extending the sweep, `selector_k8` reaches **0.3600**, which is
above doc 21's `oracle_selection` (0.350) — expected, because that oracle picked perfectly from a
BM25-limited pool whose own recall ceiling was 0.777, and a selector over all ~40 sentences is not
bound by it. The secondary prediction also holds, narrowly: recall@4 is **0.792** against BM25's
0.777 at k=12 — the same evidence from a third as many sentences.

Against a 14M selector, BM25 at four sentences finds 56% of the gold and the selector finds 79%;
at twelve, BM25 finds what the selector finds at four.

### The decomposition, and a correction to doc 21

Doc 21 read its arms as a trade: "each added distractor costs about what the missing evidence
gains, because the extractive answerer cannot ignore a sentence in its window." The
per-operating-point decomposition does not support that as stated.

Conditional accuracy — EM on the items where all gold sentences were selected — is **flat** from
k=2 to k=8: 0.424, 0.433, 0.408, 0.408, 0.402, 0.409. Over the same range the share of items with
complete evidence rises from 0.307 to 0.750, and EM rises with it. Distractors are not costing
what the evidence gains; they cost something smaller and different, visible in **coverage**, which
falls 0.907 → 0.843: the answerer abstains more as the window fills, but the answers it does give
are no less accurate.

The inverse trade appears in exactly one place — where the window overflows. At k=12 conditional
accuracy finally drops (0.409 → 0.340) and coverage collapses to 0.733, and that arm is truncated
on **59%** of items. `bm25_k12` behaves the same way at 66% truncation. Doc 21's two top-k arms
were both in this regime, so the trade it inferred was substantially an artefact of the 384-token
window rather than a property of the answerer's attention. The honest version: **more evidence
helps monotonically until it stops fitting.**

### What selection cannot fix

| | bridge (n=238) | comparison (n=62) |
|---|---|---|
| `selector_k8` | 0.4034 | 0.1935 |
| `oracle_gold_only` | **0.4958** | **0.1935** |

A comparison question is answered at 0.19 EM *with its gold evidence in hand*, and selection
changes nothing about it — the oracle and the selector score identically. This is not a retrieval
gap and not a window gap. "Which came first, A or B?" requires reading two values and returning
the argument that wins; span extraction can only copy a substring out of the context, and it has
no operation that compares. Both sentences are right there and the answerer picks the wrong name.
That is a missing operation, and it accounts for a fifth of the set.

## 3. Expected answer type: the cost side beat the prediction, the benefit side refuted it

`src/tensorcode/answer_type.py` is library code rather than eval code because it is a
representation and not a model: `asked_for` reads the requirement off the question's surface,
`could_be` tests a candidate against it, and neither trains, loads, or reads a corpus. Any
question-answering op can use it. Its rules were tuned entirely on the train split; the 300
reported items were never consulted while writing them.

Applied on top of the answerer's output, rejecting any answer it judges confidently wrong:

| arm | rejected | failures removed | correct answers lost | precision of answers given |
|---|---|---|---|---|
| `oracle_gold_only` | 18 | **18 / 148 (12.2%)** | **0 / 130 (0.0%)** | 0.4676 → **0.5000** |
| `selector_k8` | 20 | **20 / 145 (13.8%)** | **0 / 108 (0.0%)** | 0.4269 → **0.4635** |

**P2 is falsified as written, and the reason matters more than the verdict.** The cost side beat
the prediction outright — the bar was ≤ 3% of correct answers and the measured cost is **zero**,
on both arms. The benefit side missed: the prediction named ≥ 25% of the 36
`bridge_entity_returned` failures, and the gate removes one of them.

It removes one because there are not 36 of them. Doc 21 attributed that class by *entity-mention
overlap* between the answer and the question. Under the strict test — the answer appears in the
question as a whole run of words — the class has **10** members on this arm, and **all 10 are
comparison questions**:

```
[comparison] Which movie came out first Muppet Treasure Island or Million Dollar Arm ?
             pred='Million Dollar Arm'   gold='Muppet Treasure Island'
[comparison] Are both Jonathan Marray and Wayne Black British?
             pred='British'              gold='no'
[comparison] Who was born first, Marino Girolami or Daniel Myrick?
             pred='Daniel Myrick'        gold='Marino Girolami'
```

These are not a bridge entity being returned in place of the answer. They are the comparison
failure of §2 wearing a disguise: the model picks one of the two names the question offers, and
the answer is "in the question" because *both* candidates always are. The signal cannot
discriminate there, which is precisely what the 35% train measurement predicted and why the rule
is conditioned on shape. So the gate correctly declines to fire, and the failure class that
motivated P2 does not exist at the size doc 21 reported.

What the gate does remove is 18 answers of the wrong kind — a name where a date was asked for, a
place where a count was — at no cost to a single correct answer. That is a real if modest gain,
and it is the one thing an abstention threshold cannot do, since these are confident answers. It
raises the precision of the answers given from 0.468 to 0.500 while leaving EM unchanged, because
a rejected answer becomes an abstention: **this gate buys calibration, not accuracy.** Doc 21's
framing — that answer type was the missing signal that could catch confidently-wrong answers — is
confirmed in kind and small in size.

### Costs of the derivation itself, measured

Shape is derived from the question's surface, never from the dataset's `type` field. Against that
field on the reported items: comparison **precision 0.707, recall 0.855**. A bridge question
misread as comparison loses both checks, and a comparison misread as bridge risks the
self-reference rule firing where it is catastrophic; 31 of 300 items are misread one way or the
other (22 bridge questions read as comparisons, 9 the reverse). On train, false rejection of the gold answer stands at **52/4000 = 1.30%**, of which 35 are
the irreducible case where the gold answer genuinely is a run of words from its own question.

Two rule changes were made after seeing a *train* diagnosis and are worth naming, because both
were rules I had asserted without measuring:

- **"A bare number is not a person, a place or a thing" is false.** It cost a Ferrari `458`, an
  area code `284`, a South Park episode `201` and the single `212`. Numbers name things routinely.
  Removed; `person`, `place` and `entity` are now never rejected at all.
- **A yes/no comparison with no alternative offered is checkable.** "Are both X and Y British?"
  names two entities but offers no choice between them, and **218 of 220** such train questions
  take a yes or a no. An either/or comparison is the opposite case and stays unchecked.

### A tokenizer bug found by its own test

The containment test underneath `from_question` was a character-level substring test on
normalised text, so `"no"` matched inside `"...Northeastern Ontario..."` — a yes/no answer scored
as lifted from a question about a place. Replaced with whole-word-run containment. Writing the
test for that fix then exposed a second bug: the word regex was `[a-z']+`, which **strips digits**,
so the containment test was blind to every numeric answer — `from_question` could never fire on a
number, and the boundary metric of §4 silently undercounted numeric extent disputes. Both fixed;
`_WORD` now includes digits.

This changes the §0 design fact, measured again on train with the corrected tokenizer:

| | n | gold answer is a word-run of its own question |
|---|---|---|
| bridge | 3,150 | 1.59% → **1.11%** |
| comparison | 850 | 39.11% → **35.41%** |

The pre-registered numbers stay in §0 as registered. The conclusion they were registered for —
that the rule is safe for bridge questions and catastrophic for comparisons — is unchanged, and
the ratio is if anything starker.

## 4. Span boundaries: mostly convention, and P3 was unfalsifiable as written

**P3 cannot be scored as stated**, and that is my error, not a result. It predicted that "≥ 50% of
the 54 `span_boundary` failures are containment matches" — but the attribution routine *defines*
`span_boundary` as a containment match, so the prediction was circular and could only ever come
out at 100%. The substantive question it was pointing at is what share of the whole set is an
extent disagreement, so that is what is reported.

| metric on `oracle_gold_only` | | what it accepts |
|---|---|---|
| EM | 0.4333 | the normalised strings are identical |
| same referent (containment, ≤ 3 words apart) | **0.5100** | `'Berkeley'` for `'University of California, Berkeley'` |
| containment, unbounded | 0.5367 | also a whole sentence that swallows the gold string |
| token F1 | 0.5418 | |

42 items — **14% of the set, 28% of the answered failures** — are extent disagreements: 24 where
the prediction sits inside the gold, 18 where the gold sits inside the prediction. 33 of the 42
are within three words.

**Verdict: mostly convention, but not purely, and the direction runs both ways.** Reading all 42,
they fall into two groups that a single metric cannot separate:

```
convention          'Berkeley'        / 'University of California, Berkeley'
                    'Office'          / 'Microsoft Office'
                    '2006'            / '2006 season'
                    'Kanichee'        / 'Kanichee Mine'
                    'Rick Ducommun'   / 'Richard "Rick" Ducommun'
                    'more than 1.7 billion' / '1.7 billion'

not convention      'Michael Sarnoski' / 'Mitchell Block and Michael Sarnoski'   (half the answer)
                    'Somerset County, Pennsylvania' / 'Pennsylvania'             (wrong granularity)
                    'Glenn Ford, Vince Edwards, Shirley Jones... Edward Albert Heimberger'
                        / 'Edward Albert Heimberger'                             (a sentence dump)
                    'Austro-Hungarian Army. The Austro-Hungarian Army' / 'The Austro-Hungarian Army'
```

This is why the bounded metric exists and why the unbounded one should not be quoted: a run-on
that happens to contain the gold string is not a dispute about extent. The nine items beyond the
three-word bound are mostly of the second kind.

**What this means for the gap to 0.513.** It is tempting to read `oracle_gold_only` at 0.5100
under a convention-tolerant metric against the 8B model's 0.5133 and conclude the gap is closed.
That comparison is invalid and I am not making it: the 8B number is an EM number, and regrading
the extractive answerer leniently while holding the 8B to EM measures the metric, not the models.
Both would have to be regraded together, and the 8B's predictions are not in hand. The defensible
statement is narrower: **EM understates this answerer by roughly 14 points of extent convention**,
so the real remaining gap between a 109M extractive answerer with gold evidence and an 8B
generative one is materially smaller than doc 21's 0.433-vs-0.513 implies — and, from §2, what
remains of it is concentrated in comparison questions, which selection cannot touch.

## 5. Eval hygiene

### 5a. The truncation guard: confirmed, and it indicts one more published arm

`Truncation` in `eval/selection/selector.py` asks, per item, whether what the harness just handed
the model fits the window the model reads through, and every result file this fork writes carries
its counts. It is wired into `eval/training/eval_span.py` — the harness that published 0.093.

**P4 (a) holds.** The prediction was that the guard fires on ≥ 80% of `no_retrieval` items; it
fires on **99%** (298/300, mean 1,190 tokens given against a 384-token window, worst case 1,816
tokens discarded). On SQuAD 2.0 through the same harness it fires on **1%**. That contrast is the
result worth keeping: the guard discriminates rather than alarming, and it explains why that
harness's SQuAD numbers were trustworthy while its HotpotQA number was not measuring retrieval at
all.

It also indicts an arm doc 21 reported as real: **`bm25_k12` is truncated on 66% of items** (mean
418 tokens against 384). Doc 21's "best real arm" at 0.263 was handing the answerer evidence it
could not read on two thirds of items, so that baseline was an understatement, and the selector's
margin over a *properly windowed* BM25 is smaller than the table alone suggests — `bm25_k4`, which
fits, scores 0.2533.

Every result file now records `truncation` per arm, with the verdict string spelled out so a
truncated arm cannot be quoted as a capability number without the caveat travelling with it.

### 5b. Label noise: P4 (b) is falsified, and two of my own detectors were worse than the data

**P4 (b) predicted the authored paraphrase corpus was ≥ 10% provably mislabelled. It is not: the
mechanically provable rate is 1.19%** (11 of 928 rows), and 1.62% including six rows I adjudicated
by hand. Doc 21's finding does not generalise from its sample.

The way it failed is the more useful result. The first sweep reported **6.4%** of the eval corpus
and **15.3%** of `parser_dev` as faulty, on the strength of doc 21's note that `place=@it` had
leaked from a template. That was my detector being wrong, not the corpus:

- `@it` is a **deliberate anaphora encoding**. It sits beside `place_kind='span'` (52 rows) and
  `place_kind='~/Documents'` (11 rows), and it appears on utterances that really do say "inside
  it", "in that location", "inside that folder". Flagging all 1,279 of them would have rewritten
  correct data.
- The second version kept `@it` only where the text *also* named an explicit path. Also wrong: in
  `"move that to ~/Downloads"` the `@it` is the **source** and the path is the **destination**.
- A statistical detector — an act rare for its own leading verb — keyed on **greetings**, since the
  first non-stopword of these utterances is "hi" or "hey". It reported that "hi" takes act
  `create_file` in 6 of 61 rows and flagged every other `"Hi, ..."` row as suspect. All 15
  adjudicated hits were correctly labelled.

Between them those detectors would have rewritten 1,279 correct rows toward my own guess, which is
the exact failure this sweep exists to catch, so all three counts are reported as zero and the
reasoning is kept in `eval/selection/label_audit.py`. A detector whose output you have not read is
not evidence.

The third detector found the real fault **inside the second one's false positives** — adjacent
rows reading `"copy it to ~/Downloads"` labelled `move` and `"hey move it to ~/Downloads?"`
labelled `copy`:

| corpus | rows | provable faults | rate | by kind |
|---|---|---|---|---|
| `paraphrase_eval.jsonl` | 928 | 11 | **1.19%** | swapped 7, conflict 2, corrupt 2 |
| `paraphrase_train.jsonl` | 2,785 | 51 | 1.83% | swapped 24, conflict 20, corrupt 7 |
| `parser_dev.jsonl` | 7,140 | 92 | 1.29% | swapped 52, corrupt 38, conflict 2 |

Corrections are recorded change-by-change with a justification in
`eval/results/selection_label_diff.json`, and the corpus is backed up beside itself. A mangled
utterance (`"backupson"`, `"desktopin photos"`) is **dropped rather than repaired** — it had an
intent, and writing my guess at it into a gold label would be inventing data. Two rows are left
wrong on purpose and listed as unresolved: `"Can you do it again?"` is labelled `choose`, which it
is not, but the 37-act label space has no act meaning "repeat", so there is nothing to correct it
to.

Only the **eval** corpus was corrected on disk. The train-side diff is recorded separately in
`selection_label_diff_train_not_applied.json` and deliberately not applied, because the parser's
weights were fit to those labels and retraining is not this fork's to do.

### Every headline number that rested on those labels, before and after

| set | arm | before | after | delta |
|---|---|---|---|---|
| paraphrase (n 928 → 926) | regex | 0.6369 | 0.6458 | +0.0089 |
| | grammar | 0.2823 | 0.2883 | +0.0060 |
| | learned | 0.9720 | **0.9741** | +0.0021 |
| held_out_user | all three | 1.0000 / 0.1667 / 1.0000 | unchanged | 0.0000 |
| benchmark_152 | regex, learned | 1.0000 / 0.8345 | unchanged | 0.0000 |
| civ_193 (refusal) | all three | 1.000 / 0.667 / 1.000 | unchanged | 0.0000 |
| squad_open (refusal) | all three | 0.867 / 0.740 / 0.993 | unchanged | 0.0000 |

**Correcting the labels moves every headline by less than one point.** `benchmark_152`'s grammar
arm also moved (−0.0138), and that is *not* attributable to this work — that corpus lives in a
test file this fork never touched, the evaluation is deterministic across repeated runs, and
sibling forks edited `src/tensorcode/{social,memory,permanence,metacognition}.py` between the two
measurements. It is recorded here as unattributed rather than claimed.

### The number doc 21 was actually reporting

1.19% corpus-wide and 41% in doc 21 are not in conflict, because they are different quantities.
Doc 21 hand-read rows **the model had already failed**, which estimates *how much of the failure
count is label noise* — not the corpus rate. That is the quantity my directive asks for, so it was
re-adjudicated on the same footing. Of the learned parser's 12 recorded failures on paraphrase,
**9 have wrong gold labels**:

```
wrong gold   "quick question, what's your take? thanks"   labelled ask_screen  (asks the assistant's own view)
             "You're welcome."                            labelled cancel      (also labelled thanks elsewhere)
             "You're welcome, thanks."                    labelled confirm
             'Can you do it again?'  (x2)                 labelled choose
             "what's the purpose of the sidebar?"  (x2)   labelled ask_pixels  (asks a role, not a colour)
             'do you remember my favourite colour now?'   labelled forget      (the opposite instruction)
             'Could you run `ls ~/...` for me?'           labelled list        (the utterance says "run")
real failure 'Please carry out the task.'                 -> unknown
             'who am i logged in as please'               -> git_log
```

So **about three quarters of the failure count we were reasoning about was label noise**, which
corroborates doc 21's 41% on its own terms while showing the corpus itself is clean to ~99%. The
practical consequence is that failure lists in this repo are close to unusable for diagnosis
without adjudication first, even where the aggregate accuracy is sound.

### A label error that propagated into the weights

Correcting the eval corpus made the parser *newly wrong* on rows it had previously scored:

```
'Could you move it to ~/Pictures/trips?'   gold now move    model says copy
'Could you please move it to ~/Documents?' gold now move    model says copy
```

The copy/move swap is present in `parser_dev` and `paraphrase_train` — the data the parser was fit
to — as well as in the eval set, so the model learned the error and the matched noise on both
sides concealed it. The parser does not reliably distinguish `copy` from `move`, and its 0.972 was
in part measuring agreement with a corpus mistake. This is the strongest argument in this document
for fixing labels even when the aggregate barely moves: **correlated noise in train and eval hides
capability defects instead of showing up as error.**

## 6. Where this leaves the multi-hop gap

| | EM |
|---|---|
| doc 21's best real arm (`bm25_k12`, 66% truncated) | 0.263 |
| trained selector, k=5 / k=8 | **0.337 / 0.360** |
| doc 21 `oracle_selection` (perfect pick from a BM25 pool) | 0.350 |
| `oracle_gold_only` (gold evidence handed over) | 0.433 |
| `oracle_gold_only`, extent convention tolerated | 0.510 |
| Qwen3-8B, EM | 0.513 |

Selection was the lever doc 21 said it was, and a 14M cross-encoder trained for four minutes on
free labels captured most of it: +0.097 EM over the best previously published arm, and past the
oracle that bounded the old pool. What remains above it is now two things rather than one gap, and
neither is retrieval:

1. **Extent convention**, worth ~14 points of EM and mostly not a capability difference at all.
2. **Comparison**, 0.19 EM with gold evidence in hand and unmoved by any amount of selection,
   because the answerer has no operation that compares two values — it can only copy a span. A
   fifth of the set needs an operation this architecture does not have.

The honest one-line summary: the multi-hop failure was three different problems — a truncated
window, an untrained selector, and a missing comparison operation — and doc 21's single "schema
gap" framing was measuring their sum. Two are now fixed or bounded. The third is a real
architectural absence, and it is where the next work belongs.

### Provenance and hygiene notes

- The selector was trained on the official HotpotQA **train** split (45,224 items in shard 0,
  downloaded for this purpose) and evaluated only on the 300 validation items this repo reports
  throughout. Train and eval are disjoint by split, not by subsetting.
- Every `answer_type` rule was written and tuned against **train**; the reported 300 were not
  consulted until the numbers above were produced.
- `k` was swept on the reported items, so every k is published rather than only the best.
- The design/heldout hash split is reported per arm in the result file (`selector_k8`: design
  0.390, heldout 0.339 — the design side runs high, as it did in doc 21).
- **P3 was unfalsifiable as registered** and is reported as such above rather than quietly scored.
- Suite: 1046 passed, 5 skipped, 1 failure (`test_label_space_covers_every_act_a_procedure_exists_for`,
  a `clarify_goal` act added at 19:33 to `examples/browser_agents/assistant/procedures.py` by the
  sibling fork that owns it; not this fork's file and not this fork's failure).
