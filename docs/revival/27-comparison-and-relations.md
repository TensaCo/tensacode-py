# 27. Comparison as an operation: relations over two values

Doc 26 measured a gap it could not close. An extractive answerer scores **0.1935 EM on HotpotQA
comparison questions with the gold evidence already in hand** — the same 0.1935 the trained
selector gets, and the same 0.1935 a perfect oracle gets — against **0.4958** on bridge
questions. Selection is not the gap. Retrieval is not the gap. The 384-token window is not the
gap. About a fifth of the dataset asks for something a span cannot be: the name of whichever of
two things came first, or a yes, or a no.

This doc builds the missing operation and measures it. Section 0 is written before any
measurement on the reported set.

## 0. Pre-registration (written before the run)

**Provenance.** Design decisions were made against the HotpotQA **train** shard (600 comparison
questions sampled with seed 0 from `hotpot_distractor_train_shard0.parquet`), which the selector
was also trained on and which is not reported below. Everything reported is
`hotpot(300, seed=0)` from the distractor **validation** split — the same 300 items as doc 26, so
the arms are comparable. The train measurement that informed the design is in §2; it is shown
because a bar has to come from somewhere, not as a result.

**The bar.** Comparison questions, gold evidence, EM: **0.1935 → ≥ 0.40**.

| # | prediction | falsified if |
|---|---|---|
| P1 | comparison EM with gold evidence ≥ **0.40** (from 0.1935) | below 0.40 |
| P2 | comparison EM through the trained selector (k=8) ≥ **0.35** (from 0.1935) | below 0.35 |
| P3 | bridge EM does not degrade: ≥ **0.4790** with gold evidence (from 0.4958), ≥ **0.3866** through the selector (from 0.4034) | more than 4 of 238 bridge items lost |
| P4 | the yes/no subset, on which the answerer scores a structural **0.0** (it can only copy spans, and "yes" is not in the passage), reaches ≥ **0.50** | below 0.50 |
| P5 | the whole 300 with gold evidence ≥ **0.47** (from 0.4333) | below 0.47 |
| P6 | between **30%** and **65%** of comparison questions are refused rather than answered, and every refusal names a reason from a closed set | outside that band, or an unnamed refusal |
| P7 | the operation answers *better than the answerer on the same items*: on comparison items where the relation resolves, relation EM exceeds the answerer's EM on those same items by ≥ **0.15** | below that |

**The transfer test.** The same machinery is pointed at GSM8K, where the current arithmetic
solver scores **0.0 accuracy at 0.0267 coverage** (284 of 300 abstentions, never forming a
candidate — `eval/results/structures_gsm8k.json`).

| # | prediction | falsified if |
|---|---|---|
| P8 | the relation operation does **not** move GSM8K materially: overall accuracy stays ≤ **0.05** | accuracy above 0.05 |
| P9 | it nonetheless answers the *difference* questions it recognises: coverage ≥ **0.05** with accuracy-on-covered ≥ **0.30** | either below |

P8 and P9 together are the discriminating pair. If P8 holds and P9 holds, then comparison in
prose and arithmetic in word problems are **two faculties that share a comparison step**, and doc
21's "two coincidences, not one" conclusion stands and is strengthened: the shared part is real
but small, and the rest of GSM8K needs composition, not comparison. If P8 is falsified — if this
operation moves GSM8K off zero on its own — then the faculty is one faculty and doc 21 was wrong.

**Metric hygiene.** EM is the headline everywhere, because that is what 0.1935 and 0.4958 are.
Token F1 and word-run containment are reported beside it and never in place of it: a relation
that returns "Cid Corman" against a gold of `Cid (Sidney) Corman` is right and EM scores it zero,
and doc 26 was explicit that a lenient metric must not be compared to an EM one. The comparison
subset is 62 items, so every rate carries a Wilson interval. Every arm records how often its
evidence overflowed the answerer's window, per the truncation discipline that doc 26 established.

## 1. The operation

`src/tensorcode/relation.py`. Four parts, each able to refuse:

| part | what it does | refuses with |
|---|---|---|
| `read(question, titles)` | the relation asked, the two things compared, the attribute, and what kind of answer is wanted | `not_a_comparison`, `relation_unsupported`, `candidates_unclear` |
| `values(comparison, evidence)` | one value per candidate, typed by what the relation needs | records which candidate it is missing |
| `apply(comparison, values)` | the relation over those values | `value_missing`, `incomparable`, `dimension_mismatch`, `tied`, `no_shared_value` |
| `resolve(question, evidence)` | the three in sequence | any of the above |

Eight relations in four families, which cover 66% of comparison questions on train:

```
earlier / later      → a date per candidate, ordered; returns the winning candidate's name
greater / less       → a quantity per candidate, compared; returns the winning candidate's name
same / different     → the asked attribute's value per candidate; returns yes or no
both                 → the question's predicate tested against each candidate; returns yes or no
shared               → the longest phrase said of both; returns that phrase
```

The remaining third are property selection — "Which is a comedy film, The Million Dollar Duck or
Don Quixote?" — which is not a relation over two values at all. It asks which of two things has a
property, and a span answerer is the right tool for it. Those refuse with `relation_unsupported`
and are handed back, which is why the refusal rate below is not a loss.

**Detection is surface rules, not the dataset's label and not the grammar.** My brief allowed
HotpotQA's `type` field for detecting comparison-vs-bridge and asked me to say what I used for
the relation and its two arguments. The unification grammar in `tensorcode.language` parses these
questions — coverage 0.91–0.92, one skipped token each — but its frames do not carry what is
needed:

```
"Who was born first, Pablo Trapero or Aleksander Ford?"
   → predicate='be', recipient=Entity(text='born first')
"Are both The New Pornographers and Kings of Leon American rock bands?"
   → predicate='located', Entity(text='both New Pornographers')
```

The parse succeeds and the meaning is wrong for this domain: the comparative is read as a
description and the coordination is lost. So the question side is `answer_type.shape` and
`answer_type.asked_for` (which *are* correct on all three: comparison, and person/yes_no/entity)
plus regular expressions for the relation cue and the coordination. Detection measured against
HotpotQA's `type` field on the reported 300: **precision 0.833, recall 0.726** — and see §5 for
why the precision figure understates it.

## 2. What the design was fitted to (train, not reported)

600 comparison questions from the train shard, gold evidence, relation operation alone with no
fallback. This is the measurement that set the bar, and it is on the split the selector was
trained on.

| family | n | answered | refused | EM given answered | EM over all |
|---|---|---|---|---|---|
| `which_of_order` | 92 | 80 | 12 | 0.812 | 0.707 |
| `yes_no` | 166 | 153 | 13 | 0.752 | 0.693 |
| `shared_attribute` | 85 | 80 | 5 | 0.475 | 0.447 |
| `which_of_magnitude` | 90 | 9 | 81 | 0.889 | 0.089 |
| not read at all | 167 | — | — | — | — |
| **all 600** | | | | | **0.377** |

`which_of_magnitude` is the honest failure of the set: the operation refuses 90% of it. Reading
the refusals one at a time shows why, and it is not a reader bug. "Which magazine has a larger
target audience?", "Who is more of an independent artist?", "Which genus has a larger native
habitat?" state no number anywhere in the gold evidence; "Who has been a member of more bands?"
requires counting entities named in prose, not reading a stated count. Of the 90, only a handful
turn on two stated, comparable magnitudes — and on those the operation is right 8 times in 9.

## 3. Results (HotpotQA distractor validation, 300 items, the same items as doc 26)

`eval/results/relation_hotpot.json`. `span_only` reproduces doc 26 exactly — 0.4333 / 0.4958 /
0.1935 with gold evidence and 0.3600 / 0.4034 / 0.1935 through the selector — which is the check
that this harness and doc 26's are measuring the same thing.

**Gold evidence (`oracle_gold_only`, 0% truncated):**

| subset | n | answerer alone (doc 26) | with the relation operation | 95% CI | F1 | containment |
|---|---|---|---|---|---|---|
| comparison | 62 | 0.1935 | **0.4677** | 0.349–0.590 | 0.538 | 0.500 |
| bridge | 238 | 0.4958 | 0.4916 | 0.429–0.555 | 0.608 | 0.647 |
| all | 300 | 0.4333 | **0.4867** | 0.431–0.543 | 0.594 | 0.617 |
| gold is yes/no | 16 | **0.0000** | **0.5000** | 0.280–0.720 | 0.500 | 0.500 |

**Through the trained selector (`selector_k8`, 3% truncated):**

| subset | n | answerer alone (doc 26) | with the relation operation | 95% CI |
|---|---|---|---|---|
| comparison | 62 | 0.1935 | **0.4355** | 0.319–0.559 |
| bridge | 238 | 0.4034 | 0.4034 | 0.343–0.467 |
| all | 300 | 0.3600 | **0.4100** | 0.356–0.467 |
| gold is yes/no | 16 | **0.0000** | **0.5000** | 0.280–0.720 |

**Attribution.** The relation resolves 36 of 62 comparison questions with gold evidence. On those
36 items it scores **0.611 EM against the answerer's 0.139 on the same 36 items** (F1 0.689 vs
0.234). That is the number that attributes the change to the operation rather than to a
reshuffling of which items get attempted.

**Per relation, gold evidence:** `earlier` 0.667 (n=15), `both` 0.615 (n=13), `later` 0.500
(n=2), `shared` 0.333 (n=12). No magnitude comparison in the 62 resolved to a stated pair.

**Pre-registration, judged:**

| # | bar | result | verdict |
|---|---|---|---|
| P1 | comparison, gold evidence ≥ 0.40 | 0.4677 | **met** |
| P2 | comparison, selector ≥ 0.35 | 0.4355 | **met** |
| P3 | bridge ≥ 0.4790 gold / ≥ 0.3866 selector | 0.4916 / 0.4034 | **met** (−1 item of 238; §5) |
| P4 | yes/no subset ≥ 0.50 | 0.5000 | **met**, exactly at the bar |
| P5 | all 300, gold evidence ≥ 0.47 | 0.4867 | **met** |
| P6 | 30–65% of comparisons refused, every refusal named | 41.9% refused, 6 named reasons | **met** |
| P7 | relation beats the answerer on the same items by ≥ 0.15 | +0.472 | **met** |

**Refusals, gold evidence, over all 300 items:** `relation_unsupported` 210,
`not_a_comparison` 26, `value_missing` 10, `candidates_unclear` 10, `no_shared_value` 1,
`tied` 1. Of the 26 comparison questions refused, 7 were answered correctly by the span answerer
it handed them back to, so the refusals cost nothing — this is a refusal *to the fallback*, not
an abstention. Incomparability specifically (`dimension_mismatch` from `quantity.compare`, and
`value_missing` for "no comparable pair") accounts for 10 of the 62; on train, where more
magnitude comparisons state numbers, it is 81 of 90 in that family.

## 4. Where `quantity` and `temporal` were insufficient

Both were reused rather than reimplemented, as instructed. Four places where they did not reach,
none of which required editing them:

1. **`temporal` needs `datetime`s and a `Store`; comparisons need years.** Almost every order
   comparison in HotpotQA turns on a bare year. `relate` operates on events told to a claim store,
   so `_order` builds a two-event `Store`, converts each year to `datetime(year, 1, 1, tz=utc)`,
   and calls `relate`. This is genuine reuse — the ordering is recorded as claims with evidence,
   and the refusal for an unrecorded time comes from `event_time` — but the conversion loses the
   fact that a year is an *interval*, not an instant. Two people born in 1908 and 1908 are
   reported `tied` when one may well be older. `temporal.Interval` could represent this; nothing
   in the module constructs an interval from a year.
2. **`semantics_bridge.quantities_in_text` reads digits only.** "a band with four members" yields
   no quantity at all. Prose spells small numbers out, so `relation.digitise` substitutes 29
   number words (including `duo`, `trio`, `quartet`) before the reader runs. Without it the
   magnitude family refuses on roughly half of the cases where a number is actually stated.
3. **The same reader labels bare numbers `item` and attaches whatever noun follows.** It returns
   `('1934 inch', 'inch')` for a birth year followed by a height, `('6 august', 'august')`, and
   `('19 th', 'th')` for "19th". Those are not quantities. `quantities_offered` therefore drops
   generic-`item`, dimensionless and year-valued mentions, and requires either a noun match with
   the question or that the two candidates offer exactly one shared dimension between them.
4. **`quantity.compare` refuses two counts of different nouns, correctly, and a comparison needs
   them.** A count of `people` and a count of `residents` have different dimensions, so they
   cannot be added — that is right, and `quantity` should keep refusing it. But "which has more
   people?" has already asserted that the two counts are of one thing. `_magnitude` allows a
   count-against-count comparison when both sides are single counts, and records the crossing in
   the derivation. Physical dimension mismatches (time against money, length against mass) still
   refuse through `quantity.compare`, with its own message: `cannot compare 117 minute and 28
   item: time vs item`.

`answer_type` needed one thing it does not have: `shape` does not recognise "in common" or "of the
same" as comparison cues, so `What profession do A and B have in common?` reads as a bridge
question. Rather than edit `answer_type` (it is doc 26's, and its `COMPARISON_CUES` are tuned to a
measurement I would be disturbing), `relation.read` treats a shared/same/different/both cue as a
comparison marker in its own right and defers to `shape` otherwise. **No file outside my
ownership was modified.**

## 5. Hand-adjudicating the failures

All 62 comparison items are in `adjudication_sample` in the results JSON. Reading the 14 items
where the relation answered and EM scored it zero:

- **4 are right answers that EM cannot score.** `tennis player` against a gold of `tennis`;
  `game` against `games`; `Ernst Messerschmid` against `Ernst Willi Messerschmid`;
  `Michael Tippett` against `Michael Kemp Tippett`. The operation picked the correct candidate and
  returned the name the question itself used. This is doc 26's boundary-convention problem
  arriving in a new place, and it is why containment is reported beside EM (0.532 vs 0.500 on the
  comparison subset after §6's fixes).
- **5 were false "no"s from two defects**, found here and fixed — see §6.
- **1 is a question whose dataset answer is a different type than the question asks.** "Which case
  was decided first, Selle v. Gibb or Reynolds v. Sims?" has the gold answer `1964`. The operation
  returned `Reynolds v. Sims`, which is what "which case" asks for. Not fixable without guessing.
- **4 are genuine misses**, all in `shared`: the shared-value reader looks only at copular
  complements, so "Margaret Wilson and Edna St. Vincent Millay have both been given what?"
  (`Pulitzer Prize`) finds only `American`, and "In what country are both Ugni and Stenomesson
  native plants?" (`Chile`) finds `genus`. A shared value that lives in a verb phrase or that
  needs a place-typed reader is out of reach.

**The bridge cost, item by item.** The operation fired on 6 of 238 bridge questions. Four of those
six are comparison questions that HotpotQA labelled `bridge` — "Who was born first, Yanka
Dyagileva or Alexander Bashlachev?", "British Airways and EasyJet are both based out of where?",
"What type of profession does X and Y have in common?", "Who wrote lyrics for both A and B?" — so
the detection precision of 0.833 in §1 is measured against labels that are themselves wrong here,
and the rules are right more often than that figure says. The **whole −1 EM** on bridge is one
item: the operation answered `Alexander Bashlachev` where the answerer had returned
`Alexander Nickolaevich Bashlachev`, which is the same person and the gold string. One item, a
name-form difference, on a question the dataset mislabelled. The other five were wrong before and
after.

## 6. Two defects found by that adjudication, and a post-hoc number

The five false "no"s had one cause each, and both are defects rather than tuning knobs:

1. The stemmer mapped `games` → `gam` but `game` → `game`, so a question asking whether two things
   are board **games** could never match evidence saying either is a board **game**. Same for
   `magazines`/`magazine`. A stemmer that is not self-consistent makes equal words unequal.
2. The `both` test read a candidate's sentences but not its own document **title**.
   "Chrysalis: A Magazine of Women's Culture" is a magazine, and the sentences under that title
   need not say so.

Both were fixed and **verified on train first** (yes/no family EM given answered 0.725 → 0.752,
all-600 EM 0.370 → 0.377), then the validation set was re-run once:

| subset | pre-registered run | after the two defect fixes |
|---|---|---|
| comparison, gold evidence | 0.4677 | **0.5000** |
| comparison, selector k8 | 0.4355 | **0.4677** |
| gold is yes/no (n=16) | 0.5000 | **0.6250** |
| all 300, gold evidence | 0.4867 | **0.4933** |
| bridge, gold evidence | 0.4916 | 0.4916 (unchanged) |
| relation vs answerer on the 36 items it resolves | 0.611 vs 0.139 | **0.667** vs 0.139 |

**The pre-registered numbers are the result.** The second column is post-hoc: the defects were
found by reading validation failures, so that column has seen the reported set and is inflated by
an unknown amount. It is here because the fixes are in the source and the source must reproduce
something — `eval/results/relation_hotpot.json` is the pre-registered run and carries a note
saying it does not reproduce from current source;
`eval/results/relation_hotpot_after_defect_fixes.json` is the post-hoc one and does.

## 7. The transfer test: one faculty or two?

`eval/results/relation_gsm8k.json`. The operation was pointed at GSM8K unchanged — no
GSM8K-specific solver, no chaining, no unstated constants. `difference_in` reads two quantities of
one kind out of the problem and returns the gap, which is `_magnitude` with a subtraction in place
of a winner.

| | baseline (doc 21 structures) | relation operation |
|---|---|---|
| coverage | 0.0267 | 0.0433 |
| accuracy over 300 | 0.0 | **0.0** (95% CI 0.000–0.013) |
| accuracy when answered | 0.0 (8 wrong) | **0.0** (13 wrong) |

- **P8 holds.** GSM8K does not move. 0 of 300.
- **P9 is falsified.** I predicted the operation would at least get the difference questions it
  recognises right, at ≥ 0.30. It got **0 of 13**.

The 13 are worth reading, because they say exactly where the faculty boundary is:

```
"A loaf of bread costs $2. Bagels cost $1 each. How much more do 3 loaves cost than 2 bagels?"
    gold 4    got 1     ['2 dollar and 1 dollar', 'difference 1']
"Kimberly bought 8 packages of cat food and 6 packages of dog food. Each package of cat food
 contained 11 tins, and each of dog food 6. How many more tins of cat food than dog food?"
    gold 52   got 2     ['8 package and 6 package', 'difference 2']
"Juice Box A is 4 dollars. B is 5 dollars more than A. C is 7 more than A. How much more is C than B?"
    gold 2    got 1     ['4 dollar and 5 dollar', 'difference 1']
```

Hand-adjudicated: **13 of 13 require a composition before the comparison.** 3 loaves × $2 against
2 bagels × $1; 8 × 11 tins against 6 × 6; (4+7) against (4+5). The two values being compared are
never among the numbers the problem states. A better pair-picker would not change this, and
`difference_in`'s crude choice of the first two mentions — a real weakness — is not what makes
these wrong.

**Verdict: two faculties, and doc 21's "two coincidences, not one" stands strengthened, with a
sharper boundary than it drew.** The comparison *step* transfers perfectly well: the question-side
reader recognises 25 of 300 GSM8K questions as two-candidate comparisons and reads the relation
off 13 of them correctly. What does not transfer is the assumption underneath the values slot.
The faculty this doc built is *read two stated values out of evidence and relate them*; HotpotQA
prose states its values and GSM8K computes them. The missing piece is composition, which doc 21
already named as GSM8K's 217-of-300 stage and which no amount of comparison machinery supplies.

That is a more useful boundary than "two faculties" alone, because it is constructive: if the
values slot could be filled by a *derived* quantity — `quantity.derive` already records
derivations, and `quantity.OPS` already has the arithmetic — the relation step above it would be
reusable as it stands. The faculty boundary is value production, not relation.

## 8. What this does not do

- **Magnitude comparisons are 90% refused** and that is the state of the art here, not a bug to
  fix. Most of them do not state two comparable numbers anywhere in the gold evidence. Several are
  not computable at all ("who is more of an independent artist?").
- **Shared values in verb phrases are out of reach** (`Pulitzer Prize`, `Chile`); the reader looks
  at copular complements and at words beside the attribute's own name.
- **Year-only ordering treats a year as an instant**, so same-year comparisons report `tied`
  instead of ordering by month when the month is stated.
- **Nationality comparison uses a 120-entry demonym lexicon.** It is declared in the source as a
  lexicon, not a model, because "were A and B from the same country?" is answered by comparing two
  nationality words and sentence overlap cannot substitute for knowing which words those are.
- **Detection is regular expressions.** The grammar could not supply the relation or its
  arguments (§1). A small tagger trained on the train split's comparison questions is the obvious
  next step, and would be measured the same way.
- **The 62-item comparison subset is small.** Every interval above is wide: the headline 0.1935 →
  0.4677 is 12 items to 29 items, CI 0.349–0.590. It clears the bar and it is not precise.

## 9. Files

| path | what |
|---|---|
| `src/tensorcode/relation.py` | the operation: `read`, `values`, `apply`, `resolve`, `difference_in` |
| `eval/relations/eval_relations.py` | the HotpotQA arms, span vs relation vs combined |
| `eval/relations/eval_gsm8k_transfer.py` | the transfer test |
| `eval/results/relation_hotpot.json` | **the pre-registered run** (does not reproduce from current source; see §6) |
| `eval/results/relation_hotpot_after_defect_fixes.json` | the post-hoc run, reproduces from current source |
| `eval/results/relation_gsm8k.json` | the transfer test's result and all 13 adjudicated items |
| `tests/test_relation_reading.py` | 12 cases, each a regression the surface rules made on train |
| `tests/test_relation_operations.py` | 17 cases, half of them refusals |
