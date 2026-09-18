# 21 — Schema induction from clustered failures

*Method: instead of training on the failures, read them, name the distinction each one needs, implement
the cheapest honest version of it, and measure whether it pays. Examples in, cognitive structures out.*

Everything below is measured. Where a number is inherited from an earlier run it is labelled with its
file. Where a change did not pay, the negative is reported in place rather than at the end, because on
this task the negatives are the result.

**The two headline findings are both negative, and both are load-bearing:**

1. **The language evals contain almost no schema gaps.** Of 961 failing items, **51** defeat all three
   tiers, and **21 of those 51 have wrong gold labels**. The genuine population of "no representation
   exists for this" is ~30 items, 3.1% of the failures. The large clusters are phrasing coverage, which
   the learned tier already largely solves. See [§2](#2-clusters).
2. **The multi-hop gap is not a composition deficit.** Re-seeding retrieval with an intermediate result
   — the specific mechanism proposed — buys **+0.007 EM** over a control that re-seeds with the original
   question. The gap decomposes into a truncation bug (+0.16, already fixed by adding retrieval at all),
   evidence selection (+0.087), retrieval (+0.083), and answer-extent convention. See [§4](#4-the-multi-hop-experiment).

---

## 1. The corpus

`eval/schema/corpus.py` harvests every language eval into one row per (input, tier), recording what each
tier did, what was wanted, whether it was right, and the failure kind.

| | |
|---|---|
| rows | 3,905 |
| tiers | regex (`examples/browser_agents/assistant/language.parse_message`), grammar (`tensorcode.language.domains.desktop.read_request`), learned (`NeuralRequestParser`, 13.5M params) |
| failing items | 961 |
| kinds | `wrong_action` > `wrong_answer` > `missed_answer` > `miscalibration`, in severity order |
| provenance | recorded per row as inputs-author / gold-author / tunable_against, per `11-evidence-audit.md` |

### 1.1 A measurement error of my own, found and fixed

The first corpus graded every tier against the learned parser's **internal closed-head vocabulary**. It
injected `place_kind`, `info_topic` and similar as gold slot keys, so the hand-written tiers were being
marked wrong for not emitting names they are never asked to emit. A key histogram showed `place_kind`
appearing 157 times in gold and **zero** times in any tier's output.

Fixed by decoding gold exactly as `NeuralRequestParser` decodes it. Failures fell from 1,821 to 961:
**roughly 250 of the original "failures" were fabricated by the harness.** This is class (d), our own
wiring, and it is reported here rather than quietly corrected because it is the largest single
correction in this document.

## 2. Clusters

Clustered by *the distinction the system cannot make*, not by task or surface form. The unit is an
**item**, and the verdict is a property of items: a cluster can have every tier among its failures while
containing no single item that defeats all three.

| cluster | items | weighted | wrong-action | defeat all 3 | most common failing set |
|---|---|---|---|---|---|
| phrasing_coverage | 463 | 463 | 9 | 26 | regex+grammar (196) |
| act_confusion | 202 | 202 | **25** | 8 | grammar (99) |
| self_and_memory | 86 | 86 | 1 | 8 | regex+grammar (46) |
| perception_query | 55 | 55 | 1 | 6 | grammar (28) |
| domain_membership | 53 | 53 | 2 | 0 | grammar (32) |
| reference_kind | 49 | 49 | 2 | 2 | regex+grammar (40) |
| speech_act | 17 | 17 | 0 | 0 | grammar only (17) |
| content_span_boundary | 16 | 16 | 3 | 1 | regex+grammar (14) |
| capability_calibration | 6 | 81 | 0 | 0 | — |
| functional_description | 2 | 2 | 0 | 0 | grammar (1) |
| multi_value | 1 | 1 | 0 | 0 | learned (1) |
| *quantity_composition* | 2 | 300 | 0 | — | attributed from recorded stages |
| *knowledge_gap* | 1 | 300 | 0 | — | attributed from recorded stages |
| *evidence_to_answer* | 4 | 228 | 0 | — | attributed from recorded stages |
| *retrieval_semantics* | 2 | 134 | 0 | — | attributed from recorded stages |
| *our_wiring* | 1 | 14 | 0 | — | attributed from recorded stages |
| *answer_type_coverage* | 1 | 5 | 0 | — | attributed from recorded stages |

Italic rows are open-domain stage attributions from `schema_brittleness.json`, weighted by the item
counts recorded there; no tier was run on them here, so they carry no failing-set evidence.

**What the "defeat all 3" column means.** Writing more hand-written rules is a treadmill the learned
tier exists to end, and this column says how much of each cluster is actually past that treadmill.
For `phrasing_coverage`, 196 of 463 items are failed by both hand-written tiers and **parsed correctly
by the learned tier** — a data problem that is already solved. Only 26 are beyond all three.

## 3. Adjudication: 41% of the hard tail is bad labels

The paraphrase set was produced by a local model rewriting templates with the template's label carried
over. On easy items that is fine; on the hard tail the rewrite often no longer means what the template
meant. Grading a tier against a wrong label manufactures a schema gap that does not exist, so I read all
51 all-tier-fail items and recorded a verdict for each in `eval/schema/adjudicate.py`.

| verdict | n |
|---|---|
| gold_ok — a real gap | 21 |
| **gold_wrong — the label does not describe the sentence** | **21** |
| ambiguous | 9 |

The wrong labels have recognisable causes, not random noise: acts swapped (`copy` labelled `move`),
`place=@it` leaked from the template's context into sentences that name no place, corrupted paraphrases
(`"desktopin photos"`, `"backupson"`), and plain nonsense (`"You're welcome, thanks."` labelled
`confirm`). All 21 are excluded from the measurement below and counted as class (d).

## 4. Schema changes, implemented and measured

Six distinctions, in `eval/schema/repair.py`, written as a **projection over any tier's output** rather
than inside a parser — so one change is measurable against all three tiers at once and can be deleted
cheaply. Every rule keys on linguistic structure (a determiner, a possessive, a negation, a place word,
a screen noun), never on a specific input; a rule that only helped the sentences that motivated it would
be a lookup table, not a schema change.

| | distinction |
|---|---|
| D1 `referent_ontology` | a screen object is not a filesystem object |
| D2 `typed_place` | a place word denotes a path, not itself |
| D3 `span_introducer` | a content span does not include the words that introduce it |
| D4 `memory_polarity` | a negated memory instruction is a retraction |
| D5 `app_identity` | an app named by function or possession is still that app |
| D6 `needle_head` | a search term is the term, not the noun phrase around it |

**Split.** `design` = the 30 all-tier-fail items with usable labels. I read every one of them, so they
are **not held out**. `heldout` = every other failing item *plus every item that already passed*, none
inspected individually. The second group is the one that matters: a change that helps the design set and
breaks passing items shows up here as a loss.

### 4.1 Results, held-out half (`eval/results/schema_repair.json`)

Strict scoring, fixed/broke:

| rule | regex | grammar | learned |
|---|---|---|---|
| D1 referent_ontology | **3 / 0** | **5 / 0** | 0 / 0 |
| D2 typed_place | 2 / 0 | **7 / 0** | 8 / 2 |
| D3 span_introducer | 3 / 1 | 0 / 1 | 0 / 1 |
| D4 memory_polarity | 3 / 0 | 1 / 0 | 0 / **4** |
| D5 app_identity | 0 / 0 | 0 / 0 | 1 / **3** |
| D6 needle_head | 0 / 0 | 0 / 0 | 0 / 0 |
| all together | 11 / 1 | 13 / 1 | 9 / **10** |

### 4.2 The corpus does not agree with itself, and that caps three of the six

D2, D4 and D5 all break on the same thing — gold that uses two conventions for one value:

```
want delete({'target': 'downloads'})        vs  want list({'place': '~/Music'})
want tell({'topic': 'phone number'})        vs  want ask_memory({'topic': 'phone'})
want open_app({'app': 'the system monitor'}) vs  want open_app({'app': 'terminal'})
```

Under strict comparison a canonicalising rule is therefore penalised for the corpus's inconsistency
rather than for being wrong. `measure.py --lenient` collapses exactly those three conventions **on both
sides** and reports separately (`schema_repair_lenient.json`). Neither score is the headline alone:

| rule | learned, strict | learned, lenient |
|---|---|---|
| D2 typed_place | 8 fixed / 2 broke | 0 / 0 |
| D4 memory_polarity | 0 / **4** | 0 / **0** |
| D5 app_identity | 1 / **3** | 0 / **0** |
| all together | 9 / **10** | 0 / **1** |

So the rules are not wrong. They were being scored against spelling. But note what the lenient column
also says: **0 held-out items fixed on the learned tier.** These distinctions are real and they are rare.

### 4.3 Verdicts

| rule | verdict |
|---|---|
| D1 referent_ontology | **KEEP.** The only clean win: +3 regex, +5 grammar held out, zero breakage under either scoring. |
| D2 typed_place | **KEEP for the hand-written tiers** (+7 grammar). Neutral on the learned tier once spelling is not being scored. |
| D6 needle_head | **KEEP, unmeasurable.** Correct by construction, moves nothing held out. |
| D3 span_introducer | **DISCARD.** 3 fixed / 3 broke across tiers; no honest reading makes it a win. |
| D4 memory_polarity | **DISCARD as a projection.** Its whole strict-scoring loss was spelling, but with that removed it fixes 4 design items and 0 held-out items. Not worth a rule. |
| D5 app_identity | **DISCARD.** Same shape, smaller. |

**One honesty note that weakens 4.2.** I narrowed D2 and D4 once after looking at their *held-out*
breakage examples (the quoted-content case for D4, the target-vs-place case for D2). That is design
information taken from the held-out half, so the post-narrowing held-out numbers for those two rules are
optimistic. I stopped after one round rather than deepen the leak, and the lenient column — which was
computed without reference to any example — is the one to trust.

### 4.4 Two distinctions I would have proposed are already being trained

`speech_act`/`world_statement` and typed `place_kind`/`target_kind` are already in
`eval/training/schema.py` and in the deployed artifact's config. Proposing them would have been
re-deriving the in-flight retrain. Recorded so the next pass does not spend the effort again.

## 5. The multi-hop experiment

The hypothesis under test, as posed: multi-hop reasoning and GSM8K's "which numbers combine" failures
are the same missing composition faculty, and the load-bearing piece is that **retrieval is nucleated by
the words of the utterance**, so evidence reachable only *through* an intermediate result is unreachable.
The proposed test: run the single-hop answerer twice with the second retrieval seeded by the entity found
in the first, against a control seeded by the original question.

`eval/schema/multihop.py`, HotpotQA distractor validation, n=300, k=6 sentences per hop, the existing
109M-param `span-answerer` artifact (trained on SQuAD 2.0 single paragraphs), the repo's own
`BM25Ranker`, dataset labels as the only grader. **Prediction stated before running:** re-seeded reaches
≥0.30 EM while the control stays <0.15.

| arm | sentences | EM all | EM att | coverage | gold-fact recall | all gold present |
|---|---|---|---|---|---|---|
| `no_retrieval` (the published 0.093) | 40.5 | 0.0933 | 0.2373 | 0.393 | 1.000 | 1.000 |
| `single_hop_k` | 6.0 | 0.2567 | 0.3392 | 0.757 | 0.635 | 0.350 |
| `single_hop_2k` | 12.0 | 0.2633 | 0.3465 | 0.760 | 0.777 | 0.553 |
| `chain_selected` | 2.9 | 0.2633 | 0.3211 | 0.820 | 0.549 | 0.233 |
| `precision_control` | 3.0 | 0.2500 | 0.3219 | 0.777 | 0.495 | 0.190 |
| **`two_hop_reseeded`** | 12.0 | **0.2733** | 0.3644 | 0.750 | 0.770 | 0.560 |
| **`two_hop_control`** | 12.0 | **0.2667** | 0.3540 | 0.753 | 0.766 | 0.547 |
| `oracle_selection` (gold sentences present in the pool) | 1.9 | 0.3500 | 0.3818 | 0.917 | 0.777 | 0.553 |
| `oracle_gold_only` (gold sentences, nothing else) | 2.5 | **0.4333** | 0.4676 | 0.927 | 1.000 | 1.000 |

Reference points: local 8B model prompted plainly **0.5133** EM / 0.657 F1 at coverage 1.00; the rules
arm 0.0584 (`eval/results/open_domain.json`).

### 5.1 The prediction fails, and the confound was in the baseline

**`two_hop_reseeded` 0.2733 vs `two_hop_control` 0.2667.** A difference of two items in 300. On bridge
questions alone — where the chain is actually required — 0.3067 vs 0.3025, one item in 238. Gold-fact
recall is the same to within noise (0.770 vs 0.766), and `single_hop_2k`, a **single** query, has the
highest recall of the three (0.777). Re-seeding with the found entity surfaces nothing that the
question's own terms do not already surface.

It is not even "two retrievals beat one": the two-hop arms and the one-query arm with the same sentence
budget are indistinguishable. The seed is irrelevant; only the budget matters.

**And the 0.093 figure does not mean what it was taken to mean.** `eval/training/eval_span.py` hands the
answerer all ~40 sentences, which the 384-token window truncates — it performs **no retrieval at all**.
Adding BM25 retrieval of six sentences, with no architectural change whatsoever, moves the same weights
from 0.0933 to 0.2567. Roughly **two-thirds of the apparent multi-hop deficit was a window bug**, and
0.093 could never have supported a claim about composition.

### 5.2 A second prediction, also stated in advance, also failed

If evidence must *connect* rather than merely score, then selecting the pair of sentences that jointly
cover the question and share an entity should beat selecting the top 3 by relevance. Predicted:
`chain_selected` ≥0.33 and ≥0.03 above `precision_control`. Result: 0.2633 vs 0.2500, **+0.013**. The
chain criterion does find better evidence than BM25 alone (recall 0.549 vs 0.495, and conditional
accuracy 0.529 vs 0.579 on the items it gets right) but it retains both gold facts for only 70 of 300
items. The lexical implementation of the compositional idea is far too weak to pay.

### 5.3 What the gap actually consists of

The arms above bracket it, each step measured:

| from → to | what changes | EM |
|---|---|---|
| `no_retrieval` → `single_hop_k` | fixing the truncation bug | 0.093 → 0.257 (**+0.164**) |
| everything tried about routing, chaining, re-seeding | | 0.257 → 0.273 (**+0.016**) |
| best real arm → `oracle_selection` | a perfect selector on a realistic pool | 0.263 → 0.350 (**+0.087**) |
| `oracle_selection` → `oracle_gold_only` | a perfect retriever as well | 0.350 → 0.433 (**+0.083**) |
| `oracle_gold_only` → 8B model | what remains | 0.433 → 0.513 (**+0.080**) |

Selection and retrieval are worth almost exactly the same, ~0.085 each, and **neither was reachable by
any criterion tested.** Note also that the answerer's accuracy is close to (gold recall × conditional
accuracy) along every axis varied, and the two move inversely: `chain_selected` (0.549 recall × 0.529)
and `single_hop_2k` (0.777 × 0.368) give the identical 0.263. Each added distractor costs about what the
missing evidence gains, because the answerer has no way to ignore a sentence in its window.

### 5.4 Where the oracle loses: attribution, not narrative

`eval/schema/multihop_failures.py` assigns each `oracle_gold_only` failure to one mechanism by rule
(reproducible, so it can be wrong in bulk rather than selectively). Retrieval is perfect by construction
here, so this is the whole of the remaining gap.

| mechanism | items | share | confident while wrong |
|---|---|---|---|
| correct | 130 | 0.433 | — |
| `span_boundary` — right referent, wrong extent | 54 | 0.180 | 28 |
| `wrong_span` | 42 | 0.140 | 21 |
| **`bridge_entity_returned`** — returns the entity the question travels *through* | 36 | **0.120** | 17 |
| `abstained` | 22 | 0.073 | — |
| `answer_type_unavailable` — gold is yes/no, the tier emits spans | 16 | 0.053 | — |

`bridge_entity_returned` is the genuine composition failure and it has a clean signature: *"Who replaced
the manager of Aston Villa that began at Leeds United?"* → `David O'Leary`, who **is** the manager who
began at Leeds — the intermediate, returned as the answer, at confidence 0.417. *"What was the nickname
of Judy Lewis's father?"* → `Clark Gable`, the father rather than the nickname. It cannot distinguish a
subgoal's value from the goal's value.

But it is **12%**, and it is smaller than `span_boundary` at 18%. And `span_boundary` is largely not a
capability gap at all: the errors run both ways (28 predictions strictly inside gold, 17 strictly
containing it) and many are pure extent convention — gold `Alan Mathison Turing` against predicted
`Alan Turing`, gold `Richard "Rick" Ducommun` against `Rick Ducommun`. The referent is right and EM is
measuring the dataset's arbitrary extent choice. Under F1 the oracle scores 0.585 over attempted against
the 8B model's 0.657.

**Which means: hand this 109M-param SQuAD-trained answerer the right two sentences and it is close to
the 8B model.** Almost none of the multi-hop gap is composition.

### 5.5 One faculty or two coincidences — GSM8K

The gating experiment failed, so per the brief no subgoaling machinery was built, and the shared-machinery
test could not be run. But the question is answerable from the recorded evidence, and the answer is that
these are **two different deficits wearing one word**:

| | HotpotQA | GSM8K (`schema_brittleness.json`) |
|---|---|---|
| attempts | 75–93% of items | **abstains on 284/300** |
| dominant stage | evidence selection under distraction | `arithmetic_composition` 263, reason `no_single_step_pattern` |
| given perfect evidence | 0.433 EM — the machinery works | nothing to give: no candidate is ever formed |

On HotpotQA the answering machinery exists and loses on which sentences enter the window. On GSM8K the
arm has one-step sum/difference patterns and **no representation of a chain at all**, so it never
produces a candidate to route. A composition faculty would have nothing to compose. "Composition"
describes the *task* in both cases and the *deficit* in neither the same way: one is selection, the other
is absent representation. Only 37 of 300 GSM8K failures (12%) are selection-shaped, at quantity
extraction.

## 6. Calibration

Required as a first-class output: each change states when it should abstain, measured.

- **D1, D2, D6** never abstain and never need to. They are structural projections: they fire on a
  determiner or a place word or not at all, and their held-out breakage is zero.
- **The answerer's abstention is an artefact of its window contents, not of its uncertainty.** Coverage
  is 0.393 with 40 sentences, 0.760 with 12, and **0.927 with the 2 gold sentences**. Same weights, same
  questions. It abstains when distractors confuse it, so its confidence does not measure answerability
  and cannot be used as an abstention signal on this task.
- **The mechanisms that matter are confident while wrong.** 28 of 54 `span_boundary` failures and 17 of
  36 `bridge_entity_returned` failures come in at confidence ≥0.5. Abstention cannot recover them; only
  an expected answer type could, by rejecting a span whose type does not match what was asked.
- **Prior dead ends, unchanged by this work:** the learned parser's own posterior bought +0.9 points for
  2.8% refusals; the rule tier's threshold was capability-blind. Task-declared invariants remain the only
  calibration mechanism that has worked here (3 false successes → 0).

## 7. What needs knowledge rather than structure

The brief asked for this prominently, and it is most of the answer.

1. **`knowledge_gap`, 300 weighted items.** The fact is not in the input, the graph or the grammar. No
   symbolic structure supplies it. Unfixable by schema change, by definition.
2. **`quantity_composition`, 300 weighted items.** Needs a multi-step arithmetic representation that does
   not exist. This is a build, not a distinction — and §5.5 shows it is *not* the same build as multi-hop.
3. **Evidence selection on HotpotQA, ~0.087 EM.** A better selector is worth this much, and three
   criteria (BM25 relevance, entity-linked pair chaining, question-coverage) all landed on the same
   precision/recall curve at 0.25–0.27. The compositional criterion is the right *idea* — chained
   evidence does score better — and the lexical version of it is nowhere near strong enough. This needs a
   learned pair scorer: capability, not structure.
4. **`phrasing_coverage`, 196 items failed by both hand-written tiers and parsed correctly by the learned
   tier.** Already a solved data problem; writing rules for it is the treadmill.
5. **Answer extent convention, 18% of the oracle's loss.** Not a gap at all. EM is partly measuring the
   dataset's arbitrary choice of span boundary, and F1 already shows the referent is right.

The one place where structure is genuinely and measurably missing, and is not knowledge:
**`bridge_entity_returned`, 12% of items, at high confidence** — no distinction between the value of a
subgoal and the value of the goal. That is a real schema gap, it is small, and it is the only item on the
proposed faculty list that this evidence supports.

## 8. Reproducing

```
PYTHONPATH=src:. venv-eval/bin/python eval/schema/corpus.py            # 3,905 rows
venv-browser/bin/python  eval/schema/cluster.py                        # schema_clusters.json
venv-browser/bin/python  eval/schema/measure.py [--lenient]            # schema_repair[_lenient].json
PYTHONPATH=src:. venv-eval/bin/python eval/schema/multihop.py --n 300  # schema_multihop.json
PYTHONPATH=src:. venv-eval/bin/python eval/schema/multihop_failures.py # schema_multihop_failures.json
```

`venv-eval` has torch; `venv-browser` does not and is enough for the corpus and repair layers.
