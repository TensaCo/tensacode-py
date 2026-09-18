# 13. Where the cognitive schemas break: a failure taxonomy

The benchmarks in [12](12-open-domain.md) are used here as a diagnostic instrument rather than a scoreboard. Nothing was tuned. Every wrong or abstained item was attributed to the earliest stage of the arm's own pipeline that is responsible, then labelled by the deficiency it reveals.

## Ranked findings

**1. Half of what the arm gets right on HotpotQA is chance.** Of its 16 correct answers, 13 were picked from a sentence holding several candidates of the asked-for type with nothing discriminating between them — the arm takes the first by position. Expected correct by chance alone: **8.52 of 16**. On SQuAD 2.0 it is 1.5 of 5. So the 5.8% and 9.1% reported in 12 overstate the arm: roughly half the HotpotQA credit and a third of the SQuAD credit is luck, not discrimination. This belongs in any headline that quotes those numbers.

**2. A prior of mine was wrong: retrieval is not the bottleneck on SQuAD 2.0.** I expected string-shaped matching to be the binding constraint. In fact BM25 already places a gold-answer-bearing sentence in the top 3 for **91.5%** of answerable items, and crude stemming changes that by **exactly nothing** (91.5% to 91.5%). The evidence is found and then wasted downstream. Retrieval *is* the top failure on HotpotQA, where bridge facts share few words with the question (130 of 300 items) — so the same deficiency binds on one benchmark and not the other, and a single story about 'lexical matching' would have been wrong.

**3. The real constraint on extraction is that the schema cannot name things.** When the gold sentence *is* retrieved (65 of 71 items), the candidate generator can produce the gold span only **33.9%** of the time. Dropping the answer-type filter changes it by zero (22 vs 22 items), so type filtering is not what stops it: there is simply no entity recogniser. Regexes over capitalised runs, numbers and dates cannot see two thirds of the answers people ask for.

**4. Abstention is miscalibrated in both directions, not just one.** On SQuAD 2.0 the arm abstained on 54 answerable questions it had the evidence for, and answered 23 questions that have no answer. Both come from the same cause: the threshold is on evidence strength, and evidence strength is not capability.

**5. Multi-step numeric reasoning is absent, not weak.** GSM8K: 263 of 300 items need a chain the one-step sum/difference patterns cannot express, and 37 do not even yield two quantities. Zero correct out of 300.

**6. ARC-Easy is unanswerable by construction, and that is the honest bound.** All 300 items are a knowledge gap: nothing in the question, the options or the grammar supplies the fact. The arm refuses every one, which is the correct behaviour and worth zero points.

## The four-way split

Across 1058 diagnosed failures:

| Class | Meaning | Count | Share |
| --- | --- | ---: | ---: |
| **(a)** | missing representation | 605 | 57.2% |
| **(b)** | missing mechanism | 139 | 13.1% |
| **(c)** | knowledge gap | 300 | 28.4% |
| **(d)** | our wiring error | 14 | 1.3% |

(a) and (b) are work we can do. (c) bounds what any symbolic structure can claim. (d) is our own error, and it is small but real: on HotpotQA 14 items are labelled answerable while the gold string is not literally in the sentences we split out — mostly yes/no questions, which the arm has no projection for at all.

## Per-stage counts

| Benchmark | Stage the failure is attributed to | Class | Count |
| --- | --- | :---: | ---: |
| squad2 | span not produced | (a) | 87 |
| squad2 | abstained though answerable | (b) | 54 |
| squad2 | answered an unanswerable question | (b) | 23 |
| squad2 | selection chose wrong candidate | (b) | 6 |
| squad2 | retrieval missed evidence | (a) | 4 |
| hotpot | retrieval missed evidence | (a) | 130 |
| hotpot | span not produced | (a) | 84 |
| hotpot | selection chose wrong candidate | (b) | 51 |
| hotpot | answer not in passage but labelled answerable | (d) | 14 |
| hotpot | boolean or other type | (b) | 5 |
| gsm8k | arithmetic composition | (a) | 263 |
| gsm8k | no quantities parsed | (a) | 37 |
| arc_easy | world knowledge | (c) | 300 |

What each stage would have required:

- **abstained though answerable** (b): capability-aware abstention. The threshold is on evidence strength, not on whether a span of the right type was actually found.
- **span not produced** (a): an entity/number/date recogniser over the retrieved sentence. The candidate generator is regex-shaped, so it cannot see spans it has no pattern for.
- **selection chose wrong candidate** (b): a mechanism that scores candidates against the question. Right now the first candidate of the right type in the best sentence wins, which is position, not reasoning.
- **retrieval missed evidence** (a): lexical semantics: synonymy, hypernymy and paraphrase over claim objects. BM25 matches surface strings, so a question that shares no words with its evidence cannot retrieve it.
- **answer not in passage but labelled answerable** (d): the gold span is not literally in the sentences we split out; our sentence splitter or passage handling dropped it.
- **boolean or other type** (b): yes/no and comparison handling; the arm has no projection for them.
- **arithmetic composition** (a): a representation of quantities, their relations and an order of operations. One-step sum/difference patterns cannot express a chain.
- **no quantities parsed** (a): quantity extraction with units and referents, not bare numbers.
- **world knowledge** (c): facts about the world. Nothing in the passage or the grammar can supply them.

## Probes, with the predictions they were written to falsify

Each probe ran on a 150-item SQuAD 2.0 slice disjoint from both the calibration and diagnosed test slices. These are probes, not fixes: nothing was adopted.

**typeless candidates**

- *Prediction:* If the answer-type filter is the binding constraint, dropping it should raise answerable accuracy above the arm's 15.6% EM on attempted answerable items. If it does not, type filtering was not what was stopping it.
- *Result:* `{"attempted": 71, "correct": 8, "accuracy_over_attempted": 0.1127}`
- *Verdict:* **falsified.** Accuracy over attempted is 11.3%, no better than the gated arm, so the answer-type filter was never the binding constraint.

**span coverage**

- *Prediction:* Retrieval is not the constraint on SQuAD 2.0 (the stem probe shows BM25 already puts the gold-bearing sentence in the top 3 for ~92% of items). So the gold span should usually be UNPRODUCIBLE by the regex candidate generator. If instead it is usually producible, the failure is selection, not representation, and the (a) label on span_not_produced is wrong.
- *Result:* `{"answerable_with_gold_in_passage": 71, "gold_sentence_retrieved": 65, "gold_span_producible_with_asked_type": 22, "gold_span_producible_with_any_type": 22, "producible_rate_asked_type": 0.3385, "producible_rate_any_type": 0.3385}`
- *Verdict:* **confirmed.** Two thirds of gold spans are unproducible even when the right sentence is in hand, and the type filter is irrelevant to that. The (a) label on `span_not_produced` stands.

**stem retrieval**

- *Prediction:* If string-shaped matching is the binding constraint at retrieval, crude stemming should lift recall@3 of the gold-bearing sentence by at least 5 points. A smaller gain means morphology is not the problem and real lexical semantics is needed.
- *Result:* `{"items_with_gold_in_passage": 71, "bm25_recall_at_3": 0.9155, "stem_overlap_recall_at_3": 0.9155}`
- *Verdict:* **falsified.** No change at all, and the baseline was already 91.6%. Morphology is not the problem on this benchmark, and neither is retrieval.

## What to change, and the number that would falsify each

Ordered by the count of failures each would address. None is implemented.

1. **An entity/number/date recogniser over retrieved sentences** (class a; addresses 171 items). The claim objects are strings; they need to be typed entities. *Prediction:* raising gold-span producibility from 33.9% to >70% should lift SQuAD answerable EM on attempted items from 15.6% to at least 35% at unchanged coverage. If it does not, span production was not the binding constraint and selection is.
2. **Candidate scoring against the question** (class b; addresses 57 items directly and most of the luck credit). Today the first candidate by position wins. *Prediction:* any scoring that beats position should cut the chance-expected share of HotpotQA correct answers from 53.2% to under 25%. If the chance share stays put, the selector is still not discriminating.
3. **A second hop** (class a; addresses much of HotpotQA's 130 retrieval misses). The arm issues one query and stops; a bridge question needs the first answer as a term in a second query. *Prediction:* two-hop retrieval should lift HotpotQA gold-fact coverage from 31.6% to at least 55%. If coverage barely moves, the failure is lexical, not structural.
4. **Capability-aware abstention** (class b). Already measured and **falsified as a cascade strategy** — see the routing section below. Abstention should still be fixed for its own sake (it refuses 54 answerable items), but not on the grounds that it makes a cascade pay.
5. **Quantity representation with an operation chain** (class a). *Prediction:* representing quantities, their referents and an order of operations should take GSM8K from 0/300 to at least 15% on items whose gold solution has two steps. If a two-step representation still scores near zero, the parse is the problem, not the arithmetic.
6. **Nothing for ARC-Easy** (class c). No symbolic change helps; this bounds the claim rather than inviting work.

## Routing: is any cascade worth having? No.

Asked directly, on the same 300-item test slices, with a router trained to predict whether the cheap tier will be right (features from the item and the tier's own output; trained on a disjoint 400-item slice).

| Benchmark | Always model | Always rules | **Oracle router** | Best learned router | Calls saved at best |
| --- | ---: | ---: | ---: | ---: | ---: |
| squad2 | 67.0% | 11.3% | **68.3%** | 67.0% | 0.0% |
| hotpot | 51.3% | 5.3% | **53.3%** | 51.3% | 0.0% |

**No routing beats asking the model every time, and there is almost nothing to route on.** The *oracle* — a router with perfect foresight, keeping the cheap answer exactly when it is right and the model is wrong — gains 1.3 points on squad2 and 2.0 points on hotpot. The learned router's best operating point saves 0% of calls on both benchmarks: it degenerates to 'always ask the model', which is the correct thing for it to do.

Signal quality, as AUC for predicting whether the cheap tier will be right (0.5 = no information):

| Benchmark | Learned capability router | The old BM25 signal |
| --- | ---: | ---: |
| squad2 | 0.5478 | 0.5077 |
| hotpot | 0.4569 | 0.6364 |

Both are at or near chance, and on HotpotQA the learned router is *below* chance (0.457), meaning it did not generalise from its training slice at all. The honest reading is not 'our router is bad' but 'when the cheap tier is right 5-11% of the time, there is no subset worth routing to it'. A cascade needs a cheap tier that is right often enough for its correct region to be findable.

**And the one hypothesis that looked most promising fails too.** The arm refuses 84.0% of unanswerable SQuAD questions against the model's 57.6%, which suggested using it purely as an answer-not-present detector. But that figure is a high abstention rate, not detection: P(abstain | unanswerable) = 84.0% against P(abstain | answerable) = 79.5%, a lift of just 4.5 points. Wiring it up that way — rules decide answerability, model answers the rest — scores 53.3% against the model's 67.0%. The abstention carries almost no information about answerability.

## What this says about generality

The user's framing was right: these benchmarks are most useful as a brittleness probe. The taxonomy says the arm is not 'a bit behind' on open-domain work — it is missing three representations (typed entities, composed hops, quantities with operations) and two mechanisms (candidate scoring, capability-aware abstention), and one whole benchmark is outside what any symbolic structure can do.

The parts that did survive contact are worth naming precisely, because they are the ones to build on: retrieval into a claim store with provenance works and is not the bottleneck; abstention exists, fires with a stated reason, and is honest even when miscalibrated; and the pipeline is legible enough that every one of these failures could be attributed to a stage. A system that cannot tell you where it broke could not have produced this document.
