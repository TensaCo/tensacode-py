# 12. Open-domain benchmarks: what the cognition does off its home turf

**Headline, and it is not flattering: on public open-domain benchmarks the no-model tensacode arm answers almost nothing, and what it does answer it mostly gets wrong.** Coverage is 18.3% on SQuAD 2.0, 5.3% on GSM8K and 0.0% on ARC-Easy, where it has no knowledge source and correctly refuses every question. It is not a general question answerer and this measurement says so plainly.

The same local model (Qwen/Qwen3-8B), prompted plainly with no tensacode structure, is far better everywhere: squad2 67.0%, hotpot 51.3%, gsm8k 81.3%, arc_easy 80.3% correct overall, against the rule arm's squad2 42.0%, hotpot 5.3%, gsm8k 0.0%, arc_easy 0.0%.

**And the cascade this architecture proposes is WORSE than the model alone on 3 of 4 benchmarks**, not better: squad2 56.0% vs 67.0%; hotpot 11.7% vs 51.3%; gsm8k 77.3% vs 81.3%. It ties on arc_easy, and the reason is instructive: there the rule arm abstains on everything, so the cascade simply *is* the model.

This inverts the Banking77 result, where a cheap tier plus abstention beat a general model. The mechanism is visible in the equal-coverage tables below: the model is better than the rules **on the very items the rules chose to answer** (SQuAD 2.0 61.0% vs 9.1%, HotpotQA 49.3% vs 5.8%). So the cheap tier is not selecting the items it is good at. Its abstention is capability-blind: it fires on weak lexical overlap, not on whether it can actually answer. A cascade is only worth having when the cheap tier knows when to shut up, and here it does not.

**The sharpest single number: on SQuAD 2.0 the rule arm scores 42.0% overall, which is worse than the 48.0% it would get by refusing every question.** Its answers are net negative. A system whose output is worse than its own silence has no business answering.

One real, narrow win for the abstention machinery, visible only because SQuAD 2.0 labels unanswerable questions: the model **answers** far better (77.1% exact-match on the answerable questions it attempted, against 15.6%) but **refuses** worse (57.6% of unanswerable questions correctly declined, against the rule arm's 84.0%). Knowing when there is no answer in the passage is the one thing the cheap tier does better than the model, and it is exactly the thing this project claimed for it. It is also not enough to make the cascade worth it.

What the exercise does establish: the abstention machinery is real and measurable — it refuses rather than inventing answers, says why, and on ARC-Easy refuses 100% of questions it cannot ground. Provenance is real but thin: on HotpotQA the arm names the sentences its answer rests on, and those citations are checkable against the published supporting facts, covering 31.6% of gold sentences on average but *all* of them on only 3.3% of items. The citation mechanism works; the retrieval behind it is weak at three sentences.

## How this was measured

| | |
| --- | --- |
| Benchmarks | SQuAD 2.0 (extractive QA, half the questions unanswerable), HotpotQA distractor (multi-hop, gold supporting facts published), GSM8K (grade-school multi-step arithmetic), ARC-Easy (science multiple choice, world knowledge) |
| Grader | public dataset labels only; no model judges anything |
| Splits | disjoint calibration (150) and test (300) slices of one shuffled pool, seed 0 |
| Model arm | Qwen/Qwen3-8B, greedy, batched; multiple choice scored by option likelihood |
| Arm `rules` | tensacode only: patterns + BM25 rank + Unknown; no model |
| Arm `model` | the same local model prompted plainly, no tensacode structure |
| Arm `cascade` | rules answer what they can; only their abstentions reach the model |
| Model cost | 2055 calls, 138669 generated tokens, 2519.8s of generation, 55.0 tok/s, 79.5s to load |

No model judged any answer. Every score is exact match or the dataset's own label. The abstention threshold for the extractive arm was chosen on a calibration slice that is disjoint from the test slice, and a threshold sweep on the test slice is reported separately as a curve so the trade-off is visible rather than tuned.

## Results

Accuracy is over *attempted* items; coverage is the share attempted. A system that answers 10% of questions at 50% accuracy is not better than one that answers everything at 45%, so read the two columns together.

| Benchmark | Arm | Coverage | Accuracy over attempted | 95% CI | Correct over all items | Model calls | s/item |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| squad2 | tensacode only (no model) | 18.3% | **9.1%** | 4.0%–19.6% | 42.0% | 0 | 0.0001 |
| squad2 | local model alone (no tensacode) | 71.3% | **55.1%** | 48.4%–61.7% | 67.0% | 300 | 0.366 |
| squad2 | cascade: tensacode first, model on abstentions | 75.7% | **43.2%** | 36.9%–49.7% | 56.0% | 245 | 0.0001 |
| hotpot | tensacode only (no model) | 91.3% | **5.8%** | 3.6%–9.3% | 5.3% | 0 | 0.0004 |
| hotpot | local model alone (no tensacode) | 100.0% | **51.3%** | 45.7%–56.9% | 51.3% | 300 | 1.245 |
| hotpot | cascade: tensacode first, model on abstentions | 100.0% | **11.7%** | 8.5%–15.8% | 11.7% | 26 | 0.0004 |
| gsm8k | tensacode only (no model) | 5.3% | **0.0%** | 0.0%–19.4% | 0.0% | 0 | 0.0 |
| gsm8k | local model alone (no tensacode) | 100.0% | **81.3%** | 76.5%–85.3% | 81.3% | 300 | 2.949 |
| gsm8k | cascade: tensacode first, model on abstentions | 100.0% | **77.3%** | 72.3%–81.7% | 77.3% | 284 | 0.0 |
| arc_easy | tensacode only (no model) | 0.0% | **0.0%** | 0.0%–0.0% | 0.0% | 0 | 0.0 |
| arc_easy | tensacode only, forced to guess by word overlap | 27.7% | **26.5%** | 18.2%–36.9% | 7.3% | 0 | 0.0 |
| arc_easy | local model alone (no tensacode) | 100.0% | **80.3%** | 75.5%–84.4% | 80.3% | 300 | 0.389 |
| arc_easy | cascade: tensacode first, model on abstentions | 100.0% | **80.3%** | 75.5%–84.4% | 80.3% | 300 | 0.0 |

### Floors and random controls

| Benchmark | Floor / control | Value |
| --- | --- | ---: |
| squad2 | always abstain | 48.0% |
| squad2 | always answer random span | 1.3% |
| hotpot | random first word of a random sentence | 0.7% |
| gsm8k | random number from the question | 1.7% |
| arc_easy | random choice | 24.0% |
| arc_easy | always first option | 28.0% |

- **squad2:** 48.0% of the sample is unanswerable, so 'always abstain' scores that much
- **hotpot:** free-form span: chance is ~0
- **gsm8k:** free-form numeric answer: chance is ~0
- **arc_easy:** 4-way choice: chance is 0.25

## SQuAD 2.0 (extractive QA, half the questions unanswerable)

n = 300 test items, 150 calibration items. Calibration: threshold maximising overall correctness on the calibration slice (threshold 8.0)

- **tensacode only (no model)**: on the 156 answerable questions it attempted 32 and got 15.6% exact-match on those (3.2% of all answerable). On the 144 unanswerable ones it correctly refused 84.0%.
- **local model alone (no tensacode)**: on the 156 answerable questions it attempted 153 and got 77.1% exact-match on those (75.6% of all answerable). On the 144 unanswerable ones it correctly refused 57.6%.
- **cascade: tensacode first, model on abstentions**: on the 156 answerable questions it attempted 153 and got 64.0% exact-match on those (62.8% of all answerable). On the 144 unanswerable ones it correctly refused 48.6%.

This is the benchmark that prices abstention honestly, because refusing is *sometimes the right answer* and the dataset says when. Note the asymmetry it creates: with roughly half the sample unanswerable, a system that refuses everything already scores about half overall, which is why the calibration step drifts toward refusing.

Threshold sweep on the test slice (a curve, not a tuned number):

| min BM25 score | Coverage | Accuracy over attempted | Correct over all |
| ---: | ---: | ---: | ---: |
| 0.0 | 96.7% | 9.3% | 11.3% |
| 1.0 | 95.7% | 9.1% | 11.0% |
| 2.0 | 88.3% | 9.4% | 14.7% |
| 3.0 | 76.7% | 9.1% | 19.3% |
| 4.0 | 62.3% | 10.7% | 26.3% |
| 6.0 | 30.7% | 12.0% | 39.3% |
| 8.0 | 18.3% | 9.1% | 42.0% |
| 12.0 | 4.7% | 7.1% | 46.0% |

**Equal-coverage comparison** (the question the Banking77 work taught us to ask: is the cheap tier adding anything, or just answering the easy items?)

- On the 55 items the rules answered, the rules scored 9.1% and the model scored 61.0% on those same items.
- On the 245 items the rules refused, the model scored 53.8%.

Cascade tiers: model answered 172 of 245 at 94.8%; rules answered 55 of 55 at 9.1%

## HotpotQA distractor (multi-hop, gold supporting facts published)

n = 300 test items, 150 calibration items. Calibration: threshold maximising overall correctness on the calibration slice

- **tensacode only (no model)** cited evidence containing *all* the gold supporting sentences on 3.3% of items, and on average covered 31.6% of the gold sentences (300 items had usable gold labels).
- **cascade: tensacode first, model on abstentions** cited evidence containing *all* the gold supporting sentences on 0.7% of items, and on average covered 29.4% of the gold sentences (274 items had usable gold labels).

The provenance number is the one claim of ours this benchmark supports directly: the arm does not merely produce an answer, it names the sentences it read, and those citations can be checked against HotpotQA's published supporting facts. The answers themselves are mostly wrong.

Threshold sweep on the test slice (a curve, not a tuned number):

| min BM25 score | Coverage | Accuracy over attempted | Correct over all |
| ---: | ---: | ---: | ---: |
| 0.0 | 91.3% | 5.8% | 5.3% |
| 1.0 | 91.3% | 5.8% | 5.3% |
| 2.0 | 91.3% | 5.8% | 5.3% |
| 3.0 | 91.3% | 5.8% | 5.3% |
| 4.0 | 90.7% | 5.9% | 5.3% |
| 6.0 | 87.0% | 6.1% | 5.3% |
| 8.0 | 76.7% | 7.0% | 5.3% |
| 12.0 | 53.0% | 6.9% | 3.7% |

**Equal-coverage comparison** (the question the Banking77 work taught us to ask: is the cheap tier adding anything, or just answering the easy items?)

- On the 274 items the rules answered, the rules scored 5.8% and the model scored 49.3% on those same items.
- On the 26 items the rules refused, the model scored 73.1%.

Cascade tiers: rules answered 274 of 274 at 5.8%; model answered 26 of 26 at 73.1%

## GSM8K (grade-school multi-step arithmetic)

n = 300 test items, 150 calibration items. Calibration: no threshold applies

**Equal-coverage comparison** (the question the Banking77 work taught us to ask: is the cheap tier adding anything, or just answering the easy items?)

- On the 16 items the rules answered, the rules scored 0.0% and the model scored 93.8% on those same items.
- On the 284 items the rules refused, the model scored 80.6%.

Cascade tiers: model answered 284 of 284 at 81.7%; rules answered 16 of 16 at 0.0%

## ARC-Easy (science multiple choice, world knowledge)

n = 300 test items, 150 calibration items. Calibration: no threshold applies

**Equal-coverage comparison** (the question the Banking77 work taught us to ask: is the cheap tier adding anything, or just answering the easy items?)

- On the 0 items the rules answered, the rules scored 0.0% and the model scored 0.0% on those same items.
- On the 300 items the rules refused, the model scored 80.3%.

Cascade tiers: model answered 300 of 300 at 80.3%

## What this changes

1. **The project's cognition claims do not transfer to open domain.** Everything the browser agents and the assistant do well is narrow, scripted competence in environments we wrote. Faced with public questions, the no-model arm has no knowledge, no arithmetic planning beyond one step, and no way to answer anything not lexically present in a passage.
2. **Abstention is genuine, not decorative.** The arm refuses cleanly and says why (`weak_evidence`, `no_span_of_type`, `no_knowledge_source`, `multi_step`). On ARC-Easy it refuses everything rather than guessing, and the separately reported forced-guess variant shows what guessing by word overlap would buy against the 25% chance floor. That is the behaviour the design promised.
3. **The cascade actively harms results off home turf, and that is the most useful finding here.** A cheap tier that answers confidently and wrongly is worse than no cheap tier at all: on HotpotQA the rules answered 274 of 300 items at 5.8% while the model would have scored 49.3% on those same items, dragging the cascade to 11.7%. Where the rule arm abstains completely (ARC-Easy) the cascade is exactly the model, with no harm done. The lesson is not 'cascades work' or 'cascades fail', it is that a cascade's value is entirely determined by the *calibration* of its cheap tier, and ours is calibrated on lexical overlap, which has nothing to do with whether it can answer. Banking77 looked good because the threshold there was fitted on in-domain validation data.
4. **Provenance is the one part that travels, and it is thinner than advertised.** The claim store and the ranked-evidence path work identically on public data, and a citation can be checked against gold supporting facts — which is more than most systems offer. But at three retrieved sentences the arm covered only 31.6% of gold sentences on average and *all* of them on 3.3% of items. Checkable citation is a mechanism we have; good retrieval is not.

Two follow-ups live elsewhere. [13 — schema brittleness](13-schema-brittleness.md) turns these benchmarks into a failure taxonomy: where the arm breaks along its own pipeline, what share of its *correct* answers are chance (about half on HotpotQA), and the measured answer to "does any routing beat always asking the model?" — which is no, with only 1.3-2.0 points of headroom even for an oracle router. The abstention-as-answerability-detector idea fails there too.

Read this together with [11 — evidence audit](11-evidence-audit.md), which labels every measured claim in this project by who wrote the environment and who graded the answer. Most of the headline results were graded by code we wrote in environments we wrote. This file is one of the few where neither the questions nor the grader is ours, and it is the least flattering.
