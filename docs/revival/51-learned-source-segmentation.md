# 51 — Learned source segmentation

*2026-09-19. This checkpoint learns token-boundary proposals from original source text.
It advances the input boundary identified by [49](49-source-faithful-language-evaluation.md)
and the [structured workspace objective](36-structured-cognitive-workspace.md). It does not
establish general semantic interpretation or visual scene understanding.*

## The active capability and supplied structure

A character-boundary model proposes competing segmentations of unchanged input text.
The learner estimates boundaries from annotated UD English EWT training examples instead
of consulting the chart tokenizer's contraction list or quotation regex. Character spans
retain the original source, including apostrophes, punctuation, case, and repeated words.
The runtime integration preserves candidate provenance and does not grant the first
segmentation permission to act.

The model is an averaged perceptron over character windows, Unicode categories, short
character sequences, and previous boundary decisions. Those features and the start/join
representation are supplied architecture. Word boundaries are externally annotated training
supervision. Whitespace is an explicit forced gap: tokens cannot contain whitespace and
all non-whitespace characters must be covered. The complete local corpus satisfies that
assumption; it is not a universal linguistic claim.

The standalone segmentation evaluation uses a beam of four and retains at most three
complete candidates, with a maximum of 100,000 expansions. The integrated learned reader
currently retains at most two segmentation candidates under its shared sentence budget.
The three-candidate oracle below therefore describes decoder availability, not the exact
set surviving runtime retention. These are bounded proposals with uncalibrated scores.
Beam truncation does not make the surviving candidate certain. Missing or corrupt model
artifacts fail explicitly; the removed tokenizer is not reinstated as a production fallback.
An evaluation-only comparison with the historical chart tokenizer measures boundary behavior.

## Training and evaluation

[`eval/parsing/train_segmentation.py`](../../eval/parsing/train_segmentation.py) reuses the
source-faithful CoNLL-U loader. It reads `# text` unchanged and uses exact multiword-token
child spans. No normalized text, grammar tokenization, paraphrases, or held-out annotations
enter training. Local corpus inventory:

| Split | Sentences | Basic words | Original characters | Alignment errors |
|---|---:|---:|---:|---:|
| Train | 12,544 | 204,578 | 995,550 | 0 |
| Development | 2,001 | 25,148 | 123,372 | 0 |
| Test | 2,077 | 25,094 | 122,619 | 0 |

The initial pilot used 1,000 training sentences, two epochs, and 250 deterministically
sampled development inputs. It did not evaluate the test split. Its first segmentation
matched the exact gold token sequence on 86.8% of development inputs, compared with 73.6%
for the historical tokenizer; one of the retained candidates matched on 93.2%.
These are development diagnostics, not fresh generalization evidence.

The full configuration uses **training split only**, five epochs, seed `20260922`, and
all development and test records. The algorithm and decoding bounds are unchanged from
the pilot. The test split has previously been inspected and evaluated in this project;
this report makes no pristine-held-out claim.

```sh
.venv/bin/python -m eval.parsing.train_segmentation \
  --epochs 5 --seed 20260922 --dev-sample 0 --test-sample 0 \
  --output eval/results/segmentation.json
```

The default artifact is
`~/.cache/tensorcode/models/ud_ewt_segmenter.json`. It is versioned JSON containing learned
weights and training provenance. It is generated locally and is not committed to the repo.
There are no downloads, paid inference calls, or binary model additions.

Reports record data/model/source hashes, sampling details, exact candidate counts, errors,
truncation, and timing. Whole-candidate oracle recall chooses one complete segmentation
using gold spans; it never unions incompatible boundaries from different candidates.
Oracle availability is not evidence that the agent knows which segmentation is correct.
Span precision/recall are exact character-span metrics, not semantic accuracy.

The first full run measured useful boundary proposals, but a concurrent evaluator source
edit invalidated one post-run source-hash check. That edit changed alternate-token validation
and model provenance reporting, while `load_records` was unchanged. The pre-freeze summary
is retained in
[`segmentation_pre_freeze_summary.json`](../../eval/results/segmentation_pre_freeze_summary.json).
A final rerun uses identical data, algorithm, seed, epochs, and decoding bounds after source
freeze; it is a provenance correction, not quality-based tuning.

**Final training/evaluation status: completed.** The rerun used the same configuration,
and all source and dataset hash checks passed. It reproduced the pre-freeze quality metrics
exactly. Final learned artifact SHA-256:
`089bde655768ac1e74199f09b6367bec0f75a4c0db306d4c5b393636bdc9f9a8`.
The JSON artifact is 490,216 bytes. Full training took 15.53 seconds on the local host.

| Split / method | Exact token sequences | Exact-sequence rate | Span precision | Span recall |
|---|---:|---:|---:|---:|
| Dev: historical tokenizer | 1,508 / 2,001 | 75.36% | 92.67% | 93.29% |
| Dev: first learned candidate | 1,832 / 2,001 | 91.55% | 98.66% | 98.66% |
| Dev: whole-candidate oracle, up to three | 1,923 / 2,001 | 96.10% | 99.28% | 99.42% |
| Test: historical tokenizer | 1,565 / 2,077 | 75.35% | 92.27% | 92.51% |
| Test: first learned candidate | 1,901 / 2,077 | 91.53% | 98.43% | 98.62% |
| Test: whole-candidate oracle, up to three | 1,995 / 2,077 | 96.05% | 99.12% | 99.46% |

There were no empty candidate sets, decoding errors, invalid source partitions, or expansion
budget exhaustions in either split. Search still reported truncation on 1,980 development
inputs and 2,052 test inputs because beam/output limits discard alternatives. Median test
segmentation latency was 0.599 ms; its 95th percentile was 2.42 ms. These are narrow token
boundary measurements on a shared development host, not total agent response latency.

The full per-input reports are
[the pilot](../../eval/results/segmentation_pilot.json) and
[the final train/dev/test measurement](../../eval/results/segmentation.json).
No test annotations were passed to training. The test split's previous exposure and the
UD boundary convention limit broader generalization claims, even though the comparison
uses real original text rather than authored parser fixtures.

## Downstream reader smoke and its costs

The same twelve original-text inputs from doc49 were evaluated after runtime integration,
without tuning on their outcomes. The report is
[`parsing_spans_segmented_smoke.json`](../../eval/results/parsing_spans_segmented_smoke.json).
All source/model/data hash checks passed, and no reader or metadata errors occurred.

| Measurement | Prior reader smoke | Reader with learned segmentation |
|---|---:|---:|
| First-proposal LAS, full denominator | 71 / 103 = 68.93% | 75 / 103 = 72.82% |
| Whole-candidate oracle LAS, full denominator | 80 / 103 = 77.67% | 84 / 103 = 81.55% |
| Oracle conditional LAS | 80 / 98 = 81.63% | 84 / 101 = 83.17% |
| Exactly aligned spans / 120 gold words | 115 | 116 |
| Predicted words | 123 | 118 |
| Oracle exact trees | 5 / 12 | 6 / 12 |
| Inputs with retention discards | 6 / 12 | 12 / 12 |
| Inputs reporting search truncation | 12 / 12 | 12 / 12 |
| Median latency | 2.646 s | 3.855 s |
| Total timed reader work | 27.198 s | 49.947 s |

The retained syntax proposals contain four more correct labeled attachments under this
oracle comparison. First-proposal correctness also rises on this small cohort, but that
proposal remains uncommitted: reader order is not evidence of user intent. Conditional
accuracy now covers three more representable endpoints; the full-denominator scores keep
that coverage change visible.

**Compute and retention costs increased.** Every input now discards proposals under the
joint cap, compared with half previously. Median latency rose by about 46%, and total timed
work by about 84%. Concurrent verification on the shared host prevents an isolated speed
claim, but these observed costs must accompany the small quality improvement. Additional
segmentation branches do not automatically receive sufficient downstream reasoning budget.

This is a tiny reused diagnostic cohort, not a fresh generalization benchmark. The result
establishes that the new source proposals reach the actual reader and can change its
available syntax. It does not establish reliable semantic selection, improved execution,
or generalized cognition.

## Evaluation cleanup

The obsolete `evaluate_candidates --active-reader` branch has been removed. It assumed
that every interpretation used `Sentence.tokens`, which can misassign dependency indices
when segmentation candidates have different token boundaries. `evaluate_candidates` now
measures fixed-UD-token decoder comparisons; actual reader measurements use the validated
per-candidate source anchors in `span_evaluation`. The earlier doc47 command and measurements
remain explicitly historical. There is no compatibility switch for the removed path.

## Repository verification

Verification exercised the complete repository suite: **1,822 passed, 5 skipped**,
with one outdated test assertion failing because an absent semantic projection now
explicitly reports `False` instead of `None`. The assertion was corrected to match
whether a candidate has acts, and the failed-test rerun passed. No runtime change
followed that full run. Thus all **1,823 applicable tests** have passing verification.
The initial verification process had separately ended on SIGTERM; the completed run
used an independent log and took 229.97 seconds. Compilation/help checks also verify
the removal of the obsolete active-reader evaluation option.

## What remains unresolved

Learning the source boundaries removes one authored lexical preprocessing dependency. It
does not infer entity identity, user intent, goal semantics, or world dynamics. Subsequent
POS, dependency, and semantic proposal budgets can still discard the correct interpretation.
The current representation also forces gaps at whitespace, so it does not propose multiword
units as individual tokens; those would require another explicit representation or later
composition.

The nine focused training/evaluation tests use supplied candidates to verify source fidelity,
no boundary-union oracle inflation, deterministic selection, and explicit alignment errors.
They establish the evaluation contract. Actual learned behavior is measured separately on
the corpus; new schemas or fixtures are not counted as learned capabilities.
