# 52 — Global interpretation retention

*2026-09-19. This checkpoint addresses the compute and retention costs measured in
[51 — Learned source segmentation](51-learned-source-segmentation.md). It concerns which
inferred alternatives reach the workspace and how much semantic projection work is spent
producing them. It does not establish autonomous semantic selection or general cognition.*

## The problem and the bounded change

The previous reader projected semantic alternatives inside individual segmentation/tag
branches, truncated those branches, and then applied another final cap. It could spend work
materializing meanings that were discarded immediately and let local ordering determine
which syntax families remained available globally. In the twelve-input measurement from
doc51, every input discarded proposals; median reader latency reached 3.855 seconds.

The new reader first organizes source-anchored syntax families across its segmentation and
tag proposals. It uses one global final-alternative cap and advances semantic frontiers
incrementally. Deferred syntax retains evidence and provenance instead of being represented
as an already evaluated meaning. This is an authored resource-allocation policy operating
on learned source/syntax proposals; it is not a learned judgment of which meaning is true.

Existing numeric bounds remain matched to doc51: segmentation beam four/two candidates,
tag beam four/four candidates, parse beam eight/four candidates, local-margin parse ranking,
100,000 expansions per search, 600,000 sentence search expansions, sixteen final alternatives,
four semantic emissions per family, 64 semantic expansions per family, and 2,048 sentence
semantic expansions. `max_alternatives` also bounds syntax families before projection.
The effects of changed allocation must be measured separately from merely enlarging budgets.

A semantic frontier can resume while the API or active read owns it. Reader outputs record
pending counts, reasons, and deferred syntax metadata; they do **not** persist a live cursor
or queued binding snapshot. This checkpoint therefore provides no cross-turn or
restart-resumable reader search. Repeating a read may repeat earlier work. Dependency
candidate search itself is still eager.

Syntax breadth also consumes places that previously held additional semantic variants of
the same tree. Under the unchanged sixteen-alternative cap, more syntax families can mean
fewer materialized role alternatives per family. Pending counts do not expose those missing
meaning candidates to workspace investigation. Without persisted cursors, producing them
requires further reading/search work. Increased syntax diversity alone therefore establishes
neither general semantic preservation nor better understanding.

## Instrumentation and metric boundaries

The source-faithful evaluator now records per-sentence `retention_stats` and
`reader_phase_ms`. These dictionaries are copied onto final alternatives by the reader;
the evaluator counts each dictionary once per sentence, checks that copies agree, and
rejects malformed telemetry rather than multiplying counts by the number of alternatives.

Retention counters distinguish generated, retained, and deferred syntax; semantic work
explored; emitted semantic candidates; pending work; families not yet expanded; and emitted
semantics subsequently discarded. A pending-work counter is the implementation's frontier
accounting, not a count of every possible interpretation of the input.

The old `proposals_discarded` value counted flattened semantic/segmentation truncation;
the new value counts excluded syntax families. Those raw numbers have different units and
cannot establish reduced discarded work by subtraction. Before/after comparisons instead
show inputs with truncation, distinct retained syntax, and retained final alternatives.
The earlier report did not record a separate emitted-role-variant count; it cannot supply
that missing baseline retrospectively. New semantic-emission counters are reported with
this limitation.

Timed phases are segmentation, syntax, semantics, and retention. Their sums describe the
instrumented code sections. Total evaluator latency additionally includes reader overhead,
source validation, and scoring; phase times are not assumed to partition that total exactly.
Missing telemetry and malformed/conflicting copies remain explicit in the report.

Quality metrics retain the [doc49](49-source-faithful-language-evaluation.md) contracts:
exact source anchors, partial attachment credit, explicit conditional denominators, and
oracle selection of whole candidate trees. First retention order is not an agent commitment.
Generated syntax or a deferred record is not counted as a selected or understood meaning.

Ten new focused tests verify once-per-sentence counting, malformed/conflicting metadata,
explicitly uninstrumented readers, exact cohort replay, and deterministic exclusions. Along
with the existing sixteen span-evaluation tests, all 26 focused tests passed before real
reader runs. These are measurement-contract fixtures, not cognition demonstrations.

## Predeclared measurements

[`retention_cohorts.json`](../../eval/results/retention_cohorts.json) was written before the
changed-reader inference runs. It fixes two cohorts and the previous reader budgets:

1. Replay the exact twelve test-split IDs from doc51. This is a reused diagnostic comparison.
2. Sample 24 development inputs, seed `20260923`, with 2–20 annotated words, excluding all
   twelve previous IDs. The prior IDs do not occur in the development split; the explicit
   exclusion remains recorded. This is a newly predeclared syntax-retention audit sample,
   not pristine held-out data: the development split was already evaluated for segmentation.

No gold outcomes select runtime policies or change the cohorts. New reports preserve the
old reports and record source/model/dataset hashes with post-run checks.

```sh
.venv/bin/python -m eval.parsing.span_evaluation \
  --cohort-report eval/results/parsing_spans_segmented_smoke.json \
  --max-tokens 20 --output eval/results/parsing_spans_retention_diagnostic.json

.venv/bin/python -m eval.parsing.span_evaluation \
  --treebank "$HOME/.cache/tensorcode/seeds/UD_English-EWT/en_ewt-ud-dev.conllu" \
  --sample 24 --seed 20260923 --max-tokens 20 \
  --exclude-results eval/results/parsing_spans_segmented_smoke.json \
  --output eval/results/parsing_spans_retention_dev.json
```

## Exact twelve-input comparison

The final [diagnostic report](../../eval/results/parsing_spans_retention_diagnostic.json)
replays the earlier twelve inputs with identical configured budgets and model artifacts.
All source, model, and dataset post-run hashes agree. No corpus alignment, reader,
candidate-validation, or telemetry errors occurred.

| Measurement | Previous segmented reader | Global retention |
| --- | ---: | ---: |
| First retained LAS | 75/103 (72.82%) | 75/103 (72.82%) |
| Whole-candidate oracle LAS | 84/103 (81.55%) | 84/103 (81.55%) |
| Oracle conditional LAS | 84/101 (83.17%) | 84/101 (83.17%) |
| Oracle exact trees | 6/12 | 6/12 |
| Distinct retained span/head/label trees, summed | 133 | 153 |
| Final alternatives, including unresolved | 192 | 192 |
| Final alternatives containing acts | 153 | 152 |
| Inputs with truncated search | 12/12 | 12/12 |
| Median measured input latency | 3.855 s | 3.916 s |
| Total measured latency | 49.947 s | 51.447 s |

Both runs align 116 of 120 gold words and produce 118 predicted words in the selected
oracle trees. The broader retained syntax set does **not** improve measured oracle quality
on these inputs. The measured latency also provides no speedup evidence. Measurements ran
on a shared host alongside verification; small timing differences cannot establish a stable
performance regression or gain.

The new counters report 399 generated syntax families, 192 retained and 207 deferred;
453 semantic expansions, 190 emitted candidates, 1,578 pending frontier items, zero
available retained frontiers left wholly unexpanded, and zero emitted semantic candidates
subsequently discarded.
Syntax-family identity includes tag distinctions, whereas the table deduplicates only
spans, heads, and labels; its 153 trees must not be equated with 192 retained families.
The two unresolved final placeholders also explain why 190 emitted semantic candidates
can accompany 192 final alternatives. Emission itself does not imply an executable act.

Measured phases total 50.729 seconds of syntax, 0.124 seconds of semantics, 0.0089 seconds
of segmentation, and 0.0011 seconds of retention. Another 0.583 seconds lies outside those
instrumented sections, including metadata handling and evaluation. Syntax search remains
the dominant measured cost. Earlier reports lack these counters and phases, so this does
not quantify the semantic-work saving relative to the old reader.

A first completed twelve-input run is preserved as
[the pre-repair report](../../eval/results/parsing_spans_retention_pre_repair.json).
A subsequent independent API audit found that a semantic batch exception could lose an
already completed but unreturned candidate. The frontier now queues that prefix for a later
resume. The identical cohort was rerun after the repair and source freeze, without policy
or quality tuning; quality and work counters stayed identical. The pre-repair report's own
post-run hashes were valid, but it is not the final runtime evidence.

## Development audit

The [development report](../../eval/results/parsing_spans_retention_dev.json) matches all
24 predeclared IDs, their order, and the diagnostic budgets. Every source/model/dataset
post-run hash agrees. All 24 inputs have telemetry; no alignment, reader, candidate-validation,
or telemetry error occurred. There is no previous-reader comparison on this cohort, so its
scores describe the current proposal set rather than an improvement.

| Development measurement | Result |
| --- | ---: |
| First retained LAS | 155/193 (80.31%) |
| Whole-candidate oracle LAS | 171/193 (88.60%) |
| First conditional LAS | 155/191 (81.15%) |
| Oracle conditional LAS | 171/190 (90.00%) |
| First / oracle exact trees | 14/24 / 18/24 |
| Distinct retained span/head/label trees, summed | 316 |
| Final alternatives / alternatives with acts | 384 / 327 |
| Inputs with truncated search | 24/24 |
| Median / total measured latency | 3.710 s / 98.336 s |

First candidates align 223/225 gold words with 224 predicted words; oracle candidates
align 221/225 with 223 predictions. A whole-candidate attachment oracle need not maximize
word alignment, which explains the lower oracle alignment and conditional denominator.
Neither conditional LAS nor oracle quality measures actual user-intent selection.

Counters report 796 generated syntax families, 384 retained and 412 deferred; 1,588 semantic
expansions, 380 emitted semantic candidates, 7,305 pending items, zero available retained
frontiers wholly unexpanded, and zero emitted semantics discarded. Four unresolved
placeholders occupy the remaining final slots. Every input has at least one proposal
containing an act; that does not establish the act's correctness, grounding, or authorization.

Syntax accounts for 97.203 measured seconds, semantics 0.156 seconds, segmentation 0.0175
seconds, and retention 0.0019 seconds. The remaining 0.958 seconds is outside these phase
sections. The next compute problem is the eager syntax search, while the pending semantic
work makes the current coverage limit visible. Optimizing one must not silently erase the
other's alternatives or promote retention order into semantic authority.

Both cohorts favor short inputs and are small, reused or development data. They cannot
establish general semantic understanding, transfer to other domains, reliable user-intent
selection, or end-to-end task success. The change demonstrably alters active retention and
makes costs and unfinished work inspectable; the diagnostic quality remains unchanged.

## Verification

The full repository suite passed: **1,848 passed, 5 skipped** in 237.41 seconds.
That process loaded the semantic frontier before the final failure-atomicity repair.
After that repair, all **75 affected reader, semantic, and telemetry regression tests**
passed in two focused runs (49 and 26 tests). Both final inference reports were also
produced after the repair, with matching post-run source hashes. These checks establish
bounded search and measurement behavior; they do not establish general cognition.
