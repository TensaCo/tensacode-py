# Next bounded milestone: learn response completeness and constraints

Status: bounded assessor implementation and GB10 training completed; checkpoint
**not promoted**. No active chatbot improvement is claimed. See
[the protocol](response-quality-implementation.md) and
[measured results](../docs/results/response-quality-pilot.json). Continues the
[big picture](big-picture.md) after [coverage diagnostics](cognitive-answer-coverage.md).

## Pilot outcome

The owned Electra assessor has three binary heads, masked assistant labels,
explicit evidence coverage, replayable training and safe complete artifacts.
55 candidates from 20 questions trained for five epochs / 70 AdamW updates;
17 candidates from six questions calibrated temperatures, and 16 candidates from
six separate questions assessed development behavior. Source documents are disjoint.
All full inputs fit 512 tokens. Encoder and head parameters changed. Training
loss decreased 0.7093 → 0.4656; weights, optimizer continuation and all 88
fresh-process receipts reproduced exactly on GB10.

The result fails promotion. Support and constraints accept every calibration and
development candidate, including four and five labelled development failures,
respectively. Their accuracy equals the all-positive baseline. Completeness
rejects all three development negatives but also eight positives: accuracy 0.50
versus 0.8125 for the all-positive baseline. Its calibrated temperature reaches100.
The joint gate is therefore only a completeness gate on these partitions. It
accepts five known-good development variants across four questions (five of eight
known-good candidates retained), but also accepts two bad calibration variants
that answer “racing” to a question explicitly excluding racing.

No tool integration, checkpoint publication, new release, or final-set evaluation
was justified. The trained artifact, optimizer, 70 saved experience batches and
executed script remain on GB10 at the location recorded in the result. The runner
now checks pinned local HF metadata and ETags before loading; the same audit
passed separately on this run's actual foundation files.

## Broader supervised run (v2)

The source-disjoint 192-question corpus is prepared and reviewed: 367 training,
92 calibration and 91 development candidates. All 550 full assessor inputs fit
512 tokens. Five epochs / 460 updates on GB10 reduced training loss from 0.6267
to 0.3640; complete artifacts, exact optimizer continuation and all 550
fresh-process receipts were verified. [Results](../docs/results/response-quality-v2.json)
record the failed predeclared gate: support and constraints each reject only2/19
labelled development negatives. Completeness rejects4/10 and retains63/81
positives. Joint screening admits28 known-good,20 known-bad and8 unresolved
candidates. Evidence removal/shuffling changes mean support scores by only
−0.011/−0.018. More data alone did not establish evidence-sensitive judgment.

Next controlled comparison: native tokenizer paired question/candidate and
evidence inputs, with identical data, initialization seed, optimizer and schedule.
Separately review authored source-intervention pairs from training questions to
make evidence dependence identifiable. Preserve these as distinct experiments;
do not attribute multiple simultaneous changes to one cause. Joint-evidence NLI
is now explicitly configurable and tested, but its candidate-level improvements
also admit additional failures and do not justify promotion.

## Next implementation target

Improve the supervision before adding another gate. The current 55-candidate
training split is too narrow to establish three distinct judgments. Build a
larger document-disjoint training corpus from natural generated proposals,
including source-supported wrong answer types, excluded activities, entity and
number errors, missing source chains, and correct concise alternatives. Record
assistant supervision and any authored contrasts separately. Keep question,
source and candidate variants together; mask unresolved judgments.

Predeclare per-axis negative detection and positive retention requirements, plus
joint useful coverage and failure counts, before fitting another model. Compare
question/candidate-only and evidence-shuffled ablations to establish whether an
assessor actually uses source evidence. Do not reuse this development split as a
fresh final test. Only integrate a model after those checks; then freeze and run
the complete chatbot comparison on reserved questions.

## Problem established by real outputs

Source-wise NLI accepts true statements that fail the question. A QNLI relevance
model catches obvious tautologies but misses requested answer types and compound
constraints. Neither score can be relabeled as correctness. More accepted answers
alone failed the explicit promotion gate.

## Governing experiment sequence

1. Build reviewed question/evidence/candidate examples with distinct labels for
   source support, question completeness and constraint satisfaction. Retain exact
   sources and target authorship. Natural failures matter alongside any explicitly
   authored corruptions. Gold answers and reviewer rationales are targets/metadata,
   never inference inputs. Assistant-produced labels must be identified as such,
   not described as human ground truth.
2. Use document-disjoint training, calibration and development partitions. Include
   correct aliases, short but sufficient answers, wrong requested types, omitted
   qualifiers, entity/number swaps, multi-source requirements, unrelated sources,
   contradiction and unanswered cases. Audit labels before expensive training.
   Avoid a classifier that succeeds only by detecting synthetic formatting.
3. Train a bounded evidence-conditioned assessor or proposal objective on GB10.
   Keep model ownership, safe complete artifacts and replayable supervision. Start
   with a fixed small budget, save losses and optimizer state, and verify reload.
   Do not introduce another production component until its diagnostic behavior
   improves over the measured source-NLI and QNLI baselines.
4. Connect a successful objective to active candidate selection and final response
   assessment through the opaque tool. Retain evidence and rejection reasons;
   source contradiction checks and revision invalidation stay active. No lexical
   exceptions keyed to the failed questions, authored semantic seeds or graph
   implementation shortcuts.
5. Freeze the full pipeline before the reserved final set. Compare correct useful
   coverage and incorrect/incomplete responses, plus omission/revision/conflict
   controls. Include same-foundation direct answering and workspace bypass when
   making a workspace claim. Report all failures and inherited capabilities.

Do not call this trained or useful until data, parameter updates and end-to-end
held-out behavior establish it. The reserved final cases have not been run.

## Paired encoding and intervention follow-up

With the same initial parameters, natural corpus, seed and five-epoch schedule,
native question/candidate versus evidence pairing accepted 26 known-good, nine
known-bad and eight unresolved development candidates. Training on an additional
128 reviewed evidence interventions accepted 30 known-good, 12 known-bad and
seven unresolved candidates. Neither meets the declared promotion requirements.
The intervention run has more updates (620 versus 460), not an equal-compute
comparison. No assessor is integrated into production selection. Raw reports
and checkpoint continuation checks remain on GB10; see
`docs/results/response-quality-comparisons.json`.
