# Next bounded milestone: learn response completeness and constraints

Status: planned, not implemented or trained. Continues the governing
[big picture](big-picture.md) after [coverage diagnostics](cognitive-answer-coverage.md).

## Problem established by real outputs

Source-wise NLI accepts true statements that fail the question. A QNLI relevance
model catches obvious tautologies but misses requested answer types and compound
constraints. Neither score can be relabeled as correctness. More accepted answers
alone failed the explicit promotion gate.

## Next experiment

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
