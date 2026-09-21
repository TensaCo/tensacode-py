# Response-quality implementation and bounded training

Continues the owner-approved response-quality milestone. Work on main; heavy
inference/training on GB10 only. Do not alter existing published checkpoints.

## Design

An owned, explicitly configured assessor encodes question, full supplied evidence,
and candidate text together using a native pretrained transformer. Three learned
binary heads assess source support, answer completeness and constraint satisfaction.
These are supervised model judgments, not truth guarantees. Missing/ambiguous
review labels are masked targets. Gold answers and rationales never enter inputs.
All heads and tokenizer assets exist at initialization and persist safely.

Start with the 88 natural proposals from the previously inspected 32-question
HotpotQA development run. Review each against original sources; identify labels
as assistant-authored, not human ground truth. Keep duplicate/paraphrased candidates
for a question and all connected source documents in one partition. Fixed split:
20 question groups training, 6 calibration, 6 development if documents permit;
otherwise connected components determine conservative group counts. No new final
question contents are accessed during development. Report the small sample and
shared generator/foundation exposure limitations.

Train a bounded pilot with fixed seed, learning rate and epoch count on GB10;
record initial/final per-head metrics, calibration, optimizer state, exact tensor
reload and fresh-process inference. Complete-input truncation is explicit and
prevents acceptance; it never silently turns partial evidence into full coverage.

Only connect a promising assessor to candidate selection and final response
screening. Existing NLI contradiction checks, source revisions, memory invalidation
and transactional sessions remain active. A configured assessor must be owned by
the complete tool artifact. No lexical exceptions or domain semantic seeds.

## Ownership and sequence

1. Parent: prepare immutable candidate/source records, fixed document-group split,
   coordinate assistant review, train/evaluate pilot and record decisions.
2. Model worker: `_internal/response_quality.py`, focused model/loss/artifact tests.
   Explicit foundation loading; separate axes, masked labels, strict config,
   trace/objective and training checkpoint support, source/truncation receipts.
3. Three independent reviewers: disjoint candidate ID ranges; labels plus concise
   rationale and source anchors. No model outputs as grading oracle.
4. After pilot evidence, integrate configured assessor through Investigator and
   Chatbot, test active rejection/revision/ownership and run frozen final comparison.
5. Full tests/build, independent review, coherent commits/pushes. Keep failures and
   missing capabilities visible in the pinned development notes.

## Pilot protocol

Foundation: existing local `cross-encoder/qnli-electra-base`, pinned
`c7dea87c98b2269a935686c31336e97e837cbbeb`. Newly added head semantics must be
explicitly distinguished from the foundation's relevance training.
Seed 20260921, 5 epochs, AdamW lr 2e-5, batch 4, max input 512 tokens. Use training
partition only for updates, calibration only for temperature, development only
for assessing whether integration is justified. No threshold search on development;
acceptance threshold .5 on each trained head, with existing source-NLI policy.
If connected documents reduce group count, record actual sizes before training.
Never train or choose a configuration on reserved final rows 304:336.

## Pre-training audit

All 88 exact question/evidence/candidate inputs fit the pinned tokenizer's
512-token capacity (198–415 tokens); no evidence needs truncation. Source and
question grouping will still be validated independently of this token check.
Cross-shard assistant review harmonized two ambiguous rows before training.
Final axis counts (true / false / masked): support 52 / 16 / 20; completeness
63 / 23 / 2; constraints 46 / 24 / 18. Forty-two candidates across 21 of 32
questions have all three positive labels. These labels imply a proposal-coverage
ceiling for this corpus, not an observed chatbot improvement. Train-majority and
all-positive baselines are required because labels are imbalanced.
