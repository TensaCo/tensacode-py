# Current milestone: useful cognitive answers

## Intent and approved direction

Improve the complete cognitive Chatbot's evidence-based answering before adding
more API architecture. Preserve source revisions, uncertainty and explicit action
boundaries. This is the first bounded milestone under [the compass](big-picture.md).

## Plan

1. Audit historical candidate/verification/realization receipts to localize losses.
   Treat inspected prior test examples as development evidence for this work.
2. Establish a reproducible GB10 diagnostic using the complete existing artifact.
   Keep foundation weights, prompts, policy and source/data hashes in reports.
3. Choose the smallest justified mechanism change from the observed failures;
   specify it here before implementation. Add failure/contrast tests, then run the
   real-model comparison. Do not relax thresholds to manufacture coverage.
4. Freeze the chosen configuration before evaluating fresh disjoint examples.
   Report proposed/selected/realized answers, unsupported outputs, completeness,
   abstention, and controls separately. Report negative results honestly.
5. Verify CPU tests, builds and persistence; commit/push changes with limitations.
   Record the next bounded milestone; no claim that this finishes general cognition.

## Progress

- [x] Pin objective, priorities and behavioral gates.
- [x] Diagnose historical failures and reproduce the complete model.
- [x] Implement and test an isolated verifier intervention; promotion failed.
- [x] Run fixed development comparison and review outcomes; preserve final set after failed gate.
- [x] Document diagnostic results and verification.

Coherent milestone commits and pushes are recorded in main history.

## Diagnostic findings and selected next experiment

Historical 32-case audit: 88 proposals; 80 fail strongest-source support, three
otherwise-supported wrong proposals are blocked by contradiction, one by
truncation, and four pass across two questions. Realization preserves both
selections. Ranking and decoder repair are therefore not the immediate target.

GB10 replay reproduced every historical acceptance decision. A punctuation-based
source segmentation probe, with identical verifier weights/temperature/thresholds,
kept four accepted proposals but lost the correct James Franco claim and admitted
a repetitive non-answer instead. Reject this intervention; do not ship it.

Next bounded intervention: compare an explicitly selected NLI foundation trained
on MultiNLI/FEVER/ANLI against the existing small SNLI-adapted verifier. Hold the
proposal generator, ranker, realizer, source-wise policy and 192-token verification
budget fixed. Fit its temperature on the same independent SNLI validation split,
never on these inspected question/candidate outcomes. This tests inherited
verifier quality; it is not a learned-workspace improvement. Preserve label order
from the model's explicit configuration/card, use safetensors and pin revision.

First run the historical cases as development only. If useful answers improve
without an unacceptable observed unsupported-answer increase, freeze the complete
new artifact and run baseline/intervention on fresh disjoint HotpotQA questions
with omission/revision controls. Evaluate actual text against sources and gold
answers, and report circular/non-answer outputs separately. Do not declare
NLI acceptance to be correctness or publish it as a general cognitive model.

Tokenizer audit: all 176 historical premise/hypothesis pairs have matching input
IDs and attention masks between the serialized fast tokenizer and the native
DeBERTa wrapper rebuilt from that vocabulary at the saved budget. Token-type IDs
differ but type_vocab_size=0 means they do not affect this model. No demonstrated
tokenizer reconstruction defect; do not refactor tokenization to explain these
failures without new evidence.

## Comparison protocol (recorded before new verifier results)

- Candidate foundation: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, revision
  `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`; native safetensors only.
- Calibration: 256 distinct SNLI validation pairs selected by the existing
  `train_verifier.load_records`; temperature only. No QA labels for fitting.
- Development: the 32 historical HotpotQA questions (rows 272:304), now explicitly
  reused for diagnosis, with every non-abstained response reviewed against its
  original evidence and gold answer. Whole-word containment is diagnostic only.
- Promotion gate for this bounded experiment: more source-reviewed correct,
  responsive answers than the baseline's 1/32, with no increase over its one
  incorrect/incomplete/non-answer output. Report all categories, not just coverage.
  Failure means preserve the result and do not promote the artifact as improved.
- If the gate passes, freeze complete weights/config, script hash and policy, then
  compare both artifacts on the next 32 rows (304:336) from the same pinned
  HotpotQA validation file; assert IDs disjoint from recorded prior cases.
- The first eight fixed final cases receive fresh-session omission and
  same-session evidence replacement controls. Include the existing explicitly
  conflicting-source fixture separately. No tuning after final outcomes.
- Same supplied oracle passages, generator, ranker/workspace, realizer, 192-token
  verifier budget and .7/.2/.3 policy. Foundation pretraining exposure is unknown.
  This does not test learned retrieval, workspace advantage, or action planning.

## Follow-up diagnostic: question responsiveness

Partial development inspection of the replacement verifier already identifies
multiple non-answers: naming a magazine when asked its type, and stating that an
actor played "the character" without naming it. It also permits a wrong surfing
location. Complete all 32 cases and retain the failed promotion gate; no improved
model claim follows from higher coverage alone.

Before adding a production component, probe `cross-encoder/qnli-electra-base`
revision `c7dea87c98b2269a935686c31336e97e837cbbeb` on question/candidate and
question/realized-answer pairs. Its documented sigmoid score concerns whether
text answers a question, not whether it is true. Use the fixed binary threshold
0.5 without tuning on these cases; report raw scores and truncation. It must not
replace source support or contradiction screening. Reject the approach if it
cannot distinguish the observed non-answers from useful answers. Download and
inference stay on GB10. This remains a development diagnostic; final cases stay
unseen and unselected until a complete candidate pipeline is frozen.

## Outcome and next work

The complete replacement verifier produced 6 correct responsive answers, 1 wrong
answer, 3 incomplete/nonanswers and 22 abstentions on inspected development cases.
The baseline had 1 correct, 1 nonanswer and 30 abstentions. Promotion failed:
incorrect/incomplete output count increased from 1 to 4. All 8 omission and 8
replacement controls and the separate conflict fixture abstained.

The QNLI diagnostic rejects the unnamed-character and age-range tautologies, but
assigns ~.998 to both the wrong-location answer and the magazine-name/type
mismatch. Do not ship it as a completeness or constraint guarantee. No new
production gate was added, thresholds were not relaxed, and no checkpoint was
promoted. Do not run final rows 304:336 until a complete next intervention is
frozen. Existing final history is unchanged; reused cases are development now.

Artifacts: GB10 `artifacts/coverage-20260921`; public compact evidence and assistant
review: `docs/results/cognitive-coverage-development.json`. Full raw receipts,
calibration tensors and complete unpromoted model remain on GB10, with hashes in
the report. Bootstrap's initial integer-label-map JSON failure is preserved in
its log; the regression test and normalized bootstrap pass. CPU suite: 602 passed,
1 skipped (second-device CUDA assertion).

Next bounded milestone: [train response quality](response-quality-training.md).
This milestone completed diagnostics and established the missing training target;
it did not achieve dependable cognitive answering or a learned workspace gain.
