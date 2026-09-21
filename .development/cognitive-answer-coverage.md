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
- [ ] Implement and test evidence-justified improvement.
- [ ] Run frozen real-input comparison and review outcomes.
- [ ] Document results, verify, commit and push.

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
