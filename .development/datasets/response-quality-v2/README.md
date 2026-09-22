# Natural response-quality supervision, v2

550 deduplicated natural proposals from 192 HotpotQA training questions. The
unchanged owned generator receives only question and supporting passages; source
answers are review metadata. Six assistant reviewers supplied labels, followed
by cross-shard sampling and parent adjudication in `review-audit.json`. These are
assistant judgments, not human ground truth. Original source annotations can
contain underspecified questions or misleading premises; reviewers preserve
uncertainty rather than forcing agreement with the reference answer.

The fixed selection has 128 training, 32 calibration and 32 development questions.
All variants stay together. Validation documents were reserved using only source
titles; reserved final question/answer content was not inspected. Source titles
and text hashes are disjoint across selected questions. Manifests pin source
bytes, exact cases, generation settings, model artifact and candidate output.

Targets: source **support**, answer-slot/type **completeness**, and question
**constraints** (restrictions and correct requested values). Unsupported coherent
claims fail support; genuine interpretation ambiguity is null. Wrong values of
the right type can be complete while failing support and constraints. A supported
restatement with the answer absent fails completeness, while constraints are
masked unless a restriction is explicitly violated. Correct brief answers need
not repeat their qualifiers. Ambiguous trailing names and malformed predicates
are masked; known name substitutions are not silently repaired. Null labels are
excluded from the corresponding loss, not treated as negatives.

Generation truncated inputs on 22 proposals; this is recorded on each candidate.
Assessor eligibility independently checks its complete question/evidence/candidate
input. No truncated assessor input is trained or admitted as fully covered.
This corpus is broader development supervision, not an untouched final benchmark
or evidence of general cognitive competence.

## Reviewed TRAIN-only augmentation

`augmented-training-manifest.json` binds a separate 527-row training corpus:
367 unchanged natural proposals, 128 reviewed evidence interventions and 32
independently reviewed near-correct contrasts. Its calibration and development
files are byte-identical to the original corpus. Rebuild with
`.development/experiments/prepare_quality_augmentation.py`; never use the authored
category, provenance, or reviewer rationale as model input.

`augmented-token-preflight.json` records tokenizer-only coverage using the frozen
native three-axis prompts: 490 eligible training candidates, 1,054 known axis
labels and 37 excluded over-budget candidates. At batch four, three epochs means
792 updates, versus 639 for the original native-only control. Unknown labels stay
masked. This comparison changes training data and update count; it is not a
matched-compute causal estimate. No model outcome or qualification is implied.
