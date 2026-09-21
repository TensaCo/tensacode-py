# Autonomous completion sequence

Owner request: continue autonomously through all pinned objectives. Main only,
verified milestone pushes, real-model work on GB10, artifacts under jacob-valdez
when promotion is justified. This sequence is execution work, not a completion
claim. Graph remains intentionally unimplemented.

## Active dependencies

1. **Evidence-conditioned useful chatbot.** Fix the measured single-source support
   bottleneck with explicit joint-evidence verification. Retain every source's
   contradiction veto, complete coverage, truncation abstention, source revisions
   and artifact identity. Evaluate source/joint with identical weights and policy.
2. **Response quality.** Expand natural reviewed training proposals on independent
   Hotpot training documents. Use 192 questions: 128 train, 32 calibration, 32
   development; seed 20260922. Reserve validation documents by title without
   inspecting reserved final question/answer contents. Keep all variants together.
   Preserve assistant label authorship, ambiguity and exact source evidence.
3. Train the existing three-axis assessor with full inputs, five epochs, AdamW
   2e-5, batch four, max512, seed20260922; no threshold search. Report every
   truncation exclusion. Temperature fitting uses calibration only. Compare
   train-majority, evidence-free and evidence-shuffled predictions. Ablated
   context changes support semantics; do not score them against unchanged labels
   as if they were fully reviewed truth labels.
4. Before integration, require each axis to reject at least half of its labelled
   development negatives while retaining at least 80% of positives; joint
   acceptance must retain at least half of all-known-good candidates with no
   known failures. Small-sample acceptance is provisional. Demonstrate evidence
   sensitivity on explicitly reviewed paired source interventions. Preserve
   failures and revise the approach instead of relaxing thresholds after results.
5. Integrate only qualified owned components, freeze the complete tool and run
   reserved final comparison with omission/revision/conflict controls. Judge
   actual answers using sources, not NLI/lexical approval. Include direct answering
   and identical-foundation workspace bypass.
6. Resolve workspace usefulness based on these controlled results; then implement
   and evaluate missing latent readout, multimodal relationships and action/outcome
   transfer. Schemas, authored scene graphs and simulations alone do not satisfy
   the objective. Pretrained tool releases require documented competent behavior.

## Validation and persistence

Use focused failing tests before code, exact artifact/optimizer continuation,
fresh-process inference, full suite/build/CI per coherent milestone. Store raw
training evidence and failures, update the big picture, and maintain clean main
between milestones. No arbitrary claim of “everything complete” while measured
capability gaps remain.

A separate foundation-capacity diagnostic compares the existing generator with
pinned `google/flan-t5-xl` revision `7d6315df2c2fb742f0f5b556879d730926ca9001`
on historical development questions only. Run direct answering and the same
proposal prompt before deciding whether more adaptation of the smaller model is
warranted. Any improvement is inherited foundation capability until controlled
training/workspace tests establish TensorCode-specific gains. This diagnostic
does not alter the v2 corpus's fixed original generator or data split.

## V2 result and next controlled interventions

The larger natural-proposal run failed all per-axis gates, and removing/shuffling
evidence barely moved support scores. Preserve v2 at `0a150f7`. Next compare only
input encoding (existing JSON sequence versus native tokenizer pair: question and
candidate / full evidence) using identical seed20260922, data and five-epoch
schedule. No threshold tuning. Separately prepare and review up to64 training-only
source-swap and omission pairs with the same questions/candidates. These are
explicit authored evidence interventions, not new natural outputs or human labels;
they cannot establish real-world correctness alone. No final cases are used.

## Current execution checkpoint

Trainable text/ViT OUTPUT_ENCODING now passes native-equivalence, gradient,
foundation and artifact tests, including ALBERT factorized embeddings and RoBERTa
position offsets found during independent review. Full suite: 699 passed, one
skipped; wheel and sdist build. This implements readout mechanics, not learned
shared semantics.

Paired assessor encoding reduced known-bad development approvals from 20 to 9;
adding 128 reviewed evidence interventions produced 12. Neither passes promotion.
The larger FLAN-T5-XL diagnostic completed all 32 historical cases without errors.
Direct answers appear stronger, while the existing declaration prompt often
copies source fragments; source-grounded review is in progress. Next test a
single simpler evidence-QA declaration prompt with the same foundation, then
measure the complete joint-verification tool. Keep prompt changes and foundation
changes separately identifiable. Do not access reserved final questions yet.

The larger foundation comparison is reviewed: base direct 18/32 correct, XL direct
29/32, XL old-prompt proposals 17/32, XL evidence-QA proposals 30/32. Reviews permit
clipped trailing prose when the answer is unambiguous; well-formedness is separate.
XL active/bypass first beams match; the workspace is untrained. Investigator now
owns prompt version 1 or 2, shared by generation and loss and retained in receipts.
Full XL + joint verification evaluation is running on GB10, with unchanged other
component weights and fixed policy, plus eight omission/revision controls.

Next diagnostic: frozen FLAN-T5-XL judgments for the existing reviewed quality
calibration/development candidates. Three fixed instructions assess support,
requested information and restrictions. Score the first decoder token conditional
on native `yes`/`no` token alternatives at threshold .5. No prompt search, fitting
or calibration; exclude and report any full-input overflow. Preserve original
per-axis/joint gates. These are inherited model judgments, not calibrated truth
or learned TensorCode quality. This investigates whether the stronger foundation
can assess question-conditioned quality that source-only NLI misses.

The frozen generative assessor also fails promotion (49 known-good and 14 known-bad
joint acceptances out of 91 development candidates). Next bounded learning test:
freeze FLAN-T5-XL, train only owned workspace/projection/gate on the 367 natural
training candidates' known axis labels as yes/no sequence targets. Three epochs,
batch four, AdamW .001, shuffle seed20260923; no threshold or prompt tuning. Use
ToolTrainer capture/replay and exact optimizer continuation. Compare final active
and bypassed workspace with identical frozen foundation/dtype/attention settings
on calibration/development; preserve failures. Foundation weights must remain
bitwise unchanged. This is a mechanism/capability experiment, not release approval.

Complete XL/joint run: seven correct, one incorrectly attributed answer, 24
abstentions. All eight omission controls abstain; six withdrawal controls echo
the withdrawal notice while not setting the formal abstention flag. Authored
conflict abstains. Oracle-corpus memory hit@1 is 1.0 (lexical .96875), not open
retrieval competence. No promotion. Full report and assistant reviews preserved.

Repeated fresh sessions previously rehashed all investigator weights, including
the multi-billion-parameter generator. The fingerprint cache is now shared per
Investigator under its existing session lock; evidence remains session-owned.
Regression tests verify zero extra tensor reads on fresh sessions, invalidation
on parameter updates and explicit invalidation across sessions.

Independent review found no blockers in the shared fingerprint cache; runtime
tests and manual configuration mutation, parameter replacement and concurrent
independent-session probes passed. Full local suite: 707 passed, one skipped;
wheel/sdist built. Source-only full-tool comparison is queued after workspace
training, using the identical saved XL generator, realizer, ranker and verifier.
