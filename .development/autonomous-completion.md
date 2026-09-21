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
