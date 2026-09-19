# 55 — Empirical model investigation

*2026-09-19. This extends investigation beyond caller-authored condition lists while
preserving the boundaries in [36](36-structured-cognitive-workspace.md),
[40](40-evidence-driven-interpretation.md), and
[54](54-investigation-with-pending-interpretations.md). Its question is which supplied
transition model fits a fresh response, not which interpretation a user intended.*

## What is learned, and what remains supplied

Earlier investigation required callers to supply each candidate's predictions as
`Condition` records. The new path obtains typed outcome predictions from fitted
`LearnedTransitionModel` instances. A proposed action is evaluated under every candidate's
model against the **same** retained observation. The predicted associations come from
actual recorded action/observation pairs and disjoint validation attempts, rather than
a hand-written expected-outcome list passed into investigation.

The caller still supplies the interpretation candidates, their correspondence to models
(`ModelApplicability`), the common observation/action projection, available probe actions,
and applicability justification. The current implementation does not discover latent
contexts, infer language-to-model bindings, invent measurements, or generate arbitrary
actions. These supplied pieces are part of the experiment, not learned achievements.

Candidate models must use the same provider and the same `Projection` object so their
outcome labels are compared in one measurement space. This requirement does not prove
that a mutable projection closure will stay semantically stable; that remains an explicit
caller contract. Supported predictions must resolve to retained applied transition samples,
and reprojection must agree with the actual training/validation evidence. The validator
recomputes rule membership, support, and correct counts for both splits against the current
learned artifact. Genuine sample IDs alone cannot validate a tampered rule label or
fabricated accuracy count.

## Retain forecasts before executing the probe

`Agent.propose_experience_investigation(group_id, bindings, observation_source_id, calls)`
retains the same-probe forecasts, model identities/revisions, evidence, candidate coverage,
and search policy before executing any probe. Every non-rejected candidate requires a
binding. Predictions that are unsupported or unavailable remain unknown; omitting a rival
cannot manufacture a unique answer.

Each candidate retains an empirical set of typed outcomes, with separate training and
validation attempt IDs/counts for every outcome. The learned point label remains in the
artifact but is not treated as an exclusive forecast: a minority outcome observed under
the same supported rule must not eliminate that model. These sets include outcomes from
**both training and validation samples**. Consequently, validation is no longer independent
of forecast-support construction; only the fresh operational probe is separate from that
construction. Original point-classifier validation metrics retain their narrower meaning.

The authored probe policy compares possibly overlapping outcome sets. It minimizes the
largest number of candidates surviving an outcome, including unknown candidates that
survive any response, then minimizes unknown predictions. It is not calibrated information
gain or learned attention. Equal support sets do not discriminate. A tied best probe
requires an explicit choice; list order does not authorize an action.

`Agent.execute_experience_investigation(proposal_id, call=...)` performs at most one probe
through the existing capability/receipt/observation path. It checks the retained model,
candidate group, continuation, provider, capability, and fresh observation before dispatch.
Guards run again after observation callbacks; provider/capability/model/group drift during
post-action projection also suppresses the applicability diagnostic while preserving the
actual receipt and assessments. A stale callback cannot certify applicability. The
proposal is consumed once; repeating its execution cannot repeat the action. Proposal
creation itself does not grant automatic execution in the chat loop.

After the action, one common projected observation is compared with every retained
empirical support set. “Confirmed” means compatible with an observed support outcome,
not probabilistically established or causally proven. Unknown candidates remain viable.
An unexpected response can contradict all
the supplied hypotheses; it does not force a fallback winner. Pending interpretations or
stale investigation inputs prevent a unique-applicability diagnostic.

Even a uniquely supported applicability candidate does **not** select the workspace
interpretation, assert a belief, or rewrite a task. The result deliberately reports model
applicability rather than user intent. A contradiction in the current context also does
not globally suspend a rule trained for a different context. Revising context bindings
and learning from such counterexamples remain separate work.

## Reproducible authored environment

The bounded audit uses an explicitly authored hidden-wiring simulator. Its observable
state contains a lamp value; actions press numbered buttons. Two experimenter-selected
wiring contexts produce different responses to the same button. The hidden context is
excluded from the model's observation features. The experimenter supplies the context
partition and a lamp/button projection; the fitted models induce response associations
from executed transitions. This is measured learning in a supplied small environment,
not discovered physical concepts or understanding of unstructured text or images.

The [recorded audit](../../eval/results/model_investigation.json) is reproduced with:

```sh
.venv/bin/python -m eval.learning.evaluate_model_investigation
```

Each context contributes twenty training attempts: eight button-zero attempts and twelve
button-one attempts. Each also contributes four distinct validation attempts, two per
button. Training and validation attempt IDs are disjoint; reset actions are executed and
recorded but excluded from the fitted sample partition. Setup performs 97 applied actions
in total: 48 sampled attempts, 48 accompanying resets, and one final reset before the fresh
probe. The two context models are fitted separately using the same projection object.

| Measurement | Context 1 model | Context 2 model |
| --- | ---: | ---: |
| Training / validation attempts | 20 / 4 | 20 / 4 |
| Validation predictions / all validation attempts | 2/4 | 2/4 |
| Correct / predicted validation outcomes | 2/2 | 2/2 |
| Forecast for the same fresh button-one probe | lamp 1 | lamp 2 |
| Fresh observed response in context 2 | lamp 2 | lamp 2 |
| Candidate assessment | contradicted | confirmed |

Validation coverage is **50%**, with 100% accuracy conditional on making a prediction.
The two unsupported validation cases per model are not successes. The supported
button-one prediction has twelve training and two validation examples in its retained
rule evidence. These repetitions occupy a tiny deterministic action distribution; the
validation partition is not evidence of transfer to new environments or new concepts.

The forecast record existed before the fresh probe observations. Proposal creation
performed no action; execution applied exactly one probe; replay of the same proposal
reported `proposal_already_consumed` without executing again. The result's
`supported_candidate_id` identifies the supplied context-two applicability hypothesis.
The workspace group remained unchanged and unselected, and both model snapshots remained
unchanged: neither model was globally suspended because of the other's context.

All eight checks passed in each of the two measured cases, and all recorded source hashes
agreed after the run. The report contains the learned rule conditions, labels, support,
correct counts, defaults, model identities/revisions, full projected fit examples, rule evidence,
retained forecast, actual observation sources, and the replay result. It does not contain
a downloadable pretrained model artifact; these small models are fitted locally during
this reproducible run. Deterministic setup/training took 0.688 seconds and its fresh probe path took
26.47 milliseconds on the shared host. These timings are descriptive, not a scaling claim.

An initial run reached JSON export but failed because the audit serializer did not yet
handle receipt timestamps. The serializer was corrected and the same experiment rerun;
no model or outcome tuning was performed to fix serialization. Subsequently, independent
audits found and repaired acceptance of tampered rule labels and elimination of known
minority outcomes. The final report replaces the earlier pre-repair report and reruns the
deterministic case plus the explicitly authored noisy case against frozen final sources.

## Measured overlapping support

A second measured case replaces the first context model with a noisy model fitted from
an authored execution schedule. Its button-one training outcomes are thirty lamp-one and
ten lamp-two observations, with six and two corresponding validation observations.
Another eight training and two validation baseline/button-zero attempts give totals of
48 training and ten validation samples. The caller's validation threshold is 75%.

The induced rule is `button == 1 → label 1`, with forty training matches and thirty
correct point-label predictions. Its validation point prediction covers eight of ten
samples (80%) and is correct on six of those eight (75% conditional accuracy). Its
empirical support is `{1, 2}`, with counts retained by split. The second model's support
is `{2}`. The fresh response is lamp two, which is compatible with **both** models; both
assessments remain viable and `supported_candidate_id` is absent. The majority label does
not erase the observed minority outcome.

All eight mechanism checks pass again: one probe, consumed replay, retained forecast before
observation, unchanged workspace, and no model suspension. Setup includes 214 applied
simulator actions, including the original model preparation and resets. Setup/training
took 0.237 seconds and the probe path took 57.22 milliseconds in this run; the order and
shared process can affect timings. This is a test of retained empirical uncertainty,
not a calibrated probability estimate or a claim of broad noise robustness.

## Conservative behavior and remaining gaps

Nine focused integration tests cover fitted forecasts, equal support, an unsupported rival,
unexpected outcomes, explicit tied-probe choice, exact applicability coverage, and noisy
outcome overlap. The final runtime suite passed **1,943 tests, with five skipped**, in
274.68 seconds. These counts verify mechanisms and regressions, not general cognition.

This adds an empirical source of candidate-specific predictions, rather than requiring a
caller to spell out their outcome conditions. The applicability correspondence and context
partition are still supplied. A world response that supports one transition model does
not establish that the user intended its associated language interpretation. Learned
context discovery, language/vision grounding into these models, richer uncertain models,
probe invention, and justified workspace commitment remain open work.
