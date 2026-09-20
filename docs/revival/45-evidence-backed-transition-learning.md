# Evidence-backed transition learning

This implementation connects retained action observations to the existing rule
inducer. It follows the [structured workspace objective](36-structured-cognitive-workspace.md):
experience should improve inspectable models without authored interpretations being
reported as learned understanding.

[Checkpoint 55](55-empirical-model-investigation.md) uses these fitted predictions
to compare candidate model applicability through the same actual probe. The
associations are induced from retained executions; the candidate/model bindings
and measurement contract remain supplied. A mismatch under uncertain applicability
does not automatically suspend a rule that may remain valid in its training context.

## Active behavior

`tensorcode.learning.experience.extract_transitions` consumes
`agent.interpretations.sources()` for an explicitly selected `plugin:<name>` provider.
It pairs `before_action` and `after_action` observation sources by their execution
`attempt_id`. Each admitted transition retains both source IDs, provider, detached
raw observations, action, and receipt.

The pair must contain exactly one before and one after source, both observed, with
an applied receipt and identical action linkage. Unavailable/error observations,
incomplete or duplicated phases, absent attempt IDs, and nonapplied receipts yield
explicit exclusions. Rejected, failed, and indeterminate attempts are not silently
labeled successful transitions. Duplicate source IDs are rejected. Structural action
comparison supports array-valued arguments without ambiguous array truth tests.

An applied receipt establishes that execution occurred. It does not establish
that the user's goal succeeded or that every observed difference was caused by the
action. The learned label describes an observed transition under the caller's
projection, not a controlled causal effect.

## Authored projection, learned associations

The caller supplies a named `Projection`:

- `features(before, action)` converts raw observations and an action into discrete
  feature values.
- `outcome(before, action, after)` selects the observed target to predict using
  the original action context. The earlier after-only signature is retired;
  [browser transition learning](72-browser-transition-learning.md) explains why
  target-relative observation requires this context.
- `provenance` identifies this supplied interpretation. `kind` is explicitly
  `authored` in this implementation.

The system does not learn these feature meanings or discover the outcome ontology.
It reuses `candidate_literals` and `decision_list` to induce conditions and outcome
associations from projected training examples. It retains the resulting inspectable
rules separately from their empirical support.

`fit_transitions` requires explicit `train_attempt_ids` and
`evaluation_attempt_ids`. Both must be nonempty and unique, form a disjoint partition
of the supplied transitions, and retain distinct observation source IDs. Mixed
providers and duplicate attempts are rejected. Evaluation examples never enter
candidate-literal generation or rule induction.

## Prediction and validation

`ValidationPolicy(min_training_support=2, min_evaluation_support=1,
min_accuracy=1.0)` declares support and accuracy requirements. These are caller
policy defaults, not learned confidence calibration.

Each rule records its training and evaluation attempt IDs, source IDs, correct
counts, and reasons for failing validation. `LearnedTransitionModel.predict` returns
a `TransitionPrediction` with that evidence or an explicit `Unknown`.

Only explicit induced rules with sufficient observed support and validation can
answer. The inducer's majority/default label never becomes an answer. Missing
features cannot satisfy negated conditions. Previously unseen training-feature
values cause abstention rather than uncontrolled extrapolation. Projection failures
and unverified rules also abstain. A detached `artifact` property permits inspection
without allowing edits to that returned copy to change the model.

The model exposes selective evaluation coverage and accuracy. Accuracy is `None`
when there are no admitted predictions. Evaluation gates rule admission, so these
numbers describe a validation sample; they are not an independent final test score.
A separate future trajectory provides stronger assessment after admission.

## Evidence exercised

`tests/test_learning_experience.py` executes a small latch simulator across input
combinations and collects before/after observations. A supplied projection exposes
state and action features; the inducer learns rules from those transitions. Tests
check predictions against a fresh simulator execution, source provenance, heldout
contradictions, insufficient validation support, missing and unseen features,
majority-default abstention, split leakage, source aliasing, action mismatch,
array-valued actions, unavailable observations, and nonapplied receipts.

These are executable authored fixtures, not learned perception or a learned policy.

`tests/test_gym_connection.py` additionally exercises a real Gymnasium CartPole
connection through `Agent._invoke`. The test collects actual reset/step sources over
48 episodes, fits from 32 training and 16 heldout episodes, and checks supported
predictions against actual steps on independent future seeds. The measured
validation accuracy is 1.0 at coverage of at least 0.5 for this narrow workload.
The velocity-sign projection, action feature, selected actions, and seed/split
schedule are supplied by the test. The agent learns the conditional associations;
it does not learn those features or an action-selection policy. No prediction is
silently installed as a belief or executable action model.

## Remaining work

The pathway supports discrete, explicitly projected observations. Continuous values
need an explicit projection; the model currently abstains on feature values absent
from training. It cannot induce new perceptual concepts, shared visual identities,
relational graphs, or arbitrary transition functions.

Observation provenance is a retained execution record, not a tamper-proof proof.
Callers constructing `Transition` records themselves are responsible for their
sources. Split validation rejects shared attempt/source identifiers, but separate
IDs cannot prove statistical independence. Related episodes, reused environments,
and preprocessing selected after seeing evaluation outcomes can still leak
information. Repeated model selection requires another untouched test set.

The module does not automatically collect new training curricula, choose experiments,
promote predictions into capability effects, or update policies during execution.
Integrating those steps must preserve uncertainty, empirical provenance, explicit
model revision, and the distinction between predicted and observed effects.

Projection callables must remain semantically stable between fitting and prediction.
Python functions may read mutable closures or external state; copying a record or
retaining a name/provenance string does not freeze those dependencies. The current
model does not detect such semantic drift. Validation applies to the projection
actually used during fitting. Changing it requires fitting and validation again;
reusing a model after an undetected change is unsupported.

## Model identity, action contracts, and observed counterexamples

Each fit now creates a distinct model identity and stable rule identities. Read-only
`id`/`model_id` and `revision` properties identify its current state;
`snapshot()` retains rule IDs, projection provenance, observation sources, supported
action families, and suspension history. Predictions carry the model identity,
revision, rule identity, and evidence used. `is_current(prediction)` rejects old
revisions, suspended rules, foreign models, or altered prediction content.

An `ActionFamily` comprises the plugin, capability, and exact set of argument names.
These contracts come from training attempts only. Prediction rejects unseen families
before calling the feature projection, even when that projection omits action
identity. Each rule requires training and heldout support within the action family
being predicted; support from one family cannot validate another. Argument values
remain the supplied projection's responsibility, rather than part of family identity.
The current mechanism does not prove that the projection retained every causally
relevant input.

The feature guard rejects unseen feature names and values but does not require an
entire feature tuple to have appeared during training. An induced condition can
cover a previously unseen combination of individually observed feature values.
Tests demonstrate this on a fresh combination in the executed latch fixture.
This is bounded generalization under an authored projection, not a learned ontology.

`observe_outcome(prediction, actual_outcome, source_ids=..., reason=...)` compares a
retained prediction with an explicitly observed, projected outcome. Agreement leaves
the model unchanged. A contradiction appends a `RuleSuspension` containing source
IDs, reason, predicted and observed values, and the new revision. The matching rule
then returns `Unknown("suspended_rule")` for future predictions. A suspension applies
to the rule across its action families, conservatively preventing reuse of its
contradicted association. Other rules remain available at the new revision.

Unknown outcomes cannot count as counterexamples. Foreign or altered predictions,
empty source provenance, and missing reasons are rejected. The method trusts the
caller to provide actual observation evidence; it does not independently fetch or
validate source records. The agent planning bridge performs that source validation.
Existing empirical validation metrics describe the original heldout set, and are
not recomputed or portrayed as current accuracy after suspension.

There is no unsuspend or implicit retraining operation. A new explicit fit with
separate training and heldout attempts produces a new identity and fresh validation.
The original model retains its history. Reusing the same old samples in a new fit
is technically possible but does not resolve a known counterexample: the caller
must include or otherwise account for that evidence when claiming an improved model.
