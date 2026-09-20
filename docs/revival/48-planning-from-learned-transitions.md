# 48 — Planning from learned transitions

*2026-09-19. A bounded active bridge from retained experience to action selection,
following [45](45-evidence-backed-transition-learning.md) and the governing
[structured cognitive workspace objective](36-structured-cognitive-workspace.md).*

## What becomes active

The later [contingent planning milestone](56-contingent-planning-from-experience.md)
adds multi-step reachability over a finite empirical transition graph. It retains
all observed successors and executes one freshly checked step before replanning.
The induced-rule interface described here remains a separate one-step component;
neither interface supplies language-to-goal inference or learned state abstraction.

The agent can compare explicitly supplied actions using a sample-validated learned
transition model, retain the hypothetical alternatives, choose a sole supported
one-step candidate for an explicit target, execute it after fresh checks, and compare
its prediction with the independently observed result. A contradictory outcome
suspends the responsible rule instead of allowing the same supported prediction
to remain available indefinitely.

The caller supplies the desired outcome, candidate actions, feature/outcome
projection, and validation policy. The model induces associations between projected
observations, actions, and outcomes from recorded execution samples. It does not
learn those projection meanings or invent the action set or target. This is not
general multi-step planning, automatic experimental curriculum construction, or
planning from an arbitrary natural-language request.

[Empirical model investigation](55-empirical-model-investigation.md) adds a separate
use of these models: compare their forecasts for one common probe and assess which
supplied applicability hypothesis fits its observed result. Unlike this explicit-goal
planning path, it does not suspend a model globally when its proposed applicability
is contradicted, and it does not select a workspace meaning from the assessment.
The shared dispatch validator now also recomputes current rule membership and
correctness over retained fitted samples. Authentic sample IDs cannot authorize
an altered label or conceal counterexamples newly covered by a broadened rule.
The configured accuracy policy still applies; this does not force perfect accuracy
or convert empirical validation into a guarantee about future outcomes.

The previous transition-learning checkpoint returned predictions without an active
agent consumer. This checkpoint uses those predictions in a separate hypothetical
comparison and subjects the resulting choice to execution checks. Predictions do
not become observed facts or new capability effects.

## Agent API

The public entry points in [core.py](../../src/tensorcode/agent/core.py) delegate to
[experience_planning.py](../../src/tensorcode/agent/experience_planning.py):

```python
proposal = agent.propose_experience(
    model,
    observation_source_id,
    calls,
    desired_outcome,
)
execution = agent.execute_experience(proposal.id)
```

`observation_source_id` must identify the latest successful retained observation
from the model's provider. `calls` are explicit `Call` values belonging to that
mounted plugin, each naming a declared capability with exactly its declared
argument names. Duplicate calls are rejected. An `Unknown` desired outcome cannot
supply a target.

`ExperienceProposal` retains the source observation, model ID and revision, desired
outcome, each action's prediction or explicit unknown, selected call if any, reason,
and its own retained record source ID. The comparison performs no action.
Selection requires exactly one candidate with a supported prediction of the desired
outcome. Unknown predictions for other actions remain retained and do not veto that
choice: alternative actions are not mutually exclusive explanations of the world.
The decision means a sole supported target prediction in this comparison, not that
no other action could work. Multiple supported routes defer without a supplied cost
or tie policy; no supported route also defers. The agent does not silently break
ties by action order.

`ExperienceExecution` retains the proposal ID, receipt when available, verification
as a boolean or `Unknown`, associated observation source IDs, reason, and assessment
record source ID. The proposal registry is an in-memory execution mechanism;
retaining sources does not establish restart-safe recovery of pending proposals.

## Learned support has an action scope

The [transition model](../../src/tensorcode/learning/experience.py) now names its
identity and revision and gives each induced rule a stable model-local ID. A
`TransitionPrediction` identifies that model revision, rule, action family,
projection provenance, and empirical evidence. `snapshot()` exposes the model's
rule identities, action families, source IDs, projection provenance, and suspension
history for inspection.

An `ActionFamily` consists of the plugin name, capability name, and sorted argument
names. Support for one action family does not authorize another merely because an
authored projection produces identical features for both. Rule admission checks
training and evaluation support separately for each action family. Predictions
outside trained families abstain. The existing feature-domain and explicit-rule
checks also remain: missing or unseen projected evidence cannot use the inducer's
majority/default label as a fallback.

An action family constrains a declared call shape. It does not establish equivalence
of all argument values, freeze plugin behavior, or demonstrate causal effects.
Projected sample support remains conditional on the supplied representation and
observed workload. Separate training and validation identifiers do not by themselves
prove statistical independence.

## Evidence and freshness before dispatch

A supported prediction must cite training and evaluation attempts that resolve to
retained applied transitions from the model's provider. Their observation source
IDs must agree with the cited evidence. Matching IDs alone is insufficient: a caller
could otherwise fit invented labels while copying real execution identifiers.

Fitting retains a `ProjectedExample` snapshot of the actual features and outcome
consumed, together with attempt, provider, full action, action family, source IDs,
and split. The model exposes detached copies through `examples`.
`validate_transition(actual_transition)` reproduces the projection from a retained
applied before/after pair and compares it with that fitted snapshot. The bridge
requires this validation for the cited training and evaluation attempts before
proposal and again before dispatch. Missing snapshots, changed labels or features,
and mismatched full actions or linkage decline admission. An artifact constructed
without fitted examples does not acquire authority from its shape or source IDs.

Structural evidence comparisons distinguish booleans from numeric values; `False`
does not establish observed integer state 0. Validation checks the representation
actually consumed by fitting, not every raw field the authored projection ignores.
Consequently, fields outside the projection remain outside the model's claim.
Retained records are inspectable evidence, not tamper-proof execution attestations.

Execution consumes a retained proposal once under a per-proposal lock. Calling it again does not replay the
action, including after a declined dispatch. A fresh proposal is necessary after
a stale observation or other changed precondition.

Before invocation, the bridge rechecks the model identity and revision, prediction
currency, projection object, provider binding, declared capability, and retained
sample evidence. The agent's `_invoke(..., before_dispatch=guard)` hook runs the
guard after ordinary precondition checks and before the executor. Only an explicit
`True` licenses dispatch; an unknown, false, malformed result, or exception declines
it with a retained receipt and diagnostic event.

The guard captures fresh provider observations and compares them with the proposal's
original observation. Unavailable observations and changed payloads block execution.
This prevents proceeding on a detected change even if the old predicted outcome
would have been attractive. It also means irrelevant raw-payload changes can require
a new proposal; the current comparison is not a learned relevance test.

These checks do not create an atomic world transaction. A remote environment can
change after observation and before execution. Observation callbacks, projection
callables, and plugin implementations must honor their declared contracts.
Projection object identity does not detect mutable closures or external state that
change its meaning. A changed projection requires new fitting and validation.

## Observation, verification, and rule suspension

An applied receipt means that an action was applied, not that its predicted target
occurred. The bridge requires an independently retained successful `after_action`
observation for the provider, then applies the supplied contextual
`outcome(before, action, after)` projection to that actual transition.
[Browser transition learning](72-browser-transition-learning.md) uses this context
to authenticate the same acted-on entity across observations. Missing, failed, or unprojectable evidence produces an unknown
verification. The predicted outcome is never substituted for a missing observation.

The observed projected outcome is compared with the selected prediction, which
matched the supplied target during proposal. Agreement confirms this execution's
prediction under that projection. Disagreement reports a false verification and
feeds the observed counterexample and source ID to `model.observe_outcome(...)`.

A contradiction appends a `RuleSuspension` with predicted and observed outcomes,
source IDs, and justification, and advances the model revision. Predictions from
an earlier revision cease to be current. The suspended rule returns unknown on
later prediction rather than falling through to another rule or default. A new
fit and validation are required to authorize a replacement model. Incorporating the
known counterexample into that process remains the caller's responsibility; fitting
the unchanged old data does not resolve it. The exposed evaluation metrics remain
the historical validation results of the original fit. This checkpoint
does not automatically retrain from one counterexample or invent a repaired rule.

Matching observations do not suspend the model. Unknown outcomes cannot count as
contradictory evidence. The lower-level model API accepts caller-supplied observed
labels and source IDs; it is the active bridge that obtains the actual after-action
source and performs the projection. An arbitrary call to that lower-level API is
not independent evidence of a real execution.

## Scope of verification

[The agent integration tests](../../tests/test_agent_experience_planning.py) passed:
**9 passed**. The separate transition-learning and Gym run passed **35 tests**.
The complete repository suite passed **1,737 tests, with 5 skipped**, in 177.56 seconds.
The frontend chat workspace harness also passed its connection-scoping, media-preview,
reconnect, deduplication, and failed-read checks. These are mechanism checks, not general
cognitive performance scores.

The real Gymnasium workload uses `FrozenLake-v1` with `is_slippery=False` and a
renamed connection. Six rounds apply each of four explicit actions after a reset
to state 0, producing 24 step transitions: 16 training and 8 evaluation. The supplied
projection exposes integer state and action and predicts the next integer state.
After another reset, the agent compares actions 0 and 1 for the supplied target 4.
The induced model supports action 1; actual execution independently observes state
4 and verifies the prediction. Proposing the action neither mutates the environment
nor installs claims or capability effects.

This is heldout repetition at one known start state in a deterministic external
environment. It does not demonstrate maze navigation, broad state generalization,
learned visual features, or discovery of the target. The model abstains when state
4 is subsequently used as an unseen input. An unsupported alternative remains
unknown without vetoing the supported action.

Additional tests check foreign sample evidence, wrong provider binding, stale world
observations, changed capability declarations, detached returned records, consumed
proposal replay, and unavailable after-action observations. A changed real
environment transition produces a counterexample and suspends the learned rule.
Strict structural comparison prevents a boolean target from aliasing an integer
state. Separate model tests exercise action-family validation, revision history,
and prediction currency. The forgery regression fits invented next-state labels
using genuine retained Gym source IDs, then verifies that the bridge refuses the
proposal before action or belief admission. Real identifiers cannot launder
fabricated sample content into supported behavior.

## Next boundaries

This bridge connects experience to a real choice while preserving the difference
between prediction and observation. It leaves several required capabilities open:

- Multi-step hypothetical trajectories with uncertainty over intermediate states,
  explicit constraints, and validated composed action models.
- Task-led discovery of useful actions and experiments instead of caller-supplied
  candidate lists and sampling schedules.
- Learned state abstractions, visual scene organization, and grounding across
  modalities, beyond authored feature and outcome projections.
- Automatic hypothesis repair and independent evaluation after a counterexample,
  with dependency propagation into pending tasks and plans.
- Interpretation of user language into revisable goals without restoring lexical
  role aliases, implicit identity construction, or bundled domain recipes.
- Durable recovery of pending work and environment-level concurrency guarantees
  where external systems can provide them.

The appropriate claim is one-step action selection and verified feedback using
induced, sample-validated outcome associations. General cognition remains the
broader implementation objective, not a completed property of this checkpoint.
