# Browser transition learning

[Grounded browser actions](71-grounded-browser-actions.md) authenticated a selected
DOM target and performed an explicit activation. This milestone learns outcome
associations from retained browser transitions, predicts a later target's observed
state before acting, and suspends a contradicted rule using authenticated feedback.
It does not author a checkbox toggle rule or infer an operation from language.

## Contextual outcome projection

`learning.experience.Projection` now accepts `features(before, action)` and
`outcome(before, action, after)`. Outcome projection can use the original action
context to identify which observed entity to measure after the action. Fit,
evidence replay, and verification pass detached contextual inputs. The projection
must still read an actual outcome; context does not license copying a predicted
label into the observation.

`agent.document_transition_evidence.browser_transition_projection()` supplies a
literal CDP decoder. Its input features are the acted-on node's `nodeName`,
`nodeType`, literal `type_attribute`, and `inputChecked`. Its outcome is that same
node's observed post-action `inputChecked`. No activation effect or toggle formula
is embedded in the projection. Feature choice and the measurement ontology are
authored; associations between those inputs and outcomes are learned.

The action's opaque target token is resolved through issued target evidence,
not by treating a label, index, or arbitrary Ref as a live identity. Both snapshots
must correspond to the same authenticated connection/session, document/frame
identity, and exact backend node. Missing, replaced, ambiguous, forged, or
unprojectable correspondence cannot become a training label.

`inputChecked` is decoded according to its supplied sparse Boolean format: absence
from a valid index list means false within that measurement. Missing or malformed
measurement data is not false. A CDP attribute-value index of `-1` is retained as
`None`, not guessed to be an empty string; attribute names still require valid
string-table indices. This local schema rule does not close the world
for arbitrary scene propositions or establish the target's semantic role.

## Retained fitting and prospective evidence

The public functions in `agent.document_transition_evidence` are:

- `retain_document_transition_batch(agent, provider)` collects eligible applied
  transitions, authenticates independently issued before/after observations, and
  retains accepted rows plus explicit exclusions.
- `fit_document_transitions(agent, provider, retained_batch, *, train_attempt_ids,
  evaluation_attempt_ids, policy=ValidationPolicy())` authenticates retained
  evidence and fits under an explicit document-disjoint split.
- `predict_document_transition(agent, provider, model, target_token)` records a
  fresh authenticated before-context and prediction without observing a future
  result or executing the action.
- `observe_document_transition(agent, provider, model, retained_prediction,
  attempt_id)` checks the matching actual applied attempt and observed context,
  retains feedback, and passes the actual projected outcome to rule suspension.

Different attempt IDs do not establish independent documents. Training and
validation cannot reuse the same connection/session/frame/loader identity, even
with different target tokens. The browser provider authenticates its issued raw
observations and target correspondences; a caller cannot establish provenance by
constructing a lookalike payload. Reusing a provider-issued observation ID across
attempts must not inflate independent support; those repeated observations are
excluded for all affected aliases rather than retaining a conveniently counted
copy. Authentication is process-local, not a portable
signature or restart-persistence guarantee.

Predictions retain their model and source basis before the outcome. Feedback
requires the actual action and corresponding before-features to agree with that
basis. A contradictory observed outcome suspends the supported rule; an unknown
or unpaired observation is not a valid counterexample. Retained predictions and
feedback are not silently replayed under changed evidence.

## Explicit supported residual rules

The decision-list learner can leave observed examples unmatched by its induced
partial conditions. A majority/default label is not permission to predict those
cases. The new `ResidualRule` can retain a sufficiently supported, pure observed
residual feature combination as an explicit conjunction. It requires the exact
feature-name set and typed values, and remains subject to the usual validation
and evidence requirements.

Authority also requires replaying the original projected support, rather than
merely authenticating raw observations. A residual certificate must check the
exact feature population it claims to cover; labels, counts, or support IDs cannot
be edited into a different supported rule without detection. Added, removed, or
changed features invalidate that exact-population certificate. The document
bridge retains and checks model rules, projection identity, projected examples,
policy, and revision across callbacks before prediction or feedback authority.

This is conservative memorization of a supported combination, not a general
fallback. Different feature combinations, missing fields, mixed residual outcomes,
or insufficient support do not acquire a default prediction. Such a rule must
remain visible in the learned artifact and trace, with its actual training and
held-out evidence.

## Acceptance evidence and limits

The evidence/learning checks passed **51 focused tests** after the final
rule/projection authority, duplicate-observation, residual-certificate, and CDP
attribute checks. The complete repository suite passed **2,449 tests with two
skipped in 373.91 seconds** on the final runtime and test files.
The real-browser test passed **one test in 5.19 seconds**. It uses nine fresh document
navigation episodes: two training and one held-out example for each of unchecked
checkbox→true, checked checkbox→false, and ordinary DIV→false. It then records
three new target predictions before activation and compares actual paired
feedback, with all three predicted outcomes confirmed. A page with a
`preventDefault` handler supplies a real counterexample: the model predicts true
but observes false. Feedback suspends that rule, and the next fresh unchecked
checkbox prediction abstains. These are supplied experimental classes,
not labels used as authored transition rules. Predictive features contain no
target token, backend node ID, loader ID, or capture ID; those identifiers serve
authentication and splitting rather than an outcome lookup.

Page content, operation choice, literal decoder, feature selection, split policy,
and explicit target/meaning choices remain authored. The learned component is the
outcome association within that supplied scope. This is not learned browser
ontology, causal identification under arbitrary hidden state, a learned goal
mapping, or general cognition. A successful prediction about `inputChecked` does
not prove user intent or completion of a broader task.

Remaining gaps include richer observed effects, hidden-context separation,
probabilistic outcomes, learning the relevant features, temporal identity across
navigation, and selecting actions to achieve inferred goals. Process-local
authentication and contextual measurements improve the reliability of evidence;
they do not solve those representation and planning problems.

The immediate integration gap is a dependency-bound task runner. The current
browser evaluation explicitly calls prediction, activation, and feedback in
sequence; a learned prediction does not yet gate `execute_document_action`, and
feedback does not complete a retained task. The next step must bind an explicit
desired measured outcome to the selected grounded reading, model, and task
revision; recheck that basis immediately before dispatch; and record completion
only from the actual matching outcome. Unsupported predictions must leave the
target untouched, and contradictory observations must suspend the rule without
claiming task success. This integrates the learned execution loop without
pretending to infer the desired outcome from arbitrary language.
