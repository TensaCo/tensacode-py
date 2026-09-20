# Learned measured-action goals

[Learned document tasks](73-learned-document-tasks.md) required the caller to supply
an action proposal and desired post-activation measurement. This milestone develops
a learned structured-frame projection that retains those intentions as a typed
goal. After explicit goal selection, the runtime materializes the authenticated
browser target and pursues the measured outcome. The pursuit caller no longer
needs to supply the action proposal or desired value directly.

This is bounded learning from teaching examples, not an independently understood
English request. Structured frames, teaching intentions, grounded references,
measurement meanings, and selection policies remain supplied. The improvement
must be demonstrated by actual learned intent transfer and execution, not simply
by adding a goal record.

## Intent without runtime identity

`goals.MeasuredActionGoal(target, operation, measurement, desired_outcome,
basis=())` records an explicit target Ref, operation name, measurement name, and
finite typed scalar or tuple desired value. It does not store a browser connection, capture ID, opaque target
token, action proposal, or other transient execution identity. The goal describes
what measured effect the taught operation is intended to achieve for that target;
authenticating how to act on the current browser remains a later step.

The goal-correspondence learner derives exact structured-frame-to-intent mappings
from retained teaching. Reference substitution generalizes supported variable
identities while preserving explicit fixed context and exact non-reference
structure. It does not learn synonyms, arbitrary operation vocabulary, measurement
ontologies, or transformations of desired values merely by varying target Refs.
Training/held-out evidence and unvalidated rivals must remain inspectable.

## Selection, adoption, and materialization

A learned intent is proposed in the goal interpretation workspace. Explicit goal
selection remains required; one available proposal does not authorize itself.
Adoption retains the chosen goal and its teaching/model/reading commitments so
later changes can invalidate an obsolete task rather than silently reinterpret it.

Only after selection does the document runtime materialize the target using the
current authenticated capture and selected graph/reading. Unsupported operation
or measurement combinations must not be coerced into the available browser
operation. Runtime identities are derived from current retained evidence, not
learned from training tokens or copied from a teacher's old browser session.

The materialized intent feeds the existing empirical prediction, guarded dispatch,
actual outcome assessment, and task ledger. A prediction is not observation, an
applied receipt is not completion, and a model or interpretation change must remain
able to block the action. No direct desired-outcome or hand-assembled action
proposal is needed at this pursuit boundary.

A selected measured goal reaching `Agent.request` is retained with a suspended
`measured_goal_requires_materialization` outcome; it does not automatically run
lexical/domain refiners or act. Existing explicit goal adoption can install the
chosen measured goal on the retained task.

`agent.measured_document_tasks.pursue_measured_document_task(agent, provider,
model, *, task_id, language_group_id, candidate_id, path, document_group_id,
document_candidate_id)` supplies runtime connections and retained context IDs.
The selected goal supplies the target, operation, measurement, and desired outcome.
The current exact supported contract is `activate_node` with the authenticated
browser target `inputChecked` projection; other names are not coerced into it.

Materialization verifies the chosen reading is the selected goal's parent, the
runtime target equals the goal target, and the transition model implements that
measurement. It retains a separate realization source and runtime action plan
while preserving the declarative `MeasuredActionGoal` on the original task and
revision. Selected goal, reading, graph, model, and task dependencies are checked
again during pursuit. The real-browser and full-suite results below verify this
bounded integration.

## Acceptance evidence and remaining work

Four real-browser cases in `tests/test_browser_measured_goals.py` passed in
28.07 seconds. Authentic explicit teaching on three grounded document
examples supplies two correspondence-training targets and one held-out target,
without lexical bootstrapping. An independently learned browser outcome model
uses nine actual episodes. A fresh grounded target receives a learned measured
goal, which is explicitly adopted before materialization.

The pursuit call supplies no desired outcome, operation, measurement, or action
proposal. The positive case completes from actual observed feedback. A
`preventDefault` counterexample remains unverified and suspends the transition
rule; withdrawing the goal model blocks action. The direct `Agent.request` case
retains a suspended declaration without action, then completes through authenticated
materialization on the same task revision. Both suspended and done attempts remain
in history. A separate backend focused run passed 53 tests in 6.16 seconds; these
are not additive to earlier overlapping worker runs. The final materializer run
passed 27 focused tests in 5.12 seconds, including the actual `Agent.request`
handoff. Final read-only review found no additional concrete failure. The full
suite passed **2,526 tests with two skipped in 447.60 seconds** on the final
runtime and test files.

The test's page, initial frames, teaching labels, reference grounding, operation
and measurement vocabulary, and explicit selections remain authored. The learned
components are the demonstrated correspondence and empirical outcome associations
used downstream. This is not free-English goal understanding, autonomous meaning
selection, general browser affordance learning, or pixel perception.

Remaining gaps include learning broader structured transformations, interpreting
new unstructured descriptions, learning which measurement expresses success,
selecting among competing goals, and planning multiple actions under changing
identity. Separating intent from runtime identity enables accountable transfer;
it does not by itself supply those cognitive capabilities.

## Next unstructured-input boundary

The learned goal correspondence begins with a supplied structured frame. It does
not remove the active speech-act shortcuts in `language/deps_semantics.py`:
`Reader._read()` uses punctuation and prefixes for questionhood, while
`Reader.speech_act()` assigns question roles and treats subjectless frames as
requests. `LearnedReader` still installs that authored semantic reader. Thus
learned segmentation, tags, dependencies, grounding, and goal correspondences do
not together prove that the input's communicative intent was inferred.

The next replacement should retain source-anchored syntax without granting it
request authority, learn communicative interpretations from explicit retained
teaching, and preserve competing interpretations or abstention outside validated
support. A subjectless non-request must remain non-actionable through the actual
reader path. UD syntax and morphology can support syntax learning; they are not
user-intent labels. Removing the speech-act fallback is part of the governing
objective, not an optional compatibility path.
