# Active grounding investigation

[Relational grounding](63-learning-relational-scene-grounding.md) learns queries
that associate an exact structured description with scene referents. Correlated
teaching examples can leave several queries equally compatible: a target might
always carry both a color marker and a shape marker, without establishing which property matters.
This milestone asks which offered scene would make those retained hypotheses
disagree, retains their predictions before feedback, and uses explicit new
teaching to refit the model.

The new capability is investigation over learned query hypotheses. The scenes,
descriptions, and teacher alignments remain supplied; the agent does not generate
an image, infer a graph from pixels, or understand a conversational clarification.
The scene-choice policy is authored. Its input denotations are computed from the
learned queries rather than supplied as a separate hypothesis table.

## Predictions before feedback

`learning.grounding_investigation.investigate_grounding(model, description, scenes)`
applies every retained exact-description query to every offered scene, including
queries lacking successful validation or carrying conflicting validation evidence.
Those rivals must remain visible during investigation even though they cannot
publish ordinary grounding alternatives. Each prediction retains its full referent
set, query ID, training/validation/conflict IDs, validation status, completeness,
unresolved reasons, and matching work count. The empty referent set is a possible
prediction; a partial search result is not an exact empty answer.

For each scene, queries are partitioned by identical complete referent sets. A
scene is discriminating when more than one partition remains. The policy minimizes
the size of the largest partition, counting each retained query once. This is an
authored worst-case disagreement heuristic, not a learned probability distribution,
a calibrated information-gain measure, or evidence that equally numerous query
forms are equally likely. All equally best scenes remain choices; their order
does not select the first one. This partition score assumes a hypothetical
complete denotation answer. Partial positive/negative teaching checks compatibility
only and need not eliminate an entire partition; no worst-case elimination
guarantee is claimed for partial feedback.

Incomplete model search or any incomplete offered-scene denotation suppresses the
preferred-scene recommendation. Partial predictions and reasons remain inspectable.
If every query agrees across all offered scenes, the result reports no discriminating
scene. This does not establish that no other scene could distinguish them.

## Retained investigation and correction

The agent boundary authenticates the admitted model and exact language/scene
snapshots. Only offered scenes novel relative to training and held-out scene
and entity identities are ranked; excluded offers remain recorded. Predictions
are retained before a teacher supplies feedback,
so the later alignment can be compared with what each query actually forecast.

The public sequence is preparation, explicit teacher feedback on an
offered novel scene, and an explicit refit. Feedback retains positive and optional
negative referents with a stated basis. Explicit negative-only feedback is
allowed; both label sets empty is rejected. Negative-only examples constrain
queries but do not provide positive training corroboration or positive held-out
support. Unlabeled referents are not negative answers. The new example joins training; the existing held-out partition and
model search bounds remain unchanged. The refit appends a model version to the
same model group and withdraws its earlier admission. It returns an unadmitted
model, requiring a new explicit admission before later grounding.

Historical predictions, teaching, and model versions remain attributable. A
counterexample changes later inference through fitting, without rewriting the
original prediction or silently altering the meaning commitment of an existing
task. Tasks depending on the old admission must not continue under the new model
merely because its description string is the same.

The functions in `agent.grounding_investigation` are:

- `prepare_grounding_investigation(agent, admitted_handle, language_group_id,
  candidate_id, path, scene_candidates)`, where `scene_candidates` is a tuple of
  explicit `(group_id, candidate_id)` pairs. It returns a retained proposal with
  the pre-feedback investigation and evidence source ID.
- `record_grounding_feedback(agent, proposal, scene_id, positive_refs,
  negative_refs=(), *, basis)`, where `scene_id` is an eligible offered scene's `Ref`.
  The caller names a scene explicitly, including a nonbest or nondiscriminating
  scene if desired. Scores are recommendations, not permission to teach. Global
  search and the chosen scene's predictions must be complete; excluded scenes
  with training/held-out identity overlap cannot supply feedback. Its result
  retains the teaching
  record and `confirmed_query_ids`/`contradicted_query_ids`; “confirmed” means
  compatible with these explicit labels, not independently established truth.
- `refit_grounding_from_feedback(agent, feedback)`, which returns the new
  unadmitted model handle. Subsequent `admit_grounding_model` and grounding remain
  separate operations.

These operations can return `Unknown`. Authentic proposals and feedback are
reserved before callback-bearing work; repeated recording or consuming the same
feedback is refused. Failed reserved attempts are recorded as failures rather than
implicitly replayed. If publication occurs before a later evidence check fails,
the new model candidate is rejected; earlier evidence is not erased.

## Acceptance evidence and limits

The verified integration fixture teaches correlated color/shape relations,
offers a crossed scene where the learned queries separate, and supplies explicit
teacher alignment for the color-defined target. The refit removes
queries inconsistent with that alignment while preserving other compatible
hypotheses, then grounds a target in another scene. This milestone uses an explicit
authored goal projection/refiner to isolate grounding changes and task authorization.
Actual device execution through `Agent.request` before and after correction is the
final mechanism check. The old task loses authorization after refit, and the
new target follows the newly supported query. This does not turn the supplied
scene graphs or labels into inferred perception. The earlier doc63 tests separately
verify composition with learned goal correspondence; that is not the goal mechanism
claimed for this correction fixture.

Verification: **68 focused tests passed in 5.47 seconds**: four new agent
integration tests, fifteen investigation-wrapper tests, seven pure investigation
tests, sixteen grounding-learner tests, twenty existing grounding-wrapper tests,
and six existing grounding-integration tests. The crossed scene separates
three prior denotations: no target, the color-marked target, and the shape-marked
target. Feedback preserves the prior prediction source, appends a third training
example while retaining the one held-out example, and removes contradicted query
structures. The actual device fixture performs exactly two actions: the original
request and the corrected transfer request. Rechecking the old task after refit
performs no action. Further integration tests explicitly choose among tied scenes
and accept teaching on a nonbest scene.

The fourth integration case preserves the conjunction's empty prediction beside
the single-feature matches, blocking ordinary binding publication. The teacher
explicitly excludes both crossed-scene nodes with no positive label. Refit removes
the single-feature queries, retains the learned conjunction, still predicts no
target in that crossed scene, and identifies a jointly marked target in a new
transfer scene. This is explicit negative evidence, not an assumption that
unlabeled nodes are wrong. Evidence-wrapper tests exercise authentication and
revision boundaries. The full repository suite passed **2,300 tests, with five
skipped**, in 341.43 seconds. These tests are not a semantic accuracy benchmark.

Remaining gaps include autonomous scene acquisition or construction, asking a
natural-language question whose answer is reliably interpreted, learning which
experiment costs matter, correcting erroneous teachers, and extending beyond exact
structured descriptions and bounded conjunctive graph queries. Feedback can reject
an incorrect retained query without guaranteeing that the correct query exists in
the searched class. Investigation, explicit feedback, refitting, readmission, and
later execution remain separate caller-driven steps. These records are in-memory;
this milestone does not establish restart persistence or an autonomous learning loop.
