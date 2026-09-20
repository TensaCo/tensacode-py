# Learning relational scene grounding

This milestone addresses a missing link between structured perception and goals:
learning which scene referents a structured description denotes. Earlier grounding
accepted an authored `MentionBinding`. Here supervised scene examples induce a
relational query, and that query proposes bindings in a new scene. The caller does
not supply the query or a prediction-time reference lookup.

This is a bounded graph-learning capability. Scene graphs, structured descriptions,
and positive/negative teaching alignments are supplied. It does not infer a graph
from pixels, interpret an unfamiliar phrase, or establish the user's intended target.
It extends [learned goal correspondences](62-learning-goal-correspondences.md) toward
the [structured cognitive workspace objective](36-structured-cognitive-workspace.md)
without treating an object label as complete visual understanding.

## What is induced

`learning.scene_grounding.GroundingExample` retains an example ID, exact structured
`description`, `SceneGraph`, `positive_refs`, `negative_refs`, and teaching
`basis`. `fit_scene_grounding(training, validation, *, max_atoms=3,
max_patterns=512, max_matches=4096)` enumerates nonempty connected proposition
conjunctions rooted at a labeled referent. Scene-local reference identities become
query variables; repeated references retain their joins. Predicates and role names
are opaque values, not a closed list of spatial relations or visual classes.
The root can denote the whole scene image as well as a declared scene node. A
conjunction can identify a situation through its organization and related
components; the representation is not restricted to individual objects.

For each exact description, a retained query must cover every explicitly positive
referent and exclude every explicitly negative referent in the corresponding
training examples, with support from at least two independent scenes. Duplicate annotations of one
scene do not supply independent support; inconsistent graphs under one scene
identity are rejected. An unlabeled
node is not a negative. Explicit negative-only examples are allowed, but both
label sets cannot be empty. Negative-only examples constrain queries without
counting as independent positive training corroboration or positive validation
support. Without negative teaching, broad queries can remain viable
and legitimately propose extra targets. No shortest-query or lexical-frequency
rule silently chooses a meaning.

Validation example IDs and scene/entity identities are disjoint from training.
Queries require positive validation support and no conflicting validation example
before they emit prediction matches. Missing or conflicting validation is retained
as unresolved query evidence; one successful held-out example cannot erase another
contradicting it. `validation_example_ids` records positive-supported validation;
negative-only validation remains in the model examples and contributes conflict
IDs when violated. This validates agreement with supplied alignments, not the truth
of the teacher's account of the image.

`model.propose(description, scene)` returns every retained match derivation with
its referent, query IDs, training/validation example IDs, matched proposition
indices, and variable assignments. Multiple queries or assignments can support
the same referent; multiple referents remain alternatives. A complete validated
query predicting no referent remains an unresolved `query_predicts_no_referent`
rival when other queries match; its silence cannot authorize their bindings.
Unknown descriptions
remain unresolved. Description matching preserves exact typed structure; it does
not learn paraphrases, synonyms, or the meanings of strings inside that structure.

## Matching and resource limits

The query matcher preserves typed proposition contents, including scope, validity,
modality, and polarity. A boolean marker does not become numeric `1`; a possible
relation does not become an asserted relation. Variable bindings preserve relational
direction and identity joins, with distinct query variables bound injectively.
Encoding, failed matching branches, and recursive joins consume bounded work.

`max_atoms` defines the searched hypothesis class. A complete search of conjunctions
up to that size does not establish that larger explanations are unnecessary.
Pattern, match, and work exhaustion report incomplete search rather than evidence
that unreturned rivals are false. This search does not learn its own representation,
negative evidence policy, resource allocation, or relevance criterion.

## Connection to the agent

The integration boundary requires authentic retained teaching evidence, explicit
model admission, and an explicitly selected scene interpretation. Grounding proposals
become occurrence-specific `MentionBinding` alternatives in the interpretation
workspace, with model and scene dependencies. They do not assert scene propositions
as beliefs or authorize an action merely because only one referent was returned.

A later selected reading can feed the separately admitted goal-correspondence model.
The request must retain the grounding model and scene commitments alongside its
reading and goal commitments, so withdrawing evidence cannot leave an old target
authorized. Initial description construction, model admission, scene/reading/goal
selection, and task semantics remain explicit supplied responsibilities.

The public functions live in `agent.scene_grounding`:

- `retain_grounding_example(agent, language_group_id, candidate_id, path,
  scene_group_id, scene_candidate_id, positive_refs, negative_refs=(), *, basis)`
  snapshots authentic retained candidates and explicit labels. Teaching candidates
  need not already be selected; the labels are teacher input.
- `fit_grounding_model(agent, training_records, validation_records, *, group_id=None,
  max_atoms=3, max_patterns=512, max_matches=4096)` creates a historical model
  version. Supplying an existing model group refits and withdraws its old admission.
- `admit_grounding_model(agent, handle, *, reason)` explicitly admits an authentic
  complete version and returns its revision-bound handle.
- `propose_scene_groundings(agent, admitted_handle, language_group_id, candidate_id,
  path, scene_group_id, scene_candidate_id)` applies it to the specified selected
  scene. Its report retains search evidence and published candidate IDs. Distinct
  referents become distinct reading alternatives; all match derivations remain
  in the retained report even when they support the same referent.

These operations can return `Unknown` instead of a usable result. An incomplete
prediction or unresolved rival query retains diagnostics but publishes no bindings.
Search completeness alone does not discharge missing or contradictory validation. A teaching change before
fit fails validation; changes after a completed fit do not rewrite its historical
supervision. Current model admission and prediction scene commitments still have
to remain valid. No implicit model installation or scene selection is performed.

Six focused end-to-end agent tests passed: relational grounding feeds learned
goal correspondence and an actual in-memory device action, with stale model and
scene commitments blocking dispatch. Successive grounding edits preserve their
ancestor support, and the language-turn boundary blocks dispatch after scene
withdrawal. Twenty evidence-wrapper tests cover authentic supervision, admission,
refit, incomplete search, unresolved rival queries, and dependency integrity.
Thirteen matcher tests cover relational and metadata fidelity. The full repository
suite passed **2,270 tests, with five skipped**, in 326.58 seconds. These results
verify this bounded mechanism and its integration, not generalized cognition.

## Evidence and remaining work

The pure learner's sixteen focused tests passed after the investigation milestone
added negative-only teaching and empty-denotation rival regressions. Authored graph fixtures exercise
an induced relation-plus-marker conjunction, new entity identities, reversed relation
direction, two matching targets, consistently renamed opaque predicates/roles,
typed values, modality, missing negative labels, unseen descriptions, incomplete
budgets, held-out identity leakage, and contradictory validation. These are mechanism
regressions, not an image or natural-language benchmark. A whole-scene fixture
also learns a rooted organization-plus-component-coherence conjunction across
held-out scene identities. The agent integration supplies teaching descriptions,
scene graphs, and goal labels, then applies the learned grounding query and learned
goal substitution to a fresh scene/device without a prediction-time authored
target lookup. Explicit interpretation and goal policies still authorize the
choice. This measures composition of two narrow learned mechanisms, not inference
from raw language or image pixels.

The next representational gaps remain substantial: learned scene construction from
images and video, uncertain or incomplete scene propositions, temporal relations,
new description structures, correction of mistaken teaching, and evidence-driven
selection among competing bindings. An induced graph conjunction improves an actual
reference proposal mechanism; it does not complete holistic vision or generalized
cognition. Fitted examples and current admission/dependency records remain in-memory
agent state unless a separate persistence mechanism explicitly stores them.

[Active grounding investigation](64-active-grounding-investigation.md) develops
correction of correlated teaching: compare retained queries on offered novel
scenes, retain predictions before explicit feedback, and refit without reusing
held-out examples as new training. It does not infer a teacher answer from dialogue.
