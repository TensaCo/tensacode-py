# Targeted grounding observations

[Open-world grounding alternatives](66-open-world-grounding-alternatives.md)
retains unknown referents rather than silently eliminating them. This milestone
adds a bounded way to request a missing observation relevant to one such root.
The questions come from necessary conjuncts of the learned relational queries,
not a caller-authored question list or a separate table of expected answers.

The target is a concrete information gap: a known scene root might satisfy a
learned query, but a necessary root-groundable proposition is absent. The agent
can prepare that proposition as a probe, retain what positive and negative answers
would imply, and ask an explicitly supplied observer. A response creates evidence
and, when usable, a new scene alternative. It does not select the scene or execute
the request whose grounding motivated the observation.

## Derived questions and counterfactuals

`learning.grounding_probes.propose_grounding_probes(model, description, scene,
root, *, max_probes=64, max_states=65536)` derives missing necessary propositions
for a specified known root with unknown query evidence. Predicate, roles, polarity,
scope, validity, and modality come from the learned query. The candidate proposition
must be groundable without guessing another entity. In particular, an existential
`near(root, x)` gap does not justify inventing a Ref for x or treating a failed
search for x as a negative answer. Such relational gaps remain explicit unresolved
work outside this probe class.

For each retained question, the bounded plan retains counterfactual query evidence
for an affirmative and a negative answer before any observation is made. These
are `GroundingProbe.positive_evidence` and `negative_evidence` for all relevant
exact-description queries, including rivals that did not generate the question.
`query_ids` and `atom_indices` identify only its generating origins.
These are conditional predictions from the supplied graph and learned query, not facts
about the environment or claims that the observer will answer reliably. Preparing
the alternatives does not inject them into the live scene.

The caller explicitly chooses a probe from the plan. There is no first-question
execution default, learned question-ranking policy, or autonomous retry loop.
Exhausted bounds must remain visible; not generating a question is not proof that
there is no useful observation to make. Computationally incomplete plans expose
no usable probes. A computationally complete `GroundingProbePlan` can still retain
semantic unresolved reasons, such as `unresolved_relational_witness`, together
with initial `query_evidence` and `query_info`. Thus completeness is not a claim
that all unknown roots can be resolved by this observation interface.

## Observation and scene revision boundary

The agent wrapper retains the admitted grounding model, language candidate,
selected scene, raw source, proposed questions, and counterfactual evidence before
calling the observer. The observation contract accepts the full typed proposition
and scene/source context. It does not reduce the request to a lossy planning
`Condition` that drops scope, modality, validity, or polarity.

The explicitly supplied read-only observer returns a boolean or `Unknown`. True
supports the asked proposition; false supports its exact polarity opposite. A
usable boolean produces a new `SceneProposal` with that additional fact and its
observation evidence. `Unknown` or an error supplies no inferred fact. Existing
scene candidates and the pre-observation predictions remain retained.

Observation does not assert a belief, select a scene, bind a mention, choose a
meaning, or perform a task action. The caller must explicitly select a resulting
scene alternative and request fresh grounding before any later supported request.
Changing the scene comparison invalidates old scene-dependent task commitments;
a newly observed fact does not silently rewrite their original authority.

Authentic proposal and comparison snapshots are checked around callback-bearing
work. A changed model, source, language comparison, or scene must fail closed.
Observation proposals are consumed once so retrying the same call cannot silently
repeat the external observation under a stale commitment. The functions in `agent.grounding_observation` are:

- `prepare_grounding_observation(agent, admitted_handle, language_group_id,
  candidate_id, path, scene_group_id, scene_candidate_id, root, *, max_probes=64,
  max_states=65536)` returns `GroundingObservationProposal(id, evidence_source_id,
  plan)` or `Unknown`.
- `observe_grounding_proposal(agent, proposal, probe_id, observer)` accepts an
  explicit named `Plugin` and calls its
  `observe_scene_proposition(proposition, scene, source)` method. Its successful
  wrapper result is `GroundingObservationResult(id, evidence_source_id,
  scene_candidate_id, observation)`; an unknown observation has no scene candidate.

Invalid provider output becomes `Unknown`, not truthiness-coerced evidence. A
callback that invalidates the retained comparison cannot authorize scene publication
merely by returning a boolean. If a scene candidate was already published before
a later authentication failure, it is rejected; the request, observed response,
and terminal failure evidence remain attributable. The proposal is not silently
replayed after an attempted observation fails.

## Acceptance evidence and limits

The final focused run passed **28 new tests in 2.92 seconds**: nine pure probe
planner tests, sixteen adversarial wrapper tests, and three agent integration
tests. An earlier combined planner/graph-evidence/matcher regression run passed
37 tests. The full repository suite passed **2,369 tests, with five skipped**,
in 341.33 seconds (exit status 0).

The integration starts with an unknown known root B and a stateful supplied
observer whose hidden predicate value is absent from the scene. One generated
question retains affirmative→supported and negative→refuted predictions before
the callback. True or false creates an unselected scene alternative, invalidates
the old request comparison, and requires explicit new-scene and target selection
before an actual Devices action on B or A respectively. An `Unknown` response
adds neither fact nor scene proposal, and B remains unknown. Preparation and
observation alone perform no device action.

Scene graphs, observer semantics, language frames, goal projection, and caller choices in this
fixture are authored. The query association is learned, and its missing
proposition drives the probe. That demonstrates a specific learned interpretation
influencing evidence acquisition; it does not demonstrate image understanding,
natural-language question generation, autonomous sensor choice, or an inferred
model of observer reliability.

Remaining gaps include acquisition of unknown existential witnesses, discovery of
unseen referents, temporal observation alignment, conflicting observers, learned
probe utility and cost, and automatic decisions about when to seek more evidence.
A concrete next step is to use partial relational witnesses. Given a learned
`member(root, group) ∧ radial(group)` query and observed `member(A, G)`, a future
planner should derive `radial(G)` using the retained identity G, without inventing
a witness or enumerating every node. The current root-only planner cannot do
this. Even then, observing `not radial(G)` must not refute the existential query
for A, because another group witness may exist.

An observer's boolean answer is supplied evidence, not independently established
world truth. These operations and their retained records remain in-memory unless
separate persistence explicitly stores them.
