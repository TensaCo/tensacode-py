# Relational witness observations

The preceding [targeted observation milestone](67-targeted-grounding-observations.md)
could ask about a missing proposition grounded by the target root alone. It could
not use an observed relation to identify another entity worth inspecting. This
milestone develops that next step: derive a question from a learned relational
query and a connected partial witness already present in the scene.

For a learned `member(root, group) ∧ radial(group)` query and an observed
`member(A, G)`, the useful question is `radial(G)`. G comes from retained relational
evidence, not a fabricated entity or an enumeration of every scene node. A positive
answer can complete a witness for A. A negative answer rules out that particular
completion; it does not establish that A has no other group satisfying the query.

The scene graph, description, observer semantics, and caller's choices remain
supplied. The learned query determines what information is missing from a partial
witness. This extends evidence acquisition driven by structured interpretation;
it does not infer relational scene structure from pixels or understand a question
posed in natural language.

## Connected evidence determines the question

`learning.graph_partial.match_partial_query(query, scene, root, *,
max_matches=128, max_states=2048)` retains connected partial matches rooted at
the specified known referent, including the initial root-only seed.
Existing supporting facts provide variable assignments for related identities.
A missing query proposition becomes executable only when those assignments
supply all its reference variables. No missing existential variable is silently
assigned an arbitrary node, and a disconnected fact cannot supply a witness
merely because its values happen to be convenient.

`PartialGraphMatch` retains `bindings` as `(variable_index, Ref)` pairs,
`supporting` as `(query_atom_index, scene_fact_index)` pairs, remaining atom
indices, and explicit opposite-polarity `conflicts`. `GroundingProbe.witnesses`
retains `GroundingProbeWitness(query_id, atom_index, bindings, supporting,
conflicts)` origins for the proposed question. Equivalent fully
grounded questions can share an observation while preserving their different
origins. The scene source and exact typed proposition remain available through
the existing authenticated observation wrapper. Scope, validity, modality,
polarity, and identity joins must survive the projection into the question.

Supporting facts with explicit opposite-polarity evidence cannot establish a
clean partial witness. Missing facts do not become false. Bounds constrain
connected search, matching, question construction, and retained counterfactuals;
exhaustion must remain distinct from exhausting the world's possible entities.
The existing `propose_grounding_probes` API and authenticated observation wrapper
remain the entry points. Their plan budget includes partial matching; there is
no separate unbounded related-entity search.

## Conditional evidence and explicit observation

The existing probe plan retains affirmative and negative counterfactual evidence
before the observer is invoked. Related-entity questions follow the same rule:
append the hypothetical proposition or its exact polarity opposite to a copy of
the supplied scene, then assess the relevant learned queries. The prediction is
conditional on that evidence, not a declaration that the observer's answer is true.

An explicit caller chooses a retained probe and observer. A boolean response
produces a new unselected scene proposal; uncertainty produces no inferred fact.
The caller must explicitly select the new scene and request fresh grounding
before any task action. Old scene-dependent commitments cannot silently inherit
the new observation. No first-probe choice, automatic scene selection, or
belief assertion is introduced.

For existential queries, a negative observation about G can leave A unknown.
Another observed partial witness H may remain, or a witness may be absent from
the supplied graph altogether. Even exhausting every retained partial witness
does not prove existential falsity without an additional justified closure rule.
The unseen-referent possibility is not removed by completing this computation.

This exposes an additional publication boundary: a complete, validated,
nonconflicting query assessment can have no supported match while known roots
remain unknown. Such evidence must still publish unresolved grounding alternatives,
including the unseen possibility, rather than returning only diagnostics and
losing those alternatives. It supplies no executable binding. Unvalidated or
contradictory query rivals still cannot authorize supported bindings.

## Acceptance evidence and remaining limits

The final combined focused run passed **74 tests in 4.99 seconds**, including
eight partial-matcher tests, fourteen probe-planner tests, two new relational
integration cases, three prior observation integration cases, sixteen adversarial
observation tests, and related uncertainty/evidence regressions. The full
repository suite passed **2,384 tests, with five skipped**, in 345.60 seconds.

`tests/test_relational_grounding_observation.py` learns the relational pattern
from positive/negative alignments in two independent supplied training scenes
and a held-out scene, then starts with no supported target in a partial scene.
The plan derives exactly `radial(G)` for the already observed related group G.
Its witness bindings identify G and its supporting indices identify the retained
membership fact. Counterfactuals predict support after true and continued unknown
status after false, before the observer runs.

Each case calls the supplied observer exactly once, retains the original scene
and plan unchanged, and adds a new unselected scene candidate without inventing
nodes or acting. After explicit scene selection, the true case supports A and an
explicit authored goal projection permits an actual Devices action on A. The
false case retains A as unknown with no executable binding or device action.
The unseen-referent alternative remains in both cases; no belief is asserted.

These are authored scene/provider/selection/goal-projection fixtures. Query induction and
query-derived evidence requests are the learned components; supplied graph
relations are not newly inferred perceptual facts. Tests must distinguish
successful mechanism composition from general visual cognition or semantic
accuracy on unstructured input.

Remaining gaps include acquiring unseen entities and relations, deciding which
partial witness is most informative or cheapest to inspect, learning observer
reliability, handling temporal correspondence, and selecting an interpretation
under unresolved alternatives. This milestone does not supply autonomous
perception, complete quantified negation, or a learned investigation policy.

[Neutral evidence graphs](69-neutral-evidence-graphs.md) extends the shared
relational machinery beyond image-bound scenes, with literal browser snapshots
as another evidence source. Query reasoning remains distinct from modality-specific
perception and temporal identity inference.
