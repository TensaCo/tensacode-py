# Conflicting scene evidence

The relational matcher could find a witness for `red(A)` even when the same
supplied scene also contained an otherwise identical `not red(A)` proposition.
The positive witness alone could support a learned grounding and an action target.
This milestone retains the opposing evidence on the witness and blocks its use as
clean teaching, grounding, or investigation evidence.

The audit began with a broader concern about missing scene facts. The implemented
fix is narrower: explicit opposite-polarity propositions. It does not implement
open-world evaluation of every possible root, infer negation from a missing fact,
or provide a four-valued query logic. That distinction matters when interpreting
[relational grounding](63-learning-relational-scene-grounding.md) and
[active grounding investigation](64-active-grounding-investigation.md).

## What counts as a conflict

Two propositions oppose one another when they have exactly the same typed
predicate, roles, scope, validity interval, and modality, and their top-level
boolean polarity differs. Identity and metadata are preserved: `red(A)` in one
scope and `not red(A)` in another are not this conflict; neither are asserted and
possible propositions. This does not infer contradictions between different
predicates, overlapping but unequal time intervals, incompatible numeric values,
or nested propositions whose relation requires additional reasoning.

`learning.graph_queries.GraphMatch.conflicts` stores tuples of the form
`(supporting_fact_index, opposite_fact_indices)`. A successful conjunctive witness
still retains its variable bindings and supporting fact indices, alongside the
exact opposing indices. The evidence is not deleted, automatically repaired, or
resolved by preferring the positive fact. Negative query atoms are subject to the
same check against their positive counterparts.

The matcher reports `contradictory_match_evidence` in its unresolved reasons.
`QueryMatches.complete` can still be true when the bounded search finished: search
completion and evidential consistency are different properties. A caller cannot
use `complete` alone as authority to commit a binding.

## Propagation into learning and investigation

A teaching query must have complete matching and no unresolved evidence before it
can supply support. A conflicted witness cannot become either positive training
support or a clean exclusion used to establish compatibility with negative labels.
Fitting retains the reason and marks the diagnostic model incomplete, preventing
ordinary model admission. Existing examples and their conflict evidence remain
available for inspection instead of being silently filtered into a cleaner dataset.

`GroundingMatch.conflicts` carries the same fact-index evidence into learned
predictions. Conflicted matches remain diagnostic candidates, while the unresolved
prediction prevents the agent adapter from publishing executable grounding
alternatives. An apparent unique referent does not settle contradictory support.

`QueryDenotation.matches` retains full `GraphMatch` witnesses for investigation,
including conflicts. A denotation with unresolved evidence is not complete for
investigation even if its matcher exhausted the search. The investigation therefore
suppresses preferred-scene recommendations and cannot accept teacher feedback as
though those predictions were clean complete answers. This preserves the sequence
of evidence, prediction, and explicit correction rather than hiding the conflict
inside a referent set.

## Verification and correction boundary

**93 focused tests passed** across disjoint files: 48 matcher/learner/investigation/
new-agent tests in 1.44 seconds, and 45 existing integration/wrapper tests in
5.22 seconds. The first group contains eighteen matcher tests, nineteen learner
tests, eight pure investigation tests, and three new agent tests. The full
repository suite passed **2,312 tests, with five skipped**, in 334.55 seconds.

The agent regressions in `tests/test_agent_scene_conflicts.py` verify three active
boundaries: conflicting training produces an inadmissible diagnostic fit;
investigation retains opposing fact indices without ranking scenes or accepting
feedback against unresolved predictions; and scene revision invalidates an old
request before an explicit coherent scene selection enables an actual in-memory
Devices action. Conflicted prediction publishes no actionable binding. Scope and
modality differences remain distinct from exact opposition.

These are authored graph, selection, and action fixtures. Correcting the scene is
an explicit caller decision, not a demonstrated ability to resolve the underlying
visual ambiguity. The tests establish that retained contradictory evidence affects
the active learned grounding and request path; they do not measure understanding
of image pixels or natural-language corrections.

## What remains unresolved

Absence is still not a factual negative. Explicit teacher negative labels constrain
which query patterns are compatible with the teacher's examples; a nonmatch is not
proof that a proposition about an unseen or incompletely described entity is false.
The system does not yet carry a complete open-world account of supported, opposed,
both, or unknown query answers for every possible referent.

The conflict check also does not decide which source is right, learn source
reliability, reconcile time or scope, or choose a corrected scene. A caller supplies
and explicitly selects the correction. Scene graphs remain supplied structured
interpretations rather than structures inferred from pixels. The implemented gain
is that a specific retained contradiction can now stop the learned path before an
action; broader scene construction and evidence reasoning remain separate work.
