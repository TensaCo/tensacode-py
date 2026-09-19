# 40 — Evidence-driven interpretation

*2026-09-19. Implementation checkpoint under the governing objective in
[36](36-structured-cognitive-workspace.md). This extends the retained language and
scene interpretations in [37](37-interpretation-workspace-first-slice.md) and
[38](38-scene-interpretations.md), while keeping the removals in
[39](39-removing-implicit-semantic-authority.md).*

## The active capability this adds

An agent can now investigate competing interpretations through fresh observations,
retain the observation record, and select or withdraw a selection on that evidence.
Previously, an application could retain alternatives and supply a selector, but
there was no integrated observation-driven investigation between proposal and
selection. The new path connects those operations without restoring reader-order
execution or bundled semantic knowledge.

This is a bounded investigation of **supplied, world-conditional hypotheses**. It
is not autonomous hypothesis generation, calibrated inference about a speaker's
intent, or learned understanding of language or pixels. A wrong or incomplete
hypothesis space can still yield a uniquely supported candidate within that space.
The distinction must remain visible in evaluation and product descriptions.

[Checkpoint 54](54-investigation-with-pending-interpretations.md) subsequently connects
retained semantic search to this path: a bounded expansion precedes the supplied
hypothesis producer, and known pending alternatives prevent an investigation-based
selection. The source retains the visible-candidate assessment separately. This closes
the known-pending-work gap; it does not establish completeness of the generated space.

## API and lifecycle

The implementation lives in [investigation.py](../../src/tensorcode/agent/investigation.py)
and the integration in [core.py](../../src/tensorcode/agent/core.py).

An application can construct an agent with:

```python
Agent(
    interpretation_hypotheses=producer,
    interpretation_probe_budget=8,
)
```

`producer(group)` receives a retained `InterpretationGroup` and returns
`CandidateHypothesis(candidate_id, predictions, basis)` records. `predictions`
is a tuple of grounded `Condition` values; `basis` is a nonempty tuple of strings
identifying the supplied model's justification. There is no default producer.
The hypothesis producer and direct `interpretation_selector` are mutually
exclusive constructor dependencies. This makes the decision route explicit;
providing a producer does not establish how it obtained its semantic knowledge.

For an existing group, the explicit API is:

```python
investigated = agent.investigate_interpretation(
    group_id,
    hypotheses,
    max_probes=8,
)
```

Every non-rejected candidate must appear exactly once, including candidates for
which the caller has no usable prediction. Omitting a difficult rival cannot make
another reading win by default. Duplicate, unknown, rejected, or missing candidate
IDs fail coverage validation. A hypothesis with no predictions remains a viable
alternative; lack of predictions is not a contradiction.

The returned `InvestigatedInterpretation` contains `group_id`, `result`,
`decision`, and `evidence_source_id`. `result` retains the per-provider observations
and each hypothesis's confirmed, contradicted, and unresolved predictions.
`decision` contains the chosen candidate ID or `None`, its reason, and the retained
evidence source ID.

The agent records a new observation source containing the supplied hypotheses and
investigation result. Its metadata identifies the interpretation group, original
input source, and probe budget. The subsequent workspace selection or unset
revision references that evidence source. The source input and original candidates
remain available. A provider callback changing the group during investigation
causes the operation to fail rather than silently applying a result to a changed
question.

Automatic `Agent.turn` integration currently applies this producer to **text
candidate groups**. The explicit investigation API is payload-independent and can
operate on visual groups supplied with predictions. This does not constitute an
automatic visual investigation loop or a learned scene-hypothesis producer.

## How evidence changes a decision

The engine probes distinct positive condition atoms through each plugin's
`observe_condition` contract. Negated predictions refer to the opposite expected
value of the same atom. It prefers probes that divide the remaining hypotheses;
this is a procedural search heuristic, not a learned attention policy or a
probabilistic information-gain calculation.

Providers must honor a read-only observation contract. The engine does not invoke
action capabilities to obtain evidence. A probe budget counts distinct condition
atoms, not provider calls: one atom can be queried against several providers.
The budget does not impose a timeout on a provider implementation.

Evidence handling preserves the following distinctions:

- A boolean observation confirms or contradicts an explicit prediction.
- An `Unknown` provider abstains. Another provider can still supply a usable
  boolean observation for that condition.
- Conflicting boolean observations, malformed responses, or provider exceptions
  make the aggregate condition unknown. Provider-level reasons remain in the
  retained record.
- An unobserved condition remains unknown. A candidate that makes no prediction
  about a condition survives either observed result.
- Exhausting the probe budget leaves unresolved alternatives unresolved.

Selection requires exactly one viable hypothesis, at least one confirmed
prediction for it, all of its predictions confirmed, and an observed contradiction
for every rival. An empty sole hypothesis cannot win by vacuous truth. Unknown
winner predictions prevent selection even when every rival has been contradicted.
If all hypotheses fail, the result reports that failure instead of choosing the
least contradicted candidate.

For example, two supplied readings may predict different locations for the same
explicitly referenced object. Fresh observation can distinguish those accounts if
it confirms all of one account's predictions and contradicts the other. That
establishes agreement with the supplied world model. It does not prove that the
speaker intended the supported reading, or that a third unsupplied reading is
impossible. A test that authors these predictions demonstrates the investigation
mechanism, not learned reference resolution.

## Revision without concealed execution

Calling the investigation API again obtains new observations. If the previous
winner is no longer supported, the new decision can withdraw its selection or
select another candidate. Revision history retains the prior decision and its
evidence rather than rewriting the past.

Reinvestigation does **not** replay a language act, undo an already executed action,
or automatically revise a dependent task, belief, or plan. Those are separate
obligations. Continuous observation of changing environments and propagation of
invalidated dependencies are not implemented by this checkpoint. A selected
interpretation can become stale between investigation and later use.

The lower-level `investigate` engine only returns an assessment. Workspace revision
is performed by the agent integration. Neither operation turns a scene proposal
into a belief merely because its graph was supplied.

## Explicit source-bound grounding proposals

[grounding.py](../../src/tensorcode/agent/grounding.py) adds a separate mechanism for
proposing references without overwriting a reader's original output:

```python
MentionBinding(
    path=("acts", 0, "frame", "roles", "object"),
    reference=explicit_reference,
    evidence_ids=(retained_source_id,),
    basis="Justification supplied by the binding proposer",
)
```

`propose_grounding(workspace, group_id, candidate_id, bindings)` creates a new
sentence alternative with those occurrence-specific bindings. It validates that
the cited source IDs exist, the paths identify entity occurrences, and existing
explicit references are not contradicted. Invalid proposals leave the workspace
unchanged. Binding one occurrence does not globally equate all matching strings.

The original candidate and source remain retained. Provenance records the parent
candidate, input source, paths, references, evidence IDs, and supplied justification.
The new alternative is not automatically selected and does not automatically admit
claims to the belief store. A retained evidence ID makes a justification auditable;
it does not by itself prove that the reference is correct.

The caller still supplies the identity hypothesis. This API does not discover
cross-modal identity, resolve pronouns, or derive scene correspondence from pixels.
Existing implicit identity construction such as `default_ref` remains a separate
removal and integration task; this proposal mechanism must not be described as
having already eliminated every downstream identity shortcut.

## Correctness repairs supporting the longer learning path

Two accompanying repairs improve the reliability of existing mechanisms:

- [Expectation checking](../../src/tensorcode/expectation.py) returns unknown when
  required predicted aspects were not observed and no observed aspect contradicts
  the expectation. Partial agreement does not certify the whole prediction.
  An observed contradiction still reports failure.
- [Dependency certificates](../../src/tensorcode/learning/certificate.py) distinguish
  an absent key from an explicitly stored `None`. Full scans retain the ordered
  key population, including an empty scan, so later additions and changes in
  iteration order invalidate the certificate. Serialized read sets carry `format_version=1` and explicit `reads` and
  `scanned_keys` fields. Unversioned or unsupported records and incomplete current
  records are rejected and require regeneration. No compatibility interpretation
  supplies scan coverage or missing-value distinctions absent from old records.
  Point queries still ignore unrelated additions.

These repairs prevent unsupported conclusions from partial evidence and incomplete
dependency tracking. They do not connect the learning library into an autonomous
agent model-learning loop. Certificate validity also assumes stable backing facts
during computation; recording dependencies does not supply transaction isolation.

## What must happen next

The governing target remains unstructured evidence becoming revisable structured
understanding, followed by reasoning, model formation, and realization. This
checkpoint supplies one part of that loop. The next boundaries are concrete:

1. **Produce hypotheses from real inputs.** Connect interpretation proposals and
   their predicted consequences to retained language, images, and observations.
   Evaluate on inputs and ambiguities not encoded by the fixture author. Keep the
   producer's provenance and failures visible.
2. **Replace implicit grounding in the active path.** Use explicit, source-bound
   alternatives through semantic projection and execution. Missing identity must
   remain unresolved rather than acquire authority from a string-derived ID.
3. **Propagate revision to dependents.** Track which beliefs, task specifications,
   plans, and outputs depend on an interpretation. Withdraw or revalidate them
   when its support changes without silently replaying mutations.
4. **Model observation scope and freshness.** Support temporal qualifications,
   changing scenes, and relevant rechecks before dependent action. A one-time
   observation does not continuously certify a dynamic environment.
5. **Learn and test models.** Retain predictive successes and counterexamples,
   propose revisions, and evaluate them on independent evidence. Supplied
   condition tuples are not evidence of concept induction.
6. **Extend visual investigation holistically.** Test hypotheses about grouping,
   layout, relationships, events, affordances, and the overall situation. Element
   labels alone cannot establish scene understanding or shared language grounding.

Acceptance evidence should show consequential ambiguity resolved by fresh evidence,
unknowns preserving viable rivals, changed observations withdrawing a decision,
and the source-to-decision chain remaining inspectable. Separate those mechanism
checks from measurements of learned interpretation quality. General cognition is
not complete, and no test-suite total should be used to imply otherwise.

## Checkpoint verification

The final targeted investigation, grounding, workspace, certificate, and general-agent
run passed **82 tests**. The separate expectation/certificate/learning verification
passed **65 tests**. These suites overlap and are not a general cognition score. The
filesystem integration uses supplied interpretation models and project knowledge,
while observations and resulting file effects are real.
