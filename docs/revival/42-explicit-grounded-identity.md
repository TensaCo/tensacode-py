# Explicit grounded identity in the active agent

**Status:** implemented mechanism, September 2026. This checkpoint removes an
active assumption that a description supplies its own world identity. It does not
infer identity from language or pixels. The architectural objective remains
[the structured cognitive workspace](36-structured-cognitive-workspace.md), with
[implicit semantic authority removed](39-removing-implicit-semantic-authority.md).

## Why this matters

Two mentions of “the folder” can refer to different objects. “The blue box” and
“my storage” can refer to the same object. Neither spelling nor a shared Python
object determines which world entity a particular occurrence denotes.

Previously, the active agent could namespace description text into a reference,
ask plugins in order to resolve a description, or interpret a picture category
as the most recent image. These paths settled identity outside the retained
interpretation workspace. Downstream belief retrieval could then return a
confident-looking answer to a question whose subject had never been grounded.

A second problem compounded this: unresolved question roles were dropped before
lookup. A question constrained by both subject and owner could become a broader
subject-only question. A failed identity resolver in proposition conversion could
also fall back to the original description text.

The checkpoint closes those active paths. Abstention is an explicit consequence:
raw language that previously appeared to work through fabricated identity now
requires a supplied grounding proposal. This is removal of an unsupported
capability claim, not evidence that general reference understanding is complete.

## Occurrence-specific proposals

`tensorcode.agent.grounding.MentionBinding` contains:

- A structural occurrence path, such as
  `('acts', 0, 'frame', 'roles', 'object')`.
- An explicitly supplied `Ref`.
- Nonempty evidence IDs naming retained workspace sources.
- A nonempty basis describing why the caller proposes the binding.

`propose_grounding(workspace, group_id, candidate_id, bindings)` returns a new
`InterpretationCandidate`. It validates candidate/group ownership, retained source
evidence, and the occurrence paths before making any workspace change. It retains
parent/source provenance and a per-binding audit record of the path, reference,
evidence IDs, and basis.

Paths traverse frames, arbitrary role names, entity features, mappings, and tuple
positions. There is no noun dictionary or fixed role vocabulary that decides which
occurrence can be bound. Two occurrences sharing identical descriptions or the
same Python `Entity` instance can receive different references.

The original candidate and source remain unchanged. The new alternative does not
automatically become selected, and selection does not assert a world belief.
Unbound occurrences remain unresolved. A conflicting preexisting reference causes
failure; a caller can instead propose another alternative from the original
unbound reading. Request/question meanings and their frames are rebuilt together
so dispatch does not accidentally use an older ungrounded frame.

A justification and an evidence ID are audit material, not proof of correspondence.
The current binding producer is the caller. There is no learned grounding model
behind this API, and graph fixtures supplied by tests are authored knowledge.

## Active assertion and question behavior

`language.semantics.explicit_ref` accepts an explicit `Entity.ref` or an explicitly
represented literal/number value. Names, descriptions, paths, and pronouns without
an explicit reference produce `Unknown`. Literal values remain scalar values;
“three” with a supplied numeric value of `3` is not minted into a world entity.

`Agent.tell` now uses that strict resolver. If a resolver returns `None` or
`Unknown`, `to_propositions` drops the affected clause and reports the precise
role path, including nested frames and collection indices. Failure in nested
content prevents storing the enclosing proposition with an invented filler.
Direct `Ref` fillers remain references, rather than being stringified.

Feature projection diagnostics remain: grounding identity does not establish the
truth or preservation of every modifier, quantifier, relative clause, or scope
attached to the original expression. The existing converter still reports these
losses. This checkpoint does not equate resolved identity with complete semantics.

`Agent._ref_of` now returns only an explicit reference. It no longer uses the last
image, plugin `denote` ordering, or the legacy description-namespacing resolver.

Before answering or running an informing capability, every stated question role
must resolve. One unresolved role blocks the question; it cannot be dropped to
broaden retrieval. The queried hole remains a hole. Explicitly grounded informing
queries pass their reference/value to the capability directly without asking
`Plugin.refer` to guess the intended entity again.

Grounded identity does not determine which report or inspection is intended. The
agent now collects applicable informing contracts before invoking any of them.
If more than one competes, it returns an explicit unresolved outcome and records
the alternatives. Plugin/capability declaration order cannot select one. This is
particularly relevant to the discourse plugin, whose three report capabilities
currently advertise overlapping question predicates. Their meaning must be
selected explicitly in future interpretation/action proposals; a bound topic alone
does not authorize the first report in the registry.

Default `Agent.lookup` answers are restricted to the shared scope (`None`) and
user statements (`USER`). A proposition in an arbitrary hypothesis scope does not
silently become an ordinary answer. A structured caller can deliberately query
another scope with `lookup(question, scopes=(hypothesis_ref,))`.

Plugin context is initialized separately from reference resolution: after its own
state is ready, the Agent calls an optional `plugin.attach(agent)` hook. Quantity
reporting uses that hook to receive the belief store. A grounded question therefore
does not need to trigger a semantic guess merely to initialize its tool's context.

## What verification establishes

`tests/test_agent_grounded_identity.py` exercises actual `Agent.tell`, lookup, and
informing-capability paths. It checks:

- Identical descriptions with different explicit references stay distinct.
- Different descriptions with the same explicit reference share identity.
- Partly grounded questions do not broaden into weaker queries.
- Hypothesis-scoped facts require explicit scope selection.
- A new grounding alternative can feed the real assertion path.
- Raw descriptions do not produce fabricated world references.
- Failed nested resolver results retain exact occurrence-path diagnostics.
- Explicit literal values survive without becoming entity references.
- An explicitly grounded report query does not call plugin semantic resolution.
- Reversing competing informing capabilities does not cause either to execute.

The existing reader-routing, answer-ranking, proposition-retrieval, and filesystem
reporting tests now supply their intended occurrence bindings as test fixtures.
Those fixtures select a declared reader result and give each required occurrence
an explicit reference through the grounding proposal API. They test the downstream
mechanisms under supplied interpretation, not learned language understanding.
Discourse content tests additionally supply the intended report contract, disabling
competing query contracts within that test fixture while retaining all registry
entries. This isolates report rendering and source fidelity. It does not establish
that the production agent inferred a discourse intent.

## Remaining boundaries and next work

The generic standalone `default_ref` resolver and converter defaults remain for
library callers awaiting migration. They still namespace descriptions; callers
must not mistake their output for inferred world identity. No new production
compatibility switch was introduced to restore that behavior in the active
assertion/question paths.

Discourse and request mechanisms remain a separate gap. Authored discourse topic
classification, WordNet-based report routing, and request-side plugin reference
resolution still exist. Removing them requires explicit topic/intent proposals
and grounded action-argument plans, with uncertainty retained before dispatch.
The checkpoint does not claim every plugin or every library path now uses strict
identity.

The core also lacks a general producer of grounded bindings. Necessary follow-on
work includes:

1. Propose cross-modal correspondences from actual inputs, including scene-level
   relational evidence and holistic organization—not only element categories.
2. Maintain competing correspondence hypotheses with evidence that can support,
   contradict, or leave them unresolved.
3. Select inspections or clarification that discriminate between those hypotheses.
4. Revise bindings and track downstream beliefs/tasks that depended on them.
5. Learn reusable correspondence and action models from evaluated predictions.

The useful invariant is now executable: absent a supplied grounding commitment,
active statement storage and question answering cannot silently manufacture the
missing identity from the surface description.
