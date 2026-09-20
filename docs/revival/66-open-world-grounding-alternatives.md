# Open-world grounding alternatives

A successful scene query identifies support in the stored graph. It does not
establish that every other referent is wrong or that the graph contains every
possible target. This milestone retains that distinction in the grounding
workspace: supported reference bindings coexist with unresolved known-root and
unseen-referent alternatives. Explicit selection remains necessary.

This extends [conflicting scene evidence](65-conflicting-scene-evidence.md). It
adds a conservative per-root evidence assessment, not a complete four-valued logic
or a proof that the scene representation covers the world. Scene graphs and
structured descriptions remain supplied; no new pixel inference is claimed.

## Evidence for each known root

`learning.graph_evidence.assess_query(query, scene, *, max_matches=128,
max_states=2048)` returns `QueryEvidence` with retained matching witnesses,
`RootEvidence` for the scene image and declared nodes, computation completeness,
unresolved reasons, explored work, and `unseen_referents_possible=True`.

Each root has one of four evidence statuses:

- `supported`: a matching conjunction has a supporting witness without the
  relevant exact opposing evidence detected by this assessment.
- `refuted`: an explicit exact opposite exists for a necessary query conjunct
  that can be grounded by this root alone, or that contains no reference
  variables. No supported witness or detected conflict overrides that status.
- `conflicted`: supporting and opposing evidence coexist under the implemented
  exact-polarity checks. Witnesses and opposing proposition indices remain visible.
- `unknown`: neither sufficient support nor the implemented form of refutation
  is established. Missing scene facts belong here, not in `refuted`.

For example, a necessary `red(root)` conjunct can be opposed by an exact
`not red(A)` fact for root A. For `near(root, x)`, observing `not near(A, B)` does
not refute the existential query: another x might be the witness. Failing to
find any stored x also does not refute it. Scope, validity, modality, polarity,
and typed identities retain the exact matching rules of the preceding milestone.
This does not infer contradictions across different predicates or perform
universal reasoning over all possible witnesses.

`RootEvidence.supporting_matches` retains the full graph witnesses.
`refuting_atoms` identifies the necessary query atom and the stored opposing
fact indices. Matching and subsequent root assessment share the work budget.
`QueryEvidence.complete` means this computation finished within its bounds;
it never means every fact or referent in the world is known. The unseen-referent
possibility remains even when all declared roots receive definite statuses.

## Alternatives in the active workspace

`SceneGroundingModel.propose` retains per-query evidence in `query_evidence`.
The agent adapter publishes both supported `MentionBinding`
alternatives and `UnresolvedGrounding` records. An unresolved record identifies
its known `Ref`, or `None` for a potentially unseen referent, together with query
IDs, a reason, and `evidence_source_id`.

The report separates supported `candidate_ids` from `unresolved_candidate_ids`.
They belong to the same exact interpretation comparison; a supported singleton
cannot silently erase unknown competitors. Selecting an unresolved alternative
defers rather than send its payload to the act handler. No selection policy
is learned by adding these alternatives, and an explicit policy is still
responsible for any commitment to a supported reading. That explicit commitment
does not assert that the unresolved rivals are false.

Existing barriers remain: incomplete computation, contradictory evidence, and
an entire retained query predicting no supported referent cannot be disguised
as a clean unique target. These diagnostic cases do not publish ordinary binding
alternatives. They are distinct from unresolved other roots accompanying an
otherwise supported query.

The adapter records and active turn path are implemented. Focused verification
is described below; the full suite remains pending.

## Investigation and teaching are narrower than world truth

[Active grounding investigation](64-active-grounding-investigation.md) retains
this per-query evidence, but its authored ranking policy still partitions queries
by their observed match sets on offered graphs. Those sets are not complete
world denotations. An empty stored match set can contain unknown possibilities;
equal observed sets need not establish equal meanings in the world. The policy
is a bounded disagreement heuristic over available evidence, not a guaranteed
optimal experiment or a complete elimination argument.

Teacher positive/negative labels continue to constrain graph-pattern compatibility.
A teacher's negative label is not converted into a negated scene proposition.
The fitting protocol's held-out agreement with labels is not a proof of factual
nonmembership in an open world. Neither this assessment nor the investigation
policy establishes the truth, completeness, or reliability of the teacher or
scene provider.

## Acceptance and remaining gaps

The final combined focused run passed **122 tests in 9.05 seconds** across
eleven files. Five new active-turn tests exercise the execution boundary. Selecting either a known unknown referent or the
unseen `None` alternative returns an unknown outcome without invoking the act
handler. Explicit negative evidence for a known root removes that root's
uncertainty under the supported rule, while the unseen alternative remains.
An explicit supported choice executes an in-memory Devices action with task
dependencies whose comparison includes all unresolved candidate IDs. A selector
that omits those IDs from its comparison basis is rejected.

The run also includes ten graph-evidence tests, eighteen matcher tests,
twenty-one learner tests, nine pure investigation tests, eleven uncertainty-wrapper
tests, and 48 existing agent/evidence regressions. These validate root-only
opposition, unknown missing relational witnesses, and refusal to execute a manually
selected unresolved payload through the direct request path. The full repository
suite passed **2,341 tests, with five skipped**, in 342.21 seconds. These are
supplied scene, reading, and selection fixtures, not a raw-image or language benchmark.

This is not a general solution to relational negation, quantified reasoning,
scene closure, source reliability, uncertain image formation, or active acquisition
of missing observations. There is no implicit assertion that a supplied node list
is exhaustive and no autonomous policy for seeking the missing referent. The
records preserve an actionable distinction between support and ignorance; further
learning and reasoning must decide how to reduce that ignorance.
