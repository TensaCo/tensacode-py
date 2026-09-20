# Authenticated derivations

[Learned store querying](77-learned-store-querying.md) refused records carrying
`derived_from` evidence because premise IDs alone do not authenticate an inference.
This milestone develops replayable proposition derivations with admitted operators,
retained exact inputs and outputs, and bounded recursive support validation.
Quantity totals are the first active producer; the arithmetic semantics remain
supplied rather than learned.

A derivation is not trustworthy because its label says “sum” or because its named
premises exist. The operator, parameters, premise snapshots, and exact output must
be associated with an authentic retained receipt and replay successfully under
current support. General-purpose Store access inside an opaque consequence cannot
supply that guarantee.

## Explicit operator authority and receipts

The implemented boundary registers an operator and explicitly admits its version.
It receives detached retained proposition premises and explicit parameters rather
than unrestricted Store access. A private receipt retains the operator identity,
exact premise snapshots, parameters, and resulting proposition. Public evidence
can reference that receipt; constructing a lookalike `derived_from` tuple does
not create equivalent authority.

Validation must replay the admitted operator against the retained inputs and
compare the complete output, not merely a numeric field or an answer label.
Operator withdrawal or version changes invalidate its authority. The mechanism
supports supplied deterministic operators; it does not learn the meaning,
correctness, applicability, or completeness of an arbitrary inference rule.

The live `tensorcode.derivations` APIs are:

- `admit_operator(store, name, operator, *, reason)` returns an `OperatorHandle`
  with identity/name/version; readmitting the same name withdraws its prior version.
- `withdraw_operator(store, handle, *, reason)` revokes that version.
- `derive(store, handle, premise_ids, *, params=None, basis,
  population_predicate=None, max_depth=32, max_nodes=256)` validates support,
  executes and replays the supplied callable, and returns a `DerivationReceipt`
  or `Unknown`.
- `validate_record_support(store, record_id, *, max_depth=32, max_nodes=256)`
  returns true only when current bounded support validation succeeds.

The callable receives detached proposition tuples and parameters. Receipts retain
output, operator, premise IDs, evidence, and basis; the Store's private registry
also retains full premise snapshots and parameters. Public receipt data alone
cannot reconstruct authority after a restart. Exact output replay is checked
before publication and again when a derived record is validated. This verifies
agreement with an admitted implementation, not mathematical correctness of that
implementation. Focused and full-suite verification are recorded below.

## Recursive support and alternate evidence

A derived premise can itself require an authenticated derivation. Validation is
bounded and must retain uncertainty when it cannot establish the support chain.
Retraction, changed premise content, opposing evidence, and withdrawn operators
must affect downstream answer authority rather than leave a cached proof valid.

Multiple support branches require care: an invalid derivation does not necessarily
erase independent valid direct evidence, and one bad branch must not be hidden
when no valid support remains. Traversal must account for cycles and resource
limits. Exhausting a budget is not proof that all branches are invalid, and a
successful supported branch is not a claim that the entire store is consistent.

Direct evidence branches remain explicitly supplied Evidence records, not newly
authenticated real-world observations. Admitting a callable is not sandboxing it:
reentrant or observed Store/authority mutation is refused, but arbitrary external
side effects of caller code are not certified absent.

Validation retains a final read set across all visited Stores, not just the last
one replayed. Answer batches share that read set, so a later support callback
cannot mutate an earlier answer's source Store unnoticed. Interpretation
dependencies are rechecked after replay. Typed direct observed facts and authentic
`DerivationReference` values remain distinct provider outputs.

The learned store-query path can use a derived answer only after this support
boundary succeeds. It still requires the selected full query, scope constraints,
model/reading dependencies, and retained answer records. The stricter support
interface does not automatically certify every legacy derived fact in the store.

## Quantity totals and changing membership

`QuantityPlugin.total_of_kind` now replaces its unauthenticated
`derived_from`-only assertion with an actual registered derivation producer.
The operator's arithmetic is authored. Matching owner/kind records, units,
qualifiers, and explicit measurement evidence remain subject to the quantity
contract; a valid proof of arithmetic does not establish that the observations
cover the world. The result remains `total_kind:<predicate>` rather than being
relabeled as an original directly observed predicate. The plugin admits the
supplied arithmetic operator explicitly.

A total also depends on which measurements were included. Recording only selected
premise snapshots is insufficient: adding another matching measurement would leave
an obsolete sum looking valid. The producer must retain its complete relevant
scan membership, and validation must invalidate the old total when a new relevant
measurement appears, as well as when a premise is changed or retracted.

`export_derivation` produces an authenticated `DerivationReference` for a retained
receipt, and `import_derivation` retains that reference in another Store without
laundering it into direct observation. `validate_derivation_reference` checks the
origin authority and current support. This is a process-local cross-Store link,
not a portable proof independent of its issuing Store and admitted operator.

## Acceptance evidence and remaining limits

The generic derivation worker passed **28 tests**, and the quantity producer
passed six (34 together). A root integration run passed **58 tests in 24.53
seconds** across derived store answers, store-query evidence, informing evidence,
and agent operations. After adding the batch-support mutation regression, the
derived-answer file passed eight tests in 2.87 seconds. A migration run passed
60 tests in 210.58 seconds. These runs overlap and are not additive. The full
repository suite passed **2692 tests, with 2 skipped, in 803.03 seconds**.

The active integration constructs exact owner/kind totals and retains authentic
arithmetic receipts and cross-Store references. Learned store answers validate
those receipts and support rather than accept arbitrary premise IDs. Added
measurements, contradictions, retractions, operator withdrawal, altered evidence,
and cross-Store callback mutation cannot establish a current answer from stale
support. Alternate valid support remains distinct from blanket rejection of every
record with a failed branch.

This is evidence-backed execution of supplied arithmetic and operator contracts,
not learned mathematical reasoning, proof search, or an autonomous choice of
which inference is relevant. The query correspondence remains separately learned
from explicit supervision. Remaining gaps include learning inference operators,
source reliability, broader quantified reasoning, temporal applicability, and
sound replay of other legacy derivation mechanisms. Legacy untyped total,
difference, and count paths remain outside this authenticated n-ary mechanism. No general completeness or
persistent cross-process authentication is implied.

## Remaining quantity lineage gap

The owner-only `amount_of_<predicate>` path still uses `_total_claim` to combine
an automatic rate-times-first-compatible-count choice and a sum, then reveals a
plain Claim that hides the derivation lineage. `count_properties` has a related
derived-report boundary. This checkpoint must not be read as authenticating every
quantity answer merely because kind-restricted totals now have replayable support.

The next milestone should retain new untyped measurements as original observed
propositions, introduce an authenticated owner-only operator and exported
`total_owner` result, and require explicit operands for rate products rather than
implicitly pairing them. Existing Claim aggregates must not be relabeled as direct
observations to bypass missing lineage. That migration is future work; the active
quantity derivation producer here remains the exact owner-and-kind path.

A more basic limitation precedes that migration: observations sharing an owner
are not necessarily additive operands. Proposition content identity can collapse
two independent measurements of three, while different values may be rival
measurements of the same thing rather than quantities to sum. The current
kind-total contract assumes additivity; a valid receipt proves reproducible
execution of that assumption, not its semantic legitimacy.

The next arithmetic interface should retain an explicit measurement Ref and
Evidence in `remember`, and accept ordered operand IDs plus an admitted operator
in `calculate`. Conflicting values for one measurement identity must refuse.
A membership total needs explicit collection membership or aggregation policy;
“all facts about this owner” must not silently supply that policy. These are
future representation and selection requirements, not guarantees of the current
kind-total producer.

The larger cognitive objective also remains open beyond arithmetic provenance.
Current goal correspondence preserves literal non-reference structure and learns
Ref substitution, not semantic paraphrase transformations. `Agent.request` starts
a new task; corrections still require authored `tasks.revise` calls. The first
two-turn acceptance case in [36](36-structured-cognitive-workspace.md) therefore
is not end-to-end complete: a source-grounded correction must identify the existing
task, change its destination, add the README invariant, preserve actual receipts,
and replan. Evaluation must include held-out wording, ambiguous task references,
and a correction arriving after completed mutation. No derivation receipt or
arithmetic cleanup establishes that capability.

[Explicit measurements and calculations](79-explicit-measurements-and-calculations.md)
implements the next bounded representation/selection boundary described above:
measurement identities and explicit ordered operands replace automatic aggregation.
The arithmetic and aggregation choices remain supplied.
