# Learned store querying

[Learned informing correspondences](76-learned-informing-correspondences.md)
removed role-order guesses from active observation, but passive store lookup still
interpreted a question through symmetric-predicate rules, lexical role guesses,
first-open-role selection, ranking, and default scope assumptions. This milestone
replaces that implicit query authority with a taught full-Question-to-store-plan
correspondence and explicit query selection.

The target is accountable retrieval from retained evidence. A stored fact does
not become an answer merely because it shares a participant or predicate with a
question. The selected query must preserve the taught constraints and the allowed
sources/scopes, and the resulting answer must retain its supporting records.
The implemented APIs and verification results are recorded below.

## An explicit store query plan

`learning.store_query.StoreQueryPlan(query, answer_variable, allowed_scopes)`
retains a strict Proposition query, its answer variable, and a nonempty tuple of
allowed literal scopes. `StoreQueryExample(id, source_id, text, question, plan,
basis=())` supplies teaching to `fit_store_queries(training, validation,
max_pairs=256)`. A learned correspondence transfers reference identities from
a full structured Question into that plan while preserving supported non-reference
structure. The intended scope is exact structured transfer from teaching, not
arbitrary natural-language question understanding.

Model admission and query selection remain explicit. Neither one candidate nor
the absence of an alternative gives a query authority automatically. Unvalidated
or conflicting teaching must remain visible rather than being replaced by a
first-open-role guess. Unknown questions cannot fall back to predicate symmetry,
VerbNet roles, or an implicit scope list.

Allowed scopes are part of the selected plan. Retrieval must not broaden from a
specified source to another merely because the latter yields an answer.
`query.scope=None` is a literal scope, not a wildcard. `allowed_scopes` is an
additional admission set; listing several scopes does not expand the query.
Every Question Ref must occur in the actual query, including literal scope;
merely listing a participant in allowed scopes does not preserve its meaning. Strict
matching must preserve typed values, polarity, modality, validity, and the taught
role constraints; a positive fact cannot silently answer a negative or qualified
question.

The agent uses `retain_store_query_example`, `fit_store_query_model`, and
`admit_store_query_model` in `agent.store_query_learning` to retain and admit
teaching. Assigning the handle to `agent.store_query_model` enables proposals;
`agent.store_query_selector` supplies the separate explicit query choice.
`propose_store_queries` and `select_store_query` retain the full question and
selection dependencies. No admitted model returns unknown, with no legacy lookup
fallback. Direct `Agent.lookup` also uses this boundary.

## Answers retain support and limitations

The answer path should retain the selected plan, source question, dependencies,
supporting store records, and their evidence. `evaluate_store_query(agent, selection)` returns `StoreAnswer` with answer values,
record IDs, selected plan, and a retained evidence source, or `Unknown`. Its
payload preserves the inspected records, exact support, and their original
evidence. A returned value is not a source-free fact or an independent
observation by the agent. Supporting records require nonempty, typed Evidence. Opposite-polarity records
with overlapping validity intervals block an answer; equality of interval bounds
is not required to detect that contradiction. Retraction or contradiction of
support must remain visible and must not be resolved by ranking a convenient
record first. `validate_store_answer(agent, answer)` rechecks selection/model
dependencies, retained evidence, and the entire inspected predicate population.
Any change to that population conservatively invalidates the cached answer,
even if a particular change would not alter its returned value.

A query can exhaust the stored records without exhausting what is true in the
world. Missing support means no established answer under the selected plan; it
is not an inferred negative fact or proof that the requested entity does not
exist. Conflicting and retracted records must not silently supply an apparently
unambiguous answer.

A storage prerequisite was also repaired: Proposition content identity now
includes validity intervals recursively, so distinct historical observations do
not merge their evidence and remain distinct after JSON reload. Canonical omission
of an unbounded interval preserves existing unbounded IDs. Time-qualified IDs
change; opaque external ID strings are not automatically migrated. Retraction
supersession retains a proper Evidence tuple instead of treating a datetime as
evidence. These storage changes do not make the general matcher temporally strict.

Records with `derived_from` evidence explicitly defer until derivation support
and replay can be authenticated. Merely finding the named premises cannot prove
that a claimed inference is valid, so this milestone does not authorize derived
answers through a provenance-shaped shortcut.

The stricter selected-query path does not change every use of the global
`records.matches` utility. Its other consumers and broader logical semantics
remain separate work. No complete temporal, modal, or four-valued logic is claimed
from one guarded store retrieval interface.

## Breaking behavior and verification

Removing the passive symmetry/role/ranking/default-scope shortcuts changes callers
that depended on implicit lookup behavior. Explicitly authored test models can
isolate other mechanisms, but migrating those fixtures does not demonstrate that
the new learner inferred their semantics from free text.

The records and new temporal storage tests passed **21 tests**. The pure store
query learner passed eight new tests alongside twelve existing informing tests.
After the pronoun-projection cleanup, the final actual-input/semantic run passed
**25 tests in 28.11 seconds** (24 neutral/semantic-frontier tests and one actual
store integration). The final store-wrapper run passed **22 tests**. Earlier wrapper verification
included nineteen informing tests, and the final migration run passed 56 tests,
including fifteen proposition cases that took 157.58 seconds. These focused runs
overlap and are not additive. The final full suite passed **2647 tests, with
2 skipped, in 683.98 seconds**. A final focused run of store evidence,
actual-input querying, temporal identity, and dependency semantics passed
**42 tests in 34.07 seconds**; the existing frontend harness also passed.
The test uses real trained syntax, taught speech/informing
correspondences, and an actual seven-plant read receipt stored with provider and
observation locator evidence. A store-query model taught with two training owners
and one held-out owner answers for a fresh owner without another provider call.
A populated store without an admitted model or selector remains unknown. Opposite
polarity evidence and retraction invalidate cached answers and later queries.

The audit additionally found that authored pronoun rules assigned first/second
person to “I”/“you” and third person to every other PRON, including “what.” That
entire lexical-person assignment is removed while preserving word, kind, and other
features. Taught “what” query-slot consumption no longer needs an old asked-role
skip. Other POS/lexical/frame projection remains authored; this is removal of an
incorrect commitment, not newly learned pronoun semantics.

Acceptance should show exact full-question correspondence,
held-out reference transfer, explicit scope/selection, typed and qualified answer
constraints, retained record evidence, and refusal to use contradictory or
retracted support. The APIs and verified active behavior above implement those
boundaries within the stated teaching and selection constraints.

Teaching plans, source/scope policy, upstream frames and grounding, and selection
choices remain authored. Learned reference correspondence improves the retrieval
path within that supported structure. Remaining gaps include learning richer query
transformations, source reliability, open-world reasoning, and choosing among
competing interpretations through evidence rather than caller policy.

## Next boundary: replayable quantity derivations

The next audit found that existing `Rule.then` receives unrestricted Store access,
so retained matched IDs need not include every fact read by the consequence.
Quantity `derive` also accepts supplied Claims, and `total_of_kind` proposition
premises lack complete retraction propagation. Those mechanisms cannot justify
removing the derived-answer abstention merely because a few premise IDs exist.

A sound next milestone is a registered replayable quantity derivation: detached
retained premises and explicit parameters; an admitted, versioned operator; and
a receipt retaining exact output and premise snapshots. Bounded recursive
validation must check each dependency, contradictions, retraction, and operator
withdrawal before using the result as answer support. The arithmetic operator's
semantics would remain supplied, not learned. This is a proposed next boundary,
not implementation or verification claimed by this checkpoint.
