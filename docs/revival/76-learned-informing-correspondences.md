# Learned informing correspondences

[Learned communicative interpretations](75-learned-communicative-interpretations.md)
separated a provisional frame from the decision that it is a question. The next
boundary is determining which observation could answer that full question. The
previous `_look` and `attend` paths could choose an `undergoer` from subject/object
insertion order, losing the distinction between an owner and a counted kind.
This milestone develops a taught, explicitly selected Question-to-informing-plan
correspondence instead of that order-dependent authority.

The proposed capability is bounded structured transfer. Full questions, teaching
plans, measurement semantics, and admission/selection policies remain supplied.
The model must learn a correspondence from those examples; replacing a helper
with an unexplained fixed mapping would not establish learned interpretation.
Actual-input and full-suite verification are recorded below.

## Full question, full answer plan

An informing interpretation must retain the entire Question, including queried
role, participants, polarity, and entity or frame qualifiers. A plan includes the
informing capability call and the answer query that makes its resulting evidence
relevant to that Question. The target is not merely selecting a capability name
or extracting the first participant from a frame.

The pure `learning.informing` API defines
`InformingPlan(plugin, capability, args, answer_query, answer_variable)` and
`InformingExample(id, source_id, text, question, plan, basis=())`, fitted through
`fit_informing(training, validation, max_pairs=256)`. The answer query is a full
`Proposition`, and its declared answer variable must occur in that query.

Only input reference identities are abstracted; words, queried roles, qualifiers,
capability names, polarity, and metadata remain literal teaching data. Dictionary
keys are canonicalized so insertion order does not supply meaning. Independent
example/source IDs and normalized teaching texts are required. Held-out variable
references cannot reuse training identities; explicit context references shared
across all examples can remain constants. Pair-budget exhaustion remains visible. Every input Question Ref must be retained
in actual arguments or the answer query; dropping a counted-kind Ref into an
owner-only plan makes the fit incomplete. This reference coverage check is not a
proof that literal qualifiers were assigned the correct effect semantics.

A singleton conflicting training plan remains an `unvalidated_training_rival`,
with `conflicting_training_example_ids`, rather than disappearing because another
plan had more examples. It blocks authorization unless an equally applicable
supported alternative covers that rival plan. Literal qualifier-to-observation
meaning still comes from teaching.

Teaching and held-out reference variation should support transfer to a fresh
owner while preserving fixed constraints and exact non-reference structure.
Reordering a frame's role dictionary must not change the chosen owner. A counted
plant restriction must not silently become an unrestricted quantity query, and
a coin restriction must not return the plant count. Unsupported qualifiers must
cause abstention, not be discarded to make an available capability applicable.

Competing supported plans remain alternatives. Explicit model admission and plan
selection are separate commitments. A singleton does not authorize itself, and
an unvalidated rival cannot vanish because another plan looks executable.
Existing interpretation dependencies must remain attached to the observation
and resulting answer.

The retained agent APIs in `agent.informing_learning` are
`retain_informing_example(agent, group_id, candidate_id, act_index, plan, *, basis)`,
`fit_informing_model(agent, training_records, validation_records, *, group_id=None,
max_pairs=256)`, and `admit_informing_model(agent, handle, *, reason)`. Assigning an
admitted handle to `agent.informing_model` enables proposals, while
`agent.informing_selector` explicitly chooses a plan. `propose_informing` retains
a full-question group and `select_informing` captures that choice and dependencies.
No configured model or no selector supplies an implicit informing operation.
`InformingSelection` retains the plan and dependencies; `Outcome.plan` retains
that selection and `Outcome.receipt` the read receipt. The exact selected full
Question parent and complete source text remain evidence. Model, inherited
reading, and plan-choice dependencies are checked around invocation and reveal;
the provider must declare the exact read capability/query contract. Legacy
`attend` and `_filler_for_role` are removed.

The audit also found the same singleton-rival loss in the speech-act and goal
correspondence learners. Those paths now retain `conflicting_training_example_ids`
and an `unrepresented_training_rival` reason when no applicable validated
alternative represents the contrary example. Fully supported competing outputs
remain alternatives. `LearnedGoalProposal` retains the conflicting IDs. The goal
wrapper preserves that metadata and blocks unresolved batches before invoking a
selection policy, immediate dispatch, or delayed commitment. This extends the uncertainty
boundary; it is not evidence that any conflicting teacher is necessarily correct.

## Observation and quantity scope

The quantity path must actually observe the taught restriction. An owner-only
capability cannot satisfy a question about a particular counted kind merely by
reporting a number. The provider's operation and result vocabulary remain an
authored contract; a learned plan can supply the correct arguments only within
what that contract expresses.

`QuantityPlugin.remember(owner, predicate, quantity, kind=...)` retains an explicit
counted-kind Ref. `amount_of_kind_<predicate>(owner, kind)` is the read capability,
backed by `total_of_kind(owner, predicate, kind)`. It sums only matching explicit
owner/kind measurements and retains arithmetic derivation provenance. Neither
noun spelling nor units infer kind identity. No matching record is unknown, not
zero. Overlapping records with unsupported extra roles, polarity, modality, time,
or scope refuse instead of being included or silently filtered away.

`count_properties(thing)` is an explicitly selected measurement of stored
value-valued records, not an inference about whether the question meant an amount,
relationship, or property. Entity relations are outside this authored counting
domain. A completely unknown thing does not become zero world properties. The
old property-intent heuristics are removed; the provider contract remains authored.

Actual observation evidence must be retained and assessed through the selected
answer query. Provider-reported answers must retain the provider as their evidence
source, with the retained observation source ID as a locator; the agent storing a
report must not make it appear independently observed by a different authority. A matching taught plan is not an observed answer, and a successful
provider call is not proof that it answered every constraint. Stale model,
question, selection, or source evidence must prevent a prior correspondence from
silently authorizing a new answer.

Removing `_look`/`attend` role-order guesses is a breaking behavior change for
callers that previously relied on them. Authored test mappings can isolate other
mechanisms, but their migration is not evidence that the new model independently
understood the original language.

## Acceptance evidence and remaining limits

The retained informing wrapper/core passed **19 evidence tests**, and the pure
informing learner passed **12 focused tests**, including full-reference
coverage and singleton conflicting-training rivals. The actual-input integration
passed in **14.54 seconds**. It uses real trained segmentation/POS/dependency
readings, speech-intent teaching on Alex/Toni questions with Shondra held out,
then the retained Shondra question within distinct Alpha/Beta/Gamma source contexts.
Explicit owner/kind bindings support two informing-training owners and one held-out
owner. Execution for a fresh owner with seven plants and nineteen coins returns
seven plants through an actual read receipt, preserving the counted noun and
qualifiers. No informing selector makes no call; withdrawing the model prevents
further calls.

The source contexts, teaching, reference bindings, inventory records, and selection
policies are authored. This test demonstrates composition of actual trained syntax
and supervised correspondences, not paraphrase transfer, general identity grounding,
or vision. Separate existing-regression and audit runs passed 96 and 86 tests
respectively; these runs overlap and are not additive. The final full repository suite passed **2,605 tests, with two skipped**, in
631.23 seconds (exit status 0). After the strict-answer repair, the final combined run passed
**35 tests in 34.51 seconds** (nineteen wrapper, one actual-input, fifteen discourse).
Exact kind and owner identity are supplied evidence rather than inferred
from names in these examples.

Passive lookup remains a separate authored path, unchanged by this informing
correspondence. The structured question and teaching still reflect authored upstream projection.
An outcome here does not establish general English question understanding,
learned measurement ontology, source reliability, or unrestricted query planning.
It should establish that a supported correspondence governs actual informing
arguments and answer retrieval without dropping the question's retained structure.

A read-only follow-up audit found a separate passive-answer failure: a stored
positive `located(subject=x, place=desk)` fact can answer a negative question or
a question whose same-Ref subject retains an unsupported red qualifier.
`_question_bindings` discards Entity qualifiers and the default positive lookup
query does not preserve the full frame features. The shared `records.matches`
also accepts Python's `True == 1` equivalence and does not enforce validity intervals.
The same gap was found in the new informing answer path, so its running full
suite was stopped for a local strict-answer matcher repair before claiming
completion. The local informing matcher is now repaired and its nineteen wrapper
tests pass: recursive typed role/metadata equality preserves `True` versus `1`,
repeated-variable type identity, and exact scope/validity without wildcard omission.
Zero returned observations yield unknown even with an applied receipt; an explicitly
observed empty tuple is a distinct valid answer value. Passive lookup and the global
matcher remain separate limitations. Next passive-answer acceptance must reject
qualifier/polarity loss, preserve typed scalar distinctions, and check the required
time/scope constraints.

Remaining work includes learning broader transformations and measurement choices,
combining multiple observations, resolving ambiguous questions through evidence,
and understanding novel unstructured descriptions. This milestone must preserve
those limits while removing a concrete insertion-order semantic shortcut.
