# Grounded goals without lexical reinterpretation

This checkpoint extends [explicit grounded identity](42-explicit-grounded-identity.md)
to requests and supplied goal specifications. It follows the governing
[structured cognitive workspace objective](36-structured-cognitive-workspace.md).
A supplied reference is an explicit commitment; it must not be reinterpreted from
its accompanying noun when the agent reaches execution.

## Removed execution shortcuts

The single-capability matcher previously sent every argument through `Plugin.refer`,
including direct references and entities with supplied references. Plugins could
reject or change the identity using noun categories, surface spelling, or nearby
arguments. Failure explanations could then replace the actual failure with another
WordNet-based judgment. These paths have been removed.

Supplied `GoalSpec` roles also previously passed through `verbnet.role_class`.
Distinct names such as `Theme`, `Patient`, and `Material` could collapse into one
role, or have their case changed. Structured callers were therefore not actually
independent of the lexical ontology. Supplied predicates and role names are now
exact domain data in both single-capability and modeled planning.

`Plugin.refer` and `Plugin.denote` are removed from the base protocol. The desktop,
discourse, and quantity implementations no longer contain their old resolution
chains. There is no compatibility flag, renamed fallback, or production import of
the removed rules. Downstream tests supply occurrence bindings and goal contracts
explicitly when they are testing execution rather than interpretation.

## Goal boundary

`GoalSpec` normalizes explicitly bound linguistic entities into their references,
and explicit literals into their values, at construction. Nested containers and
dataclass values are traversed so a hidden ungrounded entity cannot bypass the
boundary. Existing typed domain values remain typed values. Unresolved entities,
unknown values, and missing top-level role values are rejected with a location.

A bound entity contributes its explicit identity, not additional requirements
inferred from its descriptive features. Required properties must appear as separate
conditions in the supplied specification. This normalization does not establish
that every modifier in the original language has been translated into a goal;
that remains the responsibility of the interpretation/refinement boundary.

The normalization is shared by both execution paths. Modeled planning and direct
capability matching therefore do not disagree about whether a bound `Entity` or its
`Ref` is the goal value. Explicit values such as zero, false, empty strings, and the
literal string `addressee` are not absent arguments. The latter is a sentinel only
inside the lexical adapter, not a reserved word in a supplied world model.

Refinement inputs and executable specifications are distinct. An explicit recipe
may inspect a lexical proposal containing unresolved descriptions. It can emit a
`GoalSpec` only after every retained condition is grounded. An unresolved material
or product constraint cannot be discarded to make a recipe succeed: the refiner
returns `Unknown("incomplete_refinement")`. Supplying both references preserves
the complete remaining relation. No permissive executable-goal compatibility mode
was added for old tests.

For lexical `verbnet.Goal` values, role translation remains an explicit adapter.
Mapping two supplied participants into the same lexical role cannot silently
replace the first value with the second. This adapter is authored interpretation
machinery; grounding participants does not establish that it inferred the intended
result state correctly.

Capability applicability belongs to declared preconditions and actual executor
validation. A `Param.kind` label is not proof that a noun denotes the desired entity.
Preserving an explicit argument does not bypass precondition checking, receipt
handling, post-action observation, or effect verification.

## Ambiguous action choices

Several complete grounded capabilities can advertise the same coarse result.
The old parameter-count preference or declaration order cannot establish which
report or action the user intended. The single-capability path now refuses multiple
complete plans with `ambiguous_capability`; no candidate executes. The existing
operation/constraint machinery remains for a uniquely feasible action and for
explaining incomplete candidates.

This is deliberate incompleteness in the absence of sufficient models or an explicit
choice. It is not a learned action-selection policy. Tests of report contents expose
the intended report contract as authored fixture knowledge. Production does not
silently copy that fixture selection.

## Domain cleanup

The desktop adapter no longer guesses paths from words, searches candidate filename
fragments, chooses applications by substrings, classifies nouns into file kinds, or
appends a source basename to an explicitly supplied destination. Its executor checks
provider identity, declared parameters, reference namespaces, and absolute path
arguments before dispatch. Directory listing no longer substitutes the home directory
for an invalid target. Actual executor/OS observations establish applicability.
Postconditions use the exact supplied destination.

Discourse reports retain their actual trace, capability, and belief-store rendering.
Their old noun/possessor classifiers and lexical reference minting are gone.

Quantity observation now associates each counted occurrence with the explicit subject
reference in its local clause. Two identical surface descriptions with different
references do not share an owner merely because their text matches. Subjectless or
unresolved quantity mentions do not acquire synthetic `entity:<noun>` identities.
Number and unit extraction remain supplied interpretation machinery, not learned
concept discovery.

## Evidence and remaining gaps

Tests exercise exact custom roles and role case, bound and direct references,
contradictory surface nouns, both planning paths, unresolved inputs, ambiguity under
reversed declaration order, unchanged explicit destinations, executor rejection
before mutation, and quantity occurrence identity. Actual computerworld tests execute
valid writes/renames and verify malformed calls do not mutate the environment.

These tests isolate mechanisms with authored goals, action models, and reference
bindings. They do not establish that arbitrary language or pixels can generate those
specifications. The language request path still uses lexical goal construction;
selecting an interpretation of participants is not yet selection of a fully evidenced
intention. General goal inference, learned applicability, and choosing between
competing action models remain open.

Standalone language converter defaults still require migration away from
surface-derived identity. Other authored semantic machinery, including lexical
result-state conventions and discourse context, remains outside this checkpoint.
No claim of completed generalized cognition follows from removing these execution
shortcuts.
