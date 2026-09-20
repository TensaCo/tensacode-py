# 59 — Semantic preservation at the goal boundary

*2026-09-19. This strengthens the transition from a selected interpretation to an
executable goal, following [58](58-interpretation-dependent-tasks.md). It blocks identified
loss of meaning; it does not supply a general interpreter for the rejected qualifications.*

## A stable selection can still become the wrong goal

An interpretation dependency establishes which reading authorized a task and whether that
reading is still current. It does not establish that a later adapter preserved the reading's
meaning. A negated request can remain correctly attached to its source while a lexical
result-state adapter produces an affirmative destructive goal. A referenced entity can
retain its identity while normalization discards a count or modifier. Those are semantic
losses before planning, not failures of task revision tracking.

The reproduced failure was concrete. With installed VerbNet, an explicitly constructed,
selected, grounded negative `delete` frame reached an in-memory deletion capability and
returned `done`. An additional preservation role was ignored, and a `count=3` qualification
on a referenced target could collapse to one deletion; a `temporary` modifier was also
lost. These were authored interpretation/action fixtures, not a demonstration that the
learned reader correctly parsed a user sentence or that real files were deleted.

A prohibited deletion must not become a request
to achieve the ordinary deletion result, and a qualified reference such as “three targets”
must not become an unqualified single identity merely because a `Ref` is available. A
prohibition is not automatically represented by negating the verb's final-state predicate;
for example, maintaining a constraint against an action and requesting its opposite state
are different obligations. The current adapter cannot decide that distinction generically.

## What the active boundary now rejects

The lexical request adapter explicitly consumes imperative mood as the request wrapper.
Every other frame feature remains an obligation. Features including polarity, modality,
tense, or arbitrary future qualifications cannot disappear because the adapter has no field
for them. An explicit subject is also rejected at this boundary: the existing result-state
adapter does not account for it. Failure reports identify the unconsumed feature/role path.

The frame guard runs before domain refinement so a custom refiner cannot silently turn a
negative request into a positive goal by ignoring its frame features. Lexical goals with
unmapped roles are also barred from direct goal execution and planning. These same checks
apply to supplied/resumed lexical goals; bypassing the initial request path does not bypass
the obligation. A lack of support yields unresolved behavior rather than partial action.

`normalize_goal_value` now rejects any `Entity` with unresolved candidate alternatives,
even if it also carries an explicit reference. A bound entity with features cannot collapse
to that reference: count, modifiers, definiteness, noun labels, and unfamiliar feature names
all require an explicit semantic projection first. No list of supposedly harmless entity
features is silently stripped. Nested containers and goal conditions/invariants share this
boundary and report the offending value path.

Two narrow representation cases remain supported:

- A featureless entity with an explicit `Ref` is an identity wrapper. Its display text is
  not itself interpreted as an additional constraint.
- An unbound `number` or `literal` may consume its scalar `value` field. Extra qualifiers,
  such as a unit, must still be represented explicitly. A world-bound entity cannot use
  this exception to discard a `value` qualification.

Callers can explicitly represent a quantity as a domain condition and preservation as an
invariant, then supply domain values. That representation is an authored semantic decision.
The normalizer does not certify that a refiner encoded every qualification correctly, and
a supplied `GoalSpec` is not proof of entailment from the original language.

## Evidence and compatibility consequences

Regression fixtures supply frames, reference bindings, lexical result-state entries, and
capabilities to isolate the dangerous conversion. They verify rejection before dispatch,
not learned understanding of arbitrary negative or quantified language. Entity tests also
cover unresolved alternatives, nested qualification paths, scalar representation exceptions,
and explicitly authored quantity/preservation conditions.

Previously accepted inputs may now defer because their semantics were only partly handled.
That is intentional: acting after deleting a constraint is not equivalent to understanding
the request. No compatibility switch restores qualification stripping or permits a custom
refiner to erase frame obligations implicitly. Positive featureless requests and genuinely
explicit domain goals remain subject to their existing grounding and execution checks.

The entity-preservation/task subset passed 30 tests. The request boundary passed 13
tests, including three cases using the installed VerbNet resource and an in-memory
file executor. A separate 68-test run covered request preservation, refinements,
grounded execution, task revisions, and interpretation dependencies. The positive
execution fixture migration passed 46 tests; make/move tests now supply explicit,
authored goal projections, while a separate test confirms that identity alone does
not authorize the qualified request. These overlapping runs are not additive.
The full repository run completed with 2,094 passing tests, five skips, and three
failures in the old discourse request fixtures (314.61 seconds). Those fixtures
also needed explicit report-goal projections rather than assuming that topic
identity consumed possessive/nominal qualifications. After correcting that shared
test helper, all 15 discourse tests passed in 11.72 seconds. No production code
changed after the full run began; its other tests and the corrected discourse
module cover the final runtime. These guards
do not establish autonomous meaning selection, general negation semantics, quantity
reasoning, or generalized cognition.

## The next unresolved boundary

Lexical goal construction still uses lexical-resource frequency priors and ordering. Preposition-to-role
projection can also offer competing mappings whose selection is not generally established.
Retaining those alternatives does not guarantee that the executable goal is the right one.
The next step is to preserve competing semantic obligations through explicit, inspectable
goal interpretations and justify their selection with evidence—not to replace a lost
qualification with another hardcoded action rule.

The audit also found that the current lexical resource loader retains category/role
pairs but drops some syntax restrictions. The interpreted frame does not preserve
all complement ordering and literal prepositions either. Comparing flattened slot
lists therefore cannot prove a construction incompatible. A future goal-proposal
enumerator must retain unsupported derivations as unresolved alternatives, preserve
competing injective preposition-role bindings, and expose incomplete search. Neither
resource order, a frequency prior, an executable recipe, nor one surviving proposal
after lossy filtering establishes intended meaning.

The governing requirement remains source-to-goal meaning preservation: every relevant
qualification must be represented, explicitly discharged with evidence, or left unresolved.
Blocking the known lossy boundary is progress toward that requirement, not completion of it.
