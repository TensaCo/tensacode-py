# Explicit measurements and calculations

[Authenticated derivations](78-authenticated-derivations.md) made an arithmetic
result replayable, but did not justify treating every measurement about an owner
as an addend. Equal-valued independent measurements could collapse under content
identity, and different values could be rival observations rather than quantities
to sum. This milestone makes measurement identity, operation, operands, and
calculation selection explicit and removes the remaining automatic quantity
interpretation paths.

Arithmetic remains supplied. A learned informing correspondence can map a taught
Question to an explicitly selected calculation capability; it does not discover
which measurements should be combined or which operation expresses the user's
intention. This is a stronger evidence and selection boundary, not generalized
mathematical reasoning.

## Measurements are identified observations

A measurement has an explicit Ref and supplied Evidence, separate from its owner,
optional kind, predicate context, and quantity value. Two distinct measurement
identities with equal values remain distinct observations. Repeating the same
identity and value preserves repeated evidence rather than creating another
operand. Conflicting values for the same identity must not silently become two
addends.

The retained `quantity_measurement` envelope carries the supplied predicate as
context data. It does not assert a quantity under an arbitrary predicate and then
infer additivity from that predicate or its owner. Observation evidence remains
supplied; measurement identity does not independently establish that the sensor
is reliable or the measurement is correct.

## Explicit ordered operations and selection

A calculation registration names an admitted arithmetic operator, ordered operand
record IDs, explicit context, parameters, and basis. Selecting that registration
is a separate commitment. Neither storage order, the first compatible unit, nor
all facts about an owner chooses the operands. Ordered subtraction or multiplication
must retain exactly the caller-selected inputs.

The supplied operator set includes `sum`, `sub`, `mul`, `div`, `scale`, `ratio`,
`percent_of`, `compare`, `convert`, and `count_selected`. Typed units and operation
arity constrain execution. A selected empty collection can support an explicit
count of zero when that nullary operation is admitted; it does not establish zero
objects or properties in the world. The explicit empty-count case is covered by the calculation tests below.

The active plugin API is:

- `remember(owner, predicate, quantity, *, measurement, evidence, kind=None)`
  retains an explicit `quantity_measurement` proposition and supplied Evidence.
- `register_calculation(operator, ordered_operand_ids, *, context, params=None,
  basis)` returns a calculation Ref. `CalculationContext(owner, predicate,
  kind=None)` supplies output context. Conversion requires a unit parameter;
  scaling requires an explicit factor.
- `select_calculation(reference, *, reason)` admits a registration-bound operator
  version for that context and records the separate choice. Replacing the choice
  withdraws the earlier version's authority.
- `calculate(reference)` refuses an unselected registration and returns an
  authenticated derivation result or `Unknown`.

Only selected registrations expose `calculate_<operation>_kind_<predicate>` or
`calculate_<operation>_owner_<predicate>` informing capabilities. Results retain
`calculated:<operation>:<predicate>` identity, rather than being relabeled as an
original measurement. Ordered operand selection remains in the retained registration.

The detached `.registrations` view and `selection_history` retain operand plans
and explicit selection reasons/revisions. Old `POSSESSION`/`world_predicate`
compatibility declarations are removed with the implicit plugin path.

No automatic owner/kind scan, rate-times-first-compatible-count rule, property
classification, or observation-time ownership inference remains in the intended
quantity plugin path. Old `derive`/`tell_quantity` shortcuts are removed rather
than renamed into a compatibility fallback. Historical observations and registered
calculations must not be relabeled as direct facts to bypass lineage validation.

A related unit audit removed implicit currency parity from `BASE_UNITS`: coin,
dollar, euro, sterling, and cent are distinct dimensions. An unqualified cent is
not silently USD, and exchange requires explicit rate evidence rather than equal
unit factors. This does not remove every authored unit interpretation. Existing
unit definitions, `ALIASES`, and `normalize_unit` stemming remain in `quantity.py`
and are used by `Unit.of` and language-related projection paths. Current arithmetic
must not be described as preserving every literal input unit unchanged; migration
of those semantic shortcuts is separate work.

## Replay includes the inspected population

Authenticated derivation receipts still retain exact premises, parameters, output,
and operator version. They also retain the full relevant measurement population
needed to detect another value for a selected measurement identity. A new conflicting
measurement or changed record invalidates a prior proof instead of leaving its
selected premise snapshot apparently sufficient. Even a new unrelated measurement
can conservatively invalidate the retained full-population receipt; recomputation
still uses the same explicitly selected operand IDs rather than adding the new
record to the calculation.

The generic derivation interface replaces a single optional population
predicate with canonical `population_predicates`, so all relevant inspected
populations can participate in replay. That read set is evidence for what the
operator inspected, not authority to use every record as an arithmetic operand.
Selected operands and validation population remain separate concepts. The full
measurement population is `quantity_measurement`; derived results used as operands
are exact retained premises, not an inferred aggregation of every result sharing
a predicate. This allows a same-operation calculation chain without treating its
new result as an unselected input. Derived
calculation outputs can be operands only through authenticated derivation support;
a `calculated:` predicate prefix creates no authority. Required-derived support
persists through replay, blocking both a forged observed calculation output and
a later direct-observation copy used to bypass withdrawal of its operator.
Registration `operand_ids` are the actual ordered operands; receipt `premise_ids`
include the full inspected population used to validate rival measurements.

The provider exposes only explicitly selected calculations as informing
capabilities. It reveals an authenticated derivation reference after execution,
not an unauthenticated Claim hiding arithmetic lineage. Selection, operator,
premise, or population changes must invalidate stale revelation and learned store
answers using it.

## Acceptance evidence and remaining limits

The final quantity-focused run passed **52 tests**: eighteen calculation tests,
six migrated derivation tests, and twenty-eight unit tests. A downstream fast run passed 28 tests; an earlier broader 31-test run
included English fixtures. These runs overlap and are not additive. A root fast run passed **101 tests in 3.09 seconds** across derivations, quantity
calculations/derivations/units, and derived store answers. Five actual-input tests
passed in 133.77 seconds. The same-operation derived-chain regression also passes, including upstream
operator withdrawal invalidating the chain. The final full suite passed **2718 tests, with 2 skipped, in 782.89 seconds**. Required integration covers distinct equal-valued
measurements, repeated evidence for one identity, rival values, exact ordered
operands, explicit selection, typed arithmetic, replay invalidation, and learned
informing/store answers over the selected calculation. The actual-input teaching
must be described as mapping to an explicit calculation choice, not learning an
autonomous operand-selection policy.

This milestone changes the quantity API and removes behavior that used to guess
what a quantity question meant. Supplied fixtures may register calculations to
isolate downstream mechanisms; those registrations are not learned arithmetic
intent. Broader task correction, paraphrase understanding, selection among rival
measurements, aggregation policy learning, and reliable source interpretation
remain unfinished. No claim that the governing general-cognition objective is
complete follows from explicit calculations or replayable receipts.

## Next unit-interpretation boundary

A concrete follow-up audit found premature normalization collisions:
`Unit.of('gas') == Unit.of('ga')`, `news`/`new`, and case-distinct `MS`/`ms` can
collapse through aliases or stemming before evidence establishes equivalence.
The global `BASE_UNITS` interpretation is also mutable. An authenticated scale
of two metres by two can retain the same output structure and still validate
after the metre definition is changed to a time dimension with factor 999;
the retained quantity's dimension and base interpretation have changed underneath
that structural proof.

The current process-local receipt therefore does not freeze unit-interpretation
semantics. The next fix needs literal identity plus stable source-bound unit
definitions and explicit conversion evidence. Merely comparing mutable globals
after an already stored measurement has been reinterpreted would not repair its
lost original meaning. These are documented remaining defects, not capabilities
or fixes claimed by this checkpoint.

[Literal units and supported conversions](80-literal-units-and-supported-conversions.md)
addresses the normalization and mutable-definition defects above. It preserves
literal identities and requires selected conversion evidence; it does not learn
unit-word meanings or choose a rate autonomously.
