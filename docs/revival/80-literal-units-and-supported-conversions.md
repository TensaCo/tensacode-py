# Literal units and supported conversions

[Explicit measurements and calculations](79-explicit-measurements-and-calculations.md)
exposed a remaining semantic defect: unit spelling normalization could collapse
distinct symbols, and mutable global definitions could reinterpret a stored
quantity without changing its structural derivation receipt. This milestone
removes implicit unit interpretation and requires supported, explicitly selected
conversion evidence.

A literal unit is not a learned physical concept. The operation, measurement,
conversion definition, application scope/time, and selected operands remain
supplied. An actual-question correspondence may learn to use that contract;
it does not learn which spelling denotes a unit or which exchange rate applies.

## Immutable literal unit algebra

`quantity.Unit` is an immutable product of literal string symbols and integer
exponents. `Unit.of(symbol, power=1)` preserves the supplied symbol exactly;
`Unit()` is dimensionless. Case, spelling, plurality, and punctuation are not
normalized into another symbol. The formal `dimension` signature is the symbol
product, not a table lookup of physical dimensions.

The implicit `BASE_UNITS`, `ALIASES`, normalization, base/factor interpretation,
and unit-text parser paths are retired. There is no mutable global definition
that can turn an already retained metre into a time unit. Addition/subtraction
require exact unit equality; multiplication and division combine formal symbols.
No missing equivalence or scale is guessed from conventional names.

This is a breaking boundary. Callers that mean two symbols to be related must
supply that relationship as evidence rather than depend on spelling aliases or
built-in conversion knowledge. Pure arithmetic does not read such evidence from
its environment implicitly.

## A conversion is selected evidence

A `quantity_conversion` proposition retains an explicit definition Ref, source and
target Unit, factor, scope, validity interval, and supplied Evidence. A registered
conversion calculation names its ordered measurement and definition operands.
Its parameters explicitly state the application scope and interval, and the
result preserves that metadata.

The source unit must match the selected definition, its scope must match the
application, and its validity must cover the requested interval. The operator
does not choose “now,” reverse a definition, construct a multi-hop path, or choose
between competing rates. Conflicting definitions or overlapping rivals cannot
be silently resolved by taking the first stored conversion.

Authenticated replay retains both measurement and conversion populations so
later retraction, changed definitions, or new rivals invalidate stale support.
Selecting a definition is an explicit interpretation commitment, not proof that
its factor is correct. Source reliability and the meaning of time/scopes remain
separate responsibilities.

`QuantityPlugin.remember_conversion(definition, source_unit, target_unit, factor,
*, scope, valid, evidence)` records the supplied definition. The factor must be
finite and positive. `remember` now preserves explicitly supplied measurement
scope, validity, polarity, and modality rather than discarding them.

The caller registers `"convert"` with the ordered IDs of a measurement (or an
authenticated derived numeric result) followed by its definition, and with
`params={"scope": scope, "valid": interval}`. Registration still requires an
explicit calculation context and basis; `select_calculation(..., reason=...)`
is a separate commitment. The numeric operand and definition must both cover
the application interval and have exactly its scope. The output and declared
answer query preserve the requested scope and interval. Other arithmetic
operations refuse qualifications they cannot preserve.

Adding any definition changes the retained conversion population and invalidates
the old receipt until recomputation. Recalculation uses the originally selected
operands; it does not adopt the new rate. Incompatible calculation contracts
sharing a capability name are unavailable rather than chosen by insertion order.
These guarantees preserve supplied literal identities and checked definition
support, not universal physical-unit understanding.

## Retired relation and numeric shortcuts

The unused heuristic `relation.py` reader is retired rather than adapted into
an alternative route around the new boundary. `semantics_bridge` removes `Mention`, `WORD_NUMBERS`, `MONEY`, numeric
`quantities_in`/text projection, and `tell_mentions`, without a fallback.
`eval/relations/eval_relations.py`, `eval_gsm8k_transfer.py`, and
`eval/structures/gsm8k.py` now report `retired` with `measured: false`, without
scores, network access, or output-file writes. Historical JSON results remain
unchanged. Twelve retirement/bridge tests passed, including all three entry
points under isolated Python (`-I -S`) outside the repository, checking that no
files or scores are generated and historical artifact bytes remain unchanged.
There is no replacement language-arithmetic benchmark claim.

Saved measurements describe their historical implementations. They do not establish
current unstructured-language arithmetic or unit grounding. Other authored
conditional and likelihood code remains a separate semantic gap; removing this
reader does not remove every hardcoded interpretation in the repository.

## Acceptance evidence and remaining limits

A combined unit, literal-storage, conversion, calculation, and bridge run passed
89 tests in 0.26 seconds. The separate retirement/bridge run passed 12 tests.
These overlapping runs are not additive coverage counts. They cover distinct
literal symbols such as `gas`/`ga`, `news`/`new`, and `MS`/`ms`, immutable retained
interpretations, explicit operands and context, scoped chaining, and rejection
of unsupported, conflicted, withdrawn, inverse, or out-of-interval conversions.

`tests/test_learned_conversion_inputs.py` passed one actual-input integration
case in 19.65 seconds. It uses the trained segmentation/POS/dependency reader,
explicitly taught communicative interpretation, and full-Question informing
correspondences trained with Alpha/Beta contexts and validated with Gamma. For
a fresh explicitly bound owner, a supplied measurement of two `plant_bundle`
and an explicitly selected factor-four definition produce eight `plant` through
the learned informing correspondence. The imported derivation proof is checked.
Adding a rival factor-five definition invalidates the cached support and makes
the next real-input query unknown/rejected; withdrawing the rival and selected
definition cannot restore the old answer.

The learned part is the full-question reference correspondence. Measurement
identity, unit symbols, grounding, rate evidence, operation, operand selection,
and interpretation selection are supplied. The test does not establish unit-word
understanding or autonomous rate choice. A further audit found malformed interval
endpoints could admit booleans. `Interval` now validates datetime endpoints,
ordering, and compatible timezone awareness during construction and temporal
operations. Valid naive datetimes remain supported without assigning a timezone.
Replay also rejects malformed retained intervals. The combined interval,
conversion, storage, and calculation regression run passed 54 tests in 0.21 seconds.
The full repository run completed with **2,738 passed, 2 skipped in 850.16
seconds**, exit status zero. Its log is
`/tmp/tensorcode-literal-unit-suite/output.log`. New contextual-correction work
started separately after collection and is not covered by this checkpoint's
full-suite result.

Broader goals remain unfinished: learning unit meanings and equivalences from
observations, acquiring or judging conversion sources, planning conversion chains,
understanding novel questions, and applying source-grounded task corrections.
Literal algebra and replayable conversions repair a concrete semantic instability;
they do not complete generalized cognition.
