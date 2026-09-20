"""Explicit quantity measurements and dimension-checked arithmetic.

Amounts retain their units. Counted-kind measurements additionally retain an
explicit Ref for the kind; descriptions, spelling, and unit names never create
that identity. Typed capabilities report a caller-selected owner and kind.
Language-to-measurement correspondences must be independently taught/admitted.
Untyped amounts remain available through explicit owner-only measurements, which
abstain when kind-tagged records would make that request underspecified.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence
from datetime import datetime, timezone

from ..derivations import admit_operator, derive as derive_proposition, export_derivation, DerivationReference
from ..language import Entity, Frame
from ..outcomes import Receipt, Unknown
from ..quantity import Quantity, Unit, add, convert, derive, tell_quantity
from ..quantity import compare as compare_quantities
from ..records import Claim, Evidence, Proposition, Ref, Store, Var
from ..learning.experience import _same
from ..semantics_bridge import quantities_in
from .plugin import Call, Capability, Informs, Param, Plugin

#: What this plugin cannot do from inside the plugin protocol, so that the gaps stay
#: visible rather than becoming permanent (the same discipline as
#: ``semantics_bridge.needed_from_grammar``).
needed_from_the_agent = (
    "a statement is never shown to a plugin, so `observe` has to be called by hand; one "
    "call to it from `Agent.tell` is what would let a word problem's numbers be used",
    "quantity extraction needs explicit occurrence identities and preserved count "
    "features; identity alone does not project quantity semantics",
    "counted-kind questions need explicit kind identities and an admitted correspondence "
    "to the owner-and-kind measurement; preserving a noun does not ground its kind",
    "no reachable question shape binds two owners, so `difference` and `compare` are "
    "callable but unreachable from English ('how many more X does A have than B?' parses "
    "as be(subject=A, object=name:'have B'))",
)

#: Explicit quantity-domain relation; language adapters must supply any alignment.
POSSESSION = "has_possession"

#: The unit of "how many properties": a property is a thing recorded about something, and
#: counting them is a count like any other, so it gets a unit like any other.
PROPERTY = Unit.of("property")

_OWNER = "owner_of_"


def world_predicate(predicate: str) -> str:
    """Preserve the caller's predicate; no implicit lexical translation."""
    return predicate


def _sum_kind_measurements(premises, params):
    """Supplied arithmetic policy, replayed solely against detached operands."""
    if (type(params) is not dict or set(params) != {'owner', 'kind', 'predicate'}
            or type(params['owner']) is not Ref or type(params['kind']) is not Ref
            or type(params['predicate']) is not str or not params['predicate']):
        return Unknown('explicit_quantity_identity_required')
    if not premises:
        return Unknown('nothing_recorded_for_kind')
    owner, kind, predicate = params['owner'], params['kind'], params['predicate']
    result = None
    for proposition in premises:
        if type(proposition) is not Proposition:
            return Unknown('qualified_kind_measurement')
        quantity = proposition.role('object')
        plain = Proposition(predicate, {'subject': owner, 'kind': kind, 'object': quantity})
        if type(quantity) is not Quantity or not _same(proposition, plain):
            return Unknown('qualified_kind_measurement',
                'Overlapping measurement has unsupported roles, polarity, modality, time, or scope')
        result = quantity if result is None else add(result, quantity)
        if isinstance(result, Unknown):
            return result
    return Proposition('total_kind:' + predicate,
        {'subject': owner, 'kind': kind, 'object': result})


class QuantityPlugin(Plugin):
    """Amounts, with the arithmetic that combines them and the refusals that guard it."""

    def __init__(self, name: str = "quantity") -> None:
        super().__init__(name=name)
        self.mind = Store()
        self.source = Ref(f"plugin:{name}")
        self._kind_sum_operator = admit_operator(self.mind, 'quantity:sum-explicit-kind',
            _sum_kind_measurements, reason='Supplied dimension-checked arithmetic policy; not learned semantics')
        self._kind_derivations = {}
        # Lifecycle context is independent of whether a question needs reference resolution.
        self._world: Store | None = None
        self._worked_out: dict[str, tuple[Call, Any, Receipt]] = {}

    def attach(self, agent: Any) -> None:
        """Bind the active world store without guessing identities or running actions."""
        self._world = agent.store

    # ------------------------------------------------------------- being told

    def remember(self, owner: Ref, predicate: str, quantity: Quantity, *, kind: Ref | None = None,
                 method: str = "quantity:told") -> Claim | Proposition:
        """Record an explicit measurement; kind identity is never inferred from its unit."""
        if type(owner) is not Ref or type(quantity) is not Quantity or type(predicate) is not str or not predicate:
            raise TypeError('quantity measurement requires an owner Ref, predicate, and Quantity')
        if kind is not None:
            if type(kind) is not Ref:
                raise TypeError('counted kind must be an explicit Ref')
            proposition = Proposition(predicate, {'subject': owner, 'kind': kind, 'object': quantity})
            self.mind.assert_(proposition, Evidence(source=self.source,
                observed_at=datetime.now(timezone.utc), method=method))
            return proposition
        return tell_quantity(self.mind, owner, world_predicate(predicate), quantity,
                             source=self.source, method=method)

    def observe(self, frame: Frame) -> list[Claim]:
        """Read quantities against explicit identities at their own occurrences.

        The supplied clause structure determines ownership: an object quantity
        belongs to that clause's explicitly bound subject; a counted subject
        belongs to its own explicit reference. Descriptions and equal spellings
        never establish identity. Nested clauses are considered separately.
        Number/unit extraction remains the authored ``quantities_in`` adapter.
        """
        out: list[Claim] = []
        for clause in frame.walk():
            subject = _ref_of(clause.role("subject"))
            for role, value in clause.roles.items():
                for entity in _entity_occurrences(value):
                    owner = _ref_of(entity) if role == "subject" else subject
                    if owner is None or "count" not in entity.features:
                        continue
                    # Extract exactly this occurrence. Nested feature entities
                    # are visited independently, so their quantities cannot be
                    # mistaken for this entity's count or processed twice.
                    local = Entity(entity.kind, entity.text,
                                   {key: entity.features[key] for key in ("count", "noun")
                                    if key in entity.features}, entity.ref)
                    local_frame = Frame(clause.predicate, {"subject": local})
                    for mention in quantities_in(_numerals_as_entities(local_frame)):
                        out.append(self.remember(owner, clause.predicate, mention.quantity,
                                                 method="quantity:read"))
        return out

    # ------------------------------------------------------------ the vocabulary

    def capabilities(self) -> Sequence[Capability]:
        """One informing capability per predicate it has amounts under, plus the count.

        The predicates are read off what it holds rather than listed here, so a host that
        records amounts under "spend" gets a capability that answers "how much did she
        spend?" without anything in this file mentioning spending.
        """
        caps = [
            Capability(f"amount_of_{pred}", (Param(_OWNER + pred, "thing"),),
                       informs=(Informs(pred, "undergoer", _OWNER + pred,
                           query=Proposition(pred, {"subject": Var(_OWNER + pred), "object": Var("answer")})),),
                       effect_kind="read",
                       description=f"the total amount recorded under {pred}, or nothing if it is ambiguous")
            for pred in self._predicates()
        ]
        caps.extend(Capability(f"amount_of_kind_{pred}", (Param('owner', 'thing'), Param('kind', 'kind')),
            informs=(Informs('total_kind:' + pred, 'explicit_owner_and_kind', 'owner',
                query=Proposition('total_kind:' + pred, {'subject': Var('owner'), 'kind': Var('kind'), 'object': Var('answer')})),),
            effect_kind='read', description='Total recorded amount for exactly the supplied owner and counted kind')
            for pred in self._kind_predicates())
        caps.append(Capability("count_properties", (Param("thing", "thing"),),
                               informs=(Informs(POSSESSION, "undergoer", "thing",
                                   query=Proposition(POSSESSION, {"subject": Var("thing"), "object": Var("answer")})),),
                               effect_kind="read",
                               description="how many things the store records about something"))
        return tuple(caps)

    def _kind_predicates(self):
        return sorted({record.proposition.predicate for record in self.mind.propositions()
            if ':' not in record.proposition.predicate
            and set(record.proposition.roles) == {'subject', 'kind', 'object'}
            and type(record.proposition.role('subject')) is Ref
            and type(record.proposition.role('kind')) is Ref
            and type(record.proposition.role('object')) is Quantity})

    def _predicates(self) -> list[str]:
        """The predicates it holds amounts under. Working is filed under "<op>:<pred>" and
        is not a predicate anyone asks about, so it is left out."""
        return sorted({r.claim.predicate for r in self.mind.claims()
                       if isinstance(r.claim.object, Quantity) and ":" not in r.claim.predicate})

    def display(self, ref: Any) -> str | None:
        """An amount reads as its number and its unit; anything else is not this plugin's."""
        return str(ref) if isinstance(ref, Quantity) else None

    # ------------------------------------------------------------- acting

    def execute(self, act: Call, *, key: str | None = None) -> Receipt:
        """Measure exact explicit arguments; unavailable measurements reject the call."""
        caps = [cap for cap in self.capabilities() if cap.name == act.capability]
        args = dict(act.args)
        if (act.plugin != self.name or len(caps) != 1 or len(args) != len(act.args)
                or set(args) != {param.name for param in caps[0].params}
                or any(type(value) is not Ref for value in args.values())):
            return Receipt(act, 'rejected', error='Explicit provider, capability, and exact Ref arguments required')
        self._worked_out.pop(act.capability, None)
        kind = None
        pred = self._predicate_of_capability(act.capability)
        if act.capability.startswith('amount_of_kind_'):
            pred = act.capability[len('amount_of_kind_'):]
            owner, kind = args['owner'], args['kind']
            got = self.total_of_kind(owner, pred, kind)
            answered = 'total_kind:' + pred
        elif pred is not None:
            owner = act.arg(_OWNER + pred)
            got: Any = self.total(owner, pred) if isinstance(owner, Ref) else Unknown("no_owner", "no thing was named")
            answered = pred
        elif act.capability == "count_properties":
            owner = act.arg("thing")
            got = self.count_properties(owner) if isinstance(owner, Ref) else Unknown("no_owner", "no thing was named")
            answered = POSSESSION
        else:
            return Receipt(act, "rejected", error=f"{self.name} does not implement {act.capability}")
        if isinstance(got, Unknown):
            return Receipt(act, "rejected", idempotency_key=key, error=got.detail or got.reason)
        if kind is not None:
            derivation = self._kind_derivations.get((owner, pred, kind))
            if derivation is None:
                return Receipt(act, 'rejected', error='No authenticated quantity derivation')
            observed = export_derivation(self.mind, derivation)
            if isinstance(observed, Unknown):
                return Receipt(act, 'rejected', error=observed.reason)
        else:
            observed = Claim(owner, answered, got)
        receipt = Receipt(act, "applied", idempotency_key=key)
        self._worked_out[act.capability] = (act, observed, receipt)
        return receipt

    @staticmethod
    def _predicate_of_capability(name: str) -> str | None:
        return name[len("amount_of_"):] if name.startswith("amount_of_") else None

    def reveal(self, cap: Capability, args: Mapping[str, Any], receipt: Receipt) -> Iterable[Claim | Proposition | DerivationReference]:
        if receipt.status != "applied":
            return
        got = self._worked_out.get(cap.name)
        if got is None:
            return
        action, observation, actual_receipt = got
        if (not _same(dict(action.args), dict(args)) or not _same(receipt, actual_receipt)
                or not _same(receipt.action, action)):
            return
        if cap.name.startswith('amount_of_kind_'):
            # Export again at reveal time: an applied arithmetic receipt is not
            # permanent authority after an operand/population/operator changes.
            derivation = self._kind_derivations.get((args['owner'],
                cap.name[len('amount_of_kind_'):], args['kind']))
            if derivation is None:
                return
            current = export_derivation(self.mind, derivation)
            if isinstance(current, Unknown) or not _same(current, observation):
                return
            yield current
        else:
            yield observation

    # ------------------------------------------------------------- arithmetic

    def total(self, owner: Ref, predicate: str) -> Quantity | Unknown:
        """Everything recorded for one owner under one predicate, combined into one amount."""
        got = self._total_claim(owner, world_predicate(predicate))
        return got if isinstance(got, Unknown) else got.object

    def total_of_kind(self, owner: Ref, predicate: str, kind: Ref) -> Quantity | Unknown:
        """Sum only explicit matching measurements, never infer membership from spelling."""
        if type(owner) is not Ref or type(kind) is not Ref:
            return Unknown('explicit_quantity_identity_required')
        records = [record for record in self.mind.propositions()
                   if record.proposition.predicate == predicate
                   and record.proposition.role('subject') == owner
                   and record.proposition.role('kind') == kind]
        if not records:
            return Unknown('nothing_recorded_for_kind', 'No amount recorded for this exact owner and kind')
        params = {'owner': owner, 'kind': kind, 'predicate': predicate}
        projected = _sum_kind_measurements(tuple(record.proposition for record in records), params)
        if isinstance(projected, Unknown):
            return projected
        receipt = derive_proposition(self.mind, self._kind_sum_operator,
            tuple(record.proposition.id for record in records), params=params,
            basis=('Explicit owner/kind measurement operands and complete active predicate population',),
            population_predicate=predicate)
        if isinstance(receipt, Unknown):
            return receipt
        self._kind_derivations[(owner, predicate, kind)] = receipt
        return receipt.proposition.role('object')

    def difference(self, left: Ref, right: Ref, predicate: str) -> Quantity | Unknown:
        """How much more one has than the other, refused across dimensions."""
        pred = world_predicate(predicate)
        a, b = self._total_claim(left, pred), self._total_claim(right, pred)
        if isinstance(a, Unknown):
            return a
        if isinstance(b, Unknown):
            return b
        got = derive(self.mind, left, f"difference:{pred}", "sub", (a, b), source=self.source)
        return got if isinstance(got, Unknown) else got.object

    def compare(self, left: Ref, right: Ref, predicate: str) -> str | Unknown:
        """"greater", "less" or "equal" between two owners' totals, in whatever units each
        was said in — "3 feet" against "2 metres" compares, "3 feet" against "2 coins"
        refuses."""
        pred = world_predicate(predicate)
        a, b = self.total(left, pred), self.total(right, pred)
        if isinstance(a, Unknown):
            return a
        if isinstance(b, Unknown):
            return b
        return compare_quantities(a, b)

    def in_unit(self, owner: Ref, predicate: str, unit: Unit) -> Quantity | Unknown:
        """A total said in the unit that was asked for, not in the dimension's base unit."""
        got = self.total(owner, predicate)
        return got if isinstance(got, Unknown) else convert(got, unit)

    def count_properties(self, thing: Ref) -> Quantity | Unknown:
        """Count stored value-valued records under an explicitly chosen measurement.

        Entity relations do not count as property records. Their presence and any
        known quantities never infer what the caller meant. No records at all
        preserve unknown coverage rather than asserting zero world properties.
        """
        if self._world is None:
            return Unknown('no_store', 'No store was supplied for this measurement')
        properties, relations, _ = self._facts_about(thing)
        if not properties and not relations:
            return Unknown('unknown_thing', f'No stored records about {_named(thing)}')
        return Quantity(float(properties), PROPERTY)

    # ------------------------------------------------------------- internals

    def _amounts(self, owner: Ref, predicate: str) -> list[Claim]:
        return [r.claim for r in self.mind.claims(subject=owner, predicate=predicate)
                if isinstance(r.claim.object, Quantity)]

    def _total_claim(self, owner: Ref, predicate: str) -> Claim | Unknown:
        """The combined amount as a claim, so the working is on the record.

        Two shapes get combined, in this order, and both are dimension-checked by
        ``quantity`` rather than here:

        1. a **rate times a count** — "6 tickets per ride" and "10 rides" is 60 tickets,
           and it is a product because the ride cancels, not because anything read the
           word "per" twice;
        2. a **sum** of what is left, which is only meaningful if it is all of one
           dimension.
        """
        if any(record.proposition.predicate == predicate and record.proposition.role('subject') == owner
               and type(record.proposition.role('kind')) is Ref for record in self.mind.propositions()):
            return Unknown('counted_kind_required', 'Kind-tagged measurements require an explicit counted kind')
        terms = self._amounts(owner, predicate)
        if not terms:
            return Unknown("nothing_recorded", f"I have no amount recorded for {_named(owner)}")
        for rate in [c for c in terms if _is_rate(c.object)]:
            per = {dim for dim, power in rate.object.dimension if power < 0}
            against = next((c for c in terms if c is not rate
                            and per <= {dim for dim, power in c.object.dimension if power > 0}), None)
            if against is None:
                return Unknown("dangling_rate",
                               f"{rate.object} is a rate and I have nothing to multiply it by")
            got = derive(self.mind, owner, f"product:{predicate}", "mul", (rate, against), source=self.source)
            if isinstance(got, Unknown):
                return got
            terms = [c for c in terms if c is not rate and c is not against] + [got]
        dimensions = {c.object.dimension for c in terms}
        if len(dimensions) > 1:
            return Unknown("several_dimensions",
                           "the question did not say which of " +
                           " or ".join(sorted(str(c.object.unit) for c in terms)) + " to count")
        if len(terms) == 1:
            return terms[0]
        return derive(self.mind, owner, f"total:{predicate}", "sum", terms, source=self.source)

    def _facts_about(self, thing: Ref) -> tuple[int, int, frozenset[str]]:
        """The store's facts about ``thing``, split into properties and relations.

        Facts, not predicates: "the report is red" and "the report is big" are two properties
        said with one verb. A fact counts as a property when every role but the subject is
        filled by a value rather than by a :class:`~tensorcode.records.Ref` — which is the
        store's own way of saying "another thing in the world", so nothing here has to know
        what any particular predicate means.
        """
        store = self._world
        if store is None:
            return 0, 0, frozenset()
        facts: list[tuple[str, list[Any]]] = [
            (r.claim.predicate, [r.claim.object]) for r in store.claims(subject=thing)]
        facts += [(r.proposition.predicate, [v for role, v in r.proposition.roles.items() if role != "subject"])
                  for r in store.propositions() if r.proposition.role("subject") == thing]
        properties = sum(1 for _, fillers in facts if not any(isinstance(v, Ref) for v in fillers))
        return properties, len(facts) - properties, frozenset(p for p, _ in facts)


# ------------------------------------------------------------------ reading helpers


def _is_rate(quantity: Quantity) -> bool:
    return any(power < 0 for _, power in quantity.dimension)


def _numerals_as_entities(value: Any) -> Any:
    """The treebank reader's ``count="7"`` said the way the hand grammar says it.

    ``deps_semantics.Reader.entity`` stores the nummod's *word*; ``ENGLISH`` stores an
    ``Entity(number, …)``; ``semantics_bridge._number_of`` reads ``.text`` and so sees
    nothing in the first. Rewriting the feature is the smallest place to reconcile them
    from outside those modules — and it is the same rewriting either way, so a frame that
    already carries an entity comes through untouched.
    """
    if isinstance(value, Frame):
        return Frame(value.predicate, {r: _numerals_as_entities(v) for r, v in value.roles.items()}, value.features)
    if isinstance(value, Entity):
        features = {k: _numerals_as_entities(v) for k, v in value.features.items()}
        count = features.get("count")
        if isinstance(count, str):
            features["count"] = Entity("number", count)
        return Entity(value.kind, value.text, features, value.ref, value.candidates)
    if isinstance(value, tuple):
        return tuple(_numerals_as_entities(v) for v in value)
    return value


def _entity_occurrences(value: Any) -> Iterable[Entity]:
    """Visit entity occurrences; nested frames are handled by ``Frame.walk``."""
    if isinstance(value, Entity):
        yield value
        for inner in value.features.values():
            yield from _entity_occurrences(inner)
    elif isinstance(value, (tuple, list)):
        for inner in value:
            yield from _entity_occurrences(inner)


def _ref_of(description: Any) -> Ref | None:
    """Return only the caller's explicit identity, never an identity from wording."""
    if isinstance(description, Ref):
        return description
    if isinstance(description, Entity) and isinstance(description.ref, Ref):
        return description.ref
    return None


def _named(ref: Ref) -> str:
    return str(getattr(ref, "id", ref)).split(":", 1)[-1]
