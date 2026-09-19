"""Quantity for the agent: how many, how much, and the arithmetic in between.

The agent could parse "how many plants does Shondra have?" perfectly — the reader gives
``Question(have(subject=Shondra), asked="quantity")`` — and then had nowhere to send it.
Nothing in the plugin vocabulary informs on an amount, so ``Agent._look`` found no
capability, ``Agent.lookup`` found no proposition with a ``quantity`` role, and every such
question came back "I don't know". Twelve of twelve on ``reasoning.gsm8k``, and every
"how many"/"how much" anywhere else.

This plugin is the missing end of that wire. It holds amounts as
:class:`~tensorcode.quantity.Quantity` values — a number *with its unit* — and offers one
informing capability per predicate it has amounts under, so the question reaches it through
the machinery that was already there:

    plugin = QuantityPlugin()
    plugin.remember(Ref("entity:Shondra"), "have", Quantity(7, Unit.of("plant")))
    Agent([plugin]).turn("how many plants does Shondra have?").reply   # "7 plant."

Three things it refuses to do, each because the alternative is a confident wrong number:

* **It will not add across dimensions.** ``quantity.add`` returns ``Unknown`` for
  "3 sheep + 5 coins", and that refusal is carried all the way out to "I don't know".
* **It will not pick a dimension for you.** The reader drops the counted noun: "how many
  apples do I have?" and "how many pears do I have?" both arrive as
  ``?quantity in have(subject=user)``, with *apples* and *pears* gone (see
  ``deps_semantics.Reader.speech_act``, which deletes the whole wh-phrase's role). So when
  the amounts recorded for one owner span more than one dimension, the question has not
  said which one it wants and this plugin abstains — the abstention is derived from the
  units, which is exactly the information the question lost.
* **It never answers "there is nothing there."** A capability that runs, finds nothing and
  reports ``applied`` makes ``Agent._look`` return ``answered`` with an empty answer, which
  the reply renders as a confident "There is nothing there." and every scorer reads as a
  commitment. So the work happens in :meth:`QuantityPlugin.execute` and a capability that
  cannot say a number rejects its own call, with the reason, which comes out as
  "I don't know (…)".

What it cannot yet reach is recorded in :data:`needed_from_the_agent`: a plugin is never
shown a statement, so the numbers in "Shondra has 7 plants" reach the store only as the
bare reference ``entity:7 plants``, with the ``count`` feature the reader recovered thrown
away by ``to_propositions``. :meth:`QuantityPlugin.observe` is the half of that bridge that
belongs here, and it works the moment it is called with a parsed statement.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..language import Entity, Frame
from ..language.semantics import default_ref
from ..outcomes import Receipt, Unknown
from ..quantity import Quantity, Unit, convert, derive, tell_quantity
from ..quantity import compare as compare_quantities
from ..records import Claim, Proposition, Ref, Store, Var
from ..semantics_bridge import quantities_in
from .plugin import Call, Capability, Informs, Param, Plugin

#: What this plugin cannot do from inside the plugin protocol, so that the gaps stay
#: visible rather than becoming permanent (the same discipline as
#: ``semantics_bridge.needed_from_grammar``).
needed_from_the_agent = (
    "a statement is never shown to a plugin, so `observe` has to be called by hand; one "
    "call to it from `Agent.tell` is what would let a word problem's numbers be used",
    "`to_propositions` resolves every entity to `Ref(entity:<surface text>)`, so the "
    "`count` feature the reader recovered is gone by the time the store has it",
    "`Reader.speech_act` deletes the whole wh-phrase's role, so `how many PLANTS` loses "
    "*plants*: a question asking for one dimension out of several cannot be answered",
    "no reachable question shape binds two owners, so `difference` and `compare` are "
    "callable but unreachable from English ('how many more X does A have than B?' parses "
    "as be(subject=A, object=name:'have B'))",
)

#: Explicit quantity-domain relation; language adapters must supply any alignment.
POSSESSION = "has_possession"
SAID_AS_POSSESSION = frozenset({POSSESSION})

#: The unit of "how many properties": a property is a thing recorded about something, and
#: counting them is a count like any other, so it gets a unit like any other.
PROPERTY = Unit.of("property")

_OWNER = "owner_of_"


def world_predicate(predicate: str) -> str:
    """Preserve the caller's predicate; no implicit lexical translation."""
    return predicate


class QuantityPlugin(Plugin):
    """Amounts, with the arithmetic that combines them and the refusals that guard it."""

    def __init__(self, name: str = "quantity") -> None:
        super().__init__(name=name)
        self.mind = Store()
        self.source = Ref(f"plugin:{name}")
        # the store is handed to a plugin only in `refer`, and property counting needs it in
        # `execute`; the desktop plugin carries `_last_output` between the two the same way
        self._world: Store | None = None
        self._worked_out: dict[str, tuple[Ref, str, Quantity]] = {}

    # ------------------------------------------------------------- being told

    def remember(self, owner: Ref, predicate: str, quantity: Quantity, *, method: str = "quantity:told") -> Claim:
        """Record one amount of one thing, keeping the unit with the number."""
        return tell_quantity(self.mind, owner, world_predicate(predicate), quantity,
                             source=self.source, method=method)

    def observe(self, frame: Frame) -> list[Claim]:
        """Every amount a parsed statement carries, kept with the thing it was said of.

        ``semantics_bridge.quantities_in`` already lifts the ``count`` feature and the "per"
        of a rate into a :class:`Quantity`; two things have to be put right around it.

        It gives its owner back as surface text ("I"), which is not an identity the store
        can match, so the frame's own subjects are walked once for the reference each of
        them resolved to — the pronoun "I" has already been bound to the user by
        ``Agent.deixis`` — and the mention is filed under that.

        And the two readers spell ``count`` differently: the hand grammar puts an
        ``Entity(number, "7")`` there, the treebank reader puts the bare string ``"7"``, and
        ``semantics_bridge._number_of`` only reads the first. Measured: over the twelve
        ``reasoning.gsm8k`` dev problems, read by the treebank reader — the reader the agent
        actually uses — ``quantities_in`` recovered **zero** of the numbers in them.
        :func:`_numerals_as_entities` puts the second spelling into the first before handing
        the frame over, which is a correspondence between two schemes and not a reading of
        the sentence.
        """
        owners = _subject_refs(frame)
        out: list[Claim] = []
        for mention in quantities_in(_numerals_as_entities(frame)):
            # a quantity said *of the subject* ("3 sheep died") has no owner but is about the
            # thing it counted, which is how `tell_mentions` files it too
            owner = owners.get(mention.owner) if mention.owner else Ref(f"entity:{mention.of}")
            if not isinstance(owner, Ref):
                continue
            out.append(self.remember(owner, mention.predicate or "amount", mention.quantity,
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
        caps.append(Capability("count_properties", (Param("thing", "thing"),),
                               informs=(Informs(POSSESSION, "undergoer", "thing",
                                   query=Proposition(POSSESSION, {"subject": Var("thing"), "object": Var("answer")})),),
                               effect_kind="read",
                               description="how many things the store records about something"))
        return tuple(caps)

    def _predicates(self) -> list[str]:
        """The predicates it holds amounts under. Working is filed under "<op>:<pred>" and
        is not a predicate anyone asks about, so it is left out."""
        return sorted({r.claim.predicate for r in self.mind.claims()
                       if isinstance(r.claim.object, Quantity) and ":" not in r.claim.predicate})

    @staticmethod
    def _predicate_of(param: str) -> str | None:
        return param[len(_OWNER):] if param.startswith(_OWNER) else None

    # ------------------------------------------------------------- referring

    def refer(self, description: Any, param: Param, *, context: Mapping[str, Any]) -> Any | Unknown:
        """Which thing the question is about — and whether this capability can speak for it.

        Answering "no" here rather than in ``execute`` matters: ``Agent._look`` moves on to
        the next capability when ``refer`` refuses, so ``amount_of_has_possession`` failing
        lets ``count_properties`` have its turn on the same question.
        """
        self._world = context.get("store") if isinstance(context.get("store"), Store) else self._world
        ref = _ref_of(description)
        if ref is None:
            return Unknown("cannot_refer", f"{self.name} cannot tell what {_said(description)} names")
        pred = self._predicate_of(param.name)
        if pred is not None:
            if not self._amounts(ref, pred):
                return Unknown("nothing_recorded", f"I have no amount recorded for {_named(ref)}")
            return ref
        refused = self._not_a_property_count(ref)
        return Unknown(*refused) if refused else ref

    def denote(self, description: Any) -> Any | Unknown:
        """Only things it has amounts for, so it does not shadow another plugin's entities."""
        ref = _ref_of(description)
        if ref is not None and self.mind.claims(subject=ref):
            return ref
        return Unknown("cannot_refer", f"{self.name} has no amount for {_said(description)}")

    def display(self, ref: Any) -> str | None:
        """An amount reads as its number and its unit; anything else is not this plugin's."""
        return str(ref) if isinstance(ref, Quantity) else None

    # ------------------------------------------------------------- acting

    def execute(self, act: Call, *, key: str | None = None) -> Receipt:
        """Work the amount out here, and reject the call when there is no honest number.

        ``Agent._look`` treats ``applied`` as "it looked, and this is what is there", which
        for an empty result becomes the reply "There is nothing there." — a commitment. A
        rejection carries the reason instead and comes out as "I don't know (…)".
        """
        pred = self._predicate_of_capability(act.capability)
        if pred is not None:
            owner = act.arg(_OWNER + pred)
            got: Any = self.total(owner, pred) if isinstance(owner, Ref) else Unknown("no_owner", "no thing was named")
            answered = pred
        elif act.capability == "count_properties":
            owner = act.arg("thing")
            got = self.count_properties(owner) if isinstance(owner, Ref) else Unknown("no_owner", "no thing was named")
            answered = POSSESSION
        else:
            return Receipt(act, "rejected", error=f"{self.name} does not implement {act.capability}")
        self._worked_out.pop(act.capability, None)
        if isinstance(got, Unknown):
            return Receipt(act, "rejected", idempotency_key=key, error=got.detail or got.reason)
        self._worked_out[act.capability] = (owner, answered, got)
        return Receipt(act, "applied", idempotency_key=key)

    @staticmethod
    def _predicate_of_capability(name: str) -> str | None:
        return name[len("amount_of_"):] if name.startswith("amount_of_") else None

    def reveal(self, cap: Capability, args: Mapping[str, Any], receipt: Receipt) -> Iterable[Claim]:
        if receipt.status != "applied":
            return
        got = self._worked_out.get(cap.name)
        if got is None:
            return
        owner, predicate, quantity = got
        yield Claim(owner, predicate, quantity)

    # ------------------------------------------------------------- arithmetic

    def total(self, owner: Ref, predicate: str) -> Quantity | Unknown:
        """Everything recorded for one owner under one predicate, combined into one amount."""
        got = self._total_claim(owner, world_predicate(predicate))
        return got if isinstance(got, Unknown) else got.object

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
        """How many things the agent's store records about something.

        "How many properties does the report have?" reaches this plugin as
        ``?quantity in have(subject=the report)`` — indistinguishable from "how many plants
        does Shondra have?", because the reader deleted the counted noun. So the count is
        offered only when the store's own shape rules the other reading out — see
        :meth:`_not_a_property_count` for the three conditions and the wrong answer that
        put the third one there.
        """
        refused = self._not_a_property_count(thing)
        if refused:
            return Unknown(*refused)
        return Quantity(float(self._facts_about(thing)[0]), PROPERTY)

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

    def _not_a_property_count(self, thing: Ref) -> tuple[str, str] | None:
        """Why the store's facts about ``thing`` are not the number being asked for.

        The third of these conditions was written after a wrong answer, which is the one
        thing this repo does not spend. "For how many hours do they have to fundraise…"
        reaches this plugin as ``?quantity in have(subject=they)``, and the store did know
        two things about *they* — that they raised $2100 and had a goal — so the count came
        back "1 property." on a question whose answer was 9. What tells that apart from "how
        many properties does the report have?" is not the wording, which is identical once
        the counted noun is gone, but the *shape* of what is known: a property relates a
        thing to a value ("the report is red"), while a fact relating it to another entity is
        something it did or took part in, and a question about a thing with those is far
        more likely about them than about the size of the record.
        """
        if self._world is None:
            return ("no_store", "I was not given anything to count properties in")
        if self._amounts(thing, POSSESSION):
            return ("amount_known", f"I have an amount for {_named(thing)}, not a property count")
        properties, relations, predicates = self._facts_about(thing)
        if not properties:
            return ("unknown_thing", f"I know no properties of {_named(thing)}")
        if relations:
            return ("takes_part_in_things",
                    f"what I know about {_named(thing)} is mostly what it is involved with, "
                    "and the question did not say what to count")
        if predicates & SAID_AS_POSSESSION:
            return ("counted_noun_unknown",
                    f"I know what {_named(thing)} has but not how much of it, and the question "
                    "did not say what to count")
        return None

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


def _subject_refs(frame: Frame) -> dict[str, Ref]:
    """Each subject's surface text to the reference it resolved to, over the whole frame."""
    out: dict[str, Ref] = {}
    for sub in frame.walk() if isinstance(frame, Frame) else ():
        subject = sub.roles.get("subject")
        if isinstance(subject, Entity):
            ref = _ref_of(subject)
            if ref is not None:
                out.setdefault(subject.text, ref)
    return out


def _ref_of(description: Any) -> Ref | None:
    """The identity a description has for the store: its own, or the one the store mints."""
    if isinstance(description, Ref):
        return description
    if not isinstance(description, Entity):
        return None
    got = description.ref or default_ref(description)
    return got if isinstance(got, Ref) else None


def _named(ref: Ref) -> str:
    return str(getattr(ref, "id", ref)).split(":", 1)[-1]


def _said(description: Any) -> str:
    return repr(getattr(description, "text", description))
