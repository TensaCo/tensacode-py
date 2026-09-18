"""Saying a meaning out loud, with the grammar that reads it.

Generation inverts each production's semantic spec: given the mother's meaning,
work out what each daughter would have to mean, then realise the daughters. The
specs are data, so this is a search over the *same* productions the parser used
rather than a second grammar that has to be kept in step — the failure mode this
module exists to avoid is a speaker and a listener that quietly disagree.

Coverage is partial by construction: a production whose spec cannot be inverted
is skipped, and if nothing realises the meaning the caller gets ``None`` rather
than an approximation. :func:`round_trip` is the test that matters — say it, read
it back, and check the meaning survived.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .grammar import (
    ABSENT, Ask, Attach, Build, Cat, Coord, Ent, Entry, FVar, Grammar, Head, Lit, Locative, Merge, Order, Production,
    Qualify, Terminal, inflect,
)
from .semantics import Entity, Frame, Question, Request


class Any_:
    """A daughter whose meaning the mother does not constrain (a determiner, say)."""

    def __repr__(self) -> str:
        return "ANY"


ANY = Any_()

#: features that reach the meaning through inflection or a function word, and so
#: must be demanded of a daughter rather than produced by a production
GRAMMATICAL = ("tense", "aspect", "number", "degree", "person")

#: what a subject imposes on its verb. English marks agreement on the verb, so these
#: are demanded of a verbal daughter and never of the noun phrase that supplied them.
AGREEMENT = ("number", "person")

#: categories that can express agreement. Demanding person of a name would make the
#: name unsayable, since no rule inflects one.
AGREES = ("V", "Aux", "VP")

#: The one feature a production may state without the meaning carrying it. Declarative
#: is the unmarked mood, so a bare frame is a statement. Every other feature a
#: production states — a negation, a perfect, a repair mark — must be in the meaning, or
#: generation invents it: that is how "no one 'm share grain" happened, and it is what
#: lets a production be readable but unsayable ("plenty food").
DEFAULTED = ("mood",)

#: features a word *asserts* by carrying them. Using a form that carries one the
#: meaning does not have adds to the meaning, so such a form is refused rather than
#: merely dispreferred: a tenseless frame said as "gave" claims a past that nobody
#: said, which is how "should gave food" survived a fewest-words tie against "should
#: give food". Number and degree are not on this list — an unmarked noun is still
#: singular, and nothing is claimed by saying so.
OVERSTATES = ("tense", "aspect", "polarity", "modality")


@dataclass(frozen=True)
class Need:
    """What a daughter must mean, and which grammatical features it must carry."""

    meaning: Any = ANY
    features: tuple[tuple[str, Any], ...] = ()


def realize(grammar: Grammar, meaning: Any, *, cat: str | None = None, depth: int = 8) -> str | None:
    """The best surface string for a meaning, or ``None`` if the grammar cannot say it."""
    cats = [cat] if cat else list(_default_cats(meaning, grammar))
    best: _Said | None = None
    memo: dict = {}
    for name in cats:
        got = _realize(grammar, name, Need(meaning), depth, memo)
        if got is not None and (best is None or _better(got, best)):
            best = got
    return best.text if best else None


@dataclass(frozen=True)
class _Said:
    score: float
    tokens: int
    text: str


def _better(a: _Said, b: _Said) -> bool:
    """Fewest words wins; ties go to the higher-scoring derivation.

    Length first, not score first: production weights exist to *rank parses*, and
    several are positive, so a score-first search is rewarded for piling on structure
    ("the field called north had fail and been..."). Every candidate that reaches here
    already accounts for every role and feature — that is what ``_invert`` guarantees —
    so the shortest one is the one that says exactly the meaning and nothing else.
    """
    return (-a.tokens, a.score) > (-b.tokens, b.score)


def _default_cats(meaning: Any, grammar: Grammar) -> Iterable[str]:
    if isinstance(meaning, Request):
        return ("IMP",)
    if isinstance(meaning, Question):
        return ("Q",)
    if isinstance(meaning, Frame):
        return ("S", "VP")
    return ("NP",)


def _realize(grammar: Grammar, cat: str, need: Need, depth: int,
             memo: dict | None = None) -> _Said | None:
    """The best way to say ``need`` as a ``cat``, or None.

    One six-word sentence asked this 24,103 times for **74** distinct (category, need)
    pairs, because every candidate production re-explores the same daughters. So the
    answers are memoised for the duration of one :func:`realize` call.

    That memo is only sound if a result does not depend on *where* it was reached from,
    which is why the search no longer carries a set of pairs already on the stack. That
    set was there to stop left recursion, and the depth limit already does: depth falls
    by one at every level, so ``VP -> VP PP`` unwinds on its own. With the stack gone,
    an answer depends on nothing but the category, the need and the depth remaining —
    so those three are the key, and the memo is exact rather than approximate.
    """
    said, _ = _search(grammar, cat, need, depth, {} if memo is None else memo)
    return said


def _search(grammar: Grammar, cat: str, need: Need, depth: int,
            memo: dict) -> tuple[_Said | None, int]:
    """The best way to say this, and how many levels of depth that derivation used.

    The depth remaining is part of what an answer depends on, so it is part of the key.
    But an answer that used three levels is the same answer at every depth of three or
    more, and reporting the depth *used* is what lets one computation serve them all —
    without it the same sentence was rebuilt once per depth.
    """
    if depth <= 0:
        return None, 0
    try:
        base: Any = (cat, need)
        hash(base)
    except TypeError:  # a meaning that is not hashable still needs a stable key
        base = (cat, repr(need))
    found = memo.get(base)
    if found is not None and found[1] <= depth:
        return found
    exact = memo.get((base, depth))
    if exact is not None:
        return exact
    best: _Said | None = None
    used = 0

    for score, word in _lexical(grammar, cat, need):
        said = _Said(score, 1, word)
        if best is None or _better(said, best):
            best, used = said, 1

    for prod in grammar.by_lhs(cat):
        needs = _invert(prod, need, grammar)
        if needs is None:
            continue
        pieces: list[str] = []
        score, tokens, ok, deepest = prod.weight, 0, True, 0
        for i, symbol in enumerate(prod.rhs):
            if isinstance(symbol, Terminal):
                pieces.append(symbol.word)
                tokens += 1
                continue
            sub = needs.get(i, Need())
            # a prohibition is checked against the daughter's *meaning*: the tense that
            # would make "ought gave" is carried by the complement frame, not demanded
            # of it, so refusing the demand would not have caught it
            if any(v is ABSENT and _carries(sub.meaning, k) for k, v in symbol.features.items()):
                ok = False
                break
            # the production's own category features are a constraint on the daughter,
            # prohibitions included: a forbidden feature may sit on the *entry* rather
            # than the meaning ("heaps" is a partitive), so the demand has to reach the
            # lexicon rather than being checked here and dropped
            fixed = {k: v for k, v in symbol.features.items() if not isinstance(v, FVar)}
            if fixed:
                sub = Need(sub.meaning, tuple(sorted({**dict(sub.features), **fixed}.items())))
            got, sub_used = _search(grammar, symbol.name, sub, depth - 1, memo)
            if got is None:
                ok = False
                break
            deepest = max(deepest, sub_used)
            score += got.score
            tokens += got.tokens
            pieces.append(got.text)
        if ok:
            said = _Said(score, tokens, " ".join(p for p in pieces if p))
            if best is None or _better(said, best):
                best, used = said, deepest + 1
    answer = (best, used)
    memo[(base, depth)] = answer
    if best is not None:
        held = memo.get(base)
        if held is None or held[0] is None or _better(best, held[0]) or (
                best == held[0] and used < held[1]):
            memo[base] = answer
    return answer


def _lexical(grammar: Grammar, cat: str, need: Need) -> list[tuple[float, str]]:
    """Surface forms of this category that satisfy the need, inflecting when required."""
    wanted = dict(need.features)
    listed: list[tuple[float, str]] = []
    derived: list[tuple[float, str]] = []
    candidates = (grammar.entries_by_cat(cat) if need.meaning is ANY
                  else grammar.entries_saying(cat, need.meaning))
    for entry in candidates:
        if need.meaning is not ANY and not _same(entry.sem, need.meaning):
            continue
        if any(k in entry.features and k not in wanted for k in OVERSTATES):
            continue  # the form claims more than the meaning does
        if any(v is ABSENT and k in entry.features for k, v in wanted.items()):
            continue  # a forbidden feature: "heaps bread" needs its "of"
        missing = {k: v for k, v in wanted.items()
                   if v is not ABSENT and entry.features.get(k) != v}
        if not missing:
            listed.append((entry.weight, entry.word))
            continue
        form = inflect(entry, missing)
        if form is not None:
            derived.append((entry.weight, form))  # inflection is free: it is the same word
    # a listed form beats a derived one: the lexicon has "came", so the suffix rules
    # are not asked to invent "comed"
    out = listed or derived
    out.extend(_open_class(grammar, cat, need))
    return sorted(out, key=lambda o: (-o[0], o[1]))


def _open_class(grammar: Grammar, cat: str, need: Need) -> list[tuple[float, str]]:
    """A name, path, literal — or a word the lexicon never had — is said by writing it.

    The open-class rules that let an unknown word *in* also let it back *out*, which is
    what keeps a village's own vocabulary sayable: a predicate like ``fail`` that no
    lexicon lists is still inflected to "failed" by the shared suffix table.
    """
    meaning = need.meaning
    wanted = dict(need.features)
    out: list[tuple[float, str]] = []
    for pattern, spec in getattr(grammar, "_open", ()):
        if spec.cat != cat:
            continue
        if spec.sem == "word" and isinstance(meaning, str):
            # the bare form already carries the unmarked features (a noun is singular,
            # a verb is tenseless), so only the rest has to be inflected — but only for
            # what was not demanded, or the default contradicts the demand and a plural
            # noun becomes unsayable ("the fields are empty" had no way to be said)
            defaults = {"number": "singular"} if cat in ("N", "Adj") and "number" not in wanted else {}
            entry = Entry(meaning, cat, {**defaults, **dict(spec.features)}, meaning, spec.weight)
            if any(v is ABSENT and k in entry.features for k, v in wanted.items()):
                continue
            missing = {k: v for k, v in wanted.items()
                       if v is not ABSENT and entry.features.get(k) != v}
            form = meaning if not missing else inflect(entry, missing)
            if form is not None and pattern.fullmatch(form):
                out.append((spec.weight, form))
            continue
        if not isinstance(meaning, Entity) or spec.kind != meaning.kind:
            continue
        text = f"'{meaning.text}'" if spec.kind == "literal" else meaning.text
        if pattern.fullmatch(text):
            out.append((spec.weight, text))
    return out


def _agreement(value: Any) -> dict[str, Any]:
    """The person and number a referring expression imposes on its verb.

    Parsing gets this from unification: ``NP[number=?n] VP[number=?n]`` ties the two
    together and the words supply the value. Generation has the opposite problem — the
    value has to come from the entity being talked about, and a name carries no
    features at all, though it is third person singular for every purpose here.
    """
    if isinstance(value, (Request, Question)):
        value = value.frame
    if not isinstance(value, Entity):
        return {}
    out = {key: value.features[key] for key in AGREEMENT if key in value.features}
    out.setdefault("number", "singular")
    out.setdefault("person", 3)
    return out


def _carries(meaning: Any, key: str) -> bool:
    """Whether a meaning already carries a feature a daughter is forbidden to carry."""
    if isinstance(meaning, (Request, Question)):
        meaning = meaning.frame
    return key in getattr(meaning, "features", {})


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, Entity) and isinstance(b, Entity):
        return a.text.lower() == b.text.lower()
    return a == b


def _invert(prod: Production, need: Need, grammar: Grammar) -> dict[int, Need] | None:
    """What each daughter must mean for this production to produce ``need``."""
    target = need.meaning
    sem = prod.sem
    lift = {target_feature: (index, source) for target_feature, index, source in getattr(sem, "lift", ())}
    needs: dict[int, Need] = {}
    extra: dict[int, dict[str, Any]] = {}

    def demand(index: int, meaning: Any = ANY, **features: Any) -> None:
        current = needs.get(index, Need())
        merged = dict(current.features) | extra.get(index, {}) | features
        needs[index] = Need(meaning if meaning is not ANY else current.meaning, tuple(sorted(merged.items())))

    def take_lift(features: Mapping[str, Any]) -> dict[str, Any] | None:
        """Route features that come from inflection to the daughter that carries them."""
        left = dict(features)
        for name, (index, source) in lift.items():
            if name in left:
                extra.setdefault(index, {})[source] = left.pop(name)
        return left

    def agree() -> None:
        """Route agreement along the grammar's own agreement variables.

        A variable shared by two symbols is the grammar saying they agree; shared with
        the mother, that it passes through. The whole bundle travels rather than just
        the key the variable happens to be written on, because ``?n`` between a subject
        and its verb is shorthand for subject-verb agreement, and English agreement is
        person *and* number. Without this the copula was chosen alphabetically: "Nise
        am hungry".
        """
        slots: dict[str, list[int]] = {}
        for index, symbol in enumerate(prod.rhs):
            if isinstance(symbol, Cat):
                for value in symbol.features.values():
                    if isinstance(value, FVar):
                        slots.setdefault(value.name, []).append(index)
        mother = {k: v for k, v in need.features if k in AGREEMENT}
        lhs_vars = {v.name for v in prod.lhs.features.values() if isinstance(v, FVar)}
        for name, indices in slots.items():
            found = dict(mother) if name in lhs_vars else {}
            source = None
            for index in indices:
                got = _agreement(needs[index].meaning) if index in needs else {}
                if got:
                    found, source = {**got, **found}, index
                    break
            if not found:
                continue
            for index in indices:
                if index == source or not isinstance(prod.rhs[index], Cat):
                    continue
                if prod.rhs[index].name not in AGREES:
                    continue
                current = needs.get(index, Need())
                merged = dict(current.features) | found
                needs[index] = Need(current.meaning, tuple(sorted(merged.items())))

    def done() -> dict[int, Need]:
        """Materialise the demands, including features routed by ``take_lift``."""
        for index, feats in extra.items():
            current = needs.get(index, Need())
            merged = dict(current.features) | feats
            needs[index] = Need(current.meaning, tuple(sorted(merged.items())))
        agree()
        return needs

    if target is ANY:
        return {}

    # A feature lifted from the *head* daughter is inflection ("failed"); one lifted
    # from another daughter is a function word ("did fail"). If the meaning lacks the
    # feature, that word would be invented — "no one 'm share grain" — so the
    # production is refused. Inflection-carrying productions stay available, which is
    # what a tenseless frame needs.
    head = getattr(sem, "predicate_from", None)
    if head is None:
        head = getattr(sem, "index", None)
    if lift and head is not None and any(index != head for _, (index, _) in lift.items()):
        carried = target.frame.features if isinstance(target, (Request, Question)) else getattr(target, "features", {})
        if not any(name in carried for name in lift):
            return None

    if isinstance(sem, Head):
        if isinstance(sem.index, int):
            demand(sem.index, target, **dict(need.features))
            return done()
        return None

    if isinstance(sem, Lit):
        return {} if _same(sem.value, target) else None

    if isinstance(sem, Order):
        if not isinstance(target, Request) or not isinstance(sem.index, int):
            return None
        demand(sem.index, Frame(target.frame.predicate, target.frame.roles,
                                {k: v for k, v in target.frame.features.items() if k != "mood"}))
        return done()

    if isinstance(sem, Ask):
        if not isinstance(target, Question):
            return None
        inner = Frame(target.frame.predicate, target.frame.roles,
                      {k: v for k, v in target.frame.features.items() if k != "mood"})
        if sem.asked_from is not None and isinstance(sem.asked_from, int):
            demand(sem.asked_from, target.asked)
        elif sem.asked != target.asked:
            return None
        if isinstance(sem.index, int):
            demand(sem.index, inner)
            return done()
        nested = _invert(Production(prod.lhs, prod.rhs, sem.index, prod.weight, prod.name), Need(inner), grammar)
        if nested is None:
            return None
        for i, sub in nested.items():
            demand(i, sub.meaning, **dict(sub.features))
        return done()

    if isinstance(sem, Coord):
        if not isinstance(target, tuple) or len(target) != len(sem.indices):
            return None
        for ref, value in zip(sem.indices, target):
            if not isinstance(ref, int):
                return None
            demand(ref, value)
        return done()

    if isinstance(sem, Ent):
        if not isinstance(target, Entity):
            return None
        features = dict(target.features)
        for key, ref in sem.features_from:
            if not isinstance(ref, int) or key not in features:
                return None
            demand(ref, features.pop(key))
        for key, value in sem.features:
            if features.pop(key, value if key in DEFAULTED else None) != value:
                return None
        left = take_lift(features)
        if left:  # features this production cannot express
            return None
        return done()

    if isinstance(sem, Qualify):
        if not isinstance(sem.index, int):
            return None
        if isinstance(target, Entity):
            features = dict(target.features)
            for key, ref in sem.features_from:
                if not isinstance(ref, int) or key not in features:
                    return None
                demand(ref, features.pop(key))
            for key, value in sem.features:
                if features.pop(key, value if key in DEFAULTED else None) != value:
                    return None
            left = take_lift(features)
            if left is None:
                return None
            demand(sem.index, Entity(target.kind, target.text, left, target.ref, target.candidates))
            return done()
        if isinstance(target, Frame):
            features = dict(target.features)
            for key, ref in sem.features_from:
                if not isinstance(ref, int) or key not in features:
                    return None
                demand(ref, features.pop(key))
            for key, value in sem.features:
                if features.pop(key, value if key in DEFAULTED else None) != value:
                    return None
            roles = dict(target.roles)
            for role, ref in sem.roles_from:
                if not isinstance(ref, int) or role not in roles:
                    return None
                demand(ref, roles.pop(role))
            left = take_lift(features)
            if left is None:
                return None
            demand(sem.index, Frame(target.predicate, roles, left))
            return done()
        return None

    if isinstance(sem, Attach):
        if not isinstance(sem.index, int) or not isinstance(sem.modifier, int):
            return None
        holder = target if isinstance(target, (Frame, Entity)) else None
        if holder is None:
            return None
        roles = dict(holder.roles) if isinstance(holder, Frame) else dict(holder.features)
        markable = {e.sem for e in grammar.entries_by_cat("P")}
        for role in sorted(roles):
            if role in GRAMMATICAL or role in ("mood",):
                continue
            if role not in markable:
                continue  # no preposition marks it, so it is not a PP: "north field", not "field called north"
            rest = {k: v for k, v in roles.items() if k != role}
            base = (Frame(holder.predicate, rest, holder.features) if isinstance(holder, Frame)
                    else Entity(holder.kind, holder.text, rest, holder.ref, holder.candidates))
            demand(sem.index, base)
            demand(sem.modifier, Frame("_pp", {"role": role, "value": roles[role]}))
            return done()
        return None

    if isinstance(sem, Locative):
        if not isinstance(target, Frame) or target.predicate != sem.predicate:
            return None
        roles = dict(target.roles)
        theme = None
        if sem.theme is not None:
            if not isinstance(sem.theme, int) or sem.theme_role not in roles:
                return None
            theme = roles.pop(sem.theme_role)
            demand(sem.theme, theme)
        if len(roles) != 1 or not isinstance(sem.modifier, int):
            return None
        role, value = next(iter(roles.items()))
        demand(sem.modifier, Frame("_pp", {"role": role, "value": value}))
        if take_lift(dict(target.features)):
            return None
        return done()

    if isinstance(sem, (Build, Merge)):
        if not isinstance(target, Frame):
            return None
        roles = dict(target.roles)
        features = {k: v for k, v in target.features.items() if k != "mood"}
        if isinstance(sem, Build):
            if sem.predicate is not None and sem.predicate != target.predicate:
                return None
            if sem.predicate_from is not None:
                if not isinstance(sem.predicate_from, int):
                    return None
                demand(sem.predicate_from, target.predicate)
        else:
            if not isinstance(sem.index, int):
                return None
        for role, ref in sem.roles:
            if not isinstance(ref, int) or role not in roles:
                return None
            demand(ref, roles.pop(role))
        for key, value in sem.features:
            if features.pop(key, value if key in DEFAULTED else None) != value:
                return None
        # a feature demanded *of* this constituent is expressed inside it: the perfect
        # production asks its complement for ``VP[tense=past]``, and that tense has to
        # reach the verb rather than fall off at the mother ("has arrive")
        for key, value in need.features:
            if key in lift and key not in features and value is not ABSENT:
                features[key] = value
        left = take_lift(features)
        if left is None:
            return None
        if isinstance(sem, Merge):
            demand(sem.index, Frame(target.predicate, roles, left))
            return done()
        if roles or left:  # a Build must account for every role and feature itself
            return None
        return done()

    return None


def round_trip(grammar: Grammar, meaning: Any, *, cat: str | None = None) -> tuple[str | None, Any]:
    """Say it, read it back: the surface string and the meaning that came back."""
    from .chart import understand

    text = realize(grammar, meaning, cat=cat)
    if text is None:
        return None, None
    back = understand(grammar, text)
    return text, (back.meanings[0] if back.meanings else None)
