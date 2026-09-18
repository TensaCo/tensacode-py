"""Categories, productions, a lexicon with morphology, and the grammar that holds them.

Three design decisions carry most of the weight here.

**Semantics are data, not callbacks.** A production says how its mother's meaning
is built from its daughters' with a small spec (:class:`Head`, :class:`Build`,
:class:`Merge`, :class:`Attach`, :class:`Ent`, …). A Python callable would be
easier to write and impossible to run backwards; a spec can be read in reverse,
which is what lets :mod:`tensorcode.language.generate` realise a frame with the
*same* grammar that parsed it. Specs nest: anywhere a spec refers to a daughter
it may instead refer to another spec over the same daughters.

**Word order is carried apart from role assignment.** A production's roles name
daughter positions, so a dialect that moves the verb needs one more production
over the same head rather than one predicate per word order. (Taken from
``symbolic-ai-models``'s ``symbolic_ai_parsers/grammar.py``, where slots are kept
out of the surface sequence for exactly this reason.)

**A preposition's meaning is the role it marks.** "in downloads" and "to
documents" then share one production, and a domain adds a role by adding a word.

Grammars are immutable and extensible: ``grammar.extend(...)`` returns a new
grammar, so a caller can add file names, app names, or a village's drifting
words without touching the core English.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence, Union

from .features import Bindings, FVar, ground, merge, rename, resolve, unify
from .semantics import Entity, Frame, Question, Request

# ------------------------------------------------------------------ categories


@dataclass(frozen=True)
class Cat:
    """A syntactic category: a name plus a feature structure."""

    name: str
    features: Mapping[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.name, tuple(sorted(self.features.items(), key=repr))))

    def __str__(self) -> str:
        if not self.features:
            return self.name
        return f"{self.name}[{','.join(f'{k}={v}' for k, v in sorted(self.features.items()))}]"


@dataclass(frozen=True)
class Terminal:
    """A literal word in a production (``"did"`` in ``VP -> "did" "not" VP``)."""

    word: str

    def __str__(self) -> str:
        return f'"{self.word}"'


# ------------------------------------------------------------------- semantics


@dataclass(frozen=True)
class Head:
    """The mother's meaning is this daughter's meaning."""

    index: "SemRef"


@dataclass(frozen=True)
class Lit:
    """A constant meaning (an atom, an :class:`Entity`, or a :class:`Frame`)."""

    value: Any


@dataclass(frozen=True)
class Build:
    """Construct a frame: predicate from a daughter or a constant, roles from daughters."""

    predicate: str | None = None
    predicate_from: "SemRef | None" = None
    roles: tuple[tuple[str, "SemRef"], ...] = ()
    features: tuple[tuple[str, Any], ...] = ()
    #: (target feature, daughter, that daughter's *grammatical* feature) — how tense,
    #: aspect, degree and modality reach the meaning from inflection and function words
    lift: tuple[tuple[str, int, str], ...] = ()


@dataclass(frozen=True)
class Merge:
    """Take a daughter's frame and add roles/features to it."""

    index: "SemRef"
    roles: tuple[tuple[str, "SemRef"], ...] = ()
    features: tuple[tuple[str, Any], ...] = ()
    lift: tuple[tuple[str, int, str], ...] = ()


@dataclass(frozen=True)
class Coord:
    """Coordination: the meaning is the tuple of the named daughters' meanings."""

    indices: tuple["SemRef", ...]


@dataclass(frozen=True)
class Ent:
    """Build an :class:`Entity` — what a referring expression picks out."""

    kind: str = "description"
    words_from: tuple[int, ...] = ()
    features: tuple[tuple[str, Any], ...] = ()
    features_from: tuple[tuple[str, "SemRef"], ...] = ()
    lift: tuple[tuple[str, int, str], ...] = ()


@dataclass(frozen=True)
class Qualify:
    """Add features (and, for frames, roles) to a daughter's meaning."""

    index: "SemRef"
    features: tuple[tuple[str, Any], ...] = ()
    features_from: tuple[tuple[str, "SemRef"], ...] = ()
    roles_from: tuple[tuple[str, "SemRef"], ...] = ()
    extend_text_from: tuple[int, ...] = ()
    lift: tuple[tuple[str, int, str], ...] = ()


@dataclass(frozen=True)
class Attach:
    """Attach a role-marked modifier; the modifier's ``role`` says which slot it fills."""

    index: "SemRef"
    modifier: "SemRef"


@dataclass(frozen=True)
class Locative:
    """A predication whose content is a role-marked modifier ("is in downloads")."""

    predicate: str
    modifier: "SemRef"
    theme: "SemRef | None" = None
    theme_role: str = "theme"
    lift: tuple[tuple[str, int, str], ...] = ()


@dataclass(frozen=True)
class Ask:
    """An interrogative reading: a frame plus the role being asked about."""

    index: "SemRef"
    asked: str = "polarity"
    asked_from: "SemRef | None" = None  # a wh-word whose meaning names the queried role


@dataclass(frozen=True)
class Order:
    """An imperative reading: a request for this daughter's frame."""

    index: "SemRef"


Sem = Union[Head, Lit, Build, Merge, Coord, Ent, Qualify, Attach, Locative, Ask, Order]
SemRef = Union[int, Sem]


def build_sem(sem: Sem, parts: Sequence[Any], words: Sequence[Sequence[str]],
              feats: Sequence[Mapping[str, Any]] = ()) -> Any:
    """Apply a semantic spec to daughters' meanings. Pure, and therefore reversible.

    ``words[i]`` is daughter *i*'s surface words (how an entity keeps the text it
    was named with) and ``feats[i]`` its grammatical features (how tense, aspect,
    degree and modality reach the meaning).
    """

    def part(ref: SemRef) -> Any:
        return parts[ref] if isinstance(ref, int) else build_sem(ref, parts, words, feats)

    def lifted(spec: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for target, index, source in getattr(spec, "lift", ()):
            value = dict(feats[index]).get(source) if index < len(feats) else None
            if value is not None:
                out[target] = value
        return out

    def values(spec: Any) -> dict[str, Any]:
        out = {k: v for k, v in getattr(spec, "features", ())}
        for key, ref in getattr(spec, "features_from", ()):
            got = part(ref)
            if got is not None:
                out[key] = got
        out.update(lifted(spec))
        return out

    if isinstance(sem, Head):
        return part(sem.index)
    if isinstance(sem, Lit):
        return sem.value
    if isinstance(sem, Ent):
        return Entity(sem.kind, " ".join(w for i in sem.words_from for w in words[i]), values(sem))
    if isinstance(sem, Qualify):
        base, extra = part(sem.index), values(sem)
        if isinstance(base, Entity):
            prefix = " ".join(w for i in sem.extend_text_from for w in words[i])
            text = f"{prefix} {base.text}".strip() if prefix else base.text
            return Entity(base.kind, text, {**base.features, **extra}, base.ref, base.candidates)
        if isinstance(base, Frame):
            roles = {role: part(ref) for role, ref in sem.roles_from}
            return Frame(base.predicate, {**base.roles, **roles}, {**base.features, **extra})
        return base
    if isinstance(sem, Attach):
        return _attach(part(sem.index), part(sem.modifier))
    if isinstance(sem, Locative):
        roles = {} if sem.theme is None else {sem.theme_role: part(sem.theme)}
        return _attach(Frame(sem.predicate, roles, lifted(sem)), part(sem.modifier))
    if isinstance(sem, Ask):
        asked = sem.asked
        if sem.asked_from is not None:
            named = part(sem.asked_from)
            if isinstance(named, str):
                asked = named
        return Question(_as_frame(part(sem.index)).added(mood="interrogative"), asked)
    if isinstance(sem, Order):
        return Request(_as_frame(part(sem.index)).added(mood="imperative"))
    if isinstance(sem, Coord):
        flat: list[Any] = []
        for ref in sem.indices:
            value = part(ref)
            flat.extend(value if isinstance(value, tuple) else [value])
        return tuple(flat)
    if isinstance(sem, Build):
        predicate: Any = sem.predicate
        head: Any = None
        if sem.predicate_from is not None:
            head = part(sem.predicate_from)
            predicate = head.predicate if isinstance(head, Frame) else str(head)
        roles = {role: part(ref) for role, ref in sem.roles}
        features = {k: v for k, v in sem.features}
        features.update(lifted(sem))
        if isinstance(head, Frame):  # a verb that already carries roles keeps them
            roles = {**head.roles, **roles}
            features = {**head.features, **features}
        return Frame(str(predicate), roles, features)
    if isinstance(sem, Merge):
        base = _as_frame(part(sem.index))
        roles = {role: part(ref) for role, ref in sem.roles}
        features = {k: v for k, v in sem.features}
        features.update(lifted(sem))
        return Frame(base.predicate, {**base.roles, **roles}, {**base.features, **features})
    raise TypeError(f"unknown semantic spec {sem!r}")


def _attach(base: Any, modifier: Any) -> Any:
    if not isinstance(modifier, Frame) or "role" not in modifier.roles:
        return base
    role, value = str(modifier.role("role")), modifier.role("value")
    if isinstance(base, Frame):
        return Frame(base.predicate, {**base.roles, role: value}, base.features)
    if isinstance(base, Entity):
        return Entity(base.kind, base.text, {**base.features, role: value}, base.ref, base.candidates)
    return base


def _as_frame(value: Any) -> Frame:
    if isinstance(value, Frame):
        return value
    if isinstance(value, Question):
        return value.frame
    if isinstance(value, Request):
        return value.frame
    if isinstance(value, Entity):
        return Frame("be", {"subject": value})
    return Frame(str(value))


# ----------------------------------------------------------------- productions


@dataclass(frozen=True)
class Production:
    lhs: Cat
    rhs: tuple[Cat | Terminal, ...]
    sem: Sem = Head(0)
    weight: float = 0.0  # log-scale; higher wins. 0.0 is the neutral default
    name: str = ""

    def __str__(self) -> str:
        return f"{self.lhs} -> {' '.join(str(r) for r in self.rhs)}"


_CAT = re.compile(r"^(?P<name>[A-Za-z_][\w]*)(?:\[(?P<feats>[^\]]*)\])?$")


class _Absent:
    """A feature demand that the daughter must *not* carry, written ``VP[tense=!]``.

    The mirror of a literal demand: subcategorisation sometimes needs an absence.
    A modal's complement is a bare infinitive, so ``VP -> Modal VP[tense=!]`` is what
    stops "ought gave" — the prohibition lives in the grammar, where both the parser
    and the generator can see it, rather than in a special case in either.
    """

    def __repr__(self) -> str:
        return "!"


ABSENT = _Absent()


def _atom(text: str) -> Any:
    text = text.strip()
    if text == "!":
        return ABSENT
    if text.startswith("?"):
        return FVar(text[1:])
    if text in ("true", "false"):
        return text == "true"
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return text


def parse_cat(text: str) -> Cat:
    m = _CAT.match(text.strip())
    if not m:
        raise ValueError(f"not a category: {text!r}")
    feats: dict[str, Any] = {}
    for part in (m.group("feats") or "").split(","):
        if part.strip():
            key, _, value = part.partition("=")
            feats[key.strip()] = _atom(value)
    return Cat(m.group("name"), feats)


def production(text: str, sem: Sem = Head(0), *, weight: float = 0.0, name: str = "") -> Production:
    """``production('S -> NP[number=?n] VP[number=?n]', Merge(1, roles=(("subject", 0),)))``."""
    lhs_text, _, rhs_text = text.partition("->")
    if not rhs_text:
        raise ValueError(f"production needs '->': {text!r}")
    rhs: list[Cat | Terminal] = []
    for token in re.findall(r'"[^"]*"|\S+', rhs_text.strip()):
        rhs.append(Terminal(token[1:-1]) if token.startswith('"') else parse_cat(token))
    if not rhs:
        raise ValueError(f"empty production: {text!r}")  # no epsilon: the chart relies on it
    return Production(parse_cat(lhs_text), tuple(rhs), sem, weight, name or text.strip())


# --------------------------------------------------------------------- lexicon


@dataclass(frozen=True)
class Entry:
    """One reading of one word: its category, its features, and what it means."""

    word: str
    cat: str
    features: Mapping[str, Any] = field(default_factory=dict)
    sem: Any = None  # a Frame, an Entity, an atom, or None to mean "the word itself"
    weight: float = 0.0

    def __hash__(self) -> int:
        """A cached hash that builds no strings.

        This used to be ``repr`` of the meaning, and putting entries in a cache key
        made that 148 ``repr`` calls per sentence — the same mistake, in the same
        shape, as the chart keys in §9. Equality is still the dataclass's own, so a
        key collision between two entries with the same word costs nothing.
        """
        cached = getattr(self, "_hash", None)
        if cached is None:
            try:
                cached = hash((self.word, self.cat,
                               tuple(sorted(self.features.items(), key=lambda kv: kv[0])),
                               _sem_key(self.sem)))
            except TypeError:  # an unhashable feature value
                cached = hash((self.word, self.cat))
            object.__setattr__(self, "_hash", cached)
        return cached


#: Suffix rules, tried against the lexicon's known lemmas, per category: a noun's
#: ``-s`` is a plural and a verb's is a third person, and conflating them is how a
#: grammar starts agreeing with the wrong thing. English inflection is knowledge of
#: the language, so it lives in code; the *words* do not.
SUFFIX_RULES: tuple[tuple[str, str, Mapping[str, Any], tuple[str, ...]], ...] = (
    ("ies", "y", {"number": "plural"}, ("N",)),
    ("es", "", {"number": "plural"}, ("N",)),
    ("s", "", {"number": "plural"}, ("N",)),
    ("ies", "y", {"number": "singular", "person": 3, "tense": "present"}, ("V",)),
    ("es", "", {"number": "singular", "person": 3, "tense": "present"}, ("V",)),
    ("s", "", {"number": "singular", "person": 3, "tense": "present"}, ("V",)),
    ("ing", "", {"aspect": "progressive"}, ("V",)),
    ("ing", "e", {"aspect": "progressive"}, ("V",)),
    ("ied", "y", {"tense": "past"}, ("V",)),
    ("ed", "", {"tense": "past"}, ("V",)),
    ("ed", "e", {"tense": "past"}, ("V",)),
    ("d", "", {"tense": "past"}, ("V",)),
    ("ier", "y", {"degree": "comparative"}, ("Adj",)),
    ("er", "", {"degree": "comparative"}, ("Adj",)),
    ("iest", "y", {"degree": "superlative"}, ("Adj",)),
    ("est", "", {"degree": "superlative"}, ("Adj",)),
)


@dataclass(frozen=True)
class Lexicon:
    entries: Mapping[str, tuple[Entry, ...]] = field(default_factory=dict)
    #: fitted log P(token) for tokens no constituent claims. A *fitted* background
    #: rather than a hand-set skip penalty, so a partial parse and a full parse are
    #: scored on one scale (``symbolic-ai-models``, ``parsers/parsers/cky_001``).
    background: Mapping[str, float] = field(default_factory=dict)
    unseen_background: float = math.log(1e-3)

    def near(self, token: str) -> list[str]:
        """Lemmas one edit away from an unknown token (transposition counts as one).

        Deterministic and cheap: only for tokens of five characters or more, and the
        readings it offers are penalised, so a real word always wins.
        """
        low = token.lower()
        if len(low) < 5 or low in self.entries:
            return []
        out = []
        for lemma in self.entries:
            if abs(len(lemma) - len(low)) > 1 or not (set(lemma) & set(low[:2])):
                continue
            if _one_edit(low, lemma):
                out.append(lemma)
        return sorted(out)

    def lookup(self, token: str) -> list[Entry]:
        """Entries for a token: exact spellings first, then inflected readings."""
        low = token.lower()
        found = list(self.entries.get(low, ()))
        for suffix, restore, feats, cats in SUFFIX_RULES:
            if not low.endswith(suffix) or len(low) <= len(suffix):
                continue
            lemma = low[: -len(suffix)] + restore
            for entry in self.entries.get(lemma, ()):
                if entry.cat not in cats or any(k in entry.features for k in feats):
                    continue
                found.append(replace(entry, word=token, features=merge(entry.features, feats), weight=entry.weight - 0.5))
        return found

    def knows(self, token: str) -> bool:
        return bool(self.lookup(token))

    def bg(self, token: str) -> float:
        low = token.lower()
        if low in self.background:
            return self.background[low]
        if re.fullmatch(r"-{1,2}\w[\w-]*", low):
            return math.log(0.9)  # a command flag is not content: skipping one is free
        return self.unseen_background

    def extend(self, *entries: Entry, background: Mapping[str, float] | None = None) -> "Lexicon":
        merged = {k: v for k, v in self.entries.items()}
        for entry in entries:
            key = entry.word.lower()
            merged[key] = merged.get(key, ()) + (entry,)
        return Lexicon(merged, {**self.background, **(background or {})}, self.unseen_background)

    def without(self, *words: str) -> "Lexicon":
        """Drop words — a dialect that has lost one, or a domain that redefines it."""
        gone = {w.lower() for w in words}
        return Lexicon({k: v for k, v in self.entries.items() if k not in gone}, self.background, self.unseen_background)

    @classmethod
    def of(cls, spec: Mapping[str, Sequence[Entry]]) -> "Lexicon":
        return cls({word.lower(): tuple(entries) for word, entries in spec.items()})


def _one_edit(a: str, b: str) -> bool:
    """True when one substitution, insertion, deletion or transposition maps a to b."""
    if a == b:
        return False
    if len(a) == len(b):
        diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if len(diff) == 1:
            return True
        return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]
    long, short = (a, b) if len(a) > len(b) else (b, a)
    return any(long[:i] + long[i + 1:] == short for i in range(len(long)))


#: Nouns and adjectives whose bare form already carries the demand.
UNMARKED: tuple[tuple[Mapping[str, Any], tuple[str, ...]], ...] = (
    ({"number": "singular"}, ("N", "Adj")),
)

AGREEMENT_FEATURES = ("number", "person")


def _unmarked(cat: str, wanted: Mapping[str, Any]) -> bool:
    return any(cat in cats and all(spec.get(k) == v for k, v in wanted.items())
               for spec, cats in UNMARKED)


def _agreement_unmarked(entry: "Entry", wanted: Mapping[str, Any]) -> bool:
    """Whether English marks this agreement with nothing, so the bare form is the answer.

    It marks subject agreement on exactly one form: the present third singular, which
    the ``-s`` rules above produce. Plural, first or second person, and every past form
    are all the bare word — "they share", "I share", "she shared" — so a demand for
    agreement on any of those is satisfied by the word itself rather than refused. The
    tense comes from the demand when it names one and from the form otherwise, because
    "came" is past whether or not the caller said so.
    """
    asked = {key: wanted[key] for key in AGREEMENT_FEATURES if key in wanted}
    if not asked:
        return False
    # only the agreement is excused. Everything else demanded must already be true of
    # the form, or "be" would answer a demand for a past tense it does not express —
    # with the one exception that a bare verb form *is* the present tense.
    for key, value in wanted.items():
        if key in AGREEMENT_FEATURES or entry.features.get(key) == value:
            continue
        if key == "tense" and value == "present" and "tense" not in entry.features:
            continue
        return False
    tense = wanted.get("tense", entry.features.get("tense", "present"))
    marked = asked.get("number", "singular") == "singular" and asked.get("person", 3) == 3
    return not (marked and tense == "present")


#: Dimensions a word is inflected in only once. "came" is already past, so no suffix
#: rule may add tense or aspect to it — that is how "cames" and "camed" were produced.
_ONCE: tuple[frozenset[str], ...] = (frozenset({"tense", "aspect"}), frozenset({"degree"}))


def _blocked(entry: Entry, feats: Mapping[str, Any]) -> set[str]:
    """The features on which a suffix rule may not apply to an already-inflected form."""
    out = {key for key, value in feats.items()
           if key in entry.features and entry.features[key] != value}
    for dimension in _ONCE:
        shared = dimension & set(feats)
        if shared and dimension & set(entry.features):
            out |= {key for key in shared if entry.features.get(key) != feats[key]}
    return out


def inflect(entry: Entry, wanted: Mapping[str, Any]) -> str | None:
    """See :func:`_inflect`; this is the cached front door (794 calls per sentence)."""
    try:
        return _inflect(entry, tuple(sorted(wanted.items())))
    except TypeError:  # an unhashable feature value: answer it directly
        return _inflect.__wrapped__(entry, tuple(sorted(wanted.items(), key=repr)))


@lru_cache(maxsize=65536)
def _inflect(entry: Entry, demanded: tuple[tuple[str, Any], ...]) -> str | None:
    """The surface form of an entry carrying ``wanted``, by running the suffix rules backwards.

    Parsing strips a suffix to find a lemma; saying something needs the other
    direction, and using one table for both keeps "failed" and ``tense=past`` the
    same fact rather than two lists that drift apart. Where several rules apply,
    the one whose restored letters actually end the lemma wins ("share" + past is
    "shared", not "shareed"), and the ``-es`` plural is reserved for the stems that
    take it.

    Three answers, not two. ``None`` means *contradicted* — "came" cannot be made
    present, and a caller that wanted a present form must look elsewhere. The word
    unchanged means *already so, or unmarked*: "came" carries no agreement because
    English marks person and number on present verbs only, and "share" is how a
    plural subject says it. A string means a suffix expressed the difference.
    """
    wanted = dict(demanded)
    if any(key in entry.features and entry.features[key] != value for key, value in wanted.items()):
        return None
    if all(entry.features.get(key) == value for key, value in wanted.items()):
        return entry.word
    word = entry.word
    candidates: list[tuple[int, int, str]] = []
    markable = False
    for suffix, restore, feats, cats in SUFFIX_RULES:
        if entry.cat not in cats or any(feats.get(k) != v for k, v in wanted.items()):
            continue
        markable = True  # some suffix marks this, whether or not it fits *this* form
        if _blocked(entry, feats) & set(wanted):
            # a rule refused on a feature the caller asked for: the form already carries
            # a different value, so it cannot carry this one either
            return None
        if _blocked(entry, feats):
            continue  # refused on something else: the ``-s`` rule is present-tense, and
            # "came" declining *that* is not it declining agreement
        if restore:
            # "carry" -> "carries", but "say" -> "says": English swaps a final y for
            # "ie" only after a consonant. Without that check the table inflected "say"
            # to "saies", and the stemmer then refused "says" because nothing
            # round-tripped to it.
            if restore == "y" and not re.search(r"[^aeiou]y$", word):
                continue
            if word.endswith(restore):
                candidates.append((2, len(suffix), word[: -len(restore)] + suffix))
            continue
        if suffix in ("es",) and not re.search(r"(?:s|x|z|ch|sh)$", word):
            continue  # "folders", not "folderes"
        candidates.append((1, len(suffix), word + suffix))
    if not candidates:
        # The form is unmarked for something the language *does* mark by suffix, and
        # that is the right answer: English marks person and number on present verbs
        # only, so past "came" and plural "share" are simply how it is said. A feature
        # no suffix marks at all (a future) is not this function's business — it needs
        # an auxiliary, so refusing sends the caller to the production that has one,
        # rather than saying "fail" and dropping the future on the floor.
        if markable or _unmarked(entry.cat, wanted) or _agreement_unmarked(entry, wanted):
            return word
        return None
    # a restored stem beats an appended one, and a longer suffix beats a shorter
    # one ("failed", not "faild")
    candidates.sort(key=lambda row: (-row[0], -row[1], row[2]))
    return candidates[0][2]


def words(*forms: str, cat: str, sem: Any = None, weight: float = 0.0, **features: Any) -> list[Entry]:
    """Several spellings of one entry: ``words("folder", "directory", cat="N", sem="folder")``."""
    return [Entry(form, cat, dict(features), sem if sem is not None else form, weight) for form in forms]


# ----------------------------------------------------------------- open class


#: Suffixes that betray a category for a word the lexicon has never seen, with the
#: letters the stem gets back and the features the suffix carries. A suffix is real
#: evidence, so a *marked* guess outranks a bare one: without that, "the north field
#: failed" has no way to tell which of three unknown words is the verb, and "field"
#: wins by position alone.
GUESS_SUFFIXES: tuple[tuple[str, str, str, Mapping[str, Any]], ...] = (
    ("ing", "", "V", {"aspect": "progressive"}),
    ("ied", "y", "V", {"tense": "past"}),
    ("ed", "", "V", {"tense": "past"}),
    ("ies", "y", "N", {"number": "plural"}),
    ("es", "", "N", {"number": "plural"}),
    ("s", "", "N", {"number": "plural"}),
    ("ies", "y", "V", {"number": "singular", "person": 3, "tense": "present"}),
    ("es", "", "V", {"number": "singular", "person": 3, "tense": "present"}),
    ("s", "", "V", {"number": "singular", "person": 3, "tense": "present"}),
    ("est", "", "Adj", {"degree": "superlative"}),
    ("er", "", "Adj", {"degree": "comparative"}),
    ("ly", "", "Adv", {}),
)


def _lost_e(stem: str) -> bool:
    """Whether a stripped stem is one that dropped a final "e" ("shar" <- "share").

    English drops that "e" before a vowel-initial suffix and the surface form keeps no
    record of it, which is why stripping alone produced "di" for "died" and "ow" for
    "owes". Three shapes give it away: a vowel-final stem ("di"), a "v"-final one (no
    English stem ends in a bare "v", so "arriv" is always "arrive"), and a stem of one
    vowel group ending in a single consonant ("shar"). "open" has two vowel groups and
    so keeps its own final consonant, which is what stops "opene".
    """
    if not stem:
        return False
    if re.search(r"(?:s|x|z|ch|sh)$", stem):
        return False  # a sibilant stem takes "-es" ("box" -> "boxes"), so it lost nothing
    if stem[-1] in "aiou" or stem[-1] == "v":
        return True
    # "w" and "y" after the vowel spell a diphthong rather than closing a syllable, so
    # "show" keeps its own shape ("showed", not "showe" + "d"). The round-trip check
    # cannot settle this one: *both* stems inflect back to "showed", so it can only
    # reject an inconsistent stem, never choose between two consistent ones.
    return bool(re.fullmatch(r"[^aeiou]*[aeiou][^aeiouwy]", stem))


def _stems(token: str, suffix: str, restore: str, cat: str, feats: Mapping[str, Any]) -> list[str]:
    """Stems that could have produced ``token``, best first, each verified by :func:`inflect`.

    A stem is only accepted if running the *same* suffix table forward over it gives
    back the word that was actually heard. That makes stemming and inflection one
    verified pair rather than two rules that drift: "ow" is rejected for "owes"
    because it would have been said "ows", and "carr" because it would have been
    "carred". Where several stems survive, the one that restored letters wins.
    """
    base = token[: -len(suffix)]
    proposals = [(3, base + restore)] if restore else []
    if not restore:
        if _lost_e(base):
            proposals.append((2, base + "e"))
        proposals.append((1, base))
    out: list[tuple[int, str]] = []
    for tier, stem in proposals:
        if not stem:
            continue
        if inflect(Entry(stem, cat, {}, stem), dict(feats)) == token:
            out.append((tier, stem))
    out.sort(key=lambda row: (-row[0], row[1]))
    return [stem for _, stem in out]


@dataclass(frozen=True)
class OpenClass:
    """How an unknown token may still enter the grammar.

    ``sem="entity"`` makes an :class:`Entity` of ``kind``; ``sem="word"`` makes the
    stem itself the meaning, which is what a noun, verb or adjective needs. With
    ``morphology`` the suffix decides the category and contributes its features, so
    "failed" can be a past-tense verb even though no lexicon lists it.

    Entries produced this way carry ``guessed=True`` and a low weight, so any real
    lexical entry outranks them and a caller can see what was guessed at.
    """

    pattern: str
    cat: str
    kind: str = "name"
    weight: float = -0.2          # a reading whose suffix fits the category
    sem: str = "entity"
    features: Mapping[str, Any] = field(default_factory=dict)
    morphology: bool = False
    bare_weight: float | None = None  # an unmarked reading; defaults just below `weight`

    @staticmethod
    def of(spec: "OpenClass | tuple[str, str, str]") -> "OpenClass":
        return spec if isinstance(spec, OpenClass) else OpenClass(spec[0], spec[1], spec[2])


def guess_entries(token: str, spec: OpenClass) -> list[Entry]:
    """Readings an unknown token can take under one open-class rule."""
    text = _unquote(token)
    if not spec.morphology:
        sem = Entity(spec.kind, text) if spec.sem == "entity" else text.lower()
        return [Entry(token, spec.cat, {"number": "singular", "guessed": True, **dict(spec.features)}, sem, spec.weight)]
    low = text.lower()
    out: list[Entry] = []
    best: dict[tuple[str, tuple], str] = {}
    for suffix, restore, cat, feats in GUESS_SUFFIXES:
        if cat != spec.cat or not low.endswith(suffix) or len(low) <= len(suffix) + 1:
            continue
        found = _stems(low, suffix, restore, cat, feats)
        if not found:
            continue
        # two rules can reach the same features by different suffixes ("owes" as -es or
        # -s); keep the shorter stem, which is the one that stripped only the suffix
        key = (cat, tuple(sorted(feats.items())))
        if key not in best or len(found[0]) < len(best[key]):
            best[key] = found[0]
    for (cat, feats), stem in best.items():
        out.append(Entry(token, cat, {"guessed": True, **dict(feats), **dict(spec.features)}, stem, spec.weight))
    #: the bare form stays available, but a suffix that fits is the better guess —
    #: otherwise "arrived" enters as a tenseless predicate and the past is lost
    bare = spec.bare_weight if spec.bare_weight is not None else spec.weight - 0.2
    out.append(Entry(token, spec.cat, {"number": "singular", "guessed": True, **dict(spec.features)}, low, bare))
    return out


# --------------------------------------------------------------------- grammar

_QUOTES = {"'": "'", '"': '"', "`": "`", "“": "”", "‘": "’"}


def _unquote(token: str) -> str:
    if len(token) >= 2 and token[0] in _QUOTES and token[-1] == _QUOTES[token[0]]:
        return token[1:-1]
    return token


def _sem_key(value: Any) -> Any:
    """How :func:`generate._same` distinguishes meanings, as a hashable key (None: cannot)."""
    if isinstance(value, Entity):
        return ("entity", value.text.lower())
    try:
        hash(value)
    except TypeError:
        return None
    return ("value", value)


@dataclass(frozen=True)
class Grammar:
    productions: tuple[Production, ...] = ()
    lexicon: Lexicon = field(default_factory=Lexicon)
    start: tuple[str, ...] = ("S",)  # categories that may cover a whole utterance
    #: token shapes that may enter as open-class entities even when unknown:
    #: (regex, category, entity kind). This is where file names and proper names get
    #: in without being listed.
    open_class: tuple[Any, ...] = ()  # OpenClass, or a bare (regex, cat, kind) tuple
    #: Words no constituent may span. "then" sequences two requests, so letting a
    #: modifier attach across it turns "make a folder then go to documents" into one
    #: request with a destination.
    barriers: tuple[str, ...] = ("then", "afterwards", ".", ";", "?", "!")
    name: str = "grammar"

    def __post_init__(self) -> None:
        by_lhs: dict[str, list[Production]] = defaultdict(list)
        for prod in self.productions:
            by_lhs[prod.lhs.name].append(prod)
        object.__setattr__(self, "_by_lhs", dict(by_lhs))
        specs = [OpenClass.of(spec) for spec in self.open_class]
        object.__setattr__(self, "_open", tuple((re.compile(spec.pattern), spec) for spec in specs))

    def by_lhs(self, name: str) -> list[Production]:
        return getattr(self, "_by_lhs").get(name, [])

    def categories(self) -> set[str]:
        return set(getattr(self, "_by_lhs"))

    def entries_by_cat(self, cat: str) -> tuple[Entry, ...]:
        """Every entry of a category, indexed on first use (generation asks by category)."""
        index = getattr(self, "_by_cat", None)
        if index is None:
            index = defaultdict(list)
            for entries in self.lexicon.entries.values():
                for entry in entries:
                    index[entry.cat].append(entry)
            index = {k: tuple(v) for k, v in index.items()}
            object.__setattr__(self, "_by_cat", index)
        return index.get(cat, ())

    def entries_saying(self, cat: str, meaning: Any) -> tuple[Entry, ...]:
        """Entries of a category that could *mean* this, indexed on first use.

        Generation asks "what words of category C mean M?" for every daughter of every
        candidate production, and answering it by scanning every entry of C was 13.3 s
        of a 24.9 s run — the copula alone is six entries among some sixty verbs. The
        index is keyed the way :func:`generate._same` compares: an entity matches an
        entity with the same text, and anything else matches an equal value. Entries
        whose meaning is not hashable stay in a bucket that is always scanned, so the
        answer is the same set either way.
        """
        index = getattr(self, "_by_cat_sem", None)
        if index is None:
            index, loose = {}, defaultdict(list)
            for entries in self.lexicon.entries.values():
                for entry in entries:
                    key = _sem_key(entry.sem)
                    if key is None:
                        loose[entry.cat].append(entry)
                    else:
                        index.setdefault((entry.cat, key), []).append(entry)
            index = {k: tuple(v) for k, v in index.items()}
            object.__setattr__(self, "_by_cat_sem", index)
            object.__setattr__(self, "_loose_sem", {k: tuple(v) for k, v in loose.items()})
        key = _sem_key(meaning)
        found = index.get((cat, key), ()) if key is not None else ()
        loose = getattr(self, "_loose_sem").get(cat, ())
        if not loose:
            return found if key is not None else self.entries_by_cat(cat)
        if key is None:
            return self.entries_by_cat(cat)
        return found + loose

    def entries_for(self, token: str) -> list[Entry]:
        """Lexical readings of a token, plus open-class readings for unknown shapes."""
        found = self.lexicon.lookup(token)
        if not found:
            for lemma in self.lexicon.near(token):
                # a spelling correction is a guess: mark it, so a caller can see the word
                # was read as something it was not ("firefox" as "firebox")
                found.extend(replace(e, word=token, weight=e.weight - 1.2, features={**e.features, "guessed": True, "corrected_to": lemma})
                             for e in self.lexicon.entries[lemma])
        known = bool(found)
        for pattern, spec in getattr(self, "_open"):
            # a word the lexicon knows keeps its own readings: otherwise "that file"
            # reads as a thing called "that", and "no" as a name
            if known and spec.cat in ("Name", "N", "V", "Adj", "Adv"):
                continue
            if pattern.fullmatch(token) and not any(e.cat == spec.cat for e in found):
                found.extend(guess_entries(token, spec))
        return found

    def extend(self, *, productions: Iterable[Production] = (), lexicon: Lexicon | None = None,
               entries: Iterable[Entry] = (), background: Mapping[str, float] | None = None,
               start: Sequence[str] | None = None, open_class: Sequence[tuple[str, str, str]] | None = None,
               name: str | None = None) -> "Grammar":
        lex = lexicon or self.lexicon
        if entries or background:
            lex = lex.extend(*entries, background=background)
        return Grammar(
            productions=self.productions + tuple(productions),
            lexicon=lex,
            start=tuple(start) if start is not None else self.start,
            open_class=tuple(open_class) if open_class is not None else self.open_class,
            barriers=self.barriers,
            name=name or self.name,
        )


__all__ = [
    "ABSENT", "Ask", "Attach", "Bindings", "Build", "Cat", "Coord", "Ent", "Entry", "FVar", "GUESS_SUFFIXES", "Grammar",
    "Head", "Lexicon", "OpenClass", "guess_entries",
    "Lit", "Locative", "Merge", "Order", "Production", "Qualify", "Sem", "SemRef", "SUFFIX_RULES", "Terminal",
    "build_sem", "ground", "inflect", "merge", "parse_cat", "production", "rename", "resolve", "unify", "words",
]
