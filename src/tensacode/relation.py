"""Comparing two things, and returning what the comparison asks for.

A question can put two things in competition — "Which was published first, A or B?", "Are X and
Y from the same country?", "What profession do A and B have in common?" — and answering it needs
an operation that span extraction cannot perform. Measured consequence
(``docs/revival/26-selection-and-answer-type.md``): on HotpotQA an extractive answerer scores
0.1935 on comparison questions *with the gold evidence already in hand*, identically for a
trained selector and for a perfect oracle, against 0.4958 on questions that travel through an
intermediate. Selection is not the gap. The answer to "who was born first" is one of the two
names the question offers, and to "are both X?" it is a yes or a no; neither is reliably a
substring of the evidence, so no span can be copied out to produce it.

The operation has four parts, and each can refuse:

``read``     what relation is asked, over which two things, on what attribute.
``values``   one value per candidate, read out of the evidence and typed.
``apply``    the relation over those values.
``resolve``  the three in sequence, returning what the question wants.

Refusals are named rather than guessed: a missing value says which candidate it is missing for,
and two values of different dimensions say so instead of comparing their raw numbers. The
arithmetic and the ordering are :mod:`tensacode.quantity` and :mod:`tensacode.temporal`; this
module reads values and picks relations, it does not do sums.

Nothing here is trained. The question side is surface rules because the grammar in
:mod:`tensacode.language`, while it parses these questions with 0.91 coverage, does not carry a
coordination of two candidates or a comparative relation in its frames — it reads "born first"
as a description and assigns "Are both A and B American rock bands?" the predicate ``located``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from .answer_type import AnswerType, Shape, asked_for, contains_words, shape
from .outcomes import Unknown
from .quantity import Quantity, compare as compare_quantities
from .records import Ref, Store
from .semantics_bridge import quantities_in_text
from .temporal import relate, tell_event


class Relation(str, Enum):
    """What a comparison asks to be done with two values."""

    earlier = "earlier"       # which came first
    later = "later"           # which came last / is more recent
    greater = "greater"       # which is more / larger / longer
    less = "less"             # which is fewer / smaller / shorter
    same = "same"             # do they share this attribute
    different = "different"   # do they differ on it
    both = "both"             # do they both satisfy this predicate
    shared = "shared"         # what attribute do they have in common


#: the answer a family produces: one of the named candidates, a yes/no, or a shared value
FAMILIES = {
    Relation.earlier: "which_of_order", Relation.later: "which_of_order",
    Relation.greater: "which_of_magnitude", Relation.less: "which_of_magnitude",
    Relation.same: "yes_no", Relation.different: "yes_no", Relation.both: "yes_no",
    Relation.shared: "shared_attribute",
}

_ORDER_FIRST = re.compile(r"\b(?:first|earlier|earliest|sooner|older|elder|oldest|prior|before)\b", re.I)
_ORDER_LAST = re.compile(r"\b(?:last|later|latest|newer|younger|youngest|most recent|most recently|after)\b", re.I)
_MAGNITUDE_MORE = re.compile(r"\b(?:more|greater|larger|bigger|longer|taller|higher|heavier|most|farther|further|greatest|largest|longest)\b", re.I)
_MAGNITUDE_LESS = re.compile(r"\b(?:less|fewer|smaller|shorter|lower|least|fewest|closer|smallest)\b", re.I)
_SAME = re.compile(r"\b(?:same|alike|shared)\b", re.I)
_DIFFERENT = re.compile(r"\b(?:different|differ|distinct)\b", re.I)
_BOTH = re.compile(r"\b(?:both|either|all)\b", re.I)
_SHARED = re.compile(r"\bin common\b|\b(?:mutual|shared)\b|\b(?:have|share)s?\s+which\b|\bwhich\s+\w+(?:\s+\w+)?\s+(?:do|does)\s+(?:they|both)\b", re.I)

#: an interrogative anywhere in the question: "are both A and B located in which borough?" opens
#: with an auxiliary but asks for a borough, and answering it "yes" answers a different question
_WH = re.compile(r"\b(?:what|which|who|whose|where|when)\b", re.I)

#: a year, the value an order comparison almost always turns on
_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
#: cues that tell which of an entity's several years is the one being compared
_ORDER_CUES = {
    "born": ("born", "birth", "b."), "died": ("died", "death", "d."),
    "released": ("released", "release", "premiered", "premiere", "aired", "debuted", "opened"),
    "founded": ("founded", "formed", "established", "started", "begun", "began", "created", "built", "incorporated"),
    "published": ("published", "publication", "printed", "issued", "wrote", "written"),
}
_STOP = {"the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for", "is", "are", "was", "were",
         "which", "who", "whom", "what", "that", "this", "these", "those", "do", "does", "did", "has",
         "have", "had", "both", "same", "different", "common", "mutual", "first", "last", "more", "less",
         "older", "younger", "earlier", "later", "between", "by", "from", "with", "their", "they", "them",
         "be", "been", "it", "its", "as", "than", "into", "about", "also", "known"}
_WORDS = re.compile(r"[A-Za-z0-9']+")
#: a capitalised run, which is how a named candidate appears in a question
_NAME = re.compile(r"\b(?:[A-Z][\w.&'’-]*|of|the|and|de|von|van|da|del|di|for|in)(?:\s+(?:[A-Z][\w.&'’-]*|of|the|and|de|von|van|da|del|di|for|in))*")


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORDS.findall(text or "")]


def _stem(word: str) -> str:
    """Crude singularisation. Intersections are the operation here, and "rock bands" must meet
    "rock band"; a plural that does not stem costs a correct yes."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 3:
        # only a sibilant plural loses the whole "es": boxes -> box, but games -> game
        return word[:-2] if word[-3] in "sxzh" else word[:-1]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _content(text: str) -> list[str]:
    return [_stem(w) for w in _words(text) if w not in _STOP and len(w) > 2]


# --------------------------------------------------------------- the question


@dataclass(frozen=True)
class Comparison:
    """What a comparison question asks: a relation, over two named things, on an attribute."""

    relation: Relation
    candidates: tuple[str, str]
    attribute: str
    wants: AnswerType
    question: str = ""

    @property
    def family(self) -> str:
        return FAMILIES[self.relation]

    def describe(self) -> str:
        return f"{self.relation.value}({self.candidates[0]!r}, {self.candidates[1]!r}) on {self.attribute!r}"


def _relation_of(question: str) -> Relation | None:
    """Which relation the surface asks for. Order is checked before magnitude because "older"
    reads as an age comparison and answers with a date, and before same/different because
    "Which came first, A or B?" can also contain "both"."""
    low = f" {question.lower()} "
    if _SHARED.search(question):
        return Relation.shared
    if _ORDER_FIRST.search(low):
        return Relation.earlier
    if _ORDER_LAST.search(low):
        return Relation.later
    if _MAGNITUDE_MORE.search(low):
        return Relation.greater
    if _MAGNITUDE_LESS.search(low):
        return Relation.less
    if _DIFFERENT.search(low):
        return Relation.different
    if _SAME.search(low):
        return Relation.same
    if _BOTH.search(low):
        return Relation.both
    return None


_CONNECTORS = ("of", "the", "and", "de", "von", "van", "da", "del", "di", "for", "in", "a", "an")
_INTERROGATIVES = ("Which", "Who", "Whose", "What", "Were", "Was", "Are", "Is", "Do", "Does", "Did",
                   "Has", "Have", "Had", "Between", "Both", "If", "In", "The")


_LEADING = tuple(w.lower() for w in _INTERROGATIVES[:-1]) + _CONNECTORS


def _trim(name: str) -> str:
    """A name does not begin or end with a connector: "Ed Wood of the" is "Ed Wood"."""
    words = name.strip(" ,.;:").split()
    while len(words) > 1 and words[0].lower() in _LEADING:
        words.pop(0)
    while words and words[-1].lower() in _CONNECTORS:
        words.pop()
    return " ".join(words)


def _coordinated(question: str) -> tuple[str, str] | None:
    """The two names a question coordinates: "A or B", "both A and B"."""
    parts = re.split(r",?\s+\bor\b\s+|,?\s+\band\b\s+", question.strip().rstrip("?"))
    if len(parts) < 2:
        return None
    lower = tuple(w.lower() for w in _INTERROGATIVES)
    runs = [[_trim(m) for m in _NAME.findall(part) if _trim(m).lower() not in lower]
            for part in (parts[0], parts[-1])]
    if not runs[0] or not runs[1]:
        return None
    first, second = (max(r, key=len) for r in runs)
    if not first or not second or first.lower() == second.lower():
        return None
    if len(first.split()) > 12 or len(second.split()) > 12:
        return None
    return (first, second)


def _best_title(half: str, titles: list[str]) -> str | None:
    """The evidence title this half of a coordination names, if any.

    This is what rescues a greedy capitalised run: "Kings of Leon American" contains the title
    "Kings of Leon", and the title is the name of the thing being compared.
    """
    matches = [t for t in titles if contains_words(half, t) or contains_words(t, half)]
    return max(matches, key=len) if matches else None


def _candidates(question: str, titles: tuple[str, ...] = ()) -> tuple[str, str] | None:
    """The two things being compared.

    Two readers. If the evidence carries titles — a document collection names its entities — the
    candidates are the titles the question mentions, which is exact. Otherwise they are read off
    the question's own coordination, which is where a comparison puts them. When both are
    available the coordination is mapped onto the titles, which fixes a run that swallowed a word
    of the predicate.
    """
    named = [t for t in titles if t and contains_words(question, t)]
    pair = _coordinated(question)
    if named and pair is not None:
        mapped = [_best_title(half, named) for half in pair]
        if all(mapped) and mapped[0].lower() != mapped[1].lower():
            return (mapped[0], mapped[1])  # type: ignore[return-value]
    if len(named) == 2:
        return (named[0], named[1])
    return pair


def _attribute(question: str, candidates: tuple[str, str]) -> str:
    """The property the comparison is about, with the candidates and the relation words removed."""
    text = question
    for c in candidates:
        text = re.sub(re.escape(c), " ", text, flags=re.I)
    return " ".join(_content(text))


def read(question: str, titles: tuple[str, ...] = ()) -> Comparison | Unknown:
    """Read a comparison off a question, or refuse and say why.

    ``titles`` are the names the evidence collection uses, when it has them; they make candidate
    identification exact rather than a guess at a coordination.
    """
    relation = _relation_of(question)
    if relation is None:
        return Unknown("relation_unsupported", "no comparative, same/different or in-common cue in the question")
    # ``shape`` reads " or ", "both", "more than" and the comparatives; it does not read "in
    # common" or "of the same", so a same/different/shared/both cue counts as a marker in its own
    # right. An order or magnitude cue alone does not: "who was the first president of X and what
    # did he found?" is a bridge question containing the word "first".
    if relation in (Relation.both, Relation.same, Relation.different) and _WH.search(question):
        # the cue says "both", but the question asks for a value rather than for a verdict
        relation = Relation.shared
    self_naming = shape(question) is Shape.comparison
    if not self_naming and relation not in (Relation.shared, Relation.same, Relation.different, Relation.both):
        return Unknown("not_a_comparison", "the question does not name its own candidates")
    pair = _candidates(question, titles)
    if pair is None:
        return Unknown("candidates_unclear", "could not read exactly two named things being compared")
    return Comparison(relation, pair, _attribute(question, pair), asked_for(question), question)


# ---------------------------------------------------------------- the evidence


@dataclass
class Values:
    """One value per candidate, and what they were read from."""

    values: dict[str, object] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)


def _titles_for(candidate: str, evidence: tuple[tuple[str, str], ...]) -> list[str]:
    return [t for t, _ in evidence
            if t and (contains_words(t, candidate) or contains_words(candidate, t))]


def _for_candidate(candidate: str, evidence: tuple[tuple[str, str], ...]) -> list[str]:
    """The sentences that speak about this candidate: its own document's, else any that name it."""
    own = [s for title, s in evidence
           if title and (contains_words(title, candidate) or contains_words(candidate, title))]
    return own or [s for _, s in evidence if contains_words(s, candidate)]


def _cue_for(attribute: str) -> tuple[str, ...]:
    """Which event an order comparison is about, so the right year is chosen among several."""
    words = set(_words(attribute))
    for cues in _ORDER_CUES.values():
        if words & set(cues):
            return cues
    return ()


def year_in(sentences: list[str], attribute: str) -> tuple[int, str] | None:
    """The year an order comparison turns on: the one beside the event's cue, else the earliest.

    A biography states several years; "born first" is about one of them. Taking the earliest is
    the right default for birth and founding (the first year a thing is mentioned in its own
    article is usually its origin) and is stated as a default rather than a rule.
    """
    cues = _cue_for(attribute)
    earliest: tuple[int, str] | None = None
    nearest: tuple[int, int, str] | None = None  # distance, year, sentence
    for sentence in sentences:
        low = sentence.lower()
        spots = [m.start() for cue in cues for m in re.finditer(re.escape(cue), low)]
        for match in _YEAR.finditer(sentence):
            year = int(match.group(0))
            if earliest is None or year < earliest[0]:
                earliest = (year, sentence)
            if not spots:
                continue
            # English puts the year after the verb ("died in 1994"), so a year that follows its
            # cue is nearer than one the same distance in front of it — otherwise "(born 1930)
            # died in 1994" answers a question about the death with the birth year.
            gap = min((match.start() - spot) if match.start() >= spot
                      else (spot - match.start()) + 25 for spot in spots)
            if gap <= 40 and (nearest is None or gap < nearest[0]):
                nearest = (gap, year, sentence)
    if nearest is not None:
        return (nearest[1], nearest[2])
    return earliest


#: :func:`tensacode.semantics_bridge.quantities_in_text` reads digits only, so "a band with four
#: members" yields no quantity at all. Prose spells small numbers out, and a magnitude comparison
#: that cannot read "four" refuses on half its cases; these are substituted before the reader runs.
SPELLED = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
           "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
           "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
           "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
           "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000, "million": 1_000_000,
           "billion": 1_000_000_000, "dozen": 12, "solo": 1, "duo": 2, "trio": 3, "quartet": 4,
           "quintet": 5, "sextet": 6, "septet": 7, "octet": 8}
_SPELLED_RE = re.compile(r"\b(" + "|".join(SPELLED) + r")\b", re.I)


def digitise(text: str) -> str:
    """Spelled-out numbers as digits, so a digit-only quantity reader can see them."""
    return _SPELLED_RE.sub(lambda m: f"{SPELLED[m.group(0).lower()]:g}", text)


def references(question: str, candidates: tuple[str, str]) -> tuple[str, ...]:
    """Names the question measures against, other than the two candidates.

    "Which airport is closer to Washington D.C., Dulles or Gainesville Regional?" compares two
    distances *to Washington*. Without this, the operation compared Dulles's distance to
    Washington with Gainesville Regional's distance to Gainesville and answered confidently.
    """
    out = []
    for run in _NAME.findall(question):
        name = _trim(run)
        if not name or name.lower() in tuple(w.lower() for w in _INTERROGATIVES):
            continue
        if any(contains_words(c, name) or contains_words(name, c) for c in candidates):
            continue
        out.append(name)
    return tuple(out)


def _forms(text: str) -> set[str]:
    return {w for word in _words(text) for w in (word, _stem(word))}


def quantities_offered(sentences: list[str], attribute: str, require: tuple[str, ...] = ()
                       ) -> tuple[list[tuple[Quantity, str]], list[tuple[Quantity, str]]]:
    """Quantities said of a candidate: the ones the question's noun names, and the rest.

    The rest are candidates for a joint choice — see :func:`pair_of_quantities` — because which
    number a sentence offers is often only decidable by looking at what the *other* candidate
    offers. A bare number the reader labelled "item" carries no property and is not offered.
    """
    if require:
        sentences = [s for s in sentences if any(contains_words(s, r) for r in require)]
    wanted = _forms(attribute)
    matched: list[tuple[Quantity, str]] = []
    offered: list[tuple[Quantity, str]] = []
    for sentence in sentences:
        for mention in quantities_in_text(digitise(sentence)):
            q = mention.quantity
            if wanted & (_forms(mention.of) | _forms(str(q.unit))):
                matched.append((q, sentence))
                continue
            generic = q.unit.dimensionless or str(q.unit) in ("item", "")
            if generic or _YEAR.fullmatch(f"{q.value:.0f}"):
                continue
            offered.append((q, sentence))
    return matched, offered


def pair_of_quantities(a: tuple[list, list], b: tuple[list, list]
                       ) -> tuple[tuple[Quantity, str], tuple[Quantity, str]] | None:
    """One quantity per candidate, chosen so the two are of one dimension.

    "Which canal is longer, the Shinnecock or the Wiconisco?" states 4700 feet for one and 12
    miles for the other, and neither sentence repeats the question's noun. What makes them the
    right two numbers is that they are both lengths, and nothing else on offer is. When more than
    one dimension is shared, the choice is ambiguous and the operation refuses.
    """
    for left, right in ((a[0], b[0]), (a[0] + a[1], b[0] + b[1])):
        if left and right:
            dims_a = {q.dimension for q, _ in left}
            dims_b = {q.dimension for q, _ in right}
            shared = dims_a & dims_b
            if len(shared) == 1:
                dim = shared.pop()
                return (next(p for p in left if p[0].dimension == dim),
                        next(p for p in right if p[0].dimension == dim))
            if len(shared) > 1:
                return None
    return None


#: demonyms and country names, for the commonest same/different attribute there is. A lexicon,
#: not a model: "were A and B from the same country?" is answered by comparing two nationality
#: words, and no amount of sentence overlap substitutes for knowing which words those are.
NATIONALITIES = (
    "american", "british", "english", "scottish", "welsh", "irish", "canadian", "australian",
    "new zealand", "french", "german", "italian", "spanish", "portuguese", "dutch", "belgian",
    "swiss", "austrian", "swedish", "norwegian", "danish", "finnish", "icelandic", "russian",
    "polish", "czech", "slovak", "hungarian", "romanian", "bulgarian", "serbian", "croatian",
    "slovenian", "bosnian", "albanian", "greek", "turkish", "ukrainian", "belarusian", "estonian",
    "latvian", "lithuanian", "chinese", "japanese", "korean", "indian", "pakistani", "bangladeshi",
    "sri lankan", "nepali", "thai", "vietnamese", "filipino", "indonesian", "malaysian",
    "singaporean", "mongolian", "iranian", "iraqi", "israeli", "lebanese", "syrian", "jordanian",
    "saudi", "egyptian", "moroccan", "algerian", "tunisian", "libyan", "sudanese", "ethiopian",
    "kenyan", "nigerian", "ghanaian", "senegalese", "cameroonian", "ugandan", "tanzanian",
    "zimbabwean", "zambian", "south african", "mexican", "guatemalan", "cuban", "jamaican",
    "haitian", "dominican", "puerto rican", "colombian", "venezuelan", "ecuadorian", "peruvian",
    "bolivian", "chilean", "argentine", "argentinian", "uruguayan", "paraguayan", "brazilian",
    "scandinavian", "soviet", "yugoslav", "taiwanese", "hong kong", "kazakh", "uzbek", "georgian",
    "armenian", "azerbaijani", "afghan", "cambodian", "laotian", "burmese", "myanmar",
)
_COUNTRIES = tuple(c for c in (
    "united states", "america", "united kingdom", "england", "scotland", "wales", "ireland",
    "canada", "australia", "france", "germany", "italy", "spain", "japan", "china", "india",
    "russia", "poland", "brazil", "mexico", "argentina", "sweden", "norway", "denmark", "finland",
    "netherlands", "belgium", "switzerland", "austria", "greece", "turkey", "israel", "egypt",
    "south africa", "nigeria", "kenya", "korea", "vietnam", "thailand", "philippines",
) )

#: which kind of value an attribute names, and how to find it
_NATIONALITY_CUES = ("nationality", "country", "nation", "citizenship", "from the same",
                     "same country", "national")
_KIND_CUES = ("profession", "occupation", "job", "career", "genre", "type", "kind", "field",
              "role", "sport", "instrument", "discipline", "subject", "category", "industry")
CONTINENTS = ("africa", "asia", "europe", "north america", "south america", "australia",
              "antarctica", "oceania", "eurasia", "american", "african", "asian", "european")
_GEOGRAPHY_CUES = ("continent", "hemisphere")
_COPULA = re.compile(r"\b(?:is|was|were|are|being|became|remains)\b", re.I)


def complement_of(sentence: str) -> str:
    """What a sentence predicates of its subject: the text after the copula.

    "Scott Derrickson is an American director" says *American director*; the subject's own name is
    not part of what is said about it, and including it makes every pair of people who share a
    first name look alike.
    """
    stripped = re.sub(r"\([^)]*\)", " ", sentence)
    match = _COPULA.search(stripped)
    return stripped[match.end() :] if match else stripped


def attribute_kind(attribute: str) -> str:
    """Which reader an attribute needs: a nationality, a kind-of-thing, or nothing known."""
    low = f" {attribute.lower()} "
    if any(c in low for c in _NATIONALITY_CUES):
        return "nationality"
    if any(c in low for c in _GEOGRAPHY_CUES):
        return "geography"
    if any(f" {c} " in low or c in low for c in _KIND_CUES):
        return "kind"
    return "unknown"


def said_of(sentences: list[str]) -> set[str]:
    """Every content word said about a candidate — the test set for "are both X?"."""
    return {w for s in sentences for w in _content(s)}


def attribute_phrases(sentences: list[str], attribute: str, *, need_reader: bool = True) -> set[str] | None:
    """The value of the asked attribute for one candidate, or ``None`` if it cannot be read.

    Refusing here is the point. Comparing two *whole sentences* for overlap says two film
    directors of different nationalities are "the same nationality" because both sentences
    contain the word "director"; only the attribute's own value can answer the question.
    """
    kind = attribute_kind(attribute)
    text = " ".join(sentences).lower()
    if kind == "nationality":
        found = {n for n in NATIONALITIES if re.search(rf"\b{re.escape(n)}\b", text)}
        found |= {c for c in _COUNTRIES if re.search(rf"\b{re.escape(c)}\b", text)}
        return found or None
    if kind == "geography":
        return {c for c in CONTINENTS if re.search(rf"\b{re.escape(c)}\b", text)} or None
    if kind == "kind" or not need_reader:
        complement = " ".join(complement_of(s) for s in sentences)
        return (set(_content(complement)) - set(_content(attribute))) or None
    # No reader for this attribute. Its value is whatever the evidence says beside the attribute's
    # own name: "the family Cistaceae" answers a question about families. Falling back to the
    # overlap of two whole sentences instead was measured on train and was wrong in the worst
    # way — two genera both described as "flowering plants" were called the same family.
    head = attribute_head(attribute)
    if not head:
        return None
    found: set[str] = set()
    for spot in re.finditer(rf"\b{re.escape(head)}\w{{0,3}}\b", text):
        after = _content(text[spot.end() : spot.end() + 40])
        before = _content(text[max(0, spot.start() - 30) : spot.start()])
        found.update(after[:3])
        found.update(before[-1:])
    return found or None


def attribute_head(attribute: str) -> str:
    """The noun whose value is being compared: the word after "same", else the last content word."""
    words = _content(attribute)
    if not words:
        return ""
    if "same" in attribute.lower().split():
        tail = attribute.lower().split()
        i = tail.index("same")
        rest = [_stem(w) for w in tail[i + 1 :] if w not in _STOP]
        if rest:
            return rest[-1]
    return words[-1]


def values(comparison: Comparison, evidence: tuple[tuple[str, str], ...]) -> Values:
    """Read one value per candidate out of the evidence, typed by what the relation needs."""
    got = Values()
    if comparison.family == "which_of_magnitude":
        require = references(comparison.question, comparison.candidates)
        offers = {}
        for candidate in comparison.candidates:
            sentences = _for_candidate(candidate, evidence)
            if not sentences:
                got.missing.append(candidate)
            else:
                offers[candidate] = quantities_offered(sentences, comparison.attribute, require)
        if got.missing:
            return got
        a_name, b_name = comparison.candidates
        chosen = pair_of_quantities(offers[a_name], offers[b_name])
        if chosen is None:
            got.missing.extend(name for name in comparison.candidates
                               if not any(offers[name]))
            if not got.missing:  # both offered something, but not one comparable pair
                got.missing.append(f"a comparable pair for {a_name} and {b_name}")
            return got
        for name, (quantity, sentence) in zip(comparison.candidates, chosen):
            got.values[name], got.sources[name] = quantity, sentence
        return got
    for candidate in comparison.candidates:
        sentences = _for_candidate(candidate, evidence)
        if not sentences:
            got.missing.append(candidate)
            continue
        if comparison.family == "which_of_order":
            found = year_in(sentences, comparison.attribute)
        elif comparison.relation is Relation.both:
            words = said_of(sentences + _titles_for(candidate, evidence))
            found = (words, sentences[0]) if words else None
        else:
            phrases = attribute_phrases(sentences, comparison.attribute,
                                        need_reader=comparison.relation is not Relation.shared)
            found = (phrases, sentences[0]) if phrases is not None else None
        if found is None:
            got.missing.append(candidate)
        else:
            got.values[candidate] = found[0]
            got.sources[candidate] = found[1]
    return got


# --------------------------------------------------------------- the operation


@dataclass(frozen=True)
class Resolved:
    """What the comparison answers, and the working that produced it."""

    text: str
    relation: Relation
    steps: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def describe(self) -> str:
        return f"{self.text} — {' ; '.join(self.steps)}"


def _order(a_name: str, a_year: int, b_name: str, b_year: int, relation: Relation) -> Resolved | Unknown:
    """Order two years through :mod:`tensacode.temporal`, so the ordering is recorded as claims."""
    mind = Store()
    source = Ref("obs:evidence")
    a_ref, b_ref = Ref("thing:a"), Ref("thing:b")
    tell_event(mind, a_ref, at=datetime(a_year, 1, 1, tzinfo=timezone.utc), kind="compared", source=source)
    tell_event(mind, b_ref, at=datetime(b_year, 1, 1, tzinfo=timezone.utc), kind="compared", source=source)
    how = relate(mind, a_ref, b_ref)
    if isinstance(how, Unknown):
        return how
    if how == "simultaneous":
        return Unknown("tied", f"{a_name} and {b_name} are both {a_year}")
    earlier = a_name if how == "before" else b_name
    later = b_name if how == "before" else a_name
    wanted = earlier if relation is Relation.earlier else later
    return Resolved(wanted, relation,
                    (f"{a_name}: {a_year}", f"{b_name}: {b_year}", f"{earlier} is before {later}",
                     f"asked for the {relation.value} one, so {wanted}"))


def _counted(q: Quantity) -> bool:
    """Is this a count of some thing, rather than a measure in a physical unit?"""
    dims = q.dimension
    return len(dims) == 1 and dims[0][0].startswith("count:") and dims[0][1] == 1


def _magnitude(a_name: str, a: Quantity, b_name: str, b: Quantity, relation: Relation) -> Resolved | Unknown:
    if not a.comparable(b) and _counted(a) and _counted(b):
        # quantity.compare refuses a count of people against a count of residents, correctly for
        # arithmetic — you cannot add them. A question that asks which has more has already said
        # the two counts are of one thing, so the magnitudes compare and the crossing is recorded.
        how = "greater" if a.value > b.value else "less" if a.value < b.value else "equal"
        crossed = f"compared {a.unit} with {b.unit} as counts, on the question's word"
    else:
        how = compare_quantities(a, b)
        crossed = ""
    if isinstance(how, Unknown):
        return how  # dimension mismatch, named by quantity.compare
    if how == "equal":
        return Unknown("tied", f"{a_name} and {b_name} are both {a}")
    bigger = a_name if how == "greater" else b_name
    smaller = b_name if how == "greater" else a_name
    wanted = bigger if relation is Relation.greater else smaller
    steps = (f"{a_name}: {a}", f"{b_name}: {b}", f"{bigger} is greater than {smaller}",
             f"asked for the {relation.value} one, so {wanted}")
    return Resolved(wanted, relation, steps + ((crossed,) if crossed else ()))


def _yes_no(comparison: Comparison, got: Values) -> Resolved | Unknown:
    """Same, different, or both: a yes or a no, from what is said about each candidate."""
    a_name, b_name = comparison.candidates
    a, b = got.values.get(a_name), got.values.get(b_name)
    if not isinstance(a, set) or not isinstance(b, set):
        return Unknown("value_missing", "nothing said about one of the candidates")
    if comparison.relation is Relation.both:
        wanted = set(_content(comparison.attribute))
        if not wanted:
            return Unknown("attribute_empty", "the question names no predicate to test")
        holds = wanted <= a and wanted <= b
        return Resolved("yes" if holds else "no", comparison.relation,
                        (f"asked whether both are {' '.join(sorted(wanted))}",
                         f"{a_name} is missing {' '.join(sorted(wanted - a)) or 'nothing'}",
                         f"{b_name} is missing {' '.join(sorted(wanted - b)) or 'nothing'}"))
    overlap = a & b
    same = bool(overlap)
    holds = same if comparison.relation is Relation.same else not same
    return Resolved("yes" if holds else "no", comparison.relation,
                    (f"shared: {' '.join(sorted(overlap)[:6]) or 'nothing'}",
                     f"asked whether they are {comparison.relation.value}, so {'yes' if holds else 'no'}"))


def _shared(comparison: Comparison, got: Values) -> Resolved | Unknown:
    """What two things have in common: the attribute value both their descriptions carry."""
    a_name, b_name = comparison.candidates
    a, b = got.values.get(a_name), got.values.get(b_name)
    if not isinstance(a, set) or not isinstance(b, set):
        return Unknown("value_missing", "nothing said about one of the candidates")
    overlap = a & b
    if not overlap:
        return Unknown("no_shared_value", f"nothing said of both {a_name} and {b_name}")
    phrase = longest_shared_phrase(complement_of(got.sources[a_name]),
                                   complement_of(got.sources[b_name]))
    if phrase is None:
        heads = _content(complement_of(got.sources[a_name]))
        phrase = max(overlap, key=lambda w: heads.index(w) if w in heads else -1)
    return Resolved(phrase, comparison.relation,
                    (f"said of both: {' '.join(sorted(overlap)[:8])}", f"answering with {phrase!r}"))


def longest_shared_phrase(a: str, b: str) -> str | None:
    """The longest run of words the two descriptions share, in its original spelling.

    A single word under-answers: two people described as "an American actress and film director"
    and "a French film director" share *film director*, and the gold answer to what they have in
    common is the phrase, not its head. Matching is on stemmed words so a plural meets a
    singular, but what is returned is the surface text, because "genu" is not an answer.
    """
    a_words, b_words = _WORDS.findall(a), _WORDS.findall(b)
    a_stem = [_stem(w.lower()) for w in a_words]
    b_stem = [_stem(w.lower()) for w in b_words]
    best: tuple[int, int] | None = None  # length, start in a
    for i in range(len(a_stem)):
        for j in range(len(b_stem)):
            n = 0
            while i + n < len(a_stem) and j + n < len(b_stem) and a_stem[i + n] == b_stem[j + n]:
                n += 1
            while n and (a_stem[i + n - 1] in _STOP or len(a_stem[i + n - 1]) <= 2):
                n -= 1  # a phrase does not end in a stopword
            start = i
            while n and (a_stem[start] in _STOP or len(a_stem[start]) <= 2):
                start += 1
                n -= 1  # nor begin in one
            if n and (best is None or n > best[0]):
                best = (n, start)
    if best is None:
        return None
    n, start = best
    return " ".join(a_words[start : start + n])


def apply(comparison: Comparison, got: Values) -> Resolved | Unknown:
    """Apply the relation to the values, or refuse for a named reason."""
    if got.missing:
        return Unknown("value_missing", f"no value found for {', '.join(got.missing)}")
    a_name, b_name = comparison.candidates
    a, b = got.values[a_name], got.values[b_name]
    if comparison.family == "which_of_order":
        if not (isinstance(a, int) and isinstance(b, int)):
            return Unknown("incomparable", "an order comparison needs a year for each candidate")
        return _order(a_name, a, b_name, b, comparison.relation)
    if comparison.family == "which_of_magnitude":
        if not (isinstance(a, Quantity) and isinstance(b, Quantity)):
            return Unknown("incomparable", "a magnitude comparison needs a quantity for each candidate")
        return _magnitude(a_name, a, b_name, b, comparison.relation)
    if comparison.family == "yes_no":
        return _yes_no(comparison, got)
    return _shared(comparison, got)


def resolve(question: str, evidence: tuple[tuple[str, str], ...]) -> Resolved | Unknown:
    """Answer a comparison question from evidence, or refuse with a named reason."""
    comparison = read(question, tuple(dict.fromkeys(t for t, _ in evidence)))
    if isinstance(comparison, Unknown):
        return comparison
    got = values(comparison, evidence)
    out = apply(comparison, got)
    if isinstance(out, Resolved):
        sources = tuple(dict.fromkeys(got.sources.values()))
        return Resolved(out.text, out.relation, out.steps, sources)
    return out


# ----------------------------------------------------------- word problems


_DIFFERENCE = re.compile(r"how (?:many|much) (?:more|fewer|less|longer|older|younger|farther|greater)\b", re.I)
_TIMES = re.compile(r"how many times\b", re.I)


def difference_in(question: str, text: str) -> Resolved | Unknown:
    """A "how many more X than Y" word problem: the gap between two quantities of one dimension.

    This is the same operation as :func:`_magnitude` with a subtraction instead of a winner, and
    it exists to test whether the comparison faculty transfers to arithmetic word problems.
    """
    if not _DIFFERENCE.search(question):
        return Unknown("not_a_difference", "the question does not ask for a gap between two amounts")
    mentions = quantities_in_text(text)
    if len(mentions) < 2:
        return Unknown("value_missing", f"found {len(mentions)} quantities, need two")
    a, b = mentions[0].quantity, mentions[1].quantity
    if not a.comparable(b) and not (_counted(a) and _counted(b)):
        return Unknown("incomparable", f"cannot subtract {b} from {a}")
    gap = abs(a.base() - b.base()) if a.comparable(b) else abs(a.value - b.value)
    return Resolved(f"{gap:g}", Relation.greater,
                    (f"{a} and {b}", f"difference {gap:g}"))
