"""What kind of thing a question asks for, and whether a candidate answer could be it.

A question carries a requirement its answer must satisfy — a place, a date, a count, a yes or
no — and nothing in this library represented that. Measured consequence: an extractive answerer
returns the entity a question travels *through* rather than the one it asks for ("Who replaced
the manager of Aston Villa that began at Leeds United?" → the manager, not the replacement) and
does so at high confidence, so no abstention threshold can catch it. An expected answer type is
the only signal that can, because the error is not uncertainty — it is a category error.

Two structures, both deterministic and both conservative:

``asked_for`` reads the requirement off the question's own surface (wh-word, head noun, copula
shape). ``mismatch`` rejects a candidate only when it is *confidently* of the wrong kind; a
candidate whose kind cannot be established is admitted, because refusing on ignorance costs
correct answers. ``shape`` distinguishes a question that names its own candidate answers
("Which ran longer, A or B?") from one that must travel through an intermediate — a distinction
that matters because the first may answer itself from its own words and the second may not.

Nothing here is trained and nothing needs a corpus; it is a representation, not a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class AnswerType(str, Enum):
    """The kind of thing a question asks for. ``entity`` is the honest 'a thing, unspecified'."""

    person = "person"
    place = "place"
    date = "date"
    number = "number"
    yes_no = "yes_no"
    entity = "entity"


class Shape(str, Enum):
    """Whether a question names its own candidate answers."""

    comparison = "comparison"
    bridge = "bridge"


MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december")
#: nouns that name a person by role, so "which director" asks for a person
PERSON_NOUNS = ("person", "man", "woman", "author", "director", "actor", "actress", "singer", "writer",
                "player", "artist", "founder", "president", "ceo", "musician", "composer", "producer",
                "politician", "scientist", "poet", "coach", "manager", "owner", "leader", "king", "queen",
                "senator", "governor", "mayor", "judge", "athlete", "driver", "guitarist", "drummer",
                "journalist", "actor's", "father", "mother", "son", "daughter", "brother", "sister", "wife",
                "husband", "star", "host", "narrator", "designer", "architect", "publisher", "editor")
PLACE_NOUNS = ("place", "city", "country", "state", "county", "town", "village", "province", "region",
               "island", "river", "mountain", "street", "location", "venue", "stadium", "district",
               "continent", "capital", "borough", "territory", "nation")
DATE_NOUNS = ("year", "date", "month", "decade", "birthday", "anniversary")
NUMBER_NOUNS = ("number", "count", "population", "total", "amount", "height", "length",
                "distance", "duration", "percentage", "price", "cost")
YES_NO_OPENERS = ("are", "is", "was", "were", "do", "does", "did", "has", "have", "had", "can", "could",
                  "will", "would", "should", "am")
COMPARISON_CUES = (" or ", "both", "which came first", "same", "more than", "less than", "older",
                   "younger", "larger", "smaller", "bigger", "longer", "shorter", "taller", "earlier",
                   "later", "first,", "between", "greater", "which one", "most recent", "who is younger",
                   "which has a", "which was released first")

#: digits included: a tokenizer blind to numbers cannot tell that "28,776" answers a question
#: about a population, nor that a bridge answer of "1994" was lifted from the question
_WORD = re.compile(r"[a-z0-9']+")
_YEAR = re.compile(r"\b\d{3,4}\b(?:\s*(?:bc|bce|ad|ce))?", re.I)
_DIGIT = re.compile(r"\d")
_ALTERNATIVE = re.compile(r"\bor\b", re.I)
#: "43-year-old", "19-year veteran": a digit immediately before "year" makes it a modifier
_AGE_MODIFIER = re.compile(r"\d\s*-?\s*$")
_NUMBER_WORDS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                 "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                 "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
                 "eighty", "ninety", "hundred", "thousand", "million", "billion", "dozen", "zero",
                 "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
                 "tenth", "eleventh", "twelfth", "thirteenth", "twentieth")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def shape(question: str) -> Shape:
    """Does the question name its own candidates? Read off the surface, never from a dataset field."""
    low = f" {question.lower().strip()} "
    return Shape.comparison if any(c in low for c in COMPARISON_CUES) else Shape.bridge


def asked_for(question: str) -> AnswerType:
    """The kind of answer the question requires, from its wh-word and head noun.

    Read from the LAST wh-phrase, not the first: a multi-hop question states the hop it travels
    through before the thing it asks for ("...the brother of the Secretary who was born in what
    year?" asks for a year, not a person), and reading the first cue gets those backwards.
    """
    found = [(m.group(0), m.start()) for m in _WORD.finditer(question.lower())]
    words = [w for w, _ in found]
    if not words:
        return AnswerType.entity
    wh = [(i, w) for i, w in enumerate(words) if w in ("who", "whom", "whose", "where", "when", "what", "which", "how")]
    if not wh:
        return AnswerType.yes_no if words[0] in YES_NO_OPENERS else AnswerType.entity
    # English fronts the interrogative: if the question opens with one (after an optional
    # preposition), that is the ask. Otherwise the ask is the last one, and any earlier wh is a
    # relative clause describing the hop ("the school WHERE he is chancellor") rather than asking.
    i, head = wh[0] if wh[0][0] <= 2 else wh[-1]
    if head in ("who", "whom", "whose"):
        return AnswerType.person
    if head == "where":
        return AnswerType.place
    if head == "when":
        return AnswerType.date
    if head == "how":
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if nxt in ("many", "much", "old", "tall", "long", "far", "big", "large", "high", "deep", "wide"):
            return AnswerType.number
        return AnswerType.entity
    for j, w in enumerate(words[i + 1 : i + 5], start=i + 1):  # "what/which <noun>": the noun carries it
        if w == "year" and _AGE_MODIFIER.search(question[: found[j][1]]):
            continue  # "43-year-old", "19-year veteran": an age modifier, not the thing asked for
        if w in DATE_NOUNS:
            return AnswerType.date
        if w in PERSON_NOUNS:
            return AnswerType.person
        if w in PLACE_NOUNS:
            return AnswerType.place
        if w in NUMBER_NOUNS:
            return AnswerType.number
    return AnswerType.entity


def could_be(text: str, want: AnswerType) -> bool:
    """Could this string be a ``want``? Conservative: unknown kinds are admitted.

    Only the kinds with a reliable surface signature are checked — a date looks like a date and a
    number looks like a number. ``person``, ``place`` and ``entity`` share one surface (a
    capitalised name), so a person is never rejected for looking like a place; doing that needs a
    gazetteer this does not have, and a wrong rejection costs a correct answer.
    """
    s = (text or "").strip()
    if not s:
        return True
    low = s.lower()
    if want is AnswerType.yes_no:
        return low in ("yes", "no")
    if want is AnswerType.date:
        return bool(_YEAR.search(s)) or any(m in low for m in MONTHS) or bool(re.fullmatch(r"\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?", s))
    if want is AnswerType.number:
        return bool(_DIGIT.search(s)) or any(w in _words(s) for w in _NUMBER_WORDS)
    # person, place and entity are never rejected. Rejecting bare numbers here looked safe and was
    # not: on train it threw away a Ferrari '458', an area code '284', a South Park episode '201'
    # and the single '212' — numbers name things routinely, so the rule cost correct answers and
    # caught nothing that the kinds above do not already catch.
    return True


def _normal(text: str) -> str:
    return " ".join(_words(text))


def contains_words(haystack: str, needle: str) -> bool:
    """Does ``needle`` appear in ``haystack`` as a whole run of words?

    Word runs, not characters: "no" is a character substring of "northeastern Ontario" and a
    character test therefore reports that a yes/no answer was lifted from a question about a
    place. Every containment question in this library is about words.
    """
    h, n = _words(haystack), _words(needle)
    if not n or len(n) > len(h):
        return False
    return any(h[i : i + len(n)] == n for i in range(len(h) - len(n) + 1))


def from_question(span: str, question: str) -> bool:
    """Is this candidate lifted from the question's own words?

    On HotpotQA train this is true of the gold answer for 1.6% of bridge questions and 39% of
    comparison questions — a comparison names its own candidates, so the signal only means
    anything for the bridge shape.
    """
    return contains_words(question, span)


@dataclass(frozen=True)
class Rejection:
    """Why a candidate cannot be this question's answer."""

    reason: str
    detail: str


def mismatch(question: str, span: str, *, check_self_reference: bool = True) -> Rejection | None:
    """``None`` if the candidate is admissible, a :class:`Rejection` if it is confidently wrong.

    A comparison question is not checked at all: its answer is one of the options it names, so it
    may be lifted from the question, and "which has a greater population, A or B?" is answered by
    a place name rather than by a number. Measured on train, an either/or question that opens with
    an auxiliary is answered by an option rather than by yes/no often enough that even the yes/no
    check costs more than it buys there.
    """
    if not (span or "").strip():
        return None
    want, form = asked_for(question), shape(question)
    if form is Shape.comparison:
        if want is AnswerType.yes_no and not _ALTERNATIVE.search(question) and not could_be(span, want):
            # "Are both Jonathan Marray and Wayne Black British?" names two entities but offers no
            # alternative to choose between, and 218 of 220 such questions on train take a yes or a
            # no. An either/or comparison is the opposite case and stays unchecked.
            return Rejection("wrong_answer_type", f"{span!r} cannot answer a yes/no comparison")
        return None  # otherwise its answer is one of the options it names, in whatever form
    if not could_be(span, want):
        return Rejection("wrong_answer_type", f"{span!r} cannot be a {want.value}")
    if check_self_reference and from_question(span, question):
        return Rejection("taken_from_the_question",
                         f"{span!r} is already in the question, so it is what the question travels through")
    return None
