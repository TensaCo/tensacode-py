"""A core English grammar: function words, inflection, and the productions over them.

This is the part that is knowledge of *English* rather than of any domain. It
covers the constructions a cognitive program actually has to read:

* imperatives ("make a folder called recipes on my desktop"), questions (yes/no
  and wh-), and declaratives;
* reported speech ("Anem said the north field failed"), which is the construction
  that must never be flattened into the shared world;
* negation, tense and aspect, modality, quantifiers, coordination, comparatives;
* pronouns, left for :mod:`tensorcode.language.discourse` to resolve.

Content words live in the domain: a caller extends this grammar with its verbs,
nouns and names (``DESKTOP`` for the assistant; a village's drifting lexicon for
the simulation). Prepositions carry the *role* they mark as their meaning, so
"in downloads" and "to documents" share one production.
"""

from __future__ import annotations

import math
from dataclasses import replace

from .grammar import (
    Ask, Attach, Build, Coord, Ent, Entry, Grammar, Head, Lexicon, Lit, Locative, Merge, OpenClass, Order, Qualify,
    production, words,
)
from .semantics import Entity

# --------------------------------------------------------------- function words

#: Articles, possessives and demonstratives mark definiteness, not quantity. Their
#: word contributes no meaning of its own, so no production reads their ``sem``.
DETERMINERS = (
    *words("the", cat="Det", definite=True),
    *words("a", "an", cat="Det", definite=False),
    # marked determiners are dispreferred when saying something, so a meaning that
    # only asks for definiteness comes out as "the" rather than "her"
    *words("my", "your", "our", "his", "her", "their", "its", cat="Det", weight=-0.3, definite=True, possessive=True),
    *words("this", "that", cat="Det", weight=-0.3, definite=True, demonstrative=True, number="singular"),
    *words("these", "those", cat="Det", weight=-0.3, definite=True, demonstrative=True, number="plural"),
)

#: Quantifiers do contribute meaning: it becomes the entity's ``quantifier``.
QUANTIFIERS = (
    *words("every", "each", cat="Quant", sem="all", number="singular"),
    *words("all", "both", cat="Quant", sem="all", number="plural"),
    *words("some", cat="Quant", sem="some"),
    # "any" is polarity-sensitive — "any bread" belongs in a question or a negative — so
    # it is the second way to say `some`, not the first
    *words("any", cat="Quant", sem="some", weight=-0.3),
    *words("no", cat="Quant", sem="none"),
    *words("another", "other", cat="Quant", sem="other"),
    # Amounts. Unlike `all`/`some`/`none`, these keep *one meaning per word* rather than
    # a canonical one per group: which word a dialect reaches for is information its
    # world uses ("heaps" against "much" is how drift shows up), and a speaker asked for
    # "heaps" must not be given "much". Normalising them is the caller's business.
    # A mass quantifier takes a singular mass noun ("much bread"), a count one a plural
    # ("many loaves"), and the partitives need "of" — which is what `partitive` marks.
    # ``amount`` marks the ones that need "of" to reach a determiner — "much of the
    # food", never "much the food", where "all the food" is fine. ``partitive`` marks
    # the stronger case: those cannot stand in front of a noun at all without "of".
    *words("much", cat="Quant", sem="much", number="singular", amount=True),
    *words("little", cat="Quant", sem="little", number="singular", amount=True),
    *words("many", cat="Quant", sem="many", number="plural", amount=True),
    *words("few", cat="Quant", sem="few", number="plural", amount=True),
    *words("full", cat="Quant", sem="full", amount=True, weight=-0.1),
    *words("plenty", cat="Quant", sem="plenty", amount=True, partitive=True),
    *words("heaps", cat="Quant", sem="heaps", amount=True, partitive=True),
    *words("lots", cat="Quant", sem="lots", amount=True, partitive=True),
    *words("loads", cat="Quant", sem="loads", amount=True, partitive=True),
)

PRONOUNS = (
    Entry("i", "Pron", {"person": 1, "number": "singular"}, Entity("pronoun", "i", {"person": 1})),
    Entry("me", "Pron", {"person": 1, "number": "singular"}, Entity("pronoun", "me", {"person": 1})),
    Entry("we", "Pron", {"person": 1, "number": "plural"}, Entity("pronoun", "we", {"person": 1})),
    Entry("you", "Pron", {"person": 2}, Entity("pronoun", "you", {"person": 2})),
    Entry("it", "Pron", {"person": 3, "number": "singular"}, Entity("pronoun", "it", {"animate": False})),
    Entry("they", "Pron", {"person": 3, "number": "plural"}, Entity("pronoun", "they", {})),
    Entry("them", "Pron", {"person": 3, "number": "plural"}, Entity("pronoun", "them", {})),
    Entry("he", "Pron", {"person": 3, "number": "singular"}, Entity("pronoun", "he", {"animate": True, "gender": "m"})),
    Entry("him", "Pron", {"person": 3, "number": "singular"}, Entity("pronoun", "him", {"animate": True, "gender": "m"})),
    Entry("she", "Pron", {"person": 3, "number": "singular"}, Entity("pronoun", "she", {"animate": True, "gender": "f"})),
    Entry("there", "Pron", {"person": 3, "locative": True}, Entity("pronoun", "there", {"locative": True})),
    Entry("here", "Pron", {"person": 3, "locative": True}, Entity("pronoun", "here", {"locative": True})),
    Entry("everyone", "Pron", {"person": 3, "number": "singular"}, Entity("quantified", "everyone", {"quantifier": "all", "animate": True})),
    Entry("everybody", "Pron", {"person": 3, "number": "singular"}, Entity("quantified", "everybody", {"quantifier": "all", "animate": True})),
    Entry("everything", "Pron", {"person": 3, "number": "singular"}, Entity("quantified", "everything", {"quantifier": "all"})),
    Entry("someone", "Pron", {"person": 3, "number": "singular"}, Entity("quantified", "someone", {"quantifier": "some", "animate": True})),
    Entry("something", "Pron", {"person": 3, "number": "singular"}, Entity("quantified", "something", {"quantifier": "some"})),
    Entry("nobody", "Pron", {"person": 3, "number": "singular"}, Entity("quantified", "nobody", {"quantifier": "none", "animate": True})),
    Entry("nothing", "Pron", {"person": 3, "number": "singular"}, Entity("quantified", "nothing", {"quantifier": "none"})),
)

AUXILIARIES = (
    *words("is", "'s", cat="Aux", sem="be", tense="present", number="singular", person=3, copula=True),
    *words("are", "'re", cat="Aux", sem="be", tense="present", number="plural", copula=True),
    *words("am", "'m", cat="Aux", sem="be", tense="present", person=1, copula=True),
    *words("was", cat="Aux", sem="be", tense="past", number="singular", copula=True),
    *words("were", cat="Aux", sem="be", tense="past", number="plural", copula=True),
    *words("be", cat="Aux", sem="be", copula=True),
    *words("been", cat="Aux", sem="be", aspect="perfect", copula=True),
    *words("do", cat="Aux", sem="do", tense="present"),
    *words("does", cat="Aux", sem="do", tense="present", number="singular"),
    *words("did", cat="Aux", sem="do", tense="past"),
    *words("has", cat="Aux", sem="have", tense="present", number="singular", aspect="perfect"),
    *words("have", cat="Aux", sem="have", tense="present", aspect="perfect"),
    *words("had", cat="Aux", sem="have", tense="past", aspect="perfect"),
    *words("will", "'ll", cat="Aux", sem="will", tense="future"),
    *words("being", cat="Aux", sem="be", aspect="progressive", copula=True),
)

# the copular verb also appears as a main verb, which is what carries "is installed"
COPULAS = (
    # "is" is precisely the third person singular present; leaving the person off it
    # meant a demand for agreement could not find it, and the copula was chosen
    # alphabetically ("Nise am hungry")
    *words("is", "'s", cat="V", sem="be", tense="present", number="singular", person=3, copula=True),
    *words("are", "'re", cat="V", sem="be", tense="present", number="plural", copula=True),
    *words("am", "'m", cat="V", sem="be", tense="present", person=1, copula=True),
    *words("was", cat="V", sem="be", tense="past", number="singular", copula=True),
    *words("were", cat="V", sem="be", tense="past", number="plural", copula=True),
    *words("be", cat="V", sem="be", copula=True),
)

#: A modal's second form *is* its past ("could" for "can"), so the tense sits on the
#: modal and the complement stays a bare infinitive — saying a past modal frame then
#: needs no past verb, which is what "ought gave" was reaching for. Only the forms
#: with a same-meaning present partner are marked: "should" has none in this lexicon,
#: so marking it would leave a plain ``modality=should`` unsayable. "will" is an
#: auxiliary above, where it carries the future. "ought" is not a word on its own: it
#: appears below as "ought to".
MODALS = (
    *words("can", cat="Modal", sem="can"),
    *words("could", cat="Modal", sem="can", tense="past"),
    *words("may", cat="Modal", sem="may"),
    *words("might", cat="Modal", sem="may", tense="past"),
    *words("must", cat="Modal", sem="must"),
    *words("shall", cat="Modal", sem="shall"),
    *words("should", cat="Modal", sem="should"),
    *words("would", cat="Modal", sem="would"),
    # "should" and "would" are past forms with no present partner left in the language,
    # so each is listed twice: tenseless, and as its own past. The past reading is
    # dispreferred for reading (the plain modality is the commoner one) and is what
    # generation finds when a frame carries both a modality and a tense — otherwise a
    # past modal frame has no truthful way to be said at all.
    Entry("should", "Modal", {"tense": "past"}, "should", -0.1),
    Entry("would", "Modal", {"tense": "past"}, "would", -0.1),
)

NEGATIONS = (
    *words("not", "n't", cat="Neg", sem="negative"),
    # "never" reads as negation but does not *mean* only that — it quantifies over
    # time — so it is the second spelling when saying a plain negative polarity
    *words("never", cat="Neg", sem="negative", weight=-0.2),
)

CONJUNCTIONS = (
    *words("and", cat="Conj", sem="and"),
    *words("or", cat="Conj", sem="or"),
)

COMPLEMENTISERS = (
    *words("that", "whether", "if", cat="Comp", sem=None),
)

#: A preposition's meaning *is* the role it marks.
PREPOSITIONS = (
    *words("in", "inside", "on", "at", "within", "under", cat="P", sem="location"),
    *words("into", "to", "onto", cat="P", sem="destination"),
    *words("from", "out", cat="P", sem="source"),
    *words("of", cat="P", sem="of"),
    *words("for", cat="P", sem="beneficiary"),
    *words("by", cat="P", sem="agent"),
    *words("than", cat="P", sem="standard"),
    *words("as", cat="P", sem="as"),
    *words("about", cat="P", sem="topic"),
    *words("with", cat="P", sem="content"),
    *words("with", cat="P", sem="instrument", weight=-0.4),
    *words("without", cat="P", sem="lacking"),
    *words("after", cat="P", sem="after"),
    *words("before", cat="P", sem="before"),
    *words("called", "named", "titled", cat="P", sem="name"),
    *words("containing", "saying", cat="P", sem="content"),
)

#: A wh-word's meaning is the role it asks about.
WH_WORDS = (
    *words("what", "which", cat="Wh", sem="theme"),
    *words("who", "whom", cat="Wh", sem="subject"),
    *words("where", cat="Wh", sem="location"),
    *words("when", cat="Wh", sem="time"),
    *words("why", cat="Wh", sem="reason"),
    *words("how", cat="Wh", sem="manner"),
)

def _verb(lemma: str, past: str | None = None, *, sem: str | None = None, **features: Any) -> list[Entry]:
    """A verb's forms, each marked for what it *is*.

    Listing "came" as a bare alternate of "come" is how generation produced "snow
    cames": nothing said the form was already past, so the agreement rule applied to
    it. A past form carries ``tense=past`` here, which both blocks further inflection
    and lets the reader see the tense it was told.
    """
    sem = sem or lemma
    out = [Entry(lemma, "V", dict(features), sem)]
    if past is not None and past != lemma:
        out.append(Entry(past, "V", {**features, "tense": "past"}, sem))
    return out


CORE_VERBS = (
    *_verb("say", "said", reports=True),
    *_verb("tell", "told", reports=True),
    *_verb("think", "thought", reports=True),
    *_verb("believe", reports=True),
    *_verb("claim", reports=True),
    *_verb("promise", reports=True),
    *_verb("ask", "asked", reports=True),
    *_verb("want"),
    *_verb("need"),
    *_verb("have", "had"),
    Entry("has", "V", {"tense": "present", "number": "singular", "person": 3}, "have"),
    *_verb("do"),  # "did" is the auxiliary above; as a main verb it read questions as orders
    *_verb("give", "gave"),
    *_verb("take", "took"),
    *_verb("go", "went"),
    *_verb("come", "came"),
    *_verb("know", "knew"),
    *_verb("see", "saw"),
    *_verb("help"),
)

NUMBER_WORDS = tuple(
    Entry(word, "Num", {"number": "plural" if value != 1 else "singular"}, value)
    for word, value in (("one", 1), ("two", 2), ("three", 3), ("four", 4), ("five", 5),
                        ("six", 6), ("seven", 7), ("eight", 8), ("nine", 9), ("ten", 10))
)

#: Discourse connectives are cheap to skip rather than parsed: "and then" between
#: two imperatives carries sequencing that the cover already represents.
CONNECTIVE_BACKGROUND = {
    "then": math.log(0.85), "also": math.log(0.8), "next": math.log(0.7), "and": math.log(0.5),
    "please": math.log(0.9), "now": math.log(0.6), "just": math.log(0.6), "ok": math.log(0.7),
    "okay": math.log(0.7), "so": math.log(0.6), "afterwards": math.log(0.7), "finally": math.log(0.6),
    ".": math.log(0.95), ",": math.log(0.95), "?": math.log(0.95), "!": math.log(0.95), ";": math.log(0.95),
    ":": math.log(0.9), "'": math.log(0.9), "the": math.log(0.3), "of": math.log(0.3), "for": math.log(0.3),
}

def _spelling(entry: Entry) -> Entry:
    """Make a contraction the second choice of two spellings of one word.

    "'ll" and "n't" read as readily as "will" and "not" — that is the point of listing
    them — but saying a meaning should reach for the written form, and generation
    breaks ties by score. Without this the grammar said "the field 'll fail".
    """
    if entry.word.startswith("'") or entry.word == "n't":
        return replace(entry, weight=entry.weight - 0.1)
    return entry


ENGLISH_LEXICON = Lexicon(
    entries={},
    background=CONNECTIVE_BACKGROUND,
    unseen_background=math.log(2e-3),
).extend(*(_spelling(e) for e in (
    *DETERMINERS, *QUANTIFIERS, *PRONOUNS, *AUXILIARIES, *COPULAS, *MODALS, *NEGATIONS, *CONJUNCTIONS,
    *COMPLEMENTISERS, *PREPOSITIONS, *WH_WORDS, *CORE_VERBS, *NUMBER_WORDS,
)))


# ---------------------------------------------------------------- productions

#: Multiword function words: "no one" is one pronoun, not a number called "one".
MULTIWORD_FUNCTION = [
    production('Pron -> "no" "one"', Lit(Entity("quantified", "no one", {"quantifier": "none", "animate": True})), weight=0.5),
    production('Pron -> "every" "one"', Lit(Entity("quantified", "everyone", {"quantifier": "all", "animate": True})), weight=0.5),
    production('Pron -> "any" "one"', Lit(Entity("quantified", "anyone", {"quantifier": "some", "animate": True})), weight=0.5),
    # "ought" is only a modal with its "to"; without it there is no such word, which is
    # why generation must not be able to reach for one
    production('Modal -> "ought" "to"', Lit("should"), weight=0.5),
]

NOUN_PHRASES = [
    # a common noun denotes a description; adjectives and PPs refine it
    production("NBAR[number=?n] -> N[number=?n]", Ent("description", words_from=(0,), features_from=(("noun", 0),), lift=(("number", 0, "number"),))),
    production("NBAR[number=?n] -> Adj NBAR[number=?n]", Qualify(1, features_from=(("quality", 0),), extend_text_from=(0,))),
    # a nominal compound names the thing: "recipes folder", "meeting notes.txt"
    production("NBAR[number=?n] -> Name NBAR[number=?n]", Qualify(1, features_from=(("name", 0),), extend_text_from=(0,)), weight=-0.1),
    # a noun can modify a noun: "north field", "grain store"
    production("NBAR[number=?n] -> N NBAR[number=?n]", Qualify(1, features_from=(("quality", 0),), extend_text_from=(0,)), weight=-0.2),
    production("NBAR[number=?n] -> Literal NBAR[number=?n]", Qualify(1, features_from=(("name", 0),), extend_text_from=(0,)), weight=-0.1),
    # head first, name after: "folder photos", "file ideas.md"
    production("NBAR[number=?n] -> NBAR[number=?n] Name", Qualify(0, features_from=(("name", 1),), extend_text_from=(1,)), weight=-0.2),
    production("NBAR[number=?n] -> NBAR[number=?n] Path", Qualify(0, features_from=(("name", 1),), extend_text_from=(1,)), weight=-0.2),
    production("NBAR[number=?n] -> NBAR[number=?n] PP", Attach(0, 1)),
    # "my downloads folder": a place-noun modifier names the place
    production("NBAR[number=?n] -> N[place=true] NBAR[number=?n]", Head(0), weight=-0.15),
    production("NP[number=?n] -> NBAR[number=?n]", Head(0)),
    production("NP[number=?n] -> Det[number=?n] NBAR[number=?n]", Qualify(1, lift=(("definite", 0, "definite"), ("demonstrative", 0, "demonstrative"), ("possessive", 0, "possessive")))),
    production("NP[number=?n] -> Det NBAR[number=?n]", Qualify(1, lift=(("definite", 0, "definite"), ("demonstrative", 0, "demonstrative"), ("possessive", 0, "possessive"))), weight=-0.1),
    # Quantifiers, in the five shapes English allows. The quantifier word stays in the
    # entity's text as an adjective's does, so a caller that reads the text to find the
    # amount ("much food") keeps finding it there.
    production("NP[number=?n] -> Quant[number=?n,partitive=!] NBAR[number=?n]",
               Qualify(1, features_from=(("quantifier", 0),), extend_text_from=(0,))),
    production("NP[number=?n] -> Quant[partitive=!] NBAR[number=?n]",
               Qualify(1, features_from=(("quantifier", 0),), extend_text_from=(0,)), weight=-0.1),
    production("NP[number=?n] -> Quant[amount=!] Det NBAR[number=?n]",
               Qualify(2, features_from=(("quantifier", 0),), extend_text_from=(0,), lift=(("definite", 1, "definite"),)), weight=-0.1),
    # the partitive: "heaps of bread" said bare says nothing at all — `realize` returned
    # None for it, and a dialect whose word for *much* is "heaps" fell back to "heaps
    # bread". An amount reaches a determiner the same way: "much of the food".
    production('NP[number=?n] -> Quant[partitive=true] "of" NBAR[number=?n]',
               Qualify(2, features_from=(("quantifier", 0),), extend_text_from=(0,))),
    production('NP[number=?n] -> Quant[amount=true] "of" Det NBAR[number=?n]',
               Qualify(3, features_from=(("quantifier", 0),), extend_text_from=(0,), lift=(("definite", 2, "definite"),))),
    # Tolerant in, strict out. "plenty food" drops an "of" and "much of food" keeps one
    # it does not need; both are understood, and marked ``nonstandard`` so a hearer can
    # see the reading was repaired. Generation cannot use these *because* of the mark —
    # a production may not state a feature the meaning lacks — so it says "plenty of
    # food" and "much food". The second shape is what a dialect canonicaliser produces
    # when it swaps "heaps" for "much" and leaves the "of" behind.
    production("NP[number=?n] -> Quant[partitive=true] NBAR[number=?n]",
               Qualify(1, features_from=(("quantifier", 0),), features=(("nonstandard", True),),
                       extend_text_from=(0,)), weight=-0.4),
    production('NP[number=?n] -> Quant[partitive=!,amount=true] "of" NBAR[number=?n]',
               Qualify(2, features_from=(("quantifier", 0),), features=(("nonstandard", True),),
                       extend_text_from=(0,)), weight=-0.4),
    production("NP[number=?n] -> Pron[number=?n]", Head(0)),
    production("NP -> Name", Head(0)),
    production("NP -> Path", Head(0)),
    production("NP -> Literal", Head(0)),
    production("NP -> Command", Head(0)),
    production("NP[number=plural] -> Num NBAR", Qualify(1, features_from=(("count", 0),))),
    production("NP -> Num", Ent("number", words_from=(0,), features_from=(("value", 0),))),
    production("NP -> NP PP", Attach(0, 1), weight=-0.2),
    production("NP[number=plural] -> NP Conj NP", Coord((0, 2))),
    production("NP[number=plural] -> NP \",\" NP", Coord((0, 2)), weight=-0.35),
    production("PP -> P NP", Build(predicate="_pp", roles=(("role", 0), ("value", 1)))),
    production("PP -> P Literal", Build(predicate="_pp", roles=(("role", 0), ("value", 1)))),
]

VERB_PHRASES = [
    production("VP[number=?n] -> V[number=?n]", Build(predicate_from=0, lift=(("tense", 0, "tense"), ("aspect", 0, "aspect")))),
    production("VP[number=?n] -> V[number=?n] NP", Build(predicate_from=0, roles=(("object", 1),), lift=(("tense", 0, "tense"), ("aspect", 0, "aspect")))),
    # a double object is rarer than a verb with one object and a modifier, and letting
    # it compete evenly turns "make a folder on my desktop" into "make the desktop a folder"
    # a verb marked ditransitive takes "mv a.txt b.txt" as source and destination
    production("VP[number=?n] -> V[number=?n,ditrans=true] NP NP", Build(predicate_from=0, roles=(("object", 1), ("destination", 2)), lift=(("tense", 0, "tense"),))),
    production("VP[number=?n] -> V[number=?n] NP NP", Build(predicate_from=0, roles=(("recipient", 1), ("object", 2)), lift=(("tense", 0, "tense"), ("aspect", 0, "aspect"))), weight=-0.6),
    production("VP -> VP PP", Attach(0, 1)),
    # a bare locative pronoun modifies the verb: "put it there", "init a repo there"
    production("VP -> VP Pron[locative=true]", Qualify(0, roles_from=(("location", 1),))),
    # a dative "show me X" adds a recipient, and must not compete with a double object
    production("VP[number=?n] -> V[number=?n] Pron[person=1] NP", Build(predicate_from=0, roles=(("recipient", 1), ("object", 2)), lift=(("tense", 0, "tense"),)), weight=-0.1),
    # reported speech: the complement clause stays a frame of its own
    production("VP[number=?n] -> V[number=?n,reports=true] S", Build(predicate_from=0, roles=(("content", 1),), lift=(("tense", 0, "tense"),)), weight=0.8),
    production("VP[number=?n] -> V[number=?n,reports=true] Comp S", Build(predicate_from=0, roles=(("content", 2),), lift=(("tense", 0, "tense"),)), weight=0.8),
    # a reporting verb with a *named* hearer is rarer than one that just reports, and
    # letting them compete evenly splits "said the north field failed" into a dative
    production("VP[number=?n] -> V[number=?n,reports=true] NP S", Build(predicate_from=0, roles=(("recipient", 1), ("content", 2)), lift=(("tense", 0, "tense"),)), weight=0.3),
    production("VP[number=?n] -> V[number=?n,reports=true] NP Comp S", Build(predicate_from=0, roles=(("recipient", 1), ("content", 3)), lift=(("tense", 0, "tense"),)), weight=0.8),
    # auxiliaries and negation
    # an auxiliary is there to support a verb, so prefer that over taking a noun phrase
    production("VP[number=?n] -> Aux[number=?n] VP", Qualify(1, lift=(("tense", 0, "tense"), ("aspect", 0, "aspect"))), weight=0.3),
    # the perfect takes a participle: for a regular verb that is the -ed form, which is
    # what lets "has arrived" be both read and said
    production("VP[number=?n] -> Aux[number=?n,aspect=perfect] VP[tense=past]", Qualify(1, features=(("aspect", "perfect"),), lift=(("tense", 0, "tense"),)), weight=0.4),
    production("VP[number=?n] -> Aux[number=?n] Neg VP", Qualify(2, features=(("polarity", "negative"),), lift=(("tense", 0, "tense"), ("aspect", 0, "aspect")))),
    # a modal's complement is a bare infinitive: ``VP[tense=!]`` forbids the tense, so
    # neither the parser nor the generator can put "ought" next to "gave". A tensed
    # modal frame is said by the modal's own past form ("should give"), which is the
    # second production: it lifts the tense onto the modal the way an auxiliary does.
    production("VP -> Modal VP[tense=!]", Qualify(1, features_from=(("modality", 0),))),
    production("VP -> Modal[tense=?t] VP[tense=!]",
               Qualify(1, features_from=(("modality", 0),), lift=(("tense", 0, "tense"),)), weight=0.1),
    production("VP -> Modal Neg VP[tense=!]", Qualify(2, features=(("polarity", "negative"),), features_from=(("modality", 0),))),
    production("VP[number=?n] -> V[number=?n] Neg VP", Qualify(2, features=(("polarity", "negative"),), lift=(("tense", 0, "tense"),)), weight=-0.3),
    # copular predication: "is installed", "is better than the south field"
    production("VP[number=?n] -> Aux[number=?n,copula=true] AP", Merge(1, lift=(("tense", 0, "tense"),))),
    production("VP[number=?n] -> Aux[number=?n,copula=true] Neg AP", Merge(2, features=(("polarity", "negative"),), lift=(("tense", 0, "tense"),))),
    production("VP[number=?n] -> Aux[number=?n,copula=true] PP", Build(predicate="located", lift=(("tense", 0, "tense"),)), weight=-0.1),
    # "is not trustworthy": the copula negates in place. Without this the only way to
    # say a negated ``be(x, y)`` was to negate a *clause* — "am never be trustworthy",
    # two copulas and a temporal quantifier doing the work of one "not".
    production("VP[number=?n] -> V[number=?n,copula=true] Neg NP",
               Build(predicate_from=0, roles=(("object", 2),), features=(("polarity", "negative"),),
                     lift=(("tense", 0, "tense"),)), weight=0.2),
    production("VP -> VP Conj VP", Merge(0), weight=-0.6),
    # an adjective phrase predicates something, and carries the degree of its adjective
    production("AP -> Adj", Build(predicate_from=0, lift=(("degree", 0, "degree"),))),
    production("AP -> Adj PP", Attach(Build(predicate_from=0, lift=(("degree", 0, "degree"),)), 1)),
    production("AP -> AP PP", Attach(0, 1)),
]

CLAUSES = [
    production("S -> NP[number=?n] VP[number=?n]", Merge(1, roles=(("subject", 0),), features=(("mood", "declarative"),))),
    production("S -> NP VP", Merge(1, roles=(("subject", 0),), features=(("mood", "declarative"),)), weight=-0.2),
    production("S -> Comp S", Head(1), weight=-0.3),
    # an imperative is a bare VP, so the imperative reading is *ranked* against the
    # declarative one rather than chosen by a keyword
    production("IMP -> VP", Order(0), weight=-0.05),
]

#: A question is a clause with something fronted. ``QC`` is that clause, without a
#: mood, so the fronted auxiliary can contribute tense to it before it is asked.
QUESTIONS = [
    production("QC[number=?n] -> NP[number=?n] VP[number=?n]", Merge(1, roles=(("subject", 0),))),
    production("QC[number=?n] -> NP[number=?n] AP", Merge(1, roles=(("subject", 0),))),
    production("QC -> NP PP", Locative("located", modifier=1, theme=0)),
    # a bare noun phrase is a question's *subject* only after a wh-word ("who am i").
    # Allowing it as a whole yes/no clause read "the field did not fail" as a polarity
    # question about a thing called "fail", which asserts nothing and reifies nothing.
    production("QSUBJ -> NP", Build(predicate="be", roles=(("subject", 0),))),
    # yes/no questions
    production("Q -> Aux QC", Ask(Qualify(1, lift=(("tense", 0, "tense"), ("aspect", 0, "aspect"))), asked="polarity")),
    production("Q -> Aux Neg QC", Ask(Qualify(2, features=(("polarity", "negative"),), lift=(("tense", 0, "tense"),)), asked="polarity")),
    production("Q -> Modal QC", Ask(Qualify(1, features_from=(("modality", 0),)), asked="polarity")),
    # wh-questions: the wh-word names the role being asked about
    production("Q -> Wh VP", Ask(1, asked_from=0), weight=-0.05),
    production("Q -> Wh Aux QC", Ask(Qualify(2, lift=(("tense", 1, "tense"),)), asked_from=0), weight=-0.05),
    production("Q -> Wh Aux QSUBJ", Ask(Qualify(2, lift=(("tense", 1, "tense"),)), asked_from=0), weight=-0.1),
    production("Q -> Wh Aux PP", Ask(Locative("located", modifier=2, lift=(("tense", 1, "tense"),)), asked_from=0)),
    production("Q -> Wh Aux NP PP", Ask(Locative("located", modifier=3, theme=2, lift=(("tense", 1, "tense"),)), asked_from=0), weight=-0.1),
    production("Q -> Wh NP VP", Ask(Merge(2, roles=(("subject", 1),)), asked_from=0), weight=-0.15),
    production("Q -> Wh Num NBAR PP", Ask(Locative("located", modifier=3, theme=Ent("description", words_from=(2,), features_from=(("noun", 2), ("count", 1)))), asked="count"), weight=-0.2),
]

ENGLISH = Grammar(
    productions=tuple(MULTIWORD_FUNCTION + NOUN_PHRASES + VERB_PHRASES + CLAUSES + QUESTIONS),
    lexicon=ENGLISH_LEXICON,
    start=("S", "Q", "IMP"),
    open_class=(
        OpenClass(r"~(?:/.*)?|/[\w.@+\-/]+|[\w@+\-.]+/[\w./@+\-]*", "Path", "path"),
        OpenClass(r"[\w@+\-]+\.[A-Za-z0-9]{1,8}", "Path", "path"),
        OpenClass(r"`[^`]*`", "Command", "command"),
        OpenClass(r"'[^']*'|\"[^\"]*\"|“[^”]*”|‘[^’]*’", "Literal", "literal"),
        OpenClass(r"\d+(?:\.\d+)?", "Num", "number"),
        # Any unknown word may be a name — a folder or a person can be called
        # anything — and, productively, a noun, a verb or an adjective. Without the
        # last three an unseen content word has nowhere to go and takes its whole
        # clause with it: no fixed lexicon holds the words villagers use. The weights
        # keep every real entry ahead of a guess, and each guess is marked.
        # A capitalised unknown word is probably a proper name; a lowercase one is
        # more likely a common noun or a verb, and treating it as a name first is what
        # made "the north field failed" parse as one long noun phrase.
        OpenClass(r"[A-Z][\w'\-]*", "Name", "name", weight=-0.2),
        OpenClass(r"[a-z][\w'\-]*", "Name", "name", weight=-0.9),
        OpenClass(r"[A-Za-z][\w'\-]*", "N", weight=-1.0, sem="word", morphology=True, bare_weight=-1.1),
        # a verb is cheap when its suffix says so ("failed") and dear when nothing
        # marks it ("field"), which is how the clause finds its verb
        OpenClass(r"[A-Za-z][\w'\-]*", "V", weight=-0.9, sem="word", morphology=True, bare_weight=-2.2),
        OpenClass(r"[A-Za-z][\w'\-]*", "Adj", weight=-1.0, sem="word", morphology=True, bare_weight=-1.4),
    ),
    name="english-core",
)
"""The core grammar. Domains extend it; nothing here names a domain."""
