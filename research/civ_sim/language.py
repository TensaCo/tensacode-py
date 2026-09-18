"""The only seam between the simulation and whatever parses and produces sentences.

Nothing outside this module imports a grammar. The simulation asks for two things:

    say(subject, predicate, object, lexicon, ...)   -> an English sentence in that dialect
    hear(sentence, lexicon, names, context)         -> Heard(claims, score, unknown, readings)

and four about a speech community:

    base_lexicon()            the shared starting vocabulary
    dialect(base, i, rng)     a settlement's own version of it
    drift(lexicon, rng)       a generation of sound change, borrowing and forgetting
    intelligibility(a, b)     how much of one dialect's speech the other still recovers

## Which grammar does what, and why

Understanding goes through **`tensorcode.language`** (the general package: unification features, a
chart parser, semantic frames, open-vocabulary words). It is better than the local grammar at the
job that matters here — it keeps `much food` as a quantified description instead of throwing the
quantifier away, and it takes unknown words as names rather than failing, so people can talk about
each other and about grain without the clause collapsing.

Production stays with **`research/civ_sim/grammar.py`**, the local unification grammar, because at
the time of writing the general package's generator has stemming and tense bugs that would put bad
English in every mouth in the world: `realize` gives *"snow cames"*, *"Miol ought gave food"*, and
its analysis of `owes` stems to `ow`. Measured, not assumed — see `measure_language.py`.

So the seam is a **hybrid, and it is labelled as one**: generate locally, understand generally, count
which path produced each claim (`Heard.via`) so the results table can say what came from where. When
the general package's generator round-trips cleanly, `say` moves over too and this docstring is the
only thing that has to change.

Dialects, drift and intelligibility stay on this side either way: they are language *change*, which
neither grammar models. A dialect is a concept→surface map, and it is applied to the *surface string*
(substitute on the way out, un-substitute on the way in). A word the listener's dialect does not know
survives as an open-vocabulary name — which is exactly the lossy channel we want to measure, rather
than a crash.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

from . import grammar as _g

Lexicon = _g.Lexicon

# their frame predicate -> ours. Their analyser stems some verbs too far ("owes" -> "ow",
# "died" -> "di"), so both the stem and the full form are listed and that is a bug in their
# morphology, recorded here rather than worked around silently.
THEIRS_TO_OURS = {
    "hold": "has_amount", "have": "has_amount", "keep": "keeps_back", "hoard": "keeps_back",
    "die": "died", "di": "died", "owe": "owes", "ow": "owes", "give": "gives", "giv": "gives",
    "come": "coming", "cam": "coming", "show": "showed", "raid": "raided", "help": "help",
    "say": "said", "hear": "heard", "heard": "heard", "trade": "trades",
}
# a copula sentence carries the predicate in its object: "Nise is hungry" -> hungry
COPULA_ADJECTIVES = {
    "hungry": "hungry", "dear": "expensive", "expensive": "expensive", "cheap": "cheap",
    "trustworthy": "trustworthy", "ill": "ill", "cold": "cold", "dead": "died", "full": "has_amount",
}
QUANTIFIERS = ("much", "many", "heaps", "plenty", "little", "few", "some", "no", "none", "full")
QUANT_TO_AMOUNT = {"much": "much", "many": "much", "heaps": "much", "plenty": "much", "full": "much",
                   "little": "little", "few": "little", "some": "some", "no": "none", "none": "none"}
GOODS_WORDS = {"food", "grain", "bread", "forage", "wood", "timber", "stone", "tools", "tool"}


@dataclass(frozen=True)
class Heard:
    claims: tuple  # dicts: subject, predicate, object, hearsay, via
    score: float  # how well the parse fit (0 if nothing parsed)
    unknown: tuple  # words this dialect does not have
    readings: int  # how many parses the sentence had, i.e. how ambiguous it was
    via: str = "none"  # "general" | "local" | "none": which grammar recovered the meaning


# ---------------------------------------------------------------- speech community


def base_lexicon() -> Lexicon:
    return _g.Lexicon.base()


def dialect(base: Lexicon, index: int, rng) -> Lexicon:
    return _g.dialect_from(base, index, rng)


def drift(lexicon: Lexicon, rng, *, borrow_from: Lexicon | None = None, rate: float = 0.08) -> int:
    return _g.drift(lexicon, rng, borrow_from=borrow_from, rate=rate)


def intelligibility(speaker: Lexicon, listener: Lexicon) -> float:
    return _g.intelligibility(speaker, listener)


# ---------------------------------------------------------------- saying

# our predicate -> (their frame predicate, how the object is carried). The inverse of
# THEIRS_TO_OURS, and the place to look when a new kind of claim needs to be sayable.
OURS_TO_THEIRS = {
    "has_amount": ("hold", "quantified"), "keeps_back": ("keep", "noun"), "died": ("die", None),
    "owes": ("owe", "noun"), "gives": ("give", "noun"), "coming": ("come", None),
    "showed": ("show", "noun"), "raided": ("raid", "place"), "help": ("help", "object-pronoun"),
    "trades": ("trade", "noun"),
}
# the ones that are "X is <adjective>" rather than a verb
OURS_TO_ADJECTIVE = {"hungry": "hungry", "expensive": "dear", "cheap": "cheap",
                     "trustworthy": "trustworthy", "ill": "ill", "cold": "cold"}
PAST_TENSE = {"died", "showed", "raided"}  # claims that are about something that already happened
# quantifiers the general grammar can put straight in front of a mass noun. The others are
# partitives that need "of" ("heaps of bread"), which it cannot yet say at all — `realize` returns
# None for them — so a dialect using one of those falls back to the local generator, which can.
BARE_QUANTIFIERS = {"much", "little", "no", "some", "many", "few", "full"}


@lru_cache(maxsize=16384)
def _realize_cached(verb: str, subj: str, obj_spec: tuple, feats: tuple) -> str | None:
    """Saying a sentence is a pure function of the frame, so it is memoized like parsing.

    Generation turned out to cost about as much as parsing did: swapping `say` onto
    `tensorcode.language.realize` took the civ test suite from 35 s to 359 s before this cache went
    in. A village says the same shapes over and over — the same handful of predicates about a few
    hundred named people — so the memo turns that into a one-off per distinct sentence, and the
    string returned is identical to an uncached one. `CIV_PARSE_MEMO=0` turns both memos off.
    """
    from tensorcode.language import ENGLISH, Entity, Frame, realize

    roles = {"subject": Entity(kind="name", text=subj, features={})}
    kind, text, extra = obj_spec if obj_spec else (None, None, ())
    if kind == "name":
        roles["object"] = Entity(kind="name", text=text, features={})
    elif kind == "pronoun":
        roles["object"] = Entity(kind="pronoun", text=text, features={"person": 1})
    elif kind == "description":
        noun, amount = extra
        roles["object"] = Entity(kind="description", text=text,
                                 features={"noun": noun, "number": "singular",
                                           "name": Entity(kind="name", text=amount, features={})})
    try:
        said = realize(ENGLISH, Frame(verb, roles, dict(feats)))
    except Exception:  # noqa: BLE001 - a generator fault must not strike anyone mute
        return None
    return said or None


def realize_cache_info():
    return _realize_cached.cache_info()


def _their_frame(subject: str, predicate: str, obj, lexicon: Lexicon, *, mood: str, modality, negated: bool):
    """Describe a frame for `realize` as hashable pieces, with this dialect's words in it."""
    name = lambda t: ("name", str(t), ())  # noqa: E731
    subj = _g._short(subject)
    tense = "past" if predicate in PAST_TENSE else "present"
    feats = {"mood": "declarative", "tense": tense}
    if modality:
        feats["modality"] = str(modality)
    # our claims carry negation in the object ("trustworthy" = "False"), not in a flag
    if negated or str(obj) in ("False", "false") or obj is False:
        feats["polarity"] = "negative"

    if predicate in OURS_TO_ADJECTIVE:
        return "be", subj, name(lexicon.say(OURS_TO_ADJECTIVE[predicate])), tuple(sorted(feats.items()))
    if predicate not in OURS_TO_THEIRS:
        return None
    verb, shape = OURS_TO_THEIRS[predicate]
    spec: tuple = ()
    if shape == "quantified":
        good, _, amount = str(obj).partition(":")
        good_word = lexicon.say(good or "food")
        amount_word = lexicon.say(amount) if amount and amount not in ("unsaid", "some") else "some"
        if amount_word.lower() not in BARE_QUANTIFIERS:
            return None  # a partitive: the local generator says it properly, this one cannot
        spec = ("description", f"{amount_word} {good_word}", (good_word, amount_word))
    elif shape == "noun":
        spec = name(lexicon.say(_g._short(str(obj))))
    elif shape == "place":
        spec = name(_g._short(str(obj)))
    elif shape == "object-pronoun":
        spec = ("pronoun", "me", ())
    return verb, subj, spec, tuple(sorted(feats.items()))


def say(subject: str, predicate: str, obj, lexicon: Lexicon, *, mood: str = "declare", modality=None,
        negated: bool = False, secondhand: str | None = None) -> str:
    """Put one claim into a sentence in this dialect. `secondhand` makes it reported speech.

    Speech now goes through `tensorcode.language.realize`: all 17 sentence shapes this world uses
    round-trip through it with the right predicate, including the copula ("Nise is hungry"), its
    negation ("Kasa is not trustworthy") and past tense ("Anem died") — the last of which the local
    generator got wrong. The local grammar is kept as the fallback for anything their grammar cannot
    say, so a new kind of claim degrades to the old wording instead of falling silent.
    """
    frame = _g.claim_to_frame(subject, predicate, obj, mood=mood, modality=modality, negated=negated)
    inner = None
    if mood == "declare":
        theirs = _their_frame(subject, predicate, obj, lexicon, mood=mood, modality=modality, negated=negated)
        if theirs is not None:
            said = _realize_cached(*theirs) if MEMO else _realize_cached.__wrapped__(*theirs)
            if said:
                inner = said[0].upper() + said[1:] + "."
    if inner is None:
        inner = _g.generate(frame, lexicon)
    if secondhand:
        # "X said ..." rather than "I heard from X that ...": the general parser keeps the content
        # clause of the first and drops it from the second, measured, so this is the form we use.
        return f"{secondhand} said {inner}"  # the inner clause keeps its own capitalization
    return inner


# ---------------------------------------------------------------- hearing


def _canonicalize(sentence: str, lexicon: Lexicon, known: set | None = None) -> tuple[str, tuple]:
    """Map this dialect's surface forms back to the shared words, and report what it could not.

    A form the listener has no entry for is left in place: the general parser will take it as a name,
    which is how a half-understood sentence stays half-understood instead of failing outright.
    """
    base_forms = _base_forms()
    mine = {form: concept for concept, form in lexicon.forms.items()}
    known_lower = {str(k).lower() for k in (known or ())}
    stems = {f[:4] for f in base_forms.values() if len(f) >= 4} | {f[:4] for f in lexicon.entries if len(f) >= 4}
    out, unknown = [], []
    for word in re.findall(r"[\w'’]+|[.,!?;]", sentence):
        low = word.lower().strip(".,!?;")
        if not low:
            continue
        concept = mine.get(low)
        if concept is not None and concept in base_forms:
            shared = base_forms[concept]
            out.append(shared if word.islower() else shared.capitalize())
            continue
        if low in lexicon.entries or low in base_forms.values():
            out.append(word)
            continue
        if low in known_lower or (word[:1].isupper() and len(out) > 0):
            out.append(word)  # a name, not a gap in the vocabulary
            continue
        if len(low) >= 4 and low[:4] in stems:
            out.append(word)  # an inflected form of a word this dialect does have
            continue
        unknown.append(low)
        out.append(word)  # keep it: an unfamiliar word is heard as a name, not dropped
    text = " ".join(w for w in out if w not in ".,!?;")
    return text, tuple(dict.fromkeys(unknown))


_BASE_FORMS: dict | None = None


def _base_forms() -> dict:
    """The shared vocabulary, built once. It used to be rebuilt per utterance."""
    global _BASE_FORMS
    if _BASE_FORMS is None:
        _BASE_FORMS = dict(_g.Lexicon.base().forms)
    return _BASE_FORMS


MEMO = os.environ.get("CIV_PARSE_MEMO", "1") != "0"  # set CIV_PARSE_MEMO=0 to measure without it


@lru_cache(maxsize=16384)
def _understand_cached(text: str):
    """Parsing is a pure function of the sentence, so it is memoized.

    This is not a shortcut: `tensorcode.language.build_chart` costs ~112 ms per utterance here
    (measured by cProfile: it builds chart keys with `repr()` of dataclasses, 3.9M repr calls in a
    six-day run), which is ~100x the local grammar. A village says the same few hundred sentence
    shapes over and over, so the cache turns that into a one-off cost per distinct sentence and the
    parse it returns is byte-identical to an uncached one. Remove this only when that parser gets
    faster, and re-measure if you do.
    """
    from tensorcode.language import ENGLISH, understand

    u = understand(ENGLISH, text)
    meanings = tuple(m for m in (u.meanings or ()) if hasattr(m, "roles"))
    skipped = tuple(str(w).lower().strip(".,!?") for w in (u.skipped or ()))
    return meanings, skipped, max(1, len(u.meanings or ()))


def parse_cache_info():
    return _understand_cached.cache_info()


def _entity_text(value) -> str:
    return str(getattr(value, "text", value) or "").strip()


def _amount_from(text: str) -> tuple[str | None, str | None]:
    """"much food" -> ("much", "food"); "food" -> (None, "food"). The quantifier is the point."""
    words = [w.lower() for w in text.split()]
    quant = next((w for w in words if w in QUANTIFIERS), None)
    good = next((w for w in words if w in GOODS_WORDS), None)
    return (QUANT_TO_AMOUNT.get(quant) if quant else None), good


WEATHER_WORDS = {"snow", "rain", "frost", "storm", "wind", "fog"}


def _type_subject(name: str, places: set) -> str:
    """A claim about timber is not a claim about a person called Timber."""
    low = name.lower()
    if name in places:
        return f"village:{name}"
    if low in GOODS_WORDS:
        return f"good:{low}"
    if low in WEATHER_WORDS:
        return f"weather:{low}"
    if low in ("sky", "moon", "sun"):
        return f"sky:{low}"
    return f"person:{name}"


def _frame_to_triples(frame, *, speaker: str, places: set, hearsay: bool = False, via_name: str | None = None) -> list:
    """Their Frame -> our (subject, predicate, object) claims. Unmappable frames return nothing,
    which is a misunderstanding and is counted as one."""
    roles = getattr(frame, "roles", None)
    if roles is None:
        return []
    pred_raw = str(getattr(frame, "predicate", "") or "").lower()
    feats = dict(getattr(frame, "features", {}) or {})
    negated = feats.get("polarity") == "negative"
    out = []

    content = roles.get("content")
    if content is not None and hasattr(content, "roles"):  # reported speech: "Nise said X"
        source = _entity_text(roles.get("subject")) or speaker
        return _frame_to_triples(content, speaker=speaker, places=places, hearsay=True, via_name=source)

    subject = _entity_text(roles.get("subject"))
    obj_text = _entity_text(roles.get("object"))
    if not subject:
        return []
    subj_id = _type_subject(subject, places)

    pred = THEIRS_TO_OURS.get(pred_raw)
    if pred_raw in ("be", "is", "was") or pred is None:
        words = [w.lower() for w in obj_text.split()]
        adj = next((w for w in words if w in COPULA_ADJECTIVES), None)
        if adj is None:
            return []
        pred = COPULA_ADJECTIVES[adj]
        negated = negated or "false" in words or "not" in words
        obj_value = "False" if negated else "True"
        if pred == "has_amount":
            amount, good = _amount_from(obj_text)
            obj_value = f"{good or 'food'}:{amount or 'much'}"
        out.append({"subject": subj_id, "predicate": pred, "object": obj_value})
    elif pred == "has_amount":
        amount, good = _amount_from(obj_text)
        # their parser keeps "much food" and "little food" but drops "some" and "no", so an
        # unquantified object is genuinely unquantified rather than assumed to be plenty
        out.append({"subject": subj_id, "predicate": pred,
                    "object": f"{good or 'food'}:{amount or 'unsaid'}"})
    elif pred in ("gives", "owes", "keeps_back", "help"):
        _, good = _amount_from(obj_text)
        out.append({"subject": subj_id, "predicate": pred, "object": f"good:{good or 'food'}"})
    elif pred in ("died", "coming", "hungry", "raided", "showed", "trades"):
        value = obj_text or "True"
        if pred == "showed":
            value = (obj_text.split() or ["conjunction"])[-1].lower()
        elif pred == "raided":
            value = f"settlement:{obj_text}" if obj_text else subj_id
        elif pred in ("died", "coming"):
            value = "True"
        out.append({"subject": subj_id, "predicate": pred, "object": value})
    else:
        return []

    for c in out:
        c["negated"] = negated
        c["hearsay"] = hearsay
        c["via"] = via_name
    return out


def hear(sentence: str, lexicon: Lexicon, *, names: list, context: dict, speaker: str,
         settlements: list | None = None) -> Heard:
    """Understand a sentence heard in this dialect. Failure is a real outcome, not an error.

    Tried in order: the general parser in `tensorcode.language` on the canonicalized string, then the
    local grammar as a fallback. `Heard.via` records which one recovered the meaning.
    """
    places = set(settlements or ())
    text, unknown = _canonicalize(sentence, lexicon, known=set(names or ()) | places)
    try:
        meanings, skipped_words, readings = _understand_cached(text) if MEMO else _understand_cached.__wrapped__(text)
        if meanings:
            claims = _frame_to_triples(meanings[0], speaker=speaker, places=places)
            if claims:
                skipped = tuple(dict.fromkeys(unknown + skipped_words))
                score = 3.0 - 0.5 * len(skipped)
                return Heard(tuple(claims), max(0.5, score), skipped, readings, "general")
    except Exception:  # noqa: BLE001 - a parser fault must not stop the world; fall back to the local grammar
        pass

    parses = _g.parse(sentence, lexicon, known_names=names)
    if not parses:
        return Heard((), 0.0, unknown, 0, "none")
    best = parses[0]
    claims = _g.frame_to_claims(best.frame, speaker=speaker, listener_context=context)
    for c in claims:
        subj = c.get("subject")
        if isinstance(subj, str) and subj.startswith("person:") and subj.split(":", 1)[1] in places:
            c["subject"] = "village:" + subj.split(":", 1)[1]
    return Heard(tuple(claims), float(best.score), tuple(best.unknown) or unknown, len(parses), "local")
