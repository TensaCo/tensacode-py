"""Speech: claims -> English sentences -> claims again, with loss.

The substrate is claim transmission between minds. This module is the surface: a speaker
realizes a claim as a sentence (templates, no model), and a listener parses that sentence back
into a claim with a forgiving grammar. Parsing is deliberately lossy, and each village speaks a
slightly different dialect, so meaning degrades as talk passes from mouth to mouth. That is
where misunderstanding, rumour drift and divergent local beliefs come from.

Claims here are plain tuples ``(subject, predicate, object)`` of strings, so they can be carried
in a tensacode Claim without a schema per sentence.
"""

from __future__ import annotations

import re

# each village says some things its own way; listeners from elsewhere can mishear these
DIALECT = (
    {"food": "food", "granary": "granary", "much": "plenty of", "little": "hardly any", "yes": "aye"},
    {"food": "grain", "granary": "store", "much": "a good deal of", "little": "scarcely any", "yes": "yes"},
    {"food": "bread", "granary": "loft", "much": "much", "little": "little", "yes": "so"},
    {"food": "forage", "granary": "hoard", "much": "heaps of", "little": "thin", "yes": "true"},
    {"food": "food", "granary": "granary", "much": "plenty of", "little": "little", "yes": "aye"},
    {"food": "food", "granary": "granary", "much": "plenty of", "little": "little", "yes": "aye"},
)
GREETINGS = ("Well met, {other}.", "{other}. A word?", "Peace, {other}.", "You look tired, {other}.")
FAREWELLS = ("Keep well.", "Go on, then.", "Until the evening.")


def realize(claim: tuple, *, village: int, secondhand: str | None = None, lie: bool = False) -> str:
    """One claim as a sentence in the speaker's dialect. ``secondhand`` names who told them."""
    subject, predicate, obj = claim
    d = DIALECT[village % len(DIALECT)]
    who = subject.split(":")[-1]
    what = str(obj)
    if predicate == "has_food":
        amount = d["much"] if what == "much" else d["little"] if what == "little" else "no"
        body = f"{who}'s {d['granary']} holds {amount} {d['food']}"
    elif predicate == "forage_at":
        body = f"there is {d['food']} to be had at {who}"
    elif predicate == "barren":
        body = f"{who} is stripped bare"
    elif predicate == "died":
        body = f"{who} is dead of {what}"
    elif predicate == "raided":
        body = f"{who} fell on {what}"
    elif predicate == "hoards":
        body = f"{who} keeps {d['food']} back from the {d['granary']}"
    elif predicate == "trustworthy":
        body = f"{who} is {'a good neighbour' if what == 'True' else 'not to be trusted'}"
    elif predicate == "weather_coming":
        body = f"{what} is coming"
    elif predicate == "omen":
        body = f"the sky showed {what}"
    elif predicate == "festival_at":
        body = f"we keep the feast when {who} is {what}"
    elif predicate == "how_to":
        body = f"the way to {what} is patience and a worked field"
    else:
        body = f"{who} {predicate.replace('_', ' ')} {what}"
    if lie:
        body = _invert(body, d)
    if secondhand:
        return f"I heard from {secondhand} that {body}."
    return body[0].upper() + body[1:] + "."


def _invert(body: str, d: dict) -> str:
    for a, b in ((d["much"], "no"), ("no ", d["much"] + " "), ("is dead", "is well"), ("keeps", "never keeps"), ("not to be trusted", "a good neighbour")):
        if a in body:
            return body.replace(a, b, 1)
    return body


_WORDS = {w: k for k, ws in {
    "much": ("plenty", "heaps", "good deal", "much"), "little": ("hardly", "scarcely", "little", "thin"), "none": ("no", "none", "empty"),
}.items() for w in ws}
FOOD_WORDS = ("food", "grain", "bread", "forage")
STORE_WORDS = ("granary", "store", "loft", "hoard")


def parse(sentence: str, *, village: int, gain: float = 0.6, noise: float = 0.0, rng=None) -> tuple:
    """Sentence -> (claim | None, confidence, note). Lossy on purpose: wrong dialect and noise hurt."""
    s = sentence.strip().rstrip(".").lower()
    note = ""
    secondhand = None
    if m := re.match(r"i heard from (\w+) that (.*)", s):
        secondhand, s = m[1], m[2]
        note = f"secondhand via {secondhand}"
    own = DIALECT[village % len(DIALECT)]
    foreign = not any(w in s for w in (own["food"], own["granary"])) and any(w in s for w in FOOD_WORDS + STORE_WORDS)
    conf = 0.85 * (0.5 + 0.5 * gain) * (0.6 if foreign else 1.0)
    if foreign:
        note = (note + "; " if note else "") + "unfamiliar words"
    claim = None
    if m := re.search(r"(\w[\w' -]*)'s (?:%s) holds (.*?) (?:%s)" % ("|".join(STORE_WORDS), "|".join(FOOD_WORDS)), s):
        amount = next((k for w, k in _WORDS.items() if w in m[2]), None)
        claim = (f"village:{m[1].strip().title()}", "has_food", amount or "little")
        if amount is None:
            note = (note + "; " if note else "") + "amount unclear, guessed little"
            conf *= 0.6
    elif m := re.search(r"there is (?:%s) to be had at ([\w' -]+)" % "|".join(FOOD_WORDS), s):
        claim = (f"place:{m[1].strip()}", "forage_at", "yes")
    elif m := re.search(r"([\w' -]+) is stripped bare", s):
        claim = (f"place:{m[1].strip()}", "barren", "yes")
    elif m := re.search(r"(\w[\w' -]*) is dead of (\w+)", s):
        claim = (f"person:{m[1].strip().title()}", "died", m[2])
    elif m := re.search(r"(\w[\w' -]*) fell on ([\w' -]+)", s):
        claim = (f"village:{m[1].strip().title()}", "raided", m[2].strip().title())
    elif m := re.search(r"(\w[\w' -]*) (?:keeps|never keeps) (?:%s) back" % "|".join(FOOD_WORDS), s):
        claim = (f"person:{m[1].strip().title()}", "hoards", "False" if "never keeps" in s else "True")
    elif m := re.search(r"(\w[\w' -]*) is (a good neighbour|not to be trusted)", s):
        claim = (f"person:{m[1].strip().title()}", "trustworthy", "True" if "good" in m[2] else "False")
    elif m := re.search(r"(\w+) is coming", s):
        claim = ("weather:local", "weather_coming", m[1])
    elif m := re.search(r"the sky showed ([\w' -]+)", s):
        claim = ("sky:overhead", "omen", m[1].strip())
    elif m := re.search(r"we keep the feast when (\w+) is (\w+)", s):
        claim = (f"moon:{m[1].title()}", "festival_at", m[2])
    elif m := re.search(r"the way to (\w+) is", s):
        claim = ("skill:farming", "how_to", m[1])
    else:
        return None, 0.0, "not understood"
    # noise corrupts what was heard: an amount slips, or a name is taken for someone else
    if rng is not None and claim is not None and noise > 0 and rng.random() < noise:
        subject, predicate, obj = claim
        if predicate == "has_food":
            obj = {"much": "little", "little": "none", "none": "little"}.get(obj, obj)
            note = (note + "; " if note else "") + "misheard the amount"
        elif predicate in ("died", "hoards", "trustworthy"):
            note = (note + "; " if note else "") + "unsure who was meant"
            conf *= 0.5
        claim = (subject, predicate, obj)
        conf *= 0.8
    return claim, round(min(1.0, conf), 2), note


def summarize(claim: tuple) -> str:
    """Short human-readable form for the inspector."""
    subject, predicate, obj = claim
    return f"{subject.split(':')[-1]} · {predicate.replace('_', ' ')} · {obj}"
