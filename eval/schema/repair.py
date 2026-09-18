"""The schema changes, as a projection over any tier's output.

Each function here is one induced distinction from `eval/results/schema_clusters.json`. They
are written as a projection layer rather than inside a parser so that the same change can be
measured against every tier at once, and so that a change that does not pay can be deleted
without touching a parser.

Every rule is keyed on linguistic structure — a determiner, a possessive, a negation, a
place word, a screen noun — never on a specific input. A rule that only helped the sentences
that motivated it would be a lookup table, not a schema change.

    D1 referent_ontology   a screen object is not a filesystem object
    D2 typed_place         a place word denotes a path, not itself
    D3 span_introducer     a content span does not include the words that introduce it
    D4 memory_polarity     a negated memory instruction is a retraction, and topics canonicalise
    D5 app_identity        an app named by function or possession is still that app
    D6 needle_head         a search term is the term, not the noun phrase around it
"""

from __future__ import annotations

import re

#: place words and the path each denotes (D2). The values are the canonical form the rest of
#: the system uses; a bare word is never a path.
PLACE_WORDS: dict[str, str] = {
    "home": "~", "home folder": "~", "home directory": "~",
    "desktop": "~/Desktop", "documents": "~/Documents", "docs": "~/Documents",
    "downloads": "~/Downloads", "projects": "~/Projects",
    "pictures": "~/Pictures", "photos": "~/Pictures", "images": "~/Pictures",
    "music": "~/Music", "videos": "~/Videos", "movies": "~/Videos",
    "tmp": "/tmp", "temp": "/tmp",
}

#: things that exist on the screen rather than in the file system (D1). Asking how big one is,
#: or what font it uses, is a question about the display, not about a file.
SCREEN_NOUNS = {"window", "screen", "display", "sidebar", "dock", "launcher", "top bar", "menu bar",
                "title bar", "panel", "taskbar", "desktop background", "wallpaper", "cursor", "pointer"}

#: a screen property that a pixel look can answer, and the aspect name for a structural one
PIXEL_PROPERTIES = {"colour", "color", "font", "theme", "brightness", "contrast", "style", "look"}
SIZE_WORDS = {"big", "large", "small", "wide", "tall", "size", "dimensions"}

#: words that introduce a quoted or content span without being part of it (D3)
SPAN_INTRODUCERS = re.compile(
    r"^(?:the\s+)?(?:text|phrase|content|contents|words?|line|string|message|note|following)\s*[:,-]?\s+", re.I)

#: a noun phrase whose head is the search term (D6): "TODO comments" -> "TODO"
NEEDLE_TAIL = re.compile(r"\s+(?:comments?|lines?|mentions?|references?|strings?|entries|occurrences?)$", re.I)

#: articles and possessives that are not part of an app's name (D5)
APP_NOISE = re.compile(r"^(?:a|an|the|my|our|your)\s+|^(?:sending|opening|starting|running)\s+|\s+(?:window|app|application|program)$", re.I)

#: topic phrases that mean the same remembered thing (D4)
TOPIC_HEADS = {"email address": "email", "e-mail address": "email", "email": "email",
               "phone number": "phone", "favourite colour": "favourite colour", "favorite color": "favourite colour",
               "name": "name", "birthday": "birthday", "address": "address"}
TOPIC_TRAILING = re.compile(r"\s+(?:now|anymore|any more|again|today|please)$", re.I)

MEMORY_ACTS = {"tell", "ask_memory", "forget", "ask_self"}
_NEG = re.compile(r"\b(?:don'?t|do not|never|stop|no longer|isn'?t|is not|not)\b", re.I)


def _topic_from(low: str) -> str | None:
    """The remembered thing a sentence is about, from the possessive that names it."""
    m = re.search(r"\bmy\s+([\w ]+?)\s*(?:is|isn'?t|was|anymore|any more|now|please|[.?!]|$)", low)
    if not m:
        return None
    topic = TOPIC_TRAILING.sub("", m[1].strip()).lower()
    return TOPIC_HEADS.get(topic, topic) or None


def _canon_place(value: str) -> str | None:
    word = str(value).strip().strip("'\"“”").lower().lstrip("@")
    word = re.sub(r"\s+(?:folder|directory|dir)$", "", word)
    return PLACE_WORDS.get(word)


def d1_referent_ontology(text: str, act: str, slots: dict) -> tuple[str, dict]:
    """A screen object is not a filesystem object.

    'how big is the Files window' is a measurement of the display; 'how big is report.pdf' is a
    measurement of a file. The two are the same question about different ontologies, and only
    the referent's kind separates them.
    """
    low = text.lower()
    named_screen = next((n for n in SCREEN_NOUNS if re.search(rf"\b{re.escape(n)}\b", low)), None)
    if not named_screen:
        return act, slots
    asks_property = next((p for p in PIXEL_PROPERTIES if re.search(rf"\b{p}\b", low)), None)
    asks_size = any(re.search(rf"\b{w}\b", low) for w in SIZE_WORDS)
    if asks_property:
        return "ask_pixels", {"region": named_screen, "property": asks_property}
    if asks_size and act in ("size", "count", "list", "unknown"):
        return "ask_screen", {"aspect": "window_size" if named_screen == "window" else f"{named_screen}_size",
                              "region": named_screen}
    return act, slots


def d2_typed_place(text: str, act: str, slots: dict) -> tuple[str, dict]:
    """A place word denotes a path. Leaving it as a bare word defers the decision for ever."""
    out = dict(slots)
    for key in ("place",):  # a target or dest is a named object to resolve, not a location word
        if key in out and isinstance(out[key], str):
            path = _canon_place(out[key])
            if path:
                out[key] = path
    if act in ("list", "create_file", "create_folder", "find", "grep", "size", "cd") and "place" not in out:
        m = re.search(r"\b(?:in|on|inside|into|from|of)\s+(?:my|the|our)?\s*([a-z ]+?)\s*(?:folder|directory|dir)\b", text, re.I)
        if m and (path := _canon_place(m[1])):
            out["place"] = path
    return act, out


def d3_span_introducer(text: str, act: str, slots: dict) -> tuple[str, dict]:
    """A content span does not include the words that introduce it."""
    out = dict(slots)
    for key in ("text", "value", "message"):
        if key in out and isinstance(out[key], str):
            stripped = SPAN_INTRODUCERS.sub("", out[key]).strip()
            if stripped and stripped != out[key]:
                out[key] = stripped
    # a span that collapsed to a bare determiner is not a span at all: recover the quoted run
    for key in ("text", "value"):
        if out.get(key, "").strip().lower() in {"the", "a", "an", "this", "that"}:
            quoted = re.findall(r"[\"'“‘](.+?)[\"'”’]", text)
            if quoted:
                out[key] = quoted[-1]
            else:
                m = re.search(r"\b(?:put|write|containing|contains|saying|says)\s+(?:the\s+\w+\s+)?(.+?)\s+(?:into|in|to|inside)\b", text, re.I)
                if m:
                    out[key] = SPAN_INTRODUCERS.sub("", m[1]).strip()
    return act, out


def d4_memory_polarity(text: str, act: str, slots: dict) -> tuple[str, dict]:
    """A negated memory instruction is a retraction; a polar memory question is a check; topics canonicalise."""
    out = dict(slots)
    low = text.lower().rstrip(" .!?")
    #: a negation inside quoted content ("don't forget the milk") belongs to the content
    outside_quotes = re.sub(r"[\"'“‘][^\"'”’]*[\"'”’]", " ", low)
    if "topic" in out and isinstance(out["topic"], str):
        topic = TOPIC_TRAILING.sub("", out["topic"].strip()).lower()
        out["topic"] = TOPIC_HEADS.get(topic, out["topic"])  # leave an unknown topic exactly as read

    mentions_memory = re.search(r"\b(?:remember|memor(?:y|ise|ize)|know|knew|recall|forget)\b", outside_quotes)
    about_me = re.search(r"\bmy\b|\bi am\b|\bi'm\b|\bwho i am\b|\bam i\b", outside_quotes)
    #: an act that names a file or a place is about the file system, whatever words the content uses
    FILE_ACTS = {"create_file", "create_folder", "write", "read", "list", "delete", "move", "copy", "rename", "find", "grep", "run"}
    if act in FILE_ACTS:
        return act, out
    if act in MEMORY_ACTS or mentions_memory:
        # "please don't remember my name", "my name isn't Jacob anymore" -> a retraction
        if _NEG.search(outside_quotes) and (mentions_memory or about_me) and not low.endswith("?"):
            topic = out.get("topic") or _topic_from(low)
            return "forget", {"topic": topic} if topic else {}
        # "am i Jacob" / "do you know who i am" -> asking about a remembered value
        if re.match(r"^(?:am i|is my|do you know (?:who|what) i)\b", low) or "who i am" in low:
            topic = out.get("topic") or ("name" if re.search(r"\bname\b|who i am|^am i\b", low) else None)
            slots_out = {"topic": topic or "name"}
            if m := re.match(r"^am i\s+([\w.@-]+)$", low):
                slots_out["expect"] = m[1]  # a polar check carries the value it is checking
            return "ask_memory", slots_out
        # "what do you know" -> the whole of memory is the topic
        if re.match(r"^what (?:do|did) you know\b", low) or re.match(r"^what do you remember\b", low):
            return "ask_memory", {"topic": "everything"}
    return act, out


def d5_app_identity(text: str, act: str, slots: dict) -> tuple[str, dict]:
    """An app named by what it does, or by possession, is still that app."""
    if act != "open_app":
        return act, slots
    out = dict(slots)
    name = str(out.get("app", "")).strip()
    if not name:
        m = re.search(r"\b(?:open|start|launch|begin|bring up)\s+(?:up\s+)?(?:my|the|a|an)?\s*([\w ]+?)\s*(?:for me)?\s*[.?!]?$", text, re.I)
        name = m[1].strip() if m else ""
    previous = None
    while name and name != previous:
        previous = name
        name = APP_NOISE.sub("", name).strip()
    if name:
        out["app"] = name
    return act, out


def d6_needle_head(text: str, act: str, slots: dict) -> tuple[str, dict]:
    """A search term is the term, not the noun phrase built around it."""
    if act != "grep" or "needle" not in slots:
        return act, slots
    out = dict(slots)
    needle = str(out["needle"]).strip()
    quoted = re.findall(r"[\"'“‘](.+?)[\"'”’]", text)
    if quoted:
        out["needle"] = quoted[-1]
    else:
        out["needle"] = NEEDLE_TAIL.sub("", needle).strip() or needle
    return act, out


RULES = (d1_referent_ontology, d2_typed_place, d3_span_introducer, d4_memory_polarity, d5_app_identity, d6_needle_head)
NAMES = tuple(r.__name__ for r in RULES)


def repair(text: str, act: str, slots: dict, enabled: set[str] | None = None) -> tuple[str, dict]:
    """Apply the induced distinctions to one tier's reading."""
    for rule in RULES:
        if enabled is None or rule.__name__ in enabled:
            act, slots = rule(text, act, slots)
    return act, slots
