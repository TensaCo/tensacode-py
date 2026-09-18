"""The assistant's domain: the words a desktop agent needs, and the acts they map to.

This exists to be measured. ``examples/browser_agents/assistant/language.py`` reads
the same requests with ~105 regexes; this reads them with the core English grammar
plus the vocabulary below, and :func:`acts` projects the resulting frames onto the
same ``(act, slots)`` vocabulary so the two can be compared on the same benchmark.

The projection is deliberately thin: it renames roles and resolves place words to
paths. Everything structural — which verb, what it acts on, where, with what text,
and whether the utterance was an order, a question or a report — comes from the
grammar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..chart import Understanding, understand
from ..english import ENGLISH
from ..grammar import Ask, Build, Ent, Grammar, Head, Lit, Locative, Order, Qualify, production, words
from ..semantics import Entity, Frame, Question, Request

#: Place words and the paths they name.
PLACES: Mapping[str, str] = {
    "desktop": "~/Desktop", "documents": "~/Documents", "docs": "~/Documents", "downloads": "~/Downloads",
    "home": "~", "projects": "~/Projects", "pictures": "~/Pictures", "music": "~/Music", "videos": "~/Videos",
    "tmp": "/tmp", "temp": "/tmp",
}

#: Dock applications, by the words people use for them.
APPS: Mapping[str, str] = {
    "firefox": "Firefox", "browser": "Firefox", "chromium": "Chromium", "chrome": "Chromium",
    "files": "Files", "nautilus": "Files", "terminal": "Terminal", "shell": "Terminal",
    "editor": "Text Editor", "gedit": "Text Editor", "vscode": "Visual Studio Code",
    "slack": "Slack", "mail": "Mail", "email": "Mail", "settings": "Settings", "wireshark": "Wireshark",
    "rhythmbox": "Rhythmbox",
}

VERBS = (
    *words("make", "create", cat="V", sem="make"),
    *words("mkdir", cat="V", sem="make_folder"),
    *words("touch", cat="V", sem="make_file"),
    *words("add", "put", "place", cat="V", sem="add"),
    *words("delete", "remove", "trash", "erase", "rm", cat="V", sem="delete"),
    *words("read", "cat", "print", "display", "view", cat="V", sem="read"),
    *words("show", "tell", cat="V", sem="show"),
    *words("list", "ls", cat="V", sem="list"),
    *words("write", "save", "type", cat="V", sem="write"),
    *words("append", cat="V", sem="append"),
    *words("move", cat="V", sem="move"),
    *words("mv", cat="V", sem="move", ditrans=True),
    *words("copy", "duplicate", cat="V", sem="copy"),
    *words("cp", cat="V", sem="copy", ditrans=True),
    *words("mention", "contain", cat="V", sem="mention"),
    *words("rename", cat="V", sem="rename"),
    *words("find", "locate", cat="V", sem="find"),
    *words("search", "look", "grep", cat="V", sem="search"),
    *words("count", cat="V", sem="count"),
    *words("install", cat="V", sem="install"),
    *words("run", "execute", "exec", cat="V", sem="run"),
    *words("open", cat="V", sem="open"),
    *words("launch", "start", cat="V", sem="launch"),
    *words("go", "cd", "switch", "change", cat="V", sem="go"),
    *words("commit", cat="V", sem="commit"),
    *words("init", "initialize", "initialise", cat="V", sem="init"),
    *words("clean", "organize", "organise", "tidy", "fix", cat="V", sem="tidy"),  # known words, no act: honestly unknown
)

NOUNS = (
    *words("folder", "directory", "dir", cat="N", sem="folder"),
    *words("file", "note", "document", "doc", cat="N", sem="file"),
    *words("readme", cat="N", sem="readme"),
    *words("repo", "repository", cat="N", sem="repo"),
    *words("project", cat="N", sem="project"),
    *words("git", cat="N", sem="git"),
    *words("line", cat="N", sem="line"),
    *words("word", cat="N", sem="word"),
    *words("item", "thing", "stuff", cat="N", sem="item"),
    *words("space", cat="N", sem="space"),
    *words("disk", "drive", "storage", cat="N", sem="disk"),
    *words("core", "cpu", "processor", cat="N", sem="cpu"),
    *words("process", "program", cat="N", sem="process"),
    *words("hostname", cat="N", sem="hostname"),
    *words("ip", cat="N", sem="ip"),
    *words("address", cat="N", sem="address"),
    *words("time", "date", "day", cat="N", sem="time"),
    *words("uptime", cat="N", sem="uptime"),
    *words("history", cat="N", sem="history"),
    *words("message", cat="N", sem="message"),
    *words("content", "contents", cat="N", sem="content"),
    *words("pdf", cat="N", sem="pdf"),
    *words("text", cat="N", sem="text"),
    *words("system", cat="N", sem="system"),
    *words("user", "username", cat="N", sem="user"),
    *words("app", "application", cat="N", sem="app"),
    *words("command", cat="N", sem="command"),
    *words("change", cat="N", sem="change"),
    *words("weather", "joke", "sandwich", "wifi", "test", cat="N", sem="offtopic"),  # in the lexicon, out of scope
    *[e for word, path in PLACES.items() for e in words(word, cat="N", sem=f"place:{path}", place=True)],
    # "files", "mail" and "editor" are also ordinary nouns, so the app reading is
    # dispreferred and only wins when a launching verb selects it
    *[e for word, app in APPS.items() for e in words(word, cat="N", sem=f"app:{app}", weight=-0.5, app=True)],
    *words("files", cat="N", sem="file", number="plural"),
)

ADJECTIVES = (
    *words("installed", cat="Adj", sem="installed"),
    *words("empty", cat="Adj", sem="empty"),
    *words("big", "large", cat="Adj", sem="big"),
    *words("new", cat="Adj", sem="new"),
    *words("old", cat="Adj", sem="old"),
    *words("many", cat="Adj", sem="many"),
    *words("much", cat="Adj", sem="much"),
    *words("running", cat="Adj", sem="running"),
    *words("uncommitted", cat="Adj", sem="uncommitted"),
    *words("git", cat="Adj", sem="git"),
)

#: Dialogue moves. They are utterances in their own right, not clauses.
SPELLINGS = (
    *words("whats", "what", cat="Whats", sem="theme"),
    *words("pwd", cat="Q", sem="cwd"),
    *words("uptime", cat="Q", sem="uptime"),
)

DIALOGUE = (
    *words("hi", "hello", "hey", "yo", cat="Move", sem="greet"),
    *words("thanks", "thanx", "thx", "ty", cat="Move", sem="thanks"),
    *words("yes", "yeah", "yep", "sure", "ok", "okay", cat="Move", sem="confirm"),
    *words("no", "nope", "nah", cat="Move", sem="cancel"),
    *words("help", cat="Move", sem="help"),
)

#: Multiword items outrank the split reading: "how many" is not "how" + "many".
MULTIWORD = [
    production('N[app=true] -> "text" "editor"', Lit("app:Text Editor"), weight=0.4),
    production('N[app=true] -> "vs" "code"', Lit("app:Visual Studio Code"), weight=0.4),
    production('N[app=true] -> "file" "manager"', Lit("app:Files"), weight=0.4),
    production('N[app=true] -> "system" "monitor"', Lit("app:System Monitor"), weight=0.4),
    production('N[app=true] -> "app" "center"', Lit("app:App Center"), weight=0.4),
    production('N[place=true] -> "home" "folder"', Lit("place:~"), weight=0.4),
    production('N[place=true] -> "home" "directory"', Lit("place:~"), weight=0.4),
    production('Move -> "thank" "you"', Lit("thanks"), weight=0.4),
    production('Move -> "never" "mind"', Lit("cancel"), weight=0.4),
    production('Move -> "go" "ahead"', Lit("confirm"), weight=0.4),
    production('Move -> "yes" "please"', Lit("confirm"), weight=0.4),
    production('Wh -> "how" "many"', Lit("count"), weight=0.4),
    production('Wh -> "how" "much"', Lit("amount"), weight=0.4),
    production('Wh -> "how" "big"', Lit("size"), weight=0.4),
    production('Wh -> "how" "large"', Lit("size"), weight=0.4),
    production('Wh -> "what" "operating" "system"', Lit("os"), weight=0.4),
    production('U -> Move', Head(0), weight=0.4),
    # spellings and idioms people actually type
    production('V -> "get" "rid" "of"', Lit("delete"), weight=0.4),
    # a verbless request: "new folder photos"
    production('IMP -> "new" NBAR', Order(Build(predicate="make", roles=(("object", 1),))), weight=0.3),
    production('Q -> Whats PP', Ask(Locative("located", modifier=1), asked="theme"), weight=0.4),
    production('Q -> Whats NP', Ask(Build(predicate="be", roles=(("subject", 1),)), asked="theme"), weight=0.4),
    production('Q -> Wh Aux NP V', Ask(Build(predicate_from=3, roles=(("subject", 2),)), asked_from=0), weight=0.2),
]

DESKTOP: Grammar = ENGLISH.extend(
    productions=MULTIWORD,
    entries=[*VERBS, *NOUNS, *ADJECTIVES, *DIALOGUE, *SPELLINGS],
    start=("S", "Q", "IMP", "U", "NP"),
    name="english+desktop",
)


# ------------------------------------------------------------------ projection


@dataclass(frozen=True)
class Act:
    """The assistant's request shape: an act name and its slots."""

    act: str
    slots: Mapping[str, Any]

    def __repr__(self) -> str:
        return f"{self.act}({', '.join(f'{k}={v!r}' for k, v in sorted(self.slots.items()))})"


#: wh-word or noun -> the machine question it asks
INFO = {
    "time": "date", "user": "user", "disk": "disk", "space": "disk", "ip": "ip", "address": "ip",
    "hostname": "hostname", "uptime": "uptime", "process": "processes", "cpu": "cpus", "os": "os",
}


def _speaks_of_self(value: Any) -> bool:
    """"me", "us", "you", "i" name the people talking, never the file or folder."""
    return isinstance(value, Entity) and value.kind == "pronoun" and value.features.get("person") in (1, 2)


def _refers_back(value: Any) -> bool:
    """A third-person pronoun or a demonstrative: the utterance really did point at something."""
    if not isinstance(value, Entity):
        return False
    return (value.kind == "pronoun" and value.features.get("person") not in (1, 2)) or bool(value.features.get("demonstrative"))


def _place(value: Any) -> str | None:
    """A path for an entity, if it names one: a literal path, a place word, or a name."""
    if not isinstance(value, Entity) or _speaks_of_self(value):
        return None
    if value.kind == "path":
        return value.text
    noun = value.features.get("noun")
    if isinstance(noun, str) and noun.startswith("place:"):
        return noun.split(":", 1)[1]
    if value.kind in ("name", "literal", "resolved"):
        return value.text
    if value.kind == "pronoun":
        return "@it"
    return None


def _target(value: Any) -> str | None:
    """What an act operates on: a path, a name, or the reference "@it"."""
    if isinstance(value, tuple):
        return _target(value[0]) if value else None
    if not isinstance(value, Entity) or _speaks_of_self(value):
        return None
    if value.kind == "pronoun" or value.features.get("demonstrative"):
        return "@it"
    if value.kind in ("path", "literal", "name"):
        return value.text
    if value.features.get("name"):
        return _text(value.features["name"])
    place = _place(value)
    if place:
        return place
    if value.features.get("noun") in (None, "item"):
        return None
    return _as_place(value) or value.text


def _text(value: Any) -> str | None:
    if isinstance(value, Entity):
        return value.text
    return value if isinstance(value, str) else None


def _app(value: Any) -> str | None:
    if isinstance(value, Entity):
        noun = value.features.get("noun")
        if isinstance(noun, str) and noun.startswith("app:"):
            return noun.split(":", 1)[1]
        if isinstance(value.text, str) and value.text.startswith("app:"):
            return value.text.split(":", 1)[1]
    if isinstance(value, str) and value.startswith("app:"):
        return value.split(":", 1)[1]
    return None


def _noun(value: Any) -> str | None:
    if isinstance(value, Entity):
        noun = value.features.get("noun")
        return noun if isinstance(noun, str) else None
    return None


def _kind_of(value: Any) -> str:
    """folder, file or unknown — from the noun used, or from a file extension."""
    noun = _noun(value)
    if noun in ("folder", "repo", "project"):
        return "folder"
    if noun in ("file", "readme", "note", "document", "pdf"):
        return "file"
    text = _target(value) or ""
    if "." in text.rsplit("/", 1)[-1]:
        return "file"
    return "unknown"


def acts(meaning: Any, *, alone: bool = True) -> list[Act]:
    """Project one reading onto the assistant's ``(act, slots)`` vocabulary."""
    if isinstance(meaning, str):  # a dialogue move
        return [Act(meaning, {})]
    if isinstance(meaning, Request):
        return _needed(_from_request(meaning.frame), meaning.frame)
    if isinstance(meaning, Question):
        return _needed(_from_question(meaning), meaning.frame)
    if isinstance(meaning, Frame):
        return _needed(_from_request(meaning), meaning)  # a bare clause read as an instruction
    if isinstance(meaning, Entity):
        # a bare phrase is a request only when it names something outright — a pasted
        # path, a number. A loose noun is an answer or a fragment, not an instruction.
        if not alone or meaning.kind not in ("path", "literal", "number", "command"):
            return []
        path = _place(meaning)
        return [Act("read", {"target": path})] if path else []
    return []


def _from_request(frame: Frame) -> list[Act]:
    if not _understood(frame):
        return []  # part of the request has no expression in the act vocabulary
    verb = frame.predicate
    obj = frame.role("object") or frame.role("theme")
    place = frame.role("location") or frame.role("destination")
    content = frame.role("content")
    # PP attachment is genuinely ambiguous, so "a folder called x on my desktop" may
    # hang the place on the verb or on the noun. Look in both before giving up.
    if isinstance(obj, Entity):
        place = place or obj.features.get("location") or obj.features.get("destination")
        content = content or obj.features.get("content")
        if _noun(obj) == "content" and obj.features.get("of") is not None:
            obj = obj.features["of"]

    if isinstance(obj, tuple):  # "make a folder called a and a folder called b"
        out: list[Act] = []
        for item in obj:
            out.extend(_from_request(Frame(verb, {**frame.roles, "object": item}, frame.features)))
        return out

    if verb in ("make", "make_folder", "make_file"):
        named = (obj.features.get("name") if isinstance(obj, Entity) else None) or frame.role("name")
        if isinstance(named, tuple):  # "folders called drafts and final"
            return [a for item in named
                    for a in _from_request(Frame(verb, {**frame.roles, "object": Entity(
                        obj.kind, obj.text, {**obj.features, "name": item})}, frame.features))]
        name = _text(named) if named is not None else _target(obj)
        if isinstance(obj, Entity) and _noun(obj) == "readme" and named is None:
            name = "README.md"
        elif named is None and isinstance(name, str) and name.lower() in GENERIC_NOUNS:
            name = None  # "make a folder" names no folder
        if isinstance(name, str) and name.startswith("@"):
            name = None  # a reference is not a name: nothing new can be called "it"
        if place is None and isinstance(obj, Entity) and _place_word(obj) and named is None:
            place, name = obj, None  # "make a folder in documents": the noun was the place
        kind = {"make_folder": "folder", "make_file": "file"}.get(verb) or _kind_of(obj)
        slots = {"name": name, "place": _as_place(place), "text": _text(content)}
        return [Act("create_folder" if kind != "file" else "create_file", _drop(slots))]
    if verb in ("add", "write", "append", "put"):
        # "put hello in a file called hi.txt" creates the file with that text in it
        if isinstance(place, Entity) and place.features.get("name") is not None and _kind_of(place) in ("file", "folder"):
            made = _from_request(Frame("make", {k: v for k, v in frame.roles.items() if k not in ("location", "destination", "object")}
                                       | {"object": place}, frame.features))
            return [Act(a.act, _drop({**a.slots, "text": _text(obj)})) for a in made]
        if isinstance(obj, Entity) and _kind_of(obj) in ("file", "folder") and obj.features.get("name"):
            merged = Frame("make", {**frame.roles, "content": obj.features.get("content") or content or frame.role("destination")}, frame.features)
            text = _text(frame.role("destination")) or _text(content) or _text(frame.role("theme"))
            acts_ = _from_request(Frame("make", {k: v for k, v in frame.roles.items() if k != "destination"}, frame.features))
            return [Act(a.act, _drop({**a.slots, "text": text})) for a in acts_]
        target = _target(place) if place is not None else _target(obj)
        text = _text(obj) if place is not None else _text(content)
        return [Act("write", _drop({"text": text, "target": target, "append": verb in ("add", "append", "put")}))]
    if verb == "delete":
        return [Act("delete", _drop({"target": _target(obj) or _target(place)}))]
    if verb in ("move", "copy"):
        return [Act(verb, _drop({"target": _target(obj), "dest": _as_place(place)}))]
    if verb == "rename":
        return [Act("rename", _drop({"target": _target(obj), "new_name": _target(frame.role("destination") or frame.role("as"))}))]
    if verb in ("read", "show", "open", "view", "list"):
        app = _app(obj) or _app(place)
        if app and verb in ("open", "show"):
            return [Act("open_app", {"app": app})]
        kind = _kind_of(obj)
        if verb == "list":
            return [Act("list", _drop({"place": _as_place(place) or _as_place(obj)}))]
        if obj is None or (kind == "folder" and verb != "read"):
            return [Act("list", _drop({"place": _as_place(obj) or _as_place(place)}))]
        if _place_word(obj) and verb in ("show", "open"):
            return [Act("list", _drop({"place": _as_place(obj)}))]
        if verb == "show" and _refers_back(obj):
            return [Act("list", {"place": "@it"})]
        return [Act("read", _drop({"target": _target(obj), "place": _as_place(place) if place is not None else None}))]
    if verb == "launch":
        app = _app(obj)
        return [Act("open_app", {"app": app})] if app else []
    if verb == "find":
        pattern = _pattern(obj)
        return [Act("find", _drop({"pattern": pattern, "place": _as_place(place) or "~"}))]
    if verb == "search":
        needle = _text(content) or _text(frame.role("topic")) or _text(frame.role("beneficiary")) or _text(obj)
        return [Act("grep", _drop({"needle": needle, "place": _as_place(place) or "~"}))]
    if verb == "count":
        unit = _noun(obj)
        return [Act("count", _drop({"unit": unit + "s" if unit else None, "target": _target(place) or _target(obj)}))]
    if verb == "mention":
        return [Act("grep", _drop({"needle": _target(obj), "place": _as_place(place) or "~"}))]
    if verb == "install":
        return [Act("install", _drop({"package": _target(obj)}))]
    if verb == "run":
        return [Act("run", _drop({"command": _text(obj)}))]
    if verb == "go":
        return [Act("cd", {"target": _as_place(place) or _as_place(obj) or "~"})]
    if verb == "init":
        return [Act("git_init", {"target": _as_place(place) or _as_place(obj) or "@it"})]
    if verb == "commit":
        return [Act("git_commit", _drop({"target": _as_place(place) or _as_place(obj) or "@it",
                                         "message": _text(frame.role("as") or content)}))]
    return []


def _needed(acts_: list[Act], frame: Frame | None = None) -> list[Act]:
    """Withhold an act that does not say what it acts on, or that invents a reference.

    "@it" means *the thing we were just talking about*. The projection may only use it
    when the utterance actually pointed at something — a third-person pronoun or a
    demonstrative. Otherwise the reference is invented, and an invented reference in a
    core slot is how "make me a sandwich" becomes a new folder.
    """
    licensed = frame is not None and any(_refers_back(e) for e in frame.entities())
    out: list[Act] = []
    for act in acts_:
        core = NEEDS_TARGET.get(act.act)
        if core is not None and act.slots.get(core) is None:
            continue
        if not licensed and any(v == "@it" for v in act.slots.values()):
            continue
        out.append(act)
    return out


def _place_word(value: Any) -> bool:
    noun = _noun(value)
    return isinstance(noun, str) and noun.startswith("place:")


def _as_place(value: Any) -> str | None:
    """A place slot: a path, or "@name" for a folder the conversation knows by name."""
    if value is None or _speaks_of_self(value):
        return None
    path = _place(value)
    if path and (path.startswith(("~", "/")) or path == "@it"):
        return path
    if isinstance(value, Entity):
        named = value.features.get("name")
        if named is not None:
            return f"@{_text(named)}"
        if value.kind in ("name", "literal"):
            return f"@{value.text}"
        if value.kind == "pronoun":
            return "@it"
    return path


def _pattern(value: Any) -> str | None:
    if isinstance(value, Entity) and value.features.get("name") is not None:
        return _text(value.features["name"])
    noun = _noun(value)
    if noun == "pdf":
        return "*.pdf"
    target = _target(value)
    return target


def _from_question(question: Question) -> list[Act]:
    frame, asked = question.frame, question.asked
    subject, theme = frame.role("subject"), frame.role("theme")
    location = frame.role("location")
    topic = INFO.get(_noun(subject) or "") or INFO.get(_noun(theme) or "")
    if asked in ("count", "amount", "size"):
        noun = _noun(theme) or _noun(subject)
        if noun in ("line", "word"):
            return [Act("count", _drop({"unit": noun + "s", "target": _target(location) or _target(theme)}))]
        if noun in ("space", "disk"):
            return [Act("info", {"topic": "disk"})]
        if noun == "cpu":
            return [Act("info", {"topic": "cpus"})]
        if asked == "size":
            return [Act("size", _drop({"target": _target(subject) or _as_place(subject)}))]
        return [Act("list", _drop({"place": _place(location) or _place(theme), "count": True}))]
    if topic:
        return [Act("info", {"topic": topic})]
    if frame.predicate == "installed" or _noun(theme) == "installed":
        return [Act("which", _drop({"program": _target(subject)}))]
    if frame.predicate in ("say", "mention", "contain") and asked == "theme":
        return [Act("grep" if frame.predicate == "mention" else "read",
                    _drop({"needle": _target(frame.role("object")), "target": _target(subject),
                           "place": "~" if frame.predicate == "mention" else None}))]
    if asked == "location" and subject is not None:
        return [Act("find", _drop({"pattern": _target(subject), "place": "~"}))]
    if frame.predicate == "located" or location is not None:
        return [Act("list", _drop({"place": _place(location)}))]
    if asked == "subject" and frame.predicate == "be":
        return [Act("info", {"topic": "user"})]
    if frame.predicate == "running" or _noun(subject) == "process":
        return [Act("info", {"topic": "processes"})]
    return []


def _drop(slots: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in slots.items() if v is not None}


#: Roles each act can express. A frame carrying anything else is only *partly*
#: understood, and acting on a partly understood request is how an agent does the
#: wrong thing — so the act is withheld and the request reads as unknown.
EXPRESSIBLE: Mapping[str, frozenset[str]] = {
    "make": frozenset({"object", "theme", "location", "destination", "content", "name", "as"}),
    "make_folder": frozenset({"object", "theme", "location", "destination", "name"}),
    "make_file": frozenset({"object", "theme", "location", "destination", "content", "name"}),
    "add": frozenset({"object", "theme", "location", "destination", "content"}),
    "write": frozenset({"object", "theme", "location", "destination", "content"}),
    "append": frozenset({"object", "theme", "location", "destination", "content"}),
    "put": frozenset({"object", "theme", "location", "destination", "content"}),
    "delete": frozenset({"object", "theme", "location"}),
    "move": frozenset({"object", "theme", "destination", "location"}),
    "copy": frozenset({"object", "theme", "destination", "location"}),
    "rename": frozenset({"object", "theme", "destination", "as"}),
    "read": frozenset({"object", "theme", "location", "recipient", "of"}),
    "show": frozenset({"object", "theme", "location", "recipient", "of"}),
    "open": frozenset({"object", "theme", "location"}),
    "view": frozenset({"object", "theme", "location"}),
    "list": frozenset({"object", "theme", "location", "recipient"}),
    "launch": frozenset({"object", "theme"}),
    "find": frozenset({"object", "theme", "location", "name"}),
    "search": frozenset({"object", "theme", "location", "content", "topic", "beneficiary"}),
    "count": frozenset({"object", "theme", "location", "of"}),
    "install": frozenset({"object", "theme"}),
    "run": frozenset({"object", "theme"}),
    "go": frozenset({"object", "theme", "destination", "location"}),
    "init": frozenset({"object", "theme", "location", "destination", "as"}),
    "commit": frozenset({"object", "theme", "location", "destination", "as", "content"}),
    "mention": frozenset({"object", "theme", "location"}),
}

#: Words for a *kind* of thing. On their own they do not name one, so "make a folder"
#: has no name to create and the act is withheld rather than making "folder".
GENERIC_NOUNS = frozenset({"folder", "directory", "dir", "file", "note", "document", "doc", "repo", "repository",
                           "project", "item", "thing", "stuff", "content", "contents", "app", "application"})

#: Acts that change something need to know what they are changing.
NEEDS_TARGET = {"delete": "target", "move": "target", "copy": "target", "rename": "target",
                "write": "target", "install": "package", "run": "command", "read": "target",
                "create_folder": "name", "create_file": "name", "grep": "needle", "find": "pattern"}


def _understood(frame: Frame) -> bool:
    allowed = EXPRESSIBLE.get(frame.predicate)
    if allowed is None:
        return True
    return not (set(frame.roles) - allowed - {"subject", "agent"})


def read_request(text: str, *, grammar: Grammar = DESKTOP) -> tuple[list[Act], Understanding]:
    """Read one chat message: the acts it asks for, and the parse they came from."""
    got = understand(grammar, text)
    out: list[Act] = []
    alone = len(got.meanings) == 1
    for meaning in got.meanings:
        out.extend(acts(meaning, alone=alone))
    # a bare noun phrase over a half-covered utterance is not a request: saying nothing
    # is better than guessing a read, and it keeps unclear input from acting
    if got.coverage < 0.7 and all(isinstance(m, Entity) for m in got.meanings):
        return [], got
    return out, got
