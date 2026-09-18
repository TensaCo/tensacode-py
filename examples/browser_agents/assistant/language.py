"""Chat messages -> request frames, with a forgiving grammar (no model).

Assumptions that make free text tractable: a message is a sequence of clauses joined by
"then" / "and" / sentence ends; each clause is one request that starts from a verb; slots
(names, places, quoted text, paths, references like "it") can appear anywhere in the
clause. Politeness and filler are ignored. Anything that does not fit is returned as an
``unknown`` frame carrying the original words, so the agent can say what it did not get.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

PLACES = {
    "desktop": "~/Desktop", "documents": "~/Documents", "docs": "~/Documents", "downloads": "~/Downloads",
    "home": "~", "home folder": "~", "home directory": "~", "projects": "~/Projects", "pictures": "~/Pictures",
    "music": "~/Music", "videos": "~/Videos", "tmp": "/tmp", "temp": "/tmp",
}
APPS = {
    "firefox": "Firefox", "browser": "Firefox", "web browser": "Firefox", "chromium": "Chromium", "chrome": "Chromium",
    "files": "Files", "file manager": "Files", "nautilus": "Files", "terminal": "Terminal", "shell": "Terminal",
    "text editor": "Text Editor", "editor": "Text Editor", "gedit": "Text Editor", "vs code": "Visual Studio Code",
    "vscode": "Visual Studio Code", "code": "Visual Studio Code", "slack": "Slack", "mail": "Mail", "email": "Mail",
    "settings": "Settings", "system monitor": "System Monitor", "task manager": "System Monitor", "wireshark": "Wireshark",
    "app center": "App Center", "software": "App Center", "rhythmbox": "Rhythmbox", "music player": "Rhythmbox",
}
REFERENCES = {"it", "that", "this", "there", "them", "those", "these", "that folder", "the folder", "that file", "the file", "this folder", "this file", "the repo", "that repo", "the project", "that project"}
FILLER = re.compile(
    r"^(?:(?:hey|hi|ok(?:ay)?|so|now|alright|also|and|then|next|finally|first|please|pls|kindly|"
    r"can you|could you|would you|will you|can u|could u|i want you to|i'd like you to|i would like you to|i need you to|"
    r"go ahead and|help me|i want to|i need to|let's|lets|try to|you should|just)\b[\s,]*)+",
    re.I,
)
STOP = {"a", "an", "the", "my", "on", "in", "into", "to", "at", "under", "inside", "called", "named", "with", "for", "and", "of", "that", "it", "new", "empty", "some", "from", "as"}


@dataclass(frozen=True)
class Frame:
    """One request: an act plus slots. ``words`` keeps the clause as the user wrote it."""

    act: str
    words: str
    slots: dict = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.act, self.words, tuple(sorted((k, str(v)) for k, v in self.slots.items()))))


# ------------------------------------------------------------------ helpers


# single quotes open/close only at word edges, so apostrophes inside ('don't forget') stay text
_QUOTE = re.compile(r"(?<!\w)'((?:[^']|(?<=\w)'(?=\w))*)'(?!\w)|\"([^\"]*)\"|“([^”]*)”|‘([^’]*)’|`([^`]*)`")


def _mask(text: str) -> tuple[str, list[str]]:
    """Replace quoted spans with placeholders so keywords inside quotes are not read as grammar."""
    quoted: list[str] = []
    raw: list[str] = []

    def keep(m: re.Match) -> str:
        quoted.append(next(g for g in m.groups() if g is not None))
        raw.append(m[0])
        return f"QUOTE{len(quoted) - 1}"

    masked = _QUOTE.sub(keep, text)
    _mask.raw = raw  # type: ignore[attr-defined]  # the last call's spans, quotes included (for re-splitting)
    return masked, quoted


def _unmask(s: str, quoted: list[str]) -> str:
    return re.sub(r"QUOTE(\d+)", lambda m: quoted[int(m[1])], s).strip()


def split_clauses(text: str) -> list[str]:
    masked, quoted = _mask(text.strip())
    quoted = list(_mask.raw)  # type: ignore[attr-defined]  # clauses keep their quote marks
    parts = re.split(
        r"(?:[.!?;]+\s+|[.!?;]+$|\n+|,?\s+\b(?:and then|then|after that|afterwards|and also|also)\b\s*|,?\s+\band\b\s+(?=(?:%s)\b))" % "|".join(sorted(VERBS, key=len, reverse=True)),
        masked,
        flags=re.I,
    )
    clauses = [c for p in parts if p and (c := _unmask(p, quoted).strip(" ,"))]
    out: list[str] = []
    for c in clauses:  # "make a folder called a and a folder called b": the second clause borrows the verb
        for i, piece in enumerate(re.split(r",?\s+\band\s+(?=(?:a|an|another)\s+(?:new\s+)?(?:folder|file|directory|dir)\b)", c, flags=re.I)):
            if i and out and (verb := re.match(r"^\s*(?:please\s+|can you\s+|could you\s+)?(\w+)", out[-1] if i == 1 else out[-1])):
                piece = f"{verb[1]} {piece}"
            out.append(piece)
    return out


PATH = re.compile(r"(~(?:/[^\s,;]*)?|/[\w.@+-][\w./@+-]*|(?:[\w.@+-]+/)+[\w.@+-]*|[\w@+-][\w.@+-]*\.[A-Za-z0-9]{1,8})(?=[\s,;:!?)]|$|\.(?:\s|$))")


def _place(s: str) -> str | None:
    """Where: 'on my desktop', 'in documents', 'in ~/x', 'in the notes folder', 'there'."""
    m = re.search(r"\b(?:on|in|into|inside|under|within|to|at|from)\s+(?:my|the|our)?\s*(home folder|home directory|desktop|documents|docs|downloads|home|projects|pictures|music|videos|tmp|temp)(?:\s+(?:folder|directory|dir))?\b", s, re.I)
    if m:
        return PLACES[m[1].lower()]
    m = re.search(r"\b(?:on|in|into|inside|under|within|to|from)\s+(~(?:/[^\s,;]*)?|/[\w./@+-]+|[\w.@+-]+/[\w./@+-]*)", s)
    if m:
        return m[1].rstrip("/.") or m[1]
    m = re.search(r"\b(?:in|into|inside|under|within|to|from)\s+(?:the|my)\s+(QUOTE\d+|[\w.@+-]+)\s+(?:folder|directory|dir|repo|project)\b", s, re.I) or re.search(
        r"\b(?:in|into|inside|under|within|to|from)\s+(?:the\s+)?(?:folder|directory|dir|repo|project)\s+(?:called\s+|named\s+)?(QUOTE\d+|[\w.@+-]+)", s, re.I)
    if m:
        return f"@{m[1]}"  # a folder known by name; resolved against the conversation later
    if re.search(r"\b(?:in|into|inside|to)\s+(?:it|there|that folder|the folder|this folder|that one)\b|\bthere\b", s, re.I):
        return "@it"
    # "in drafts", "to final": a bare name where a place goes (not a filename, not a filler word)
    m = re.search(r"\b(?:in|into|inside|within|to)\s+([\w@+-][\w@+-]*)(?=\s*$|\s*[,?!]|\.\s|\.$|\s+(?:containing|contains|with|saying|that says|called|named|as|then|and|please|for me)\b)", s, re.I)
    if m and m[1].lower() not in STOP | BARE_PLACE_STOP and not re.fullmatch(r"quote\d+", m[1], re.I) and not re.search(r"\b(?:in order|so as|want|need|how|going|have|able)\s+to\s+\S+$", s[: m.end()], re.I):
        return f"@{m[1]}"
    return None


BARE_PLACE_STOP = {"it", "this", "that", "there", "them", "those", "these", "me", "you", "us", "here", "order", "case", "general", "full", "place", "use", "time", "total", "all", "one", "front", "mind", "bed", "line", "sync", "touch", "detail", "private", "public", "english", "text", "words", "bold", "quotes", "brackets"}


def _name(s: str, kinds: str) -> str | None:
    m = re.search(r"\b(?:called|named|titled|name it|call it)\s+(QUOTE\d+|[\w.@+-]+)", s, re.I)
    if m:
        return m[1]
    for m in re.finditer(rf"\b(?:{kinds})\b\s+(QUOTE\d+|[\w.@+-]+)", s, re.I):
        if m[1].lower() not in STOP and m[1].lower() not in PLACES:
            return m[1]
    # a filename written directly ("create note.txt in it") beats reading the word before a kind word
    if m := re.search(r"\b([\w@+-][\w.@+-]*\.[A-Za-z0-9]{1,8})\b", s):
        return m[1]
    m = re.search(rf"(QUOTE\d+|[\w.@+-]+)\s+(?:{kinds})\b", s, re.I)
    if m and m[1].lower() not in STOP | VERBS | {"new", "empty", "text", "blank", "another", "one", "git"} and m[1].lower() not in PLACES:
        return m[1]
    return None


#: "the app that is used for writing code" points at nothing the agent can resolve
_RELATIVE = re.compile(r"\b(?:the|that|this|a|an)\s+[\w-]+\s+(?:that|which|who)\b", re.I)


def _ref(s: str) -> bool:
    return bool(re.search(r"\b(?:it|that|this|them|those|these)\b(?!\s+(?:says|said|contains))|^\s*the (?:file|folder|directory|repo|project|one)\s*$", s, re.I))


def _target(s: str, quoted: list[str]) -> str | None:
    """The thing acted on: a path, a filename, a quoted name, a place, or a reference."""
    found = _target_words(s, quoted)
    return PLACES.get(found.lower(), found) if found and not found.startswith(("~", "/", "@")) else found


def _target_words(s: str, quoted: list[str]) -> str | None:
    body = re.sub(r"\b(?:on|in|into|inside|under|within|to|from)\s+(?:my|the|our)?\s*(?:home folder|home directory|desktop|documents|docs|downloads|home|projects|pictures|music|videos|tmp|temp)(?:\s+(?:folder|directory|dir))?\b", " ", s, flags=re.I)
    if m := PATH.search(body):
        return m[1]
    if m := re.search(r"\b(?:folder|directory|dir|file|repo|project)\s+(?:called|named)\s+QUOTE(\d+)", body, re.I):
        return f"@{quoted[int(m[1])]}" if not re.search(r"\bfile\b", m[0], re.I) else quoted[int(m[1])]
    if m := re.search(r"QUOTE(\d+)", body):
        return quoted[int(m[1])]
    if m := re.search(r"\b(?:folder|directory|dir|file|repo|project)\s+(?:called\s+|named\s+)?([\w.@+-]+)", body, re.I):
        if m[1].lower() not in STOP and m[1].lower() not in PLACES:
            return m[1]
    if m := re.search(r"\b(?:the|my)\s+([\w.@+-]+)\s+(?:folder|directory|dir|file|repo|project)\b", body, re.I):
        if m[1].lower() not in STOP:
            return m[1]
    if _ref(body) and not _RELATIVE.search(s):
        return "@it"
    if m := re.search(r"\b(desktop|documents|downloads|home folder|home|projects)\b", s, re.I):
        return PLACES[m[1].lower()]
    return None


def _text(s: str, quoted: list[str]) -> str | None:
    m = re.search(r"\b(?:with the text|with text|containing|contains|that says|saying|says|with the line|with the words|with content|with contents|with)\s+(QUOTE\d+)", s, re.I)
    if m:
        return _unmask(m[1], quoted)
    m = re.search(r"\b(?:containing|that says|saying|with the text|with text)\s+(.+)$", s, re.I)
    return _unmask(m[1], quoted).strip(" .") if m else None


# ------------------------------------------------------------------ grammar

VERBS = {
    "make", "create", "add", "new", "touch", "mkdir", "write", "put", "append", "save", "show", "read", "open", "cat", "display", "print",
    "list", "ls", "what", "whats", "what's", "which", "delete", "remove", "trash", "erase", "rm", "get", "move", "mv", "rename", "copy",
    "cp", "duplicate", "find", "search", "look", "locate", "where", "grep", "count", "how", "install", "launch", "start", "run", "go",
    "cd", "switch", "change", "initialize", "init", "commit", "check", "tell", "clear", "set", "who", "is", "are", "help", "yes", "no",
}


VOCAB = ("desktop", "documents", "downloads", "folder", "folders", "directory", "delete", "create", "remove", "rename", "contents",
         "projects", "install", "commit", "called", "named", "containing", "everything", "pictures", "search", "firefox", "terminal")


def _close(a: str, b: str) -> bool:
    """Edit distance <= 1, counting one adjacent transposition as a single edit."""
    if a == b or abs(len(a) - len(b)) > 1:
        return a == b
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def fix_typos(text: str) -> str:
    """Correct near-misses of a small closed vocabulary; quoted text, paths and names after 'called' are left alone."""
    masked, _ = _mask(text)
    raw = list(_mask.raw)  # type: ignore[attr-defined]
    tokens = re.split(r"(\s+)", masked)
    out = []
    for i, tok in enumerate(tokens):
        word = tok.strip(".,!?;:")
        prev = next((t for t in reversed(tokens[:i]) if t.strip()), "").lower()
        if len(word) >= 5 and word.isalpha() and word.lower() not in VOCAB and prev not in ("called", "named", "titled"):
            fix = next((v for v in VOCAB if _close(word.lower(), v) and not v.startswith(word.lower()) and not word.lower().startswith(v)), None)
            if fix:
                tok = tok.replace(word, fix)
        out.append(tok)
    return re.sub(r"QUOTE(\d+)", lambda m: raw[int(m[1])], "".join(out))


_CHOICE = re.compile(
    r"(?:(?:the|option|number|no\.?|#)\s*)?(?:\d+|first|second|third|fourth|fifth|sixth|last|\d+(?:st|nd|rd|th))(?:\s+one)?(?:\s+please)?"
    r"|(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|last)(?:\s+one)?(?:\s+please)?"
    r"|(?:the\s+)?one\s+(?:on|in|under|inside|from)\s+(?:the\s+|my\s+)?[\w~/.@ -]{1,40}"
    r"|~?/[^\s]+",
    re.I,
)


# ------------------------------------------------------- indirect requests
# A form that is not an order may still be a request, but only where its object is something
# I could act on: "I can't find my invoice" asks me to look, "I can't find my keys" does not.
# The gate is affordance, not grammar (see tensorcode.social.indirect_reading), and the
# vocabularies below are a closed floor rather than general world knowledge.

#: things that live on a computer, so a bare noun can still be my business
DOMAIN_NOUNS = {
    "file", "files", "folder", "folders", "directory", "note", "notes", "readme", "document", "documents",
    "doc", "docs", "invoice", "invoices", "receipt", "receipts", "report", "reports", "budget", "spreadsheet",
    "photo", "photos", "picture", "pictures", "screenshot", "screenshots", "log", "logs", "backup", "backups",
    "script", "scripts", "repo", "repository", "project", "draft", "drafts", "todo", "shopping list", "list",
}
#: what an embedded past/participle verb means as an order ("deleted" -> "delete")
AS_ORDER = {
    "deleted": "delete", "removed": "remove", "renamed": "rename", "moved": "move", "copied": "copy",
    "created": "create", "made": "make", "opened": "open", "listed": "list", "showed": "show", "shown": "show",
    "added": "add", "written": "write", "wrote": "write", "read": "read", "found": "find", "checked": "check",
    "committed": "commit", "installed": "install", "tidied": "tidy", "organised": "organize", "organized": "organize",
    "deleting": "delete", "removing": "remove", "moving": "move", "copying": "copy", "creating": "create",
    "making": "make", "opening": "open", "reading": "read", "writing": "write", "renaming": "rename",
}
#: evaluative complements that make "my X is Y" a complaint rather than a fact about you
MESS = r"mess|state|disaster|nightmare|shambles|tip|pigsty"
#: an utterance that wishes for something, with the proposition it wishes for
WISH = re.compile(
    r"^(?:it(?:\s+would|\s*'d|\s+is)?\s+be (?:good|nice|great|helpful|better)(?:\s+if)?"
    r"|i(?:'?d| would) (?:like|prefer|appreciate|love)(?: it)?(?:\s+if)?"
    r"|would you mind|do you mind|if you could|i wish(?: that)?|i need you to|i want you to)"
    r"[,:]?\s+(?:you\s+)?(?:to\s+|could\s+|would\s+)?(?:please\s+)?(.+)$", re.I)


def in_domain(word: str) -> bool:
    """Is this something I could act on at all? Files, folders, apps, places — not car keys."""
    w = str(word).strip().strip("\"'.!?").lower()
    if not w:
        return False
    if re.search(r"\.[a-z0-9]{1,8}$", w) or "/" in w or w.startswith("~"):
        return True
    if w in PLACES or w in APPS or w in DOMAIN_NOUNS:
        return True
    head = w.rsplit(" ", 1)[-1]
    return head in DOMAIN_NOUNS or head in PLACES


def _implications() -> list:
    from tensorcode.social import Implication

    return [
        Implication(r"\b(?:i )?(?:can'?t|cannot|could'?nt|couldn't) find (?:my |the |a )?(?P<obj>[\w.@ -]{2,40})", "complaint", "find", 0.8, "pattern"),
        Implication(r"\b(?:i'?m|i am) looking for (?:my |the |a )?(?P<obj>[\w.@ -]{2,40})", "statement", "find", 0.8, "pattern"),
        Implication(r"\bwhere (?:did |has |is )?(?:my |the )?(?P<obj>[\w.@ -]{2,40}?)\s*(?:go|gone|got to)\b", "question", "find", 0.75, "pattern"),
        Implication(r"^(?:is there|do i have|have i got) (?:a |an |any )?(?P<obj>[\w.@ -]{2,40})\s*\??$", "question", "find", 0.65, "pattern"),
        Implication(r"\b(?:my |the )?(?P<obj>[\w.@ -]{2,40}?) (?:is|are|looks?) (?:such )?an? (?:" + MESS + r")\b", "complaint", "clarify_goal", 0.6, "place"),
    ]


def act_is_affordable(text: str, frame: Frame) -> bool:
    """May an act read out of a *non-imperative* utterance stand, given what it would act on?

    The affordance gate belongs wherever an act is inferred rather than ordered, not only in
    ``indirect_frame``: the symbolic grammar also turns "I can't find my keys" into a search,
    and the object is what makes that wrong. An imperative passes untouched — "delete keys" is
    an order about a file called keys, and second-guessing an order is not this gate's job.
    """
    words = re.findall(r"[\w'@.-]+", text.strip().lower())
    objects = [v.lower() for v in frame.slots.values() if isinstance(v, str) and v]
    named = [w for w in words if in_domain(w)]  # what this utterance names that I could act on
    if named:  # whatever the shape, a reading has to be *about* what the utterance names
        return any(any(n in o for o in objects) for n in named)
    if not words or words[0] in VERBS or words[0] in ("please", "could", "can", "would", "will", "do"):
        return True  # an order, or a politeness marker in front of one
    # a bare "~" is the parser's default place, not evidence that the utterance is about a file
    probe = [v for v in objects if not v.startswith("~")]
    return not probe or any(in_domain(v) or v.startswith("@") for v in probe)


def indirect_frame(text: str) -> Frame | None:
    """Read a request out of a form that is not an order, or leave it alone.

    Two shapes: a wish whose embedded proposition is an order in disguise ("it would be good
    if you deleted notes.txt"), which is re-read as that order; and a complaint or question
    that implies looking ("I can't find my invoice"). Both are refused when the object is not
    mine to act on, which is what keeps "I can't find my keys" a remark.
    """
    from tensorcode.social import Indirect, indirect_reading

    stripped = text.strip()
    if m := WISH.match(stripped):
        inner = m[1].strip(" .!?")
        first, _, rest = inner.partition(" ")
        order = f"{AS_ORDER.get(first.lower(), first)} {rest}".strip()
        frame = parse_clause(order)
        if frame.act != "unknown" and frame.slots:
            values = [v for v in frame.slots.values() if isinstance(v, str)]
            if any(in_domain(v) or v.startswith("@") for v in values):
                return Frame(frame.act, stripped, frame.slots)
        return None
    reading = indirect_reading(stripped, _implications(), in_domain=in_domain)
    if isinstance(reading, Indirect):
        slots = dict(reading.slots)
        if reading.act == "clarify_goal":  # a mess in a *place*: name the place the way every other act does
            said = str(slots.get("place", "")).strip().lower()
            bare = re.sub(r"\s+(?:folder|directory|dir)$", "", said)  # "my downloads folder" is ~/Downloads
            slots = {"place": PLACES.get(said) or PLACES.get(bare) or slots.get("place")}
        return Frame(reading.act, stripped, {k: v for k, v in slots.items() if v})
    return None


TREAT_LONG_AS_TASK = False  # long-horizon mode: a long, multi-line message is one piece of work


def parse_message(text: str) -> list[Frame]:
    if TREAT_LONG_AS_TASK and (text.count("\n") >= 2 or len(text) > 400):
        return [Frame("unknown", text.strip())]  # a brief, not a chat line: keep it whole
    if re.search(r"\bset ?up (?:a |the |my )?(?:new )?project\b|\bnew project\b", text, re.I):
        return [Frame("setup_project", text, {"note": text})]  # a whole note, not separate clauses
    if m := re.match(r"^\s*(yes|yeah|yep|sure|ok(?:ay)?|no|nope|nah)\s*[,.!]\s+\S", text, re.I):
        return [Frame("cancel" if m[1].lower() in ("no", "nope", "nah") else "confirm", text.strip())]  # "yes, delete it" answers the question
    stripped = text.strip().rstrip(".!?")
    if re.fullmatch(r"~?/?[\w@+-][\w.@+/-]*\.[A-Za-z0-9]{1,8}|~(?:/[\w.@+-]+)*|/[\w.@+/-]+", stripped):
        return [Frame("read", text.strip(), {"target": stripped})]  # a bare path or filename: show it
    if len(stripped.split()) <= 7 and _CHOICE.fullmatch(stripped):
        return [Frame("choose", text.strip())]  # an answer to "which one?"
    frames: list[Frame] = []
    for clause in split_clauses(fix_typos(text)):
        frames.extend(_expand(clause))
    return frames or [Frame("unknown", text)]


def _expand(clause: str) -> list[Frame]:
    """'create folders x, y and z' -> one request per name."""
    masked, _ = _mask(clause)
    raw = list(_mask.raw)  # type: ignore[attr-defined]
    m = re.match(r"^(.*?\b(?:make|create|add|new|mkdir)\s+(?:(?:some|two|three|four|\d+)\s+)?(?:new\s+)?)(folders|files|directories)\s+(?:called\s+|named\s+)?(.+?)(\s+(?:on|in|into|inside|under)\s+.+)?$", masked, re.I)
    if m:
        names = [n for n in re.split(r"\s*,\s*(?:and\s+)?|\s+and\s+", m[3].strip()) if n]
        if len(names) >= 2:
            kind = {"folders": "folder", "files": "file", "directories": "folder"}[m[2].lower()]
            unmask = lambda t: re.sub(r"QUOTE(\d+)", lambda q: raw[int(q[1])], t)  # noqa: E731
            return [parse_clause(unmask(f"{m[1]}a {kind} called {n}{m[4] or ''}")) for n in names]
    return [parse_clause(clause)]


def parse_clause(clause: str) -> Frame:
    words = clause
    s, quoted = _mask(FILLER.sub("", clause.strip()))
    s = re.sub(r"\s+", " ", s).strip()
    low = s.lower().replace("’", "'")
    unq = lambda v: _unmask(v, quoted) if v else v  # noqa: E731

    def frame(act: str, **slots) -> Frame:
        # any placeholder that survived slot extraction (e.g. '@QUOTE0') becomes the quoted text
        return Frame(act, words, {k: _unmask(v, quoted) if isinstance(v, str) else v for k, v in slots.items() if v is not None})

    # dialogue
    if re.fullmatch(r"(?:y|yes|yeah|yep|sure|ok|okay|do it|go ahead|confirm(?:ed)?|please do|yes please)[.!]*", low):
        return frame("confirm")
    if re.fullmatch(r"(?:n|no|nope|nah|cancel|stop|don't|do not|never ?mind|abort)[.!]*", low):
        return frame("cancel")
    if re.fullmatch(r"(?:hi|hello|hey|yo|howdy|hiya|good (?:morning|afternoon|evening))(?: there| again| friend)?[.!]*", low) or not low:
        return frame("greet")
    if re.fullmatch(r"(?:thanks|thank you|thx|ty|cool|great|nice|perfect|awesome)(?: you)?[.!]*", low):
        return frame("thanks")
    if re.search(r"^(?:help|what can you do|what do you do|commands|what are your (?:skills|commands))\b", low):
        return frame("help")

    # things typed like shell commands map to the matching request (so they get the same checks)
    if m := re.fullmatch(r"(ls|cat|less|more|head|mkdir|rm|rmdir|touch|cd|mv|cp|pwd)((?:\s+-[a-zA-Z]+)*)(?:\s+(.+))?", s):
        cmd, args = m[1], [unq(a) for a in re.findall(r"QUOTE\d+|\S+", m[3] or "")]
        if cmd == "pwd" and not args:
            return frame("info", topic="cwd")
        if cmd == "ls" and len(args) <= 1:
            return frame("list", place=args[0] if args else None)
        if cmd in ("cat", "less", "more", "head") and len(args) == 1:
            return frame("read", target=args[0])
        if cmd == "mkdir" and len(args) == 1:
            return frame("create_folder", name=args[0])
        if cmd == "touch" and len(args) == 1:
            return frame("create_file", name=args[0])
        if cmd in ("rm", "rmdir") and len(args) == 1:
            return frame("delete", target=args[0])
        if cmd == "cd" and len(args) <= 1:
            return frame("cd", target=args[0] if args else "~")
        if cmd in ("mv", "cp") and len(args) == 2:
            src, dst = args
            if cmd == "mv" and "/" not in dst and not dst.startswith("~") and dst.lower() not in PLACES and "." in dst:
                return frame("rename", target=src, new_name=dst)
            return frame("copy" if cmd == "cp" else "move", target=src, dest=PLACES.get(dst.lower(), dst))

    # raw commands: `...`, $ ..., run: ...
    if clause.count("`") >= 2 and (m := re.fullmatch(r"(?:(?:run|execute|exec|type)\s+(?:the command\s+)?)?(?:in the terminal\s+)?QUOTE(\d+)(?:\s+in the terminal)?", low, re.I)):
        return frame("run", command=quoted[int(m[1])])
    if m := re.match(r"^(?:\$|run:|execute:|sudo\s)\s*(.+)$", clause.strip()):
        return frame("run", command=(("sudo " if clause.strip().startswith("sudo") else "") + m[1]).strip())

    # search inside files
    m = re.search(r"\b(?:search|look|grep|find|check)\b(?:\s+(?:for|through))?(?:\s+(?:mentions? of|occurrences of|the (?:text|word|phrase)))?\s+QUOTE(\d+)()", s, re.I) or re.search(
        r"\b(?:which|what) files (?:mention|contain|say|have|include)\s+(?:QUOTE(\d+)|(?:the (?:word|text)\s+)?([\w.@+-]+))", s, re.I)
    if m:
        needle = quoted[int(m[1])] if m[1] is not None else m[2]
        return frame("grep", needle=needle, place=_place(re.sub(r"QUOTE\d+", " ", s)) or "~")

    # questions about the machine
    if re.search(r"\bwhat(?:'s| is)? (?:the )?(?:time|date|day)\b|\bwhat (?:time|day|date)\b|\b(?:time|date|day) is it\b|\btoday'?s date\b|\bcurrent (?:time|date)\b|^(?:the )?(?:time|date)\??$", low):
        return frame("info", topic="date")
    if re.search(r"\bwho am i\b|\bwhat(?:'s| is) my user(?:name)?\b|\bwhich user\b", low):
        return frame("info", topic="user")
    if re.search(r"\b(?:disk|storage|free space|space left|drive space|how much space)\b", low) and not re.search(r"\b(?:folder|file|take|takes|use|uses)\b", low):
        return frame("info", topic="disk")
    if re.search(r"\bip(?: address)?\b|\bnetwork address\b", low):
        return frame("info", topic="ip")
    if re.search(r"\bhostname\b|\bcomputer(?:'s)? name\b|\bmachine name\b", low):
        return frame("info", topic="hostname")
    if re.search(r"\buptime\b|\bhow long .* (?:up|running|on)\b", low):
        return frame("info", topic="uptime")
    if re.search(r"\b(?:processes|what(?:'s| is|s)? running|running programs|running apps)\b|^top$", low):
        return frame("info", topic="processes")
    if re.search(r"\b(?:cpus?|cores|processors?)\b", low):
        return frame("info", topic="cpus")
    if re.search(r"\b(?:os|operating system|kernel|ubuntu version|version of ubuntu|system info)\b", low):
        return frame("info", topic="os")
    if re.search(r"\bwhere am i\b|\bcurrent (?:folder|directory)\b|\bpwd\b", low):
        return frame("info", topic="cwd")
    if m := re.search(r"\bis ([\w.+-]+) installed\b|\bdo (?:i|we) have ([\w.+-]+) installed\b|^which ([\w.+-]+)\??$|\bwhere is ([\w.+-]+) installed\b", low):
        return frame("which", program=next(g for g in m.groups() if g))

    # a goal that is understood but underdetermined: which way is missing, not what
    if re.match(r"^(?:organi[sz]e|tidy|clean up|sort out|sort|straighten up)\b", low) and not re.search(r"\bby (?:type|kind|name|date|month)\b", low):
        # a named place only: "tidy things up" names nothing, and guessing a folder to rearrange is not on
        obj = re.sub(r"^(?:organi[sz]e|tidy|clean|sort|straighten)(?:\s+(?:up|out))?\s*", "", s, flags=re.I).strip(" .!?")
        obj = re.sub(r"^(?:my|the|our)\s+", "", obj, flags=re.I)  # keep the case: a path is case-sensitive
        where = _place(s) or PLACES.get(obj.lower()) or (obj if re.match(r"^~?/|^[\w.-]+/", obj) else None)
        if where:
            return frame("clarify_goal", place=where)

    # ------------------------------------------------- being told, and asked
    # A question is not a command. These acts answer from memory, from what is on screen,
    # or from pixels — and each names the modality it needs, so an unanswerable one can say so.

    # something you are telling me about yourself or the world
    if m := re.match(r"^(?:remember(?: that)?|note(?: that)?|fyi|for the record)[,:]?\s+(.+)$", s, re.I):
        rest = m[1]
        if t := re.match(r"^(?:my|our)\s+([\w' -]+?)\s+(?:is|are|=)\s+(.+)$", rest, re.I):
            return frame("tell", topic=t[1].strip().lower(), value=unq(t[2]).strip(" .!"))
        return frame("tell", topic="note", value=unq(rest).strip(" .!"))
    if m := re.match(r"^(?:my|our)\s+([\w' -]+?)\s+(?:is|are|=)\s+(.+)$", s, re.I):
        topic, value = m[1].strip().lower(), unq(m[2]).strip(" .!")
        complaint = in_domain(topic) and re.match(r"^(?:such )?an? (?:" + MESS + r")$", value, re.I)
        if topic not in ("ip", "ip address", "username", "user name", "hostname", "computer name", "machine name") and not complaint:
            return frame("tell", topic=topic, value=value)
    if m := re.match(r"^(?:i am|i'm|im)\s+(?:called\s+)?([A-Z][\w-]*)$", s):
        return frame("tell", topic="name", value=m[1])
    if m := re.match(r"^(?:call me|you can call me)\s+(.+)$", s, re.I):
        return frame("tell", topic="name", value=unq(m[1]).strip(" .!"))

    # what did you learn about me / what is my X
    if re.search(r"\bwhat (?:did|have) i (?:tell|told|said to) you\b|\bwhat do you (?:know|remember) about me\b|\bwhat do you remember\b", low):
        return frame("ask_memory")
    if m := re.search(r"\b(?:what(?:'s| is| was)|do you (?:know|remember)|tell me)\s+(?:my|our)\s+([\w' -]+?)\s*\??$", s, re.I):
        return frame("ask_memory", topic=m[1].strip().lower())
    if m := re.match(r"^(?:forget|drop|unlearn)\s+(?:that|my|about my)?\s*([\w' -]*?)\s*\??$", low):
        return frame("forget", topic=(m[1].strip() or None))

    # what I did, and why
    if re.search(r"\bwhat (?:did|have) you (?:just )?(?:do|done)\b|\bwhat did you do (?:just now|last)\b", low):
        return frame("ask_self", aspect="last_action")
    if re.search(r"\bwhy did you (?:do|run|type|click)\b|\bwhy (?:that|did you)\b", low):
        return frame("ask_self", aspect="why")

    # what is on the screen, and how it looks
    if (re.search(r"\b(?:sidebar|side bar|dock|launcher|taskbar)\b", low)
            and not re.search(r"\b(?:folder|file|directory)\b", low)
            and not re.match(r"^(?:delete|remove|trash|erase|rm|make|create|move|copy|rename|write|add|put)\b", low)):
        return frame("ask_screen", aspect="icons")
    if re.search(r"\bhow many (?:icons|apps|applications|buttons) \b", low):
        return frame("ask_screen", aspect="icons")
    if re.search(r"\bwhat(?:'s|s| is)? (?:in|on) (?:the )?(?:sidebar|dock|launcher|taskbar)\b|\bwhat icons\b", low):
        return frame("ask_screen", aspect="icons")
    if m := re.search(r"\bwhat(?:'s|s| is)?\s+(?:in|inside|on)\s+(?:the\s+)?([\w .-]+?)\s+window\b", low):
        return frame("ask_screen", aspect="window", window=m[1].strip())
    if m := re.search(r"\bwhat(?: is|'s|s)?\s+(?:the\s+)?([\w.-]+)\s+(?:showing|saying|shows|says)\b", low):
        return frame("ask_screen", aspect="window", window=m[1].strip())
    if re.search(r"\bwhat changed\b.*\b(?:screen|since|last message|last time)\b|\bwhat(?:'s|s| is) (?:new|different) on (?:the )?screen\b", low):
        return frame("ask_screen", aspect="changed")
    if re.search(r"\bhow many windows\b", low):
        return frame("ask_screen", aspect="windows")
    if m := re.search(r"\bis ([\w .-]+?) (?:open|running|up)\b", low):
        return frame("ask_screen", aspect="is_open", window=m[1].strip())
    if re.search(r"\bwhich (?:apps|applications|programs|windows) are open\b|\bwhat (?:apps|applications|programs|windows) (?:are|is) open\b|\bwhat(?:'s|s| is) open\b", low):
        return frame("ask_screen", aspect="windows")
    if re.search(r"\bwhat(?:'s|s| is)? on (?:the |my )?(?:screen|display|desktop screen)\b|\bwhat do you see\b|\bdescribe (?:the )?screen\b|\bwhat does the screen (?:look like|show)\b", low):
        return frame("ask_screen", aspect="all")
    if m := re.search(r"\bwhat colou?r (?:is|are)\s+(?:the\s+)?([\w ]+?)\s*\??$", low):
        return frame("ask_pixels", aspect="color", region=m[1].strip())
    if re.search(r"\bwhat colou?r\b", low):
        return frame("ask_pixels", aspect="color", region="screen")
    if m := re.search(r"\bis (?:the |my )?([\w ]+?) (?:dark or light|light or dark|dark|light)\b", low):
        return frame("ask_pixels", aspect="color", region=m[1].strip())

    # what I asked for earlier: the conversation itself
    if re.search(r"\bwhat did i (?:ask|say|tell you to do)\b|\bwhat have i asked\b|\bwhat did i want\b", low):
        return frame("ask_self", aspect="history")

    # an app named by what it is for, rather than by its name
    if m := re.match(r"^(?:open|launch|start|run|use)\s+(?:the\s+|a\s+|an\s+|some\s+)?(?:app|application|program|thing|something|editor|tool)\b(?:\s+(?:that\s+(?:is\s+)?|to\s+|for\s+|which\s+(?:is\s+)?|used\s+(?:for|to)\s+))?(.*)$", low):
        what = re.sub(r"^(?:used\s+)?(?:for|to)\s+", "", m[1].strip(" ?.")).strip()
        if what:
            return frame("open_function", function=what)
    if m := re.match(r"^(?:which|what) (?:app|application|program) (?:do i|should i|would i|can i)?\s*(?:use\s+)?(?:for|to)\s+(.+?)\s*\??$", low):
        return frame("open_function", function=m[1].strip(), ask_only=True)

    # apps
    if m := re.match(r"^(?:open|launch|start|run|bring up|show)\s+(?:up\s+)?(?:the\s+|a\s+|my\s+)?(.+?)(?:\s+app(?:lication)?)?$", low):
        target = m[1].strip(" .")
        if target in APPS:
            return frame("open_app", app=APPS[target])

    # git
    if re.search(r"\b(?:git init|initiali[sz]e (?:a )?git|make (?:it|this|that|.+?) (?:a|into a) (?:git )?repo(?:sitory)?|turn .+ into a (?:git )?repo|init(?:ialize)? (?:a )?repo)", low):
        return frame("git_init", target=_git_target(s, quoted))
    if re.search(r"\bgit log\b|\b(?:commit )?history\b|\b(?:recent |last |the )?commits\b|\blog of\b", low):
        return frame("git_log", target=_git_target(s, quoted))
    if re.search(r"\bcommit\b", low):
        m = re.search(r"\b(?:with (?:the )?message|message|saying|as|called|:)\s+(QUOTE\d+)", s, re.I) or re.search(r"(QUOTE\d+)", s)
        return frame("git_commit", target=_git_target(s, quoted), message=unq(m[1]) if m else None)
    if re.search(r"\bgit status\b|\bwhat(?:'s| has| is)? changed\b|\bstatus of (?:the )?(?:repo|git)\b|\buncommitted\b", low):
        return frame("git_status", target=_git_target(s, quoted))

    # install
    if m := re.search(r"\binstall\s+([\w.+-]+)", low):
        return frame("install", package=m[1])

    # go to / cd
    if re.fullmatch(r"(?:go|cd)\s+(?:back(?: home)?|home|to (?:my )?home(?: folder| directory)?)", low):
        return frame("cd", target="~")
    if re.match(r"^(?:go|cd|switch|change|move)\s+(?:(?:in)?to|over to|directory to|folder to)?\s*", low) and re.match(r"^(?:go to|go into|cd\b|switch to|change (?:dir(?:ectory)? |folder )?to|move into)", low):
        return frame("cd", target=_place(s) or _target(s, quoted))


    # find files by name
    if re.search(r"^(?:find|locate|where(?:'s| is)|search for|look for)\b", low):
        body = re.sub(r"^(?:find|locate|where(?:'s| is)|search for|look for)\s+(?:all\s+|any\s+|the\s+|my\s+)?(?:files?|folders?|directories)?\s*(?:named|called|matching|with names? like|ending in|ending with)?\s*", "", s, flags=re.I)
        m = re.search(r"QUOTE(\d+)|(\*?\.?[\w@+-]+(?:\.[\w*]+)?\*?)", body)
        if m:
            name = quoted[int(m[1])] if m[1] is not None else m[2]
            if re.search(r"\bending (?:in|with)\b", low) and not name.startswith("*"):
                name = "*" + name
            if re.fullmatch(r"(?:pdfs?|txt|markdown|md|images?|pngs?|jpe?gs?|python|py)", name.lower()):
                name = {"pdf": "*.pdf", "pdfs": "*.pdf", "txt": "*.txt", "markdown": "*.md", "md": "*.md", "python": "*.py", "py": "*.py", "png": "*.png", "pngs": "*.png", "jpg": "*.jpg", "jpeg": "*.jpg", "jpgs": "*.jpg", "image": "*.png", "images": "*.png"}.get(name.lower(), name)
            return frame("find", pattern=name, place=_place(s) or "~")

    # counting / size
    if re.search(r"\bhow many (?:lines|words)\b|\bcount (?:the )?(?:lines|words)\b|\bline count\b|\bword count\b", low):
        return frame("count", unit="words" if "word" in low else "lines", target=_target(s, quoted))
    if re.search(r"\bhow (?:big|large)\b|\bsize of\b|\bhow much space does\b|\bhow much (?:room|space) (?:is|does)\b", low):
        return frame("size", target=_target(s, quoted))
    if re.search(r"\bhow many (?:files|items|things|folders)\b", low):
        return frame("list", place=_place(s) or _target(s, quoted) or "@it", count=True)

    # delete
    if m := re.match(r"^(?:delete|remove|trash|erase|get rid of|throw away|throw out)\s+(.+)$", s, re.I):
        obj = re.split(r"\s+(?:from|on|in|inside|under|off)\s+", m[1], maxsplit=1, flags=re.I)[0]
        clear = re.fullmatch(r"(?:QUOTE\d+|~?/\S*|(?:the |my )?\S+\.[A-Za-z0-9]{1,8}|it|that|this|them|those|these|(?:that|this|the) (?:file|folder|one|directory)|everything|all(?: (?:the|of the|my))? (?:files|things|stuff|items)|all of it"
                             r"|(?:the |my )?\S+ (?:folder|directory|dir|file)|(?:the |my )?(?:folder|directory|file) (?:called |named )?\S+)", obj.strip(), re.I)
        if not clear and re.match(r"^(?:delete|remove|trash|erase)\s", s, re.I) and re.fullmatch(r"[\w@+-][\w.@+-]*", obj.strip()) and obj.strip().lower() not in STOP | {"everything", "all", "stuff", "things", "noise"}:
            return frame("delete", target=obj.strip())  # "delete garden": one explicit name; the program resolves it and asks first
        target = _target(m[1], quoted) if clear else None
        if target:
            return frame("delete", target=target)
        return Frame("unknown", words)

    # move / rename / copy
    if m := re.match(r"^(?:rename)\s+(.+?)\s+(?:to|as|into)\s+(.+)$", s, re.I):
        if (t := _target(m[1], quoted)) is None:
            return Frame("unknown", words)
        return frame("rename", target=t, new_name=unq(m[2].strip(" .")).strip("'\""))
    if m := re.match(r"^(move|mv|copy|cp|duplicate)\s+(.+?)\s+(?:to|into|in|onto|over to)\s+(.+)$", s, re.I):
        dest = _place("to " + m[3]) or _target(m[3], quoted)
        if _target(m[2], quoted) is None or dest is None:
            return Frame("unknown", words)
        return frame("copy" if m[1].lower() in ("copy", "cp", "duplicate") else "move", target=_target(m[2], quoted), dest=dest)

    # write text into a file
    if m := re.match(r"^(write|put|add|append|save|type)\s+(.+?)\s+(?:to|into|in|at the end of|onto)\s+(.+)$", s, re.I):
        what = m[2]
        if n := re.fullmatch(r"(?:a |an |the )?(?:new )?(?:text )?file (?:called |named )?(QUOTE\d+|\S+)((?:\s+(?:on|in|into|inside|under)\s+.+)?)", m[3].strip(), re.I):
            if not re.fullmatch(r"(?:a |an |the )?(?:new |empty |blank )?(?:file|folder|directory|dir)\b.*", what, re.I):
                return frame("create_file", name=unq(n[1]), text=unq(what).strip(), place=_place(n[2]) if n[2] else None)
        if re.fullmatch(r"(?:a |an |the )?(?:new |empty |blank )?(?:file|folder|directory|dir)\b.*", what, re.I):
            pass  # "add a file to X" is creation, below
        else:
            text = unq(what).strip()
            text = re.sub(r"^(?:the (?:text|line|words?)|a line saying|a line|a note saying|a note)\s+", "", text, flags=re.I)
            return frame("write", text=text, target=_target(m[3], quoted), append=m[1].lower() in ("add", "append") or "end of" in low)

    # create
    if re.match(r"^(?:make|create|add|new|touch|mkdir|start|put|place|drop)\b", low):
        folder = re.search(r"\b(?:folder|directory|dir|mkdir)\b", low)
        file_ = re.search(r"\b(?:file|note|document|doc|touch|readme)\b|\.\w{1,8}\b", low)
        kind = "folder" if folder and (not file_ or folder.start() < file_.start()) else "file" if file_ else None
        if kind is None and (n := _name(s, "x")) is None:
            m = PATH.search(s)
            kind = "file" if m and "." in m[1].split("/")[-1] else "folder" if m else None
        if kind:
            name = _name(s, "folder|directory|dir" if kind == "folder" else "file|note|document|doc")
            if name is None and (m := PATH.search(re.sub(r"\b(?:on|in|into|inside|under|within|to)\s+(?:my|the)?\s*\w+(?:\s+folder)?\b", " ", s))):
                name = m[1]
            if name is None and kind == "file" and re.search(r"\breadme\b", low):
                name = "README.md"
            return frame("create_" + kind, name=unq(name) if name else None, place=_place(s), text=_text(s, quoted))

    # "read notes", "cat notes": a bare name to read (the program lists it instead if it is a folder)
    if m := re.match(r"^(?:read|cat|view)\s+(?:the\s+|my\s+)?([\w@+-][\w.@+-]*)\s*\??$", low):
        if m[1] not in STOP | BARE_PLACE_STOP and not re.fullmatch(r"quote\d+", m[1]):
            return frame("read", target=re.search(re.escape(m[1]), s, re.I)[0])

    # "what's in recipes", "list recipes", "show recipes": a bare name is a thing known by name
    if m := re.match(r"^(?:list|ls|show(?: me)?|open|view|(?:show me |tell me )?what(?:'s| is| are)?(?: in| inside)|(?:show me |tell me )?whats in|look (?:in|inside)|what do i have in)\s+(?:the\s+|my\s+)?([\w@+-][\w.@+-]*)\s*\??$", low):
        word = m[1]
        if word not in STOP and word not in PLACES and word not in ("it", "that", "this", "there", "everything", "files", "folders", "stuff", "me") and not re.fullmatch(r"(?:un)?(?:known)|up|out|quote\d+", word):
            original = re.search(re.escape(word), s, re.I)[0]
            return frame("read", target=original) if re.search(r"\.[A-Za-z0-9]{1,8}$", word) else frame("list", place=f"@{original}")

    # show / read / list
    if re.match(r"^(?:show|read|open|cat|display|print|list|ls|what(?:'s| is| are)?|whats|tell me|view|see|check|look at|look in|look inside|go through)\b", low):
        target = _target(re.sub(r"^\S+\s+(?:me\s+)?(?:what(?:'s| is)\s+)?", "", s, flags=re.I), quoted)
        place = _place(s)
        says_list = re.search(r"\b(?:list|ls|files|folders|contents of (?:the |my )?(?:folder|directory|desktop|documents|downloads|home)|what(?:'s| is) (?:in|on|inside))\b", low)
        is_file = target and target != "@it" and re.search(r"\.[A-Za-z0-9]{1,8}$", target)
        if is_file or re.search(r"\b(?:read|cat|contents? of (?:the )?file|what does .+ say|in the file)\b", low):
            return frame("read", target=target if target and not target.startswith("~/") or is_file else target, place=place)
        if says_list or place or target:
            return frame("list", place=place or target or "~")
    return Frame("unknown", words)


def _git_target(s: str, quoted: list[str]) -> str:
    if place := _place(re.sub(r"QUOTE\d+", " ", s)):
        return place
    m = re.search(r"\b(?:make|turn|init(?:iali[sz]e)?(?: git in)?)\s+(?:the\s+|my\s+)?([\w.@+~/-]+)\s+(?:folder\s+)?(?:a|into|as)\b", s, re.I) or re.search(
        r"\b(?:in|of|for|inside)\s+(?:the\s+|my\s+)?([\w.@+~/-]+)(?:\s+(?:folder|repo|repository|project|directory))?", re.sub(r"QUOTE\d+", " ", s), re.I)
    if m and m[1].lower() not in STOP | {"it", "this", "that", "here", "git", "repo"}:
        return m[1] if "/" in m[1] or m[1].startswith("~") else f"@{m[1]}"
    return "@it"
