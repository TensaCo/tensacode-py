"""The label space the learned parser predicts: the assistant's own acts and slots.

Taken from ``examples/browser_agents/assistant/procedures.py`` (the acts a procedure
exists for) and the slot names those procedures read. Shared by data generation,
training, evaluation and the runtime implementation so the four cannot drift.
"""

from __future__ import annotations

#: every act the assistant has a procedure for, plus the honest catch-all
ACTS: tuple[str, ...] = (
    "unknown",
    # dialogue
    "greet", "thanks", "help", "confirm", "cancel", "choose", "clarify_goal",
    # conversation memory
    "tell", "ask_memory", "forget",
    # questions about what it perceives, itself, and pixels
    "ask_screen", "ask_pixels", "ask_self",
    # files and folders
    "list", "read", "create_folder", "create_file", "write", "delete", "move", "copy", "rename",
    # looking things up
    "find", "grep", "count", "size", "info", "which",
    # the machine
    "cd", "open_app", "open_function", "run", "install",
    # git and projects
    "git_init", "git_commit", "git_status", "git_log", "setup_project",
)
ACT_INDEX = {a: i for i, a in enumerate(ACTS)}

#: slots whose value is a span of the utterance (recovered by character offsets)
SPAN_SLOTS: tuple[str, ...] = (
    "target", "name", "text", "dest", "new_name", "pattern", "needle", "command",
    "package", "program", "app", "message", "topic", "value", "function", "region", "place",
)
SPAN_INDEX = {s: i for i, s in enumerate(SPAN_SLOTS)}

#: BIO tag set over the span slots
TAGS: tuple[str, ...] = ("O",) + tuple(f"{p}-{s}" for s in SPAN_SLOTS for p in ("B", "I"))
TAG_INDEX = {t: i for i, t in enumerate(TAGS)}

#: a place is either a span of the utterance, one of the named places, or the thing last touched
PLACE_KINDS: tuple[str, ...] = (
    "none", "span", "~", "~/Desktop", "~/Documents", "~/Downloads", "~/Projects",
    "~/Pictures", "~/Music", "~/Videos", "/tmp", "@it",
)
PLACE_INDEX = {p: i for i, p in enumerate(PLACE_KINDS)}

#: the machine questions `info` answers
INFO_TOPICS: tuple[str, ...] = ("none", "date", "user", "disk", "ip", "hostname", "uptime", "processes", "cpus", "os", "cwd")
INFO_INDEX = {t: i for i, t in enumerate(INFO_TOPICS)}

UNITS: tuple[str, ...] = ("none", "lines", "words")
UNIT_INDEX = {u: i for i, u in enumerate(UNITS)}

#: which facet of the screen or of itself a question asks about
ASPECTS: tuple[str, ...] = ("none", "icons", "windows", "history", "why")
ASPECT_INDEX = {a: i for i, a in enumerate(ASPECTS)}

#: boolean slots, predicted by their own heads
FLAGS: tuple[str, ...] = ("append", "count", "ask_only")
FLAG_INDEX = {f: i for i, f in enumerate(FLAGS)}

#: what kind of speech act the utterance is, which decides whether it is ours to act on at all.
#: `world_statement` is the one that matters: "Nise is hungry" is a fact about the world, not a
#: request, and reading it as an act is a wrong action rather than a missed answer.
SPEECH_ACTS: tuple[str, ...] = ("command", "question", "self_disclosure", "world_statement", "other")
SPEECH_ACT_INDEX = {a: i for i, a in enumerate(SPEECH_ACTS)}

#: what the thing acted on is, when the utterance does not spell it out ("read it")
TARGET_KINDS: tuple[str, ...] = ("none", "span", "@it")
#: names the assistant supplies by convention rather than reading them off the utterance
NAME_CANON: tuple[str, ...] = ("none", "README.md")

#: heads other than the act classifier and the tagger: (name, vocabulary)
CLOSED_HEADS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("place_kind", PLACE_KINDS), ("info_topic", INFO_TOPICS), ("unit", UNITS), ("aspect", ASPECTS),
    ("target_kind", TARGET_KINDS), ("name_canon", NAME_CANON), ("speech_act", SPEECH_ACTS),
)

#: words that name a place, and the path they mean (the deterministic half of `place`)
PLACE_WORDS: dict[str, str] = {
    "desktop": "~/Desktop", "documents": "~/Documents", "docs": "~/Documents", "downloads": "~/Downloads",
    "projects": "~/Projects", "pictures": "~/Pictures", "photos": "~/Pictures", "music": "~/Music",
    "videos": "~/Videos", "movies": "~/Videos", "home": "~", "home folder": "~", "home directory": "~",
    "tmp": "/tmp", "temp": "/tmp",
}


_WH = r"(?:what|which|who|whom|whose|where|when|why|how)"
_AUX = r"(?:is|are|was|were|do|does|did|can|could|will|would|should|have|has|am)"


#: slots whose value is the entire utterance rather than a span inside it. The procedure for
#: setup_project reads the whole request back out, so there is nothing for a tagger to find:
#: the model only has to get the act right and the decoder supplies the text.
WHOLE_INPUT_SLOTS: frozenset[str] = frozenset({"note"})


def speech_act_of(text: str, act: str) -> str:
    """Read the speech act off the surface form, the way a listener does.

    Deliberately a function of shape rather than of vocabulary, so it carries to words the
    training data never contained: a third-person subject followed by a copula or a plain
    present-tense verb is a statement about the world whoever the subject is.
    """
    import re

    stripped = text.strip().rstrip(".!").strip()
    low = stripped.lower()
    if act in ("greet", "thanks", "confirm", "cancel", "choose", "help"):
        return "other"
    if act == "clarify_goal":
        # "my desktop is a mess" opens with "my" but asks for work; it is not self-disclosure
        return "command"
    if text.strip().endswith("?") or re.match(rf"^(?:{_WH}\b|{_AUX}\b)", low) or re.match(r"^(?:hey |hi, |ok |so |btw |quick question, |just wondering, )*(?:" + _WH + r"|" + _AUX + r")\b", low):
        return "question"
    filler = r"(?:just so you know, |for the record,? |btw |by the way, |fyi |hey |hi, |ok |so )*"
    if re.match(rf"^{filler}my\b", low) or re.match(rf"^{filler}i(?:'m| am| was|'ve)\b", low):
        return "self_disclosure"
    # hearsay and reporting frames are statements whoever they are about
    if re.match(r"^(?:i heard|i hear|they say|they said|apparently|word is|rumour has it|rumor has it|"
                r"someone said|everyone says|it seems|supposedly)\b", low):
        return "world_statement"
    words = stripped.split()
    if words and words[0].lower().strip(",") not in COMMAND_VERBS and re.search(r"\b(?:said|told me|says|mentioned)\b", low):
        return "world_statement"
    # a statement about the world: something that is not an order named first, then a verb of
    # state or happening. "wood is cheap" and "Coralin holds much food" qualify; "delete it"
    # does not, because "delete" is an order.
    if len(words) >= 2 and words[0].lower().strip(",") not in COMMAND_VERBS and words[1].lower() in STATEMENT_VERBS:
        return "world_statement"
    # a determiner-led subject can be several words long: "the north field failed"
    if len(words) >= 3 and words[0].lower() in ("the", "a", "an", "our", "their", "his", "her", "this", "that"):
        if any(w.lower() in STATEMENT_VERBS for w in words[1:5]):
            return "world_statement"
    return "command"


#: verbs that open an order, so the utterance is a request rather than a remark
COMMAND_VERBS: frozenset[str] = frozenset({
    "make", "create", "add", "new", "touch", "mkdir", "write", "put", "append", "save", "show", "read", "open",
    "cat", "display", "print", "list", "ls", "delete", "remove", "trash", "erase", "rm", "move", "mv", "rename",
    "copy", "cp", "duplicate", "find", "search", "look", "locate", "grep", "count", "install", "launch", "start",
    "run", "go", "cd", "switch", "change", "initialize", "init", "commit", "check", "tell", "forget", "drop",
    "wipe", "view", "call", "turn", "apt", "execute", "exec", "type", "place", "set", "get", "give", "send",
    "please", "pls", "help",
})

#: verbs that report a state or a happening rather than ask for one
STATEMENT_VERBS: frozenset[str] = frozenset({
    "is", "are", "was", "were", "isn't", "aren't", "has", "have", "had", "holds", "held", "owns", "owes",
    "wants", "needs", "comes", "came", "goes", "went", "failed", "fails", "burned", "burns", "died", "dies",
    "arrived", "arrives", "mentions", "mentioned", "says", "said", "told", "grows", "grew", "costs", "cost",
    "works", "worked", "broke", "broken", "looks", "seems", "feels", "lives", "sits", "stands", "remains",
})


def slots_of(act: str) -> frozenset[str]:
    """Which slots an act can carry, so training never labels a slot the act has no use for."""
    return _ACT_SLOTS.get(act, frozenset())


_ACT_SLOTS: dict[str, frozenset[str]] = {
    "list": frozenset({"place", "count"}),
    "read": frozenset({"target", "place"}),
    "create_folder": frozenset({"name", "place"}),
    "create_file": frozenset({"name", "place", "text"}),
    "write": frozenset({"target", "text", "append"}),
    "delete": frozenset({"target"}),
    "move": frozenset({"target", "dest"}),
    "copy": frozenset({"target", "dest"}),
    "rename": frozenset({"target", "new_name"}),
    "find": frozenset({"pattern", "place"}),
    "grep": frozenset({"needle", "place"}),
    "count": frozenset({"target", "unit"}),
    "size": frozenset({"target"}),
    "info": frozenset({"topic"}),
    "which": frozenset({"program"}),
    "cd": frozenset({"target"}),
    "open_app": frozenset({"app"}),
    "open_function": frozenset({"function", "ask_only"}),
    "run": frozenset({"command"}),
    "install": frozenset({"package"}),
    "git_init": frozenset({"target"}),
    "git_commit": frozenset({"target", "message"}),
    "git_status": frozenset({"target"}),
    "git_log": frozenset({"target"}),
    "setup_project": frozenset({"note"}),  # "note" is the whole utterance (see WHOLE_INPUT_SLOTS)
    "clarify_goal": frozenset({"place"}),
    "tell": frozenset({"topic", "value"}),
    "ask_memory": frozenset({"topic"}),
    "forget": frozenset({"topic"}),
    "ask_screen": frozenset({"aspect"}),
    "ask_pixels": frozenset({"region"}),
    "ask_self": frozenset({"aspect"}),
}
