"""Training data for the learned request parser, and an honest record of where it came from.

Four sources, each labelled in the manifest so a reader can tell authored data from
harvested data:

* ``template``   — surface templates over the act/slot space, labels exact by construction.
                   Authored by us, so a floor on obvious breakage, never evidence of coverage.
* ``grammar``    — frames sampled from ``tensacode.language`` and realized into English by the
                   same grammar, then projected to acts: distillation of the symbolic parser.
* ``paraphrase`` — a local model rewrites a template utterance; the label survives only if
                   every span value still appears verbatim, otherwise the example is dropped.
* ``negative``   — out-of-domain language (public SQuAD questions, small talk, impossible
                   requests) labelled ``unknown``: what the parser must refuse.

Evaluation sets are never generated here.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .schema import PLACE_WORDS, speech_act_of

Part = str | tuple[str, str]  # literal text, or (slot, value)


@dataclass
class Example:
    text: str
    act: str
    spans: dict[str, list[int]] = field(default_factory=dict)  # slot -> [start, end)
    closed: dict[str, str] = field(default_factory=dict)       # place_kind / info_topic / unit / aspect
    flags: dict[str, bool] = field(default_factory=dict)
    source: str = "template"

    def value(self, slot: str) -> str | None:
        if slot not in self.spans:
            return None
        a, b = self.spans[slot]
        return self.text[a:b]


#: function words that begin a spliced place phrase and must not glue to what precedes them
_GLUE = re.compile(r"^(in|on|into|under|inside)\b")


def build(parts: Sequence[Part], act: str, *, closed: dict | None = None, flags: dict | None = None, source: str = "template") -> Example:
    """Assemble an utterance from literals and (slot, value) pairs, recording exact offsets."""
    text, spans = "", {}
    for part in parts:
        val = part if isinstance(part, str) else part[1]
        # a place phrase spliced straight after a span used to run into it
        # ("...called desktopon ~/Documents"); separate before recording offsets
        if val and text and not text[-1].isspace() and _GLUE.match(val):
            text += " "
        if isinstance(part, str):
            text += part
        else:
            slot, _ = part
            spans[slot] = [len(text), len(text) + len(val)]
            text += val
    closed = dict(closed or {})
    closed.setdefault("speech_act", speech_act_of(text, act))
    return Example(text, act, spans, closed, dict(flags or {}), source)


# --------------------------------------------------------------- value pools

FILES = ["notes.txt", "todo.md", "report.pdf", "shopping.txt", "ideas.md", "chapter1.md", "a.txt", "budget.csv",
         "README.md", "log.txt", "pasta.txt", "recipe.txt", "meeting.md", "draft.txt", "invoice.pdf", "plan.md",
         "scratch.txt", "config.json", "data.csv", "notes-2026.txt", "hello.py", "index.html"]
FOLDERS = ["recipes", "drafts", "final", "garden", "photos2", "taxes", "archive", "scratch", "projects2", "work",
           "music2", "invoices", "backup", "inbox", "trip", "ledger-lab", "pet-tracker", "old-stuff",
           # names that collide with the words for places: a name is a name when nothing points at a place
           "photos", "music", "documents", "downloads", "desktop", "pictures", "temp", "home",
           "foo", "bar", "baz", "stuff", "misc", "2026", "q3", "backups", "notes"]
#: how a vague tidy-up request names the thing that is in a state
MESSY_THINGS = ["desktop", "downloads folder", "documents", "home folder", "pictures", "music folder",
                "projects folder", "inbox", "files", "photos folder", "temp folder", "screen"]
#: the states people complain about
MESS_WORDS = ["a mess", "a disaster", "a nightmare", "chaos", "a disaster zone", "cluttered", "a shambles",
              "out of control", "a pigsty", "unmanageable", "a wreck", "overflowing"]
#: verbs for an underspecified request to put something in order
TIDY_VERBS = ["tidy", "neaten", "declutter", "sort out", "straighten out", "get a handle on", "deal with",
              "do something about", "make sense of", "fix up", "reorganise", "arrange"]

#: names that need quoting because they contain spaces
QUOTED_NAMES = ["Tax 2026", "old stuff", "meeting notes.txt", "my files", "trip photos", "Q3 report.pdf",
                "final draft.md", "band practice", "receipts 2025"]
QUOTED = ["boil water", "buy milk", "it works", "first draft", "hello world", "don't forget the milk", "call Bob",
          "salt the water", "eggs, milk, bread", "meeting at 3", "remember the keys", "初期メモ", "todo: sleep"]
APPS = ["Firefox", "Chromium", "Files", "Terminal", "Text Editor", "Visual Studio Code", "Slack", "Mail",
        "Settings", "System Monitor", "App Center", "Rhythmbox", "Wireshark"]
APP_WORDS = ["firefox", "the browser", "chromium", "files", "the file manager", "the terminal", "a terminal",
             "the text editor", "vs code", "vscode", "slack", "mail", "settings", "the system monitor"]
FUNCTIONS = ["writing code", "browsing the web", "editing text", "sending email", "chatting with my team",
             "playing music", "watching processes", "capturing packets", "managing files", "installing software",
             "running commands", "editing photos"]
PACKAGES = ["cowsay", "htop", "ripgrep", "jq", "tree", "curl", "vim", "tmux", "git-lfs", "fzf"]
PROGRAMS = ["git", "python3", "node", "curl", "rustc", "docker", "make", "ffmpeg"]
COMMANDS = ["ls -la ~", "git status", "df -h", "echo hello", "uname -a", "ps aux | head", "cat /etc/hosts",
            "git log --oneline -n 5", "du -sh ~/Documents", "whoami"]
NEEDLES = ["dns", "dashboard", "budget 2026", "TODO", "password", "invoice", "hello", "error", "deadline"]
PATTERNS = ["*.pdf", "*.md", "*.txt", "report.pdf", "*.py", "notes*", "*.csv", "todo.md"]
PATHS = ["~/Desktop", "~/Documents", "~/Downloads", "~/Projects", "~/Projects/demo", "~/Documents/old",
         "~/Desktop/recipes", "/tmp", "~", "~/Music", "~/Pictures/trips"]
PLACE_PHRASES = list(PLACE_WORDS.items())
PEOPLE_TOPICS = [("name", ["Jacob", "Maya", "Anem", "Sam", "Priya", "Devin"]),
                 ("email", ["jacob@example.com", "maya@seed.local", "sam@corp.example"]),
                 ("job", ["engineer", "a teacher", "a researcher", "a nurse"]),
                 ("birthday", ["March 3rd", "the 14th of July", "2026-01-09"]),
                 ("favourite colour", ["blue", "green", "orange"]),
                 ("phone number", ["555-0134", "07700 900123"]),
                 ("city", ["Austin", "Lisbon", "Nairobi"]),
                 ("dog's name", ["Rufus", "Biscuit"])]
REGIONS = ["display", "screen", "desktop background", "sidebar", "launcher", "title bar", "terminal window",
           "top bar", "wallpaper", "window"]
INFO_PHRASES = [("date", ["what time is it", "what's the time", "what is the date", "what day is it today"]),
                ("user", ["who am i", "what's my username", "which user am i"]),
                ("disk", ["how much disk space is left", "how much free space do i have", "what's my disk usage"]),
                ("ip", ["what's my ip", "what is my ip address", "what's this machine's ip"]),
                ("hostname", ["what's the hostname", "what is this computer called", "what's this machine's name"]),
                ("uptime", ["what's the uptime", "how long has this been running"]),
                ("processes", ["what's running", "what processes are running", "show me running programs"]),
                ("cpus", ["how many cpus are there", "how many cores does this have"]),
                ("os", ["what os is this", "which operating system is this", "what kernel is this"]),
                ("cwd", ["where am i", "what folder am i in", "what's the current directory"])]

POLITE = ["", "", "", "please ", "can you ", "could you ", "would you ", "hey ", "hi, ", "ok ", "i'd like you to ",
          "i want you to ", "can you please ", "pls "]
TAIL = ["", "", "", "", " please", " for me", " thanks", "?", ".", " now"]
#: openers that fit a question rather than an order
ASKING = ["", "", "", "", "hey ", "hi, ", "ok ", "so ", "quick question, ", "just wondering, ", "btw "]
#: openers that fit telling someone something about yourself
ASSERTING = ["", "", "", "just so you know, ", "for the record, ", "btw ", "by the way, ", "fyi "]
#: acts whose surface form is a question or a bare reply
QUESTION_ACTS = frozenset({"greet", "thanks", "help", "confirm", "cancel", "choose", "ask_memory", "ask_screen",
                           "ask_pixels", "ask_self", "info", "which", "count", "size", "list", "grep", "forget"})


# ------------------------------------------------------------------ templates


def _place_part(rng: random.Random) -> tuple[list[Part], dict]:
    """A place phrase, as either a named place, a path span, or a reference to the last thing."""
    roll = rng.random()
    if roll < 0.45:
        word, path = rng.choice(PLACE_PHRASES)
        prep = rng.choice(["on my ", "in my ", "on the ", "in the ", "in ", "on ", "under "])
        return [prep, word], {"place_kind": path}
    if roll < 0.75:
        return [rng.choice(["in ", "on ", "into ", "under "]), ("place", rng.choice(PATHS))], {"place_kind": "span"}
    if roll < 0.9:
        return [rng.choice(["in ", "into ", "inside "]), rng.choice(["it", "there", "that folder"])], {"place_kind": "@it"}
    return [rng.choice(["in the ", "in my "]), ("place", rng.choice(FOLDERS)), " folder"], {"place_kind": "span"}


def generate_templates(n: int, seed: int = 0) -> list[Example]:
    rng = random.Random(seed)
    out: list[Example] = []
    while len(out) < n:
        out.extend(e for e in _one_of_each(rng) if e is not None)
    rng.shuffle(out)
    return out[:n]


def _one_of_each(rng: random.Random) -> list[Example | None]:
    """One example per act family, with independent random surface choices."""
    f, folder, quoted = rng.choice(FILES), rng.choice(FOLDERS), rng.choice(QUOTED)
    q = rng.choice(["'", '"', "“"])
    qc = {"'": "'", '"': '"', "“": "”"}[q]
    place, place_closed = _place_part(rng)
    ex: list[Example | None] = []

    def add(parts: Sequence[Part], act: str, openers: Sequence[str] | None = None, **kw) -> None:
        # a question does not take "i'd like you to"; an imperative does
        pool = openers if openers is not None else (
            ASSERTING if act == "tell" else ASKING if act in QUESTION_ACTS else POLITE)
        ex.append(build([rng.choice(pool), *parts, rng.choice(TAIL)], act, **kw))

    # --- dialogue
    add([rng.choice(["hi", "hello", "hey", "good morning", "yo"])], "greet")
    add([rng.choice(["thanks", "thank you", "cheers", "perfect, thanks", "nice"])], "thanks")
    add([rng.choice(["help", "what can you do", "what are you able to do", "show me what you can do"])], "help")
    add([rng.choice(["yes", "yep", "go ahead", "do it", "sure", "yes please"])], "confirm")
    add([rng.choice(["no", "nope", "cancel", "stop", "never mind", "don't"])], "cancel")
    add([rng.choice(["the first one", "the second one", "2", "number 3", "the last one"])], "choose")

    # --- conversation memory
    topic, values = rng.choice(PEOPLE_TOPICS)
    val = rng.choice(values)
    add([rng.choice(["my ", "my ", "my "]), ("topic", topic),
         rng.choice([" is ", " is ", " will be "]), ("value", val)], "tell")
    add([rng.choice(["what is my ", "what's my ", "do you remember my ", "tell me my "]), ("topic", topic)], "ask_memory")
    add([rng.choice(["what do you know about me", "what have i told you", "what do you remember about me"])], "ask_memory")
    add([rng.choice(["forget my ", "drop my ", "delete my "]), ("topic", topic)], "forget")
    add([rng.choice(["forget everything about me", "forget what i told you", "wipe what you know about me"])], "forget")

    # --- questions about the screen, itself, pixels
    add([rng.choice(["how many icons are in the ", "how many icons are on the ", "count the icons in the "]),
         rng.choice(["sidebar", "launcher", "dock", "left strip"])], "ask_screen", closed={"aspect": "icons"})
    add([rng.choice(["what windows are open", "which windows are open", "how many windows are open",
                     "what apps are open right now"])], "ask_screen", closed={"aspect": "windows"})
    add([rng.choice(["what's on the screen", "what do you see", "what is on screen right now",
                     "describe the screen"])], "ask_screen", closed={"aspect": "none"})
    add([rng.choice(["what color is the ", "what colour is the ", "what does the ", "how does the "]),
         ("region", rng.choice(REGIONS)), rng.choice([" look like", "", ""])], "ask_pixels")
    add([rng.choice(["what did you just do", "what have you done", "what did you do last"])], "ask_self", closed={"aspect": "none"})
    add([rng.choice(["why did you do that", "why did you do it that way", "why"])], "ask_self", closed={"aspect": "why"})
    add([rng.choice(["what have i asked you", "what did i ask you before", "what have i said"])], "ask_self", closed={"aspect": "history"})

    # --- files and folders
    add([rng.choice(["what's ", "whats ", "show me what is ", "list what's "]), *place], "list", closed=place_closed)
    add([rng.choice(["list ", "show me ", "ls "]), ("place", rng.choice(PATHS))], "list", closed={"place_kind": "span"})
    add([rng.choice(["how many files are ", "how many things are ", "count the files "]), *place],
        "list", closed=place_closed, flags={"count": True})
    add([rng.choice(["read ", "show me ", "open ", "cat ", "print ", "view ", "display "]),
         ("target", rng.choice([f, f, rng.choice(PATHS) + "/" + f]))], "read")
    add([rng.choice(["read ", "open ", "show me "]), q, ("target", rng.choice(QUOTED_NAMES)), qc], "read")
    add([rng.choice(["delete ", "remove "]), q, ("target", rng.choice(QUOTED_NAMES)), qc], "delete")
    add([rng.choice(["read ", "show me ", "open "]), ("target", f), *place], "read", closed=place_closed)
    add([rng.choice(["what does ", "what's in "]), ("target", f), rng.choice([" say", " say", ""])], "read")
    add([rng.choice(["read ", "show me "]), rng.choice(["it", "that file", "that"])], "read",
        closed={}, flags={}) if rng.random() < 0.5 else None
    add([rng.choice(["make a folder called ", "create a folder named ", "new folder ", "mkdir ",
                     "make a directory called ", "add a folder named ", "create folder "]),
         ("name", folder), *place], "create_folder", closed=place_closed)
    add([rng.choice(["make a folder called ", "create a new folder named ", "new folder "]),
         q, ("name", rng.choice(QUOTED_NAMES)), qc, *place], "create_folder", closed=place_closed)
    add([rng.choice(["make a folder called ", "new folder ", "mkdir "]), ("name", folder)], "create_folder")
    add([rng.choice(["add a folder called ", "make a folder named "]), ("name", folder),
         rng.choice([" in it", " there", " inside it"])], "create_folder", closed={"place_kind": "@it"})
    add([rng.choice(["make a file called ", "create a file named ", "touch ", "new file ", "create ",
                     "add a file called ", "make an empty file called "]), ("name", f), *place],
        "create_file", closed=place_closed)
    add([rng.choice(["create a file named ", "make a file called "]), q, ("name", rng.choice(QUOTED_NAMES)), qc,
         rng.choice([" saying ", " containing "]), q, ("text", quoted), qc], "create_file")
    add([rng.choice(["put ", "write "]), ("text", quoted), rng.choice([" in a file called ", " into a new file called "]),
         ("name", f)], "create_file")
    add([rng.choice(["make a readme", "create a readme", "add a readme"]),
         rng.choice([" in it", " there", ""])], "create_file", closed={"place_kind": "@it", "name_canon": "README.md"})
    add([rng.choice(["make a file called ", "create "]), ("name", f), *place,
         rng.choice([" saying ", " containing ", " with the text "]), q, ("text", quoted), qc],
        "create_file", closed=place_closed)
    add([rng.choice(["write ", "put "]), q, ("text", quoted), qc, rng.choice([" to ", " into ", " in "]),
         ("target", rng.choice([f, rng.choice(PATHS) + "/" + f]))], "write", flags={"append": False})
    add([rng.choice(["add ", "append "]), q, ("text", quoted), qc, rng.choice([" to ", " to the end of "]), ("target", f)],
        "write", flags={"append": True})
    add([rng.choice(["delete ", "remove ", "get rid of ", "trash ", "rm "]),
         ("target", rng.choice([f, folder, rng.choice(PATHS) + "/" + folder]))], "delete")
    for act, verbs in (("delete", ["delete ", "remove ", "trash ", "get rid of "]),
                       ("read", ["read ", "show me ", "open ", "print "]),
                       ("size", ["how big is ", "what's the size of "]),
                       ("git_status", ["what changed in ", "git status in "]),
                       ("cd", ["go to ", "cd into ", "switch to "])):
        add([rng.choice(verbs), rng.choice(["it", "that", "that file", "that folder", "this one"])], act,
            closed={"target_kind": "@it"})
    add([rng.choice(["add ", "append "]), q, ("text", quoted), qc, rng.choice([" to it", " to that file", " to the end of it"])],
        "write", closed={"target_kind": "@it"}, flags={"append": True})
    add([rng.choice(["write ", "put "]), q, ("text", quoted), qc, rng.choice([" to it", " into that file"])],
        "write", closed={"target_kind": "@it"}, flags={"append": False})
    # the act is the verb the utterance uses: drawing them apart mislabels half the rows
    mc_act = rng.choice(["move", "copy"])
    add([mc_act + " ", rng.choice(["it", "that"]), " to ", ("dest", rng.choice(PATHS))],
        mc_act, closed={"target_kind": "@it"})
    ren_verb, ren_prep = rng.choice([("rename ", " to "), ("call ", " ")])
    add([ren_verb, rng.choice(["it", "that"]), ren_prep, ("new_name", rng.choice(FILES))], "rename",
        closed={"target_kind": "@it"})
    add([rng.choice(["move ", "mv "]), ("target", f), rng.choice([" to ", " into "]), ("dest", rng.choice(PATHS))], "move")
    add([rng.choice(["copy ", "cp ", "duplicate "]), ("target", f), rng.choice([" to ", " into "]), ("dest", rng.choice(PATHS))], "copy")
    ren_verb2, ren_preps = rng.choice([("rename ", [" to ", " as "]), ("call ", [" ", " "])])
    add([ren_verb2, ("target", f), rng.choice(ren_preps), ("new_name", rng.choice(FILES))], "rename")

    # --- an underspecified request: what to do is not stated, so the act is to ask.
    #     The place is canonical (place_kind) exactly as it is for list/read/create, never a raw span,
    #     because the procedure needs a directory it can act on.
    mword, mpath = rng.choice(PLACE_PHRASES)
    add([rng.choice(["my ", "the "]), mword, rng.choice([" is ", " is such ", " has become ", " has turned into "]),
         rng.choice(MESS_WORDS)], "clarify_goal", openers=ASSERTING, closed={"place_kind": mpath})
    mword2, mpath2 = rng.choice(PLACE_PHRASES)
    add([rng.choice(TIDY_VERBS), rng.choice([" my ", " the "]), mword2], "clarify_goal",
        closed={"place_kind": mpath2})
    add([rng.choice(TIDY_VERBS), " ", ("place", rng.choice(PATHS))], "clarify_goal",
        closed={"place_kind": "span"})
    add([rng.choice(TIDY_VERBS), rng.choice([" it", " this", " that"])], "clarify_goal",
        closed={"place_kind": "@it"})

    # --- a whole multi-clause request: the procedure reads the note back out, so the act is all
    #     the model has to get right (schema.WHOLE_INPUT_SLOTS)
    proj = rng.choice(["ledger-lab", "pet-tracker", "q3", "garden", "scratch", "demo", "atlas", "mono"])
    clauses = [rng.choice([f"set up a project called {proj}", f"start a new project named {proj}",
                           f"scaffold a project {proj}", f"bootstrap {proj}", f"init a project called {proj}"])]
    if rng.random() < 0.7:
        clauses[0] += rng.choice([f" under {rng.choice(PATHS)}", f" in {rng.choice(PATHS)}", ""])
    if rng.random() < 0.7:
        clauses.append(rng.choice([f"README.md should say '{rng.choice(QUOTED)}'",
                                   f"put '{rng.choice(QUOTED)}' in the readme",
                                   f"the readme should read '{rng.choice(QUOTED)}'"]))
    if rng.random() < 0.6:
        clauses.append(rng.choice(["todo: a; b", "todo: write tests; ship it", "todos: plan; build; review"]))
    if rng.random() < 0.6:
        clauses.append(rng.choice([f"commit with message '{rng.choice(QUOTED)}'", "git init it",
                                   f"make the first commit say '{rng.choice(QUOTED)}'"]))
    add([". ".join(clauses)], "setup_project")

    # --- looking things up
    add([rng.choice(["find ", "locate ", "search for ", "where is "]), ("pattern", rng.choice(PATTERNS))], "find")
    add([rng.choice(["find all ", "list all "]), ("pattern", rng.choice(PATTERNS)), " files"], "find")
    add([rng.choice(["search for ", "look for ", "grep "]), q, ("needle", rng.choice(NEEDLES)), qc, *place],
        "grep", closed=place_closed)
    add([rng.choice(["which files mention ", "what files contain "]), ("needle", rng.choice(NEEDLES))], "grep")
    add([rng.choice(["how many lines are in ", "count the lines in ", "line count of "]), ("target", f)],
        "count", closed={"unit": "lines"})
    add([rng.choice(["how many words are in ", "count the words in ", "word count of "]), ("target", f)],
        "count", closed={"unit": "words"})
    add([rng.choice(["how big is ", "what's the size of ", "how large is "]), ("target", rng.choice([f, folder]))], "size")
    topic_info, phrases = rng.choice(INFO_PHRASES)
    add([rng.choice(phrases)], "info", closed={"info_topic": topic_info})
    add([rng.choice(["is ", "do we have ", "do i have "]), ("program", rng.choice(PROGRAMS)), " installed"], "which")

    # --- the machine
    add([rng.choice(["go to ", "cd ", "switch to ", "change to "]), ("target", rng.choice(PATHS))], "cd")
    add([rng.choice(["open ", "launch ", "start "]), ("app", rng.choice(APP_WORDS))], "open_app")
    add([rng.choice(["open the app for ", "open the app used for ", "which app do i use for ",
                     "open the program for "]), ("function", rng.choice(FUNCTIONS))], "open_function",
        flags={"ask_only": False})
    add([rng.choice(["run ", "execute ", ""]), "`", ("command", rng.choice(COMMANDS)), "`"], "run")
    add([rng.choice(["install ", "apt install ", "can you install "]), ("package", rng.choice(PACKAGES))], "install")

    # --- git
    add([rng.choice(["make ", "turn "]), ("target", folder), rng.choice([" a git repo", " into a git repository"])], "git_init")
    add([rng.choice(["commit everything in ", "commit all changes in ", "commit "]), ("target", folder),
         rng.choice([" as ", " with the message ", " saying "]), q, ("message", quoted), qc], "git_commit")
    add([rng.choice(["what changed in ", "git status in ", "what's uncommitted in "]), ("target", folder)], "git_status")
    add([rng.choice(["show the history of ", "git log in ", "what are the recent commits in "]), ("target", folder)], "git_log")
    return ex


# ------------------------------------------------------------------ negatives

SMALL_TALK = [
    "make me a sandwich", "tell me a joke", "what's the weather like", "who won the world cup",
    "book me a flight to lisbon", "order a pizza", "play some jazz on spotify", "call my mother",
    "what should i have for dinner", "write me a poem about rain", "do you love me", "sing a song",
    "what's the capital of peru", "solve this crossword", "fix my wifi", "buy bitcoin",
    "translate this into french", "summarise the news", "how do i get to the airport",
    "set an alarm for 7am", "remind me to stretch", "what's 2+2", "who are you built by",
    "organize my desktop", "clean up my files", "tidy things up", "make my computer faster",
    "tell ada im happy", "email my boss that i'm late", "post this on twitter",
]


#: subjects and predicates for statements about the world. None of these words appear in the
#: civilization's own speech, which is the held-out test for this distinction, so refusing
#: those has to come from the shape of a statement rather than from remembering its nouns.
STATEMENT_SUBJECTS = ["Mara", "Tomas", "Ilse", "Bran", "Odile", "Yusuf", "the east meadow", "the barley crop",
                      "the mill", "the west road", "barley", "iron", "salt", "the well", "the orchard",
                      "the ferry", "the smithy", "my neighbour", "the harvest", "the cellar"]
STATEMENT_PREDICATES = ["is empty", "is full", "is late", "is cheap", "is dear", "is broken", "is flooded",
                        "has failed", "has arrived", "holds little", "holds plenty", "costs too much",
                        "burned last night", "went dry", "needs repair", "seems fine", "grows slowly",
                        "owes me two sacks", "works again", "is not ready"]
STATEMENT_FRAMES = ["{s} {p}.", "{s} {p}", "I heard {s} {p}.", "{r} said {s} {p}.", "{r} told me {s} {p}.",
                    "apparently {s} {p}.", "{s} {p}, by the way.", "they say {s} {p}."]


def generate_statements(n: int, seed: int = 0) -> list[Example]:
    """Third-party statements, labelled unknown: nothing here asks the assistant for anything."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        frame = rng.choice(STATEMENT_FRAMES)
        text = frame.format(s=rng.choice(STATEMENT_SUBJECTS), p=rng.choice(STATEMENT_PREDICATES),
                            r=rng.choice(STATEMENT_SUBJECTS[:6]))
        out.append(Example(text, "unknown", {}, {"speech_act": "world_statement"}, {}, source="statement"))
    return out[:n]


def generate_negatives(n: int, seed: int = 0, squad_path: Path | None = None) -> list[Example]:
    """Out-of-domain language, labelled unknown: refusing this is the job, not a failure."""
    rng = random.Random(seed)
    pool = [Example(t, "unknown", {}, {"speech_act": speech_act_of(t, "unknown")}, {}, "negative") for t in SMALL_TALK]
    if squad_path and squad_path.exists():
        rows = [json.loads(l) for l in squad_path.read_text().splitlines()]
        rng.shuffle(rows)
        pool += [Example(r["question"].strip(), "unknown", {}, {"speech_act": "question"}, {}, "negative")
                 for r in rows[: max(0, n - len(pool))] if r.get("question")]
    rng.shuffle(pool)
    out = list(pool[:n])
    while len(out) < n and pool:
        out.append(pool[rng.randrange(len(pool))])
    return out[:n]


# ------------------------------------------------------------------ noise


_KEYS = {"a": "sq", "e": "wr", "i": "ou", "o": "ip", "u": "yi", "s": "ad", "t": "ry", "n": "bm", "r": "et", "l": "k"}


def perturb(ex: Example, rng: random.Random) -> Example:
    """Typos, casing and filler, applied outside the spans so the labels stay exact."""
    text, spans = ex.text, {k: list(v) for k, v in ex.spans.items()}
    protected = [tuple(v) for v in spans.values()]

    def safe(i: int) -> bool:
        return all(not (a <= i < b) for a, b in protected)

    roll = rng.random()
    if roll < 0.25:  # drop or swap a character outside any span
        idx = [i for i, c in enumerate(text) if c.isalpha() and safe(i)]
        if idx:
            i = rng.choice(idx)
            if rng.random() < 0.5:
                text, shift = text[:i] + text[i + 1:], -1
            else:
                text, shift = text[:i] + rng.choice(_KEYS.get(text[i].lower(), "x")) + text[i:], 1
            spans = {k: [a + shift if a > i else a, b + shift if b > i else b] for k, (a, b) in spans.items()}
    elif roll < 0.4:  # lose an apostrophe outside spans
        idx = [i for i, c in enumerate(text) if c == "'" and safe(i)]
        if idx:
            i = rng.choice(idx)
            text = text[:i] + text[i + 1:]
            spans = {k: [a - 1 if a > i else a, b - 1 if b > i else b] for k, (a, b) in spans.items()}
    elif roll < 0.55:
        text = text.upper() if rng.random() < 0.3 else text.capitalize()
    return Example(text, ex.act, spans, ex.closed, ex.flags, ex.source)


# ------------------------------------------------------------------ grammar distillation


def generate_from_grammar(n: int, seed: int = 0) -> list[Example]:
    """Ask the symbolic parser what it makes of generated utterances, and keep what it is sure of.

    Distillation with a caveat recorded in the manifest: where the grammar is wrong, the
    student inherits the error, which is why the grammar is never the evaluation.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from tensacode.language.domains.desktop import read_request  # noqa: PLC0415

    rng = random.Random(seed)
    seeds = generate_templates(n, seed=seed + 7)
    out: list[Example] = []
    for ex in seeds:
        try:
            acts, got = read_request(ex.text)
        except Exception:  # noqa: BLE001 - a parser bug is not a data bug
            continue
        if len(acts) != 1 or got.coverage < 0.95:
            continue
        act = acts[0]
        if act.act != ex.act:  # the grammar and the template disagree: keep neither, note nothing
            continue
        out.append(Example(ex.text, ex.act, ex.spans, ex.closed, ex.flags, source="grammar"))
    rng.shuffle(out)
    return out


# ------------------------------------------------------------------ writing


def write(path: Path, examples: Iterable[Example]) -> int:
    n = 0
    with path.open("w") as fh:
        for ex in examples:
            fh.write(json.dumps(asdict(ex)) + "\n")
            n += 1
    return n


def read(path: Path) -> list[Example]:
    return [Example(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


#: files whose utterances are the measurement and must never be trained on
HELD_OUT_SOURCES = (
    "tests/test_assistant_language.py",
    "tests/test_civ_language_demands.py",
    "eval/social/cases.py",
    "eval/training/eval_parser.py",
)


def held_out_utterances(root: Path) -> set[str]:
    """Every quoted string in the sets we measure on, normalised for comparison."""
    import ast as _ast

    out: set[str] = set()
    for rel in HELD_OUT_SOURCES:
        path = root / rel
        if not path.exists():
            continue
        for node in _ast.walk(_ast.parse(path.read_text())):
            if isinstance(node, _ast.Constant) and isinstance(node.value, str):
                text = node.value.strip().lower().rstrip("?.!").strip()
                if 3 <= len(text) <= 120 and " " in text:
                    out.add(text)
    return out


def _norm_utterance(text: str) -> str:
    return text.strip().lower().rstrip("?.!").strip()


def drop_contaminated(rows: Sequence[Example], held_out: set[str]) -> tuple[list[Example], dict]:
    """Remove training rows that reproduce a measured utterance verbatim.

    Both corpora are drawn from one small vocabulary of short commands, so collisions are
    inevitable rather than exceptional: "never mind", "go ahead", "mkdir foo". Left in, they
    turn a held-out score into a memorisation score, which is what happened before this check
    existed. Dropping them costs a negligible number of rows and makes the number mean what
    it claims.
    """
    kept, dropped = [], []
    for row in rows:
        (dropped if _norm_utterance(row.text) in held_out else kept).append(row)
    report = {
        "rows_dropped": len(dropped),
        "distinct_utterances_dropped": len(({_norm_utterance(r.text) for r in dropped})),
        "by_source": {s: sum(1 for r in dropped if r.source == s) for s in {r.source for r in dropped}},
        "examples": sorted({r.text for r in dropped})[:20],
    }
    # nothing may survive the filter
    assert not [r for r in kept if _norm_utterance(r.text) in held_out]
    return kept, report


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("/tmp/claude-1000/-home-brandonin-Documents-tensacode-tensacode-python/c572c14b-5662-4c07-8a7a-1ba7821d2bfa/scratchpad/training"))
    ap.add_argument("--templates", type=int, default=90_000)
    ap.add_argument("--negatives", type=int, default=6_000)
    ap.add_argument("--statements", type=int, default=12_000, help="statements about the world, labelled unknown")
    ap.add_argument("--noise", type=float, default=0.45, help="fraction of template examples perturbed")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    templates = generate_templates(args.templates, seed=args.seed)
    templates = [perturb(e, rng) if rng.random() < args.noise else e for e in templates]
    squad = Path("/tmp/claude-1000/-home-brandonin-Documents-tensacode-tensacode-python/c572c14b-5662-4c07-8a7a-1ba7821d2bfa/scratchpad/open_domain/squad2_val.jsonl")
    negatives = generate_negatives(args.negatives, seed=args.seed, squad_path=squad)

    para_path = args.out / "paraphrase_train.jsonl"
    paraphrases = read(para_path) if para_path.exists() else []
    for ex in paraphrases:  # the label travelled with the rewrite; the speech act is read off the new surface
        ex.closed.setdefault("speech_act", speech_act_of(ex.text, ex.act))
    statements = generate_statements(args.statements, seed=args.seed + 3)
    rows = templates + negatives + paraphrases + statements
    held_out = held_out_utterances(Path(__file__).parents[2])
    rows, contamination = drop_contaminated(rows, held_out)
    rng.shuffle(rows)
    cut = int(len(rows) * 0.95)
    n_train = write(args.out / "parser_train.jsonl", rows[:cut])
    n_dev = write(args.out / "parser_dev.jsonl", rows[cut:])
    manifest = {
        "seed": args.seed,
        "counts": {"train": n_train, "dev": n_dev},
        "by_source": {s: sum(1 for r in rows if r.source == s) for s in {r.source for r in rows}},
        "perturbed_fraction": args.noise,
        "contamination_check": {
            "sources": list(HELD_OUT_SOURCES),
            "candidate_strings": len(held_out),
            **contamination,
            "note": "rows reproducing a measured utterance verbatim are dropped before training. "
                    "This says nothing about vocabulary overlap, only that no measured utterance "
                    "is itself in the training data.",
        },
        "provenance": {
            "template": "authored by us (eval/training/parser_data.py): a floor on obvious breakage, not coverage",
            "negative": "public SQuAD 2.0 questions and small talk, labelled unknown",
            "statement": "third-party statements about the world, labelled unknown, built from vocabulary that "
                         "does not appear in the civilization's speech (the held-out test for this distinction)",
            "paraphrase": "training half of eval/training/paraphrase.py (a local model rewrote a template; "
                          "labels carried over only when every value survived verbatim). The evaluation half "
                          "comes from disjoint template draws and is never trained on.",
        },
        "never_trained_on": [
            "tests/test_assistant_language.py (152 cases, written before this parser existed)",
            "tests/test_civ_language_demands.py (193 cases from a running world)",
            "the five prompts the user reported failing",
        ],
    }
    (args.out / "parser_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
