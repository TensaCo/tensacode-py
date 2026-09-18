"""The body's action vocabulary, and the retired motor programs kept as a test oracle.

The dataclasses below (``Run``, ``Say``, ``Ask``, ``Look``, ``Seen``, ``Output``, …) are what
the body speaks: ``interpreter.py`` writes them and ``agent.py`` carries them out.

The generator programs after them are **no longer used by the assistant**. Procedures in
``procedures.py`` (data, walked by ``interpreter.py``) replaced them. They are kept because
``tests/test_assistant_procedures.py`` runs both engines against one fake machine and requires
the same replies and the same commands — parity evidence that outlives the port.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Generator

from .language import PLACES, Frame

HOME = "/home/agent"


# ------------------------------------------------------------------ actions


@dataclass(frozen=True)
class Run:
    command: str
    why: str


@dataclass(frozen=True)
class Output:
    text: str
    cwd: str | None = None
    timed_out: bool = False

    @property
    def lines(self) -> list[str]:
        return [ln for ln in self.text.splitlines() if ln.strip()]

    @property
    def errors(self) -> list[str]:
        return [ln for ln in self.lines if ERROR.search(ln)]


@dataclass(frozen=True)
class Ask:
    question: str
    choices: tuple[str, ...] = ("yes", "no")


@dataclass(frozen=True)
class Say:
    text: str


@dataclass(frozen=True)
class OpenApp:
    app: str


@dataclass(frozen=True)
class Click:
    """Click the visible control whose label best matches (returns True, or a string saying why not)."""

    label: str
    role: str | None = None


@dataclass(frozen=True)
class Fill:
    """Type into the visible textbox whose label best matches (returns True, or why not)."""

    label: str
    text: str
    submit: bool = False


@dataclass(frozen=True)
class Look:
    """Return what is on screen now (a Seen)."""

    settle_ms: int = 350


@dataclass(frozen=True)
class Seen:
    controls: tuple[tuple[str, str], ...]  # (role, label) of visible controls, screen order
    texts: tuple[tuple[str, str], ...]  # (window, text)

    def mentions(self, needle: str) -> bool:
        n = needle.lower()
        return any(n in t.lower() for _, t in self.texts) or any(n in label.lower() for _, label in self.controls)

    CHROME = re.compile(r"^(?:Resize window \w+|Minimize|Maximize|Close|New tab|Computers|Simulator settings|Remove from tab|Activities|System menu|Show Applications|\w{3}, \w{3} \d+ .*[AP]M)$")

    def render(self, *, dock: tuple[str, ...] = (), front: str | None = None, per_window: int = 18) -> str:
        """A compact description for the teacher: window chrome and dock icons dropped, windows listed front first."""
        labels: dict[str, list[str]] = {}
        for role, label in self.controls:
            if label and not self.CHROME.match(label) and label not in dock and label not in labels.setdefault(role, []):
                labels[role].append(label[:80])
        windows: dict[str, list[str]] = {}
        for window, text in self.texts:
            if window:
                windows.setdefault(window, []).append(" ".join(text.split())[:120])
        order = sorted(windows, key=lambda w: (w != front, list(windows).index(w)))
        lines = [f"  open windows: {', '.join(order) or 'none'}"]
        lines += [f"  {role}s: " + " | ".join(ls[:60]) for role, ls in labels.items() if role in ("button", "textbox", "combobox", "searchbox", "tab", "link", "checkbox", "menuitem", "option")]
        lines += [f"  text in {w}: " + " | ".join(list(dict.fromkeys(windows[w]))[:per_window]) for w in order]
        return "\n".join(lines)


@dataclass(frozen=True)
class Teach:
    """Ask the teacher model (only while learning something new). Returns the reply text, or None if unavailable."""

    system: str
    user: str
    max_new_tokens: int = 300


@dataclass(frozen=True)
class Remember:
    """Write plain (subject, predicate, object) triples into memory: a plan, a note, a step outcome."""

    triples: tuple[tuple[str, str, object], ...]
    why: str = ""


@dataclass(frozen=True)
class Focus:
    path: str
    kind: str  # "directory" | "regular file"


@dataclass
class Context:
    """What the conversation knows when a program starts (read from the mind)."""

    cwd: str = HOME
    focus: str | None = None
    focus_kind: str | None = None
    known: dict[str, str] = field(default_factory=dict)  # path -> kind, most recent last


Program = Generator[object, object, None]
ERROR = re.compile(r"No such file|not found|cannot |can't |File exists|Permission denied|Not a directory|Is a directory|not a git repository|fatal:|error:|invalid option|unrecognized option|unknown object|not installed|unable to|denied|not implemented|Directory not empty|nothing to commit", re.I)


def q(s: str) -> str:
    return shlex.quote(s)


def show(path: str) -> str:
    return "~" + path[len(HOME):] if path == HOME or path.startswith(HOME + "/") else path


def expand(p: str, ctx: Context) -> str:
    p = p.strip()
    if p == "~" or p.startswith("~/"):
        return (HOME + p[1:]).rstrip("/") or "/"
    if p.startswith("/"):
        return p.rstrip("/") or "/"
    return f"{ctx.cwd.rstrip('/')}/{p}".rstrip("/")


def parent(path: str) -> str:
    return path.rsplit("/", 1)[0] or "/"


def base(path: str) -> str:
    return path.rstrip("/").rsplit("/", 1)[-1]


# -------------------------------------------------------------- perception


def stat(*paths: str) -> Generator[object, object, dict[str, tuple[str, int] | None]]:
    """Look at the file system: path -> (kind, size) or None when it does not exist."""
    out = yield Run("stat -c '%F|%s|%n' " + " ".join(q(p) for p in paths), "look before acting")
    found: dict[str, tuple[str, int] | None] = {p: None for p in paths}
    for line in out.lines:
        if m := re.match(r"^(directory|regular file|symbolic link|[a-z ]+)\|(\d+)\|(.+)$", line):
            found[m[3]] = (m[1], int(m[2]))
    return found


def resolve(ref: str | None, ctx: Context, *, want: str | None = None, place: str | None = None, what: str = "that") -> Generator[object, object, str | None]:
    """Turn words for a thing ('it', '@recipes', 'notes.txt', '~/x') into one existing path, asking if unsure."""
    if ref is None:
        yield Say(f"Which {what} do you mean? I didn't catch a name.")
        return None
    if ref == "@it":
        if ctx.focus is None:
            yield Say(f"I'm not sure what “it” refers to yet. Tell me the {what} by name.")
            return None
        return ctx.focus
    name = ref[1:] if ref.startswith("@") else ref
    if name.lower() in PLACES:
        return expand(PLACES[name.lower()], ctx)
    if "/" in name or name.startswith("~"):
        return expand(name, ctx)
    candidates = [expand(f"{place}/{name}", ctx)] if place else []
    candidates += [p for p in reversed(list(ctx.known)) if base(p) == name]  # mentioned earlier in this conversation
    candidates += [f"{ctx.cwd}/{name}", f"{HOME}/{name}", f"{HOME}/Desktop/{name}", f"{HOME}/Documents/{name}"]
    candidates = list(dict.fromkeys(candidates))
    seen = yield from stat(*candidates)
    hits = [p for p in candidates if seen[p] and (want is None or seen[p][0] == want)]
    if hits and (place or len({p for p in hits}) == 1 or hits[0] in ctx.known):
        return hits[0]
    if not hits:  # search the home folder
        kind = " -type d" if want == "directory" else " -type f" if want == "regular file" else ""
        out = yield Run(f"find {HOME} -name {q(name)}{kind} 2>/dev/null | head -n 40", f"search for {name}")
        hits = [ln.strip() for ln in out.lines if ln.startswith("/") and "/." not in ln]
    if not hits:
        # a wrong name is usually a near miss: correct the belief rather than report an absence
        from tensacode.social import near_names

        where = expand(place, ctx) if place else ctx.cwd
        listing = yield Run(f"ls -1pA {q(where)}", "see what is there instead")
        pool = [ln.rstrip("/") for ln in listing.lines] + [k.rsplit("/", 1)[-1] for k in ctx.known]
        near = near_names(name, dict.fromkeys(pool))
        if not near:  # widen by prefix: a misspelling is still a near miss when the file is elsewhere
            stem = re.sub(r"[^\w.-]", "", name.rsplit("/", 1)[-1])[:3]
            if stem:
                wide = yield Run(f"find {HOME} -maxdepth 4 -iname {q(stem + '*')} 2>/dev/null | head -n 40",
                                 f"look for something like {name} elsewhere")
                pool += [ln.strip().rsplit("/", 1)[-1] for ln in wide.lines
                         if ln.strip().startswith("/") and "/." not in ln]
                near = near_names(name, dict.fromkeys(pool))
        if near:
            joined = near[0] if len(near) == 1 else ", ".join(near[:-1]) + f" or {near[-1]}"
            yield Say(f"There's no {name} — did you mean {joined}?")
            return None
        yield Say(f"I couldn't find anything called “{name}” in your home folder.")
        return None
    if len(hits) == 1:
        return hits[0]
    options = tuple(show(h) for h in hits[:6])
    answer = yield Ask(f"I found {len(hits)} things called “{name}”. Which one?\n" + "\n".join(f"{i + 1}. {o}" for i, o in enumerate(options)), options)
    pick = _choice(answer, options)
    if pick is None:
        yield Say("Okay, I'll leave it.")
        return None
    return hits[pick]


def _choice(answer: object, options: tuple[str, ...]) -> int | None:
    if not isinstance(answer, Frame):
        return None
    words = answer.words.strip().lower()
    ordinals = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "fifth": 4, "sixth": 5, "last": len(options) - 1, "the last one": len(options) - 1}
    if m := re.search(r"\b(\d+)\b", words):
        i = int(m[1]) - 1
        return i if 0 <= i < len(options) else None
    for word, i in ordinals.items():
        if re.search(rf"\b{word}\b", words):
            return i
    for i, o in enumerate(options):
        if words and (words in o.lower() or o.lower().endswith(words)):
            return i
    return None


YES = re.compile(r"^(?:y|yes|yeah|yep|yup|sure|ok(?:ay)?|do it|go ahead|please do|confirm(?:ed)?|affirmative)[.!]*$", re.I)
# commands that only read: everything else is treated as able to change the machine
READ_ONLY = re.compile(
    r"^(?:ls|cat|head|tail|wc|pwd|cd|echo|date|whoami|id|uname|hostname|uptime|df|du|ps|which|type|file|stat|find|grep|egrep|fgrep"
    r"|printenv|env|nproc|ip|ifconfig|tree|sort|uniq|cut|tr|basename|dirname|realpath|readlink|history|help|man"
    r"|git(?:\s+-C\s+\S+)?\s+(?:status|log|diff|show|branch|ls-files|rev-parse|config\s+--get))\b")
PROTECTED = (HOME, "/", f"{HOME}/Desktop", f"{HOME}/Documents", f"{HOME}/Downloads", f"{HOME}/Pictures", f"{HOME}/Music", f"{HOME}/Videos")


def changes_things(command: str) -> bool:
    """True when any part of a typed command could change the machine (so it needs a yes first).

    Separators and redirections inside quotes are part of an argument, not shell syntax
    (``stat -c '%F|%s|%n'`` only reads), so quoted spans are blanked out before splitting.
    """
    bare = re.sub(r"'[^']*'|\"[^\"]*\"", " ARG ", command)
    if re.search(r"(?<![0-9&])>", bare):  # a redirection writes
        return True
    return any(not READ_ONLY.match(part.strip()) for part in re.split(r"\|\||&&|;|\||\n", bare) if part.strip())


def confirmed(answer: object) -> bool:
    """Only used where a question was genuinely ambiguous, never as a permission gate."""
    if not isinstance(answer, Frame):
        return False
    return answer.act == "confirm" or bool(YES.match(answer.words.strip()))


def summarize_errors(out: Output) -> str:
    lines = out.errors or out.lines
    return lines[0][:200] if lines else "no output"


def listing(names: list[str], limit: int = 25) -> str:
    shown = ", ".join(names[:limit])
    return shown + (f", … and {len(names) - limit} more" if len(names) > limit else "")


# ------------------------------------------------------------------ programs


HELP = """Things you can ask me (I work this Ubuntu machine through its screen, no AI model):
• files: “make a folder called recipes on my desktop”, “create notes.txt in it saying hello”, “add 'buy milk' to notes.txt”, “read notes.txt”
• look around: “what's on my desktop?”, “find all pdfs”, “which files mention dns?”, “how big is my documents folder?”
• change things: “rename notes.txt to ideas.txt”, “move it to documents”, “copy it to the desktop”, “delete it”
• git: “make recipes a git repo”, “commit everything in recipes as 'first'”, “what changed in recipes?”, “show its history”
• the machine: “what time is it?”, “how much disk space is left?”, “what's my ip?”, “what's running?”, “open firefox”
• anything else: run a command with backticks, like `ls -la ~`
You can chain requests with “then”."""


def program_for(frame: Frame, ctx: Context) -> Program:
    return PROGRAMS.get(frame.act, unknown)(frame, ctx)


def greet(frame: Frame, ctx: Context) -> Program:
    yield Say("Hi! I'm driving this Ubuntu desktop for you. Ask me to do something with your files or the machine, or say “help”.")


def thanks(frame: Frame, ctx: Context) -> Program:
    yield Say("You're welcome.")


def help_(frame: Frame, ctx: Context) -> Program:
    yield Say(HELP)


def unknown(frame: Frame, ctx: Context) -> Program:
    # kept in step with the procedure engine's `admit`, which tells three failures apart
    yield Say(f"I didn't understand “{frame.words}”. Say “help” to see the kinds of thing I can do.")


def choose_without_question(frame: Frame, ctx: Context) -> Program:
    words = frame.words.strip().strip("'\"`")
    if re.fullmatch(r"~?/?[\w.@+ /-]*\.[A-Za-z0-9]{1,8}|~(?:/[\w.@+ -]+)*|/[\w.@+ /-]+", words):
        yield from read(Frame("read", frame.words, {"target": words}), ctx)  # a bare path or filename: show it
        return
    yield from stray_answer(frame, ctx)


def stray_answer(frame: Frame, ctx: Context) -> Program:
    yield Say("There's no open question right now." if frame.act == "choose" else "There's nothing waiting for a yes or no right now.")


def list_(frame: Frame, ctx: Context) -> Program:
    path = yield from resolve(frame.slots.get("place") or ctx.cwd, ctx, what="folder")
    if path is None:
        return
    seen = yield from stat(path)
    if seen[path] is None:
        yield Say(f"There's no {show(path)}.")
        return
    if seen[path][0] != "directory":
        yield from read(Frame("read", frame.words, {"target": path}), ctx)
        return
    out = yield Run(f"ls -1pA {q(path)}", f"list {show(path)}")
    if out.errors:
        yield Say(f"I couldn't list {show(path)}: {summarize_errors(out)}")
        return
    names = out.lines
    yield Focus(path, "directory")
    if frame.slots.get("count"):
        folders = sum(n.endswith("/") for n in names)
        plural = lambda n, word: f"{n} {word}{'s' * (n != 1)}"  # noqa: E731
        yield Say(f"{show(path)} has {plural(len(names), 'item')} ({plural(folders, 'folder')}, {plural(len(names) - folders, 'file')}).")
    elif not names:
        yield Say(f"{show(path)} is empty.")
    else:
        yield Say(f"{show(path)} has {len(names)} item{'s' * (len(names) != 1)}: {listing(names)}")


def read(frame: Frame, ctx: Context) -> Program:
    path = yield from resolve(frame.slots.get("target"), ctx, place=frame.slots.get("place"), what="file")
    if path is None:
        return
    seen = yield from stat(path)
    if seen[path] is None:
        yield Say(f"There's no {show(path)}.")
        return
    if seen[path][0] == "directory":
        yield from list_(Frame("list", frame.words, {"place": path}), ctx)
        return
    out = yield Run(f"head -n 40 {q(path)}", f"read {show(path)}")
    yield Focus(path, "regular file")
    if out.errors:
        yield Say(f"I couldn't read {show(path)}: {summarize_errors(out)}")
    elif not out.text.strip():
        yield Say(f"{show(path)} is empty.")
    else:
        more = " (first 40 lines)" if len(out.text.splitlines()) >= 40 else ""
        yield Say(f"{show(path)} says{more}:\n{out.text.rstrip()}")


def create_folder(frame: Frame, ctx: Context) -> Program:
    name = frame.slots.get("name")
    if not name:
        yield Say("What should I call the folder?")
        return
    place = yield from _place_for_new(frame, ctx)
    if place is None:
        return
    path = expand(name, Context(cwd=place))
    seen = yield from stat(path)
    if seen[path] is not None:
        yield Focus(path, seen[path][0])
        yield Say(f"There's already a {'folder' if seen[path][0] == 'directory' else 'file'} at {show(path)}, so I left it alone.")
        return
    out = yield Run(f"mkdir -p {q(path)}", f"create {show(path)}")
    after = yield from stat(path)
    if after[path] and after[path][0] == "directory":
        yield Focus(path, "directory")
        yield Say(f"Created the folder {show(path)}.")
    else:
        yield Say(f"I tried to create {show(path)} but it isn't there: {summarize_errors(out)}")


def create_file(frame: Frame, ctx: Context) -> Program:
    name = frame.slots.get("name")
    if not name:
        yield Say("What should I call the file?")
        return
    place = yield from _place_for_new(frame, ctx)
    if place is None:
        return
    path = expand(name, Context(cwd=place))
    text = frame.slots.get("text")
    seen = yield from stat(path, parent(path))
    if seen[parent(path)] is None:
        yield Say(f"The folder {show(parent(path))} doesn't exist. Want me to create it first? Say “make a folder called {base(parent(path))} in {show(parent(parent(path)))}”.")
        return
    if seen[path] is not None:
        if text is None:
            yield Focus(path, seen[path][0])
            yield Say(f"{show(path)} already exists, so I left it alone.")
            return
    command = f"printf '%s\\n' {q(text)} > {q(path)}" if text is not None else f"touch {q(path)}"
    out = yield Run(command, f"create {show(path)}")
    yield from _verify_file(path, text, out, "Created" if seen[path] is None else "Replaced the contents of")


def write(frame: Frame, ctx: Context) -> Program:
    text, append = frame.slots.get("text") or "", frame.slots.get("append")
    target = frame.slots.get("target")
    if target and target != "@it" and "/" not in target and not target.startswith("@") and "." in target:
        # a new filename is fine for writing; look for an existing one first
        path = None
        seen_known = [p for p in ctx.known if base(p) == target]
        path = seen_known[-1] if seen_known else None
        if path is None:
            here = f"{(ctx.focus if ctx.focus_kind == 'directory' else ctx.cwd)}/{target}"
            found = yield from stat(here)
            path = here if found[here] else None
        if path is None:
            path = yield from resolve(target, ctx, want="regular file", what="file")
            if path is None:
                return
    else:
        path = yield from resolve(target, ctx, want="regular file", what="file")
        if path is None:
            return
    seen = yield from stat(path)
    if seen[path] and seen[path][0] == "directory":
        yield Say(f"{show(path)} is a folder, not a file.")
        return
    out = yield Run(f"printf '%s\\n' {q(text)} {'>>' if append else '>'} {q(path)}", f"{'append to' if append else 'write'} {show(path)}")
    yield from _verify_file(path, text, out, "Added a line to" if append else "Wrote")


def _verify_file(path: str, text: str | None, out: Output, verb: str) -> Program:
    check = yield Run(f"tail -n 3 {q(path)}", f"check {show(path)}")
    if check.errors or out.errors:
        yield Say(f"Something went wrong with {show(path)}: {summarize_errors(out if out.errors else check)}")
        return
    if text is not None and text not in check.text:
        yield Say(f"I wrote to {show(path)}, but reading it back I don't see “{text}”. It ends with:\n{check.text.rstrip()}")
        return
    yield Focus(path, "regular file")
    yield Say(f"{verb} {show(path)}." + (f" It now ends with “{text}”." if text else ""))


def _place_for_new(frame: Frame, ctx: Context) -> Generator[object, object, str | None]:
    place = frame.slots.get("place")
    if place is None:
        if ctx.focus_kind == "directory" and ctx.focus:
            return ctx.focus
        return ctx.cwd
    path = yield from resolve(place, ctx, want="directory", what="folder")
    return path


def delete(frame: Frame, ctx: Context) -> Program:
    target = frame.slots.get("target")
    everything = re.search(r"\b(?:everything|all (?:the )?(?:files|things|stuff))\b", frame.words, re.I)
    path = yield from resolve(target, ctx, what="file or folder")
    if path is None:
        return
    seen = yield from stat(path)
    if seen[path] is None:
        yield Say(f"There's no {show(path)} to delete.")
        return
    is_dir = seen[path][0] == "directory"
    if everything and is_dir:
        out = yield Run(f"ls -1pA {q(path)}", f"see what is in {show(path)}")
        names = out.lines
        if not names:
            yield Say(f"{show(path)} is already empty.")
            return
        out = yield Run(" && ".join(f"rm -r {q(path + '/' + n.rstrip('/'))}" for n in names), f"empty {show(path)}")
        after = yield Run(f"ls -1pA {q(path)}", f"check {show(path)} is empty")
        yield Say(f"Deleted {len(names) - len(after.lines)} items from {show(path)}." + (f" Still there: {listing(after.lines)}" if after.lines else ""))
        return
    if path in PROTECTED:
        yield Say(f"I won't delete {show(path)} itself. If you mean what's inside it, say “delete everything in {show(path)}”.")
        return
    detail = ""
    if is_dir:
        inside = yield Run(f"ls -1pA {q(path)}", f"see what is in {show(path)}")
        detail = f" It contains {len(inside.lines)} item{'s' * (len(inside.lines) != 1)}." if inside.lines else " It's empty."
    out = yield Run(f"rm {'-r ' if is_dir else ''}{q(path)}", f"delete {show(path)}{detail}")
    after = yield from stat(path)
    if after[path] is None:
        yield Say(f"Deleted {show(path)}.")
    else:
        yield Say(f"I couldn't delete {show(path)}: {summarize_errors(out)}")


def move(frame: Frame, ctx: Context) -> Program:
    yield from _move_or_copy(frame, ctx, copy=frame.act == "copy")


def _move_or_copy(frame: Frame, ctx: Context, *, copy: bool) -> Program:
    src = yield from resolve(frame.slots.get("target"), ctx, what="file or folder")
    if src is None:
        return
    dest_dir = yield from resolve(frame.slots.get("dest"), ctx, want="directory", what="destination folder")
    if dest_dir is None:
        return
    dest = f"{dest_dir}/{base(src)}"
    seen = yield from stat(src, dest)
    if seen[src] is None:
        yield Say(f"There's no {show(src)}.")
        return
    if seen[dest] is not None:
        yield Say(f"{show(dest)} already exists, so I didn't {'copy' if copy else 'move'} anything.")
        return
    is_dir = seen[src][0] == "directory"
    out = yield Run(f"{'cp -r' if copy and is_dir else 'cp' if copy else 'mv'} {q(src)} {q(dest)}", f"{'copy' if copy else 'move'} {show(src)} to {show(dest_dir)}")
    after = yield from stat(src, dest)
    if after[dest] is None or (not copy and after[src] is not None):
        yield Say(f"That didn't work: {summarize_errors(out)}")
        return
    yield Focus(dest, seen[src][0])
    yield Say(f"{'Copied' if copy else 'Moved'} {show(src)} to {show(dest)}.")


def rename(frame: Frame, ctx: Context) -> Program:
    src = yield from resolve(frame.slots.get("target"), ctx, what="file or folder")
    if src is None:
        return
    new = frame.slots.get("new_name", "").strip()
    if not new or "/" in new:
        yield Say("What should the new name be? (Just a name, like ideas.txt.)")
        return
    dest = f"{parent(src)}/{new}"
    seen = yield from stat(src, dest)
    if seen[src] is None:
        yield Say(f"There's no {show(src)}.")
        return
    if seen[dest] is not None:
        yield Say(f"{show(dest)} already exists, so I didn't rename anything.")
        return
    out = yield Run(f"mv {q(src)} {q(dest)}", f"rename {base(src)} to {new}")
    after = yield from stat(src, dest)
    if after[dest] is None or after[src] is not None:
        yield Say(f"That didn't work: {summarize_errors(out)}")
        return
    yield Focus(dest, seen[src][0])
    yield Say(f"Renamed {show(src)} to {new}.")


def find(frame: Frame, ctx: Context) -> Program:
    pattern, place = frame.slots.get("pattern", "*"), frame.slots.get("place", "~")
    root = yield from resolve(place, ctx, want="directory", what="folder")
    if root is None:
        return
    out = yield Run(f"find {q(root)} -name {q(pattern)} 2>/dev/null | head -n 60", f"search {show(root)} for {pattern}")
    hits = [ln.strip() for ln in out.lines if ln.startswith("/") and "/." not in ln]
    if not hits:
        yield Say(f"Nothing matching “{pattern}” in {show(root)}.")
        return
    if len(hits) == 1:
        yield Focus(hits[0], "regular file")
    yield Say(f"Found {len(hits)}{'+' if len(out.lines) >= 60 else ''} matching “{pattern}” in {show(root)}: {listing([show(h) for h in hits], 15)}")


def grep(frame: Frame, ctx: Context) -> Program:
    needle = frame.slots.get("needle", "")
    root = yield from resolve(frame.slots.get("place", "~"), ctx, what="folder")
    if root is None:
        return
    out = yield Run(f"grep -ril {q(needle)} {q(root)} 2>/dev/null | head -n 80", f"search inside files for “{needle}”")
    hits = [ln.strip() for ln in out.lines if ln.startswith("/") and "/." not in ln]
    if not hits:
        yield Say(f"No files in {show(root)} mention “{needle}”.")
        return
    if len(hits) == 1:
        yield Focus(hits[0], "regular file")
    yield Say(f"{len(hits)} file{'s' * (len(hits) != 1)} in {show(root)} mention{'s' * (len(hits) == 1)} “{needle}”: {listing([show(h) for h in hits], 15)}")


def count(frame: Frame, ctx: Context) -> Program:
    path = yield from resolve(frame.slots.get("target"), ctx, want="regular file", what="file")
    if path is None:
        return
    unit = frame.slots.get("unit", "lines")
    out = yield Run(f"wc -{'w' if unit == 'words' else 'l'} < {q(path)}", f"count {unit} in {show(path)}")
    n = next((ln.strip() for ln in out.lines if ln.strip().isdigit()), None)
    yield Focus(path, "regular file")
    yield Say(f"{show(path)} has {n} {unit[:-1] if n == '1' else unit}." if n is not None else f"I couldn't count: {summarize_errors(out)}")


def size(frame: Frame, ctx: Context) -> Program:
    path = yield from resolve(frame.slots.get("target"), ctx, what="file or folder")
    if path is None:
        return
    out = yield Run(f"du -sh {q(path)}", f"measure {show(path)}")
    m = re.match(r"^(\S+)\s", out.lines[0]) if out.lines else None
    yield Say(f"{show(path)} takes up {m[1]}." if m and not out.errors else f"I couldn't measure {show(path)}: {summarize_errors(out)}")


INFO = {
    "date": ("date", lambda o: f"It's {o.lines[0]} on this machine." if o.lines else None),
    "user": ("whoami", lambda o: f"You're logged in as “{o.lines[0]}”." if o.lines else None),
    "disk": ("df -h ~", lambda o: (lambda f: f"{f[3]} free of {f[1]} ({f[4]} used) on {f[5]}.")(o.lines[-1].split()) if len(o.lines) >= 2 and len(o.lines[-1].split()) >= 6 else None),
    "ip": ("ip addr", lambda o: (lambda ips: f"This machine's IP address is {', '.join(ips)}." if ips else None)([m[1] for ln in o.lines if (m := re.search(r"\binet (\d+\.\d+\.\d+\.\d+)", ln))])),
    "hostname": ("hostname", lambda o: f"This computer is called “{o.lines[0]}”." if o.lines else None),
    "uptime": ("uptime", lambda o: f"Uptime: {o.lines[0].strip()}" if o.lines else None),
    "processes": ("ps -eo comm", lambda o: f"{len(o.lines) - 1} processes are running, including {listing([ln.split()[-1] for ln in o.lines[1:]], 12)}." if len(o.lines) > 1 else None),
    "cpus": ("nproc", lambda o: f"This machine has {o.lines[0]} CPU cores." if o.lines else None),
    "os": ("uname -a", lambda o: f"{o.lines[0]}" if o.lines else None),
    "cwd": ("pwd", lambda o: f"The terminal is in {show(o.lines[0].strip())}." if o.lines else None),
}


def info(frame: Frame, ctx: Context) -> Program:
    command, render = INFO[frame.slots["topic"]]
    out = yield Run(command, f"check {frame.slots['topic']}")
    text = None if out.errors else render(out)
    yield Say(text or f"I ran `{command}` but couldn't make sense of the result: {summarize_errors(out)}")


def which(frame: Frame, ctx: Context) -> Program:
    program = frame.slots["program"]
    out = yield Run(f"which {q(program)}", f"look for {program}")
    hit = next((ln for ln in out.lines if ln.startswith("/")), None)
    yield Say(f"Yes, {program} is installed at {hit}." if hit else f"No, {program} isn't installed (or isn't on the PATH).")


def cd(frame: Frame, ctx: Context) -> Program:
    path = yield from resolve(frame.slots.get("target"), ctx, want="directory", what="folder")
    if path is None:
        return
    out = yield Run(f"cd {q(path)}", f"go to {show(path)}")
    if out.errors:
        yield Say(f"I couldn't go there: {summarize_errors(out)}")
        return
    yield Focus(path, "directory")
    yield Say(f"Now working in {show(path)}.")


def open_app(frame: Frame, ctx: Context) -> Program:
    app = frame.slots["app"]
    opened = yield OpenApp(app)
    yield Say(f"Opened {app}." if opened else f"I clicked {app} in the dock, but I don't see its window.")


def _repo(frame: Frame, ctx: Context) -> Generator[object, object, str | None]:
    return (yield from resolve(frame.slots.get("target"), ctx, want="directory", what="folder"))


def git_init(frame: Frame, ctx: Context) -> Program:
    path = yield from _repo(frame, ctx)
    if path is None:
        return
    seen = yield from stat(f"{path}/.git")
    if seen[f"{path}/.git"] is not None:
        yield Focus(path, "directory")
        yield Say(f"{show(path)} is already a git repository.")
        return
    out = yield Run(f"git -C {q(path)} init", f"make {show(path)} a repository")
    yield Focus(path, "directory")
    if re.search(r"\breinitialized\b", out.text, re.I):
        yield Say(f"{show(path)} was already a git repository (git re-initialized it, nothing was lost).")
    elif "Initialized" in out.text:
        yield Say(f"{show(path)} is now a git repository.")
    else:
        yield Say(f"git init failed: {summarize_errors(out)}")


def git_commit(frame: Frame, ctx: Context) -> Program:
    path = yield from _repo(frame, ctx)
    if path is None:
        return
    message = frame.slots.get("message")
    if not message:
        yield Say(f"What commit message should I use? Say something like “commit everything in {base(path)} as 'first draft'”.")
        return
    status = yield Run(f"git -C {q(path)} status --short", f"see what changed in {show(path)}")
    if status.errors:
        yield Say(f"{show(path)} isn't a git repository yet. Say “make {base(path)} a git repo” first." if "not a git" in status.text else f"git status failed: {summarize_errors(status)}")
        return
    if not status.lines:
        yield Say(f"Nothing to commit in {show(path)}; everything is already committed.")
        return
    out = yield Run(f"git -C {q(path)} add -A && git -C {q(path)} commit -m {q(message)}", f"commit {len(status.lines)} changes")
    log = yield Run(f"git -C {q(path)} log --oneline -n 1", "check the commit")
    yield Focus(path, "directory")
    if log.lines and message in log.lines[0]:
        yield Say(f"Committed {len(status.lines)} change{'s' * (len(status.lines) != 1)} in {show(path)}: {log.lines[0]}")
    else:
        yield Say(f"The commit didn't show up in the log: {summarize_errors(out)}")


def git_status(frame: Frame, ctx: Context) -> Program:
    path = yield from _repo(frame, ctx)
    if path is None:
        return
    out = yield Run(f"git -C {q(path)} status --short", f"git status in {show(path)}")
    yield Focus(path, "directory")
    if out.errors:
        yield Say(f"{show(path)} isn't a git repository." if "not a git" in out.text else f"git status failed: {summarize_errors(out)}")
    else:
        yield Say(f"No uncommitted changes in {show(path)}." if not out.lines else f"{len(out.lines)} uncommitted change{'s' * (len(out.lines) != 1)} in {show(path)}:\n" + "\n".join(out.lines[:20]))


def git_log(frame: Frame, ctx: Context) -> Program:
    path = yield from _repo(frame, ctx)
    if path is None:
        return
    out = yield Run(f"git -C {q(path)} log --oneline -n 10", f"git log in {show(path)}")
    yield Focus(path, "directory")
    if out.errors:
        yield Say(f"{show(path)} has no commits yet." if "does not have any commits" in out.text or "bad default revision" in out.text else f"git log failed: {summarize_errors(out)}")
    else:
        yield Say(f"Latest commits in {show(path)}:\n" + "\n".join(out.lines) if out.lines else f"{show(path)} has no commits yet.")




def run(frame: Frame, ctx: Context) -> Program:
    command = frame.slots["command"].strip()
    out = yield Run(command, "run your command")
    text = out.text.rstrip()
    if out.timed_out:
        yield Say(f"`{command}` is still running after 20 seconds; I stopped waiting. So far:\n{text[-1500:]}")
    else:
        yield Say(f"Ran `{command}`" + (f":\n{text[-2500:]}" if text else " (no output)."))


def install(frame: Frame, ctx: Context) -> Program:
    package = frame.slots["package"]
    out = yield Run(f"sudo apt install -y {q(package)}", f"install {package}")
    yield Say(f"apt said: {summarize_errors(out)}" if out.errors else f"Installed {package}." if out.lines else "apt finished without output.")


def setup_project(frame: Frame, ctx: Context) -> Program:
    from ..tasks.desktop import ProjectTask, plan_for, read_note
    import tensacode as tc

    task = read_note(frame.slots["note"])
    if not isinstance(task, ProjectTask):
        yield Say(f"I couldn't read that as a project setup ({task.detail}). Try: “set up a project called demo under ~/Projects. README.md should say 'Demo'. todo: first; second. commit with message 'init'”.")
        return
    plan = plan_for(task)
    order = tc.plan_order(plan)
    if isinstance(order, tc.Verdict):
        yield Say("That plan doesn't hold together, so I didn't run it: " + "; ".join(order.reasons))
        return
    outputs = []
    for step in plan.steps:
        out = yield Run(step.action.command, f"step {step.id}")
        outputs.append(out)
        if out.errors and "nothing to commit" not in out.text:
            yield Say(f"Step “{step.id}” failed: {summarize_errors(out)}. I stopped there.")
            return
    root = expand(f"{task.parent}/{task.name}", ctx)
    yield Focus(root, "directory")
    ok = task.title in outputs[-1].text and task.message in outputs[-1].text
    yield Say(f"Set up {show(root)} with README, notes/todo.txt ({len(task.items)} items) and a commit “{task.message}”." if ok else f"I ran every step, but reading back {show(root)} didn't show what I expected:\n{outputs[-1].text[:500]}")


def confirm_without_question(frame: Frame, ctx: Context) -> Program:
    yield from stray_answer(frame, ctx)


PROGRAMS = {
    "greet": greet, "thanks": thanks, "help": help_, "unknown": unknown, "confirm": stray_answer, "cancel": stray_answer, "choose": choose_without_question,
    "list": list_, "read": read, "create_folder": create_folder, "create_file": create_file, "write": write, "delete": delete,
    "move": move, "copy": move, "rename": rename, "find": find, "grep": grep, "count": count, "size": size, "info": info, "which": which,
    "cd": cd, "open_app": open_app, "git_init": git_init, "git_commit": git_commit, "git_status": git_status, "git_log": git_log,
    "run": run, "install": install, "setup_project": setup_project,
}
