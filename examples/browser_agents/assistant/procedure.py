"""Procedures as data: the step language, its guards, its templates, and its primitives.

A procedure is a list of steps. Each step names one thing to do, may carry a guard
(``when``) and may bind what it learns into the environment:

    {"do": "run", "command": "ls -1pA {path|q}", "why": "list {path|show}", "bind": "out"}
    {"when": {"is": ["{kind}", "directory"]}, "do": "call", "proc": "list_dir", ...}
    {"do": "say", "text": "{path|show} has {names|count} items: {names|listing}"}

Nothing here executes anything: this module is the vocabulary (what a step may say),
the reader for templates and guards, and the registry of primitives — the few named
Python functions that do real computation, so the procedure itself stays inspectable.
``interpreter.py`` walks a procedure with its state in the mind's claim graph.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

HOME = "/home/agent"

# a step's shape: which keys each `do` accepts, and which are templates
BODY_STEPS = ("run", "click", "fill", "look", "open_app", "teach", "sample")
MENTAL_STEPS = ("say", "ask", "compute", "remember", "recall", "changes", "focus", "call", "return", "stop")
STEP_KINDS = BODY_STEPS + MENTAL_STEPS


class ProcedureError(ValueError):
    """A procedure that does not hold together: an unknown step, guard or primitive."""


@dataclass
class Procedure:
    """One way of doing one kind of request, as data."""

    id: str
    steps: list[dict]
    act: str | None = None  # the request act this handles ("list", "delete", …); None for a sub-procedure
    takes: list[str] = field(default_factory=list)  # environment names it expects from its caller
    returns: list[str] = field(default_factory=list)  # names it may return
    author: str = "hand-written"  # or "learned-from-trace"
    version: int = 1
    provenance: dict = field(default_factory=dict)
    fixtures: list = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Procedure":
        return cls(**json.loads(text))

    def check(self) -> None:
        """Raise unless every step, guard and primitive in this procedure is known."""
        _check_steps(self.steps, self.id)


def _check_steps(steps: list[dict], where: str) -> None:
    for i, step in enumerate(steps):
        do = step.get("do")
        if do not in STEP_KINDS:
            raise ProcedureError(f"{where} step {i}: unknown step {do!r}")
        if "when" in step:
            _check_cond(step["when"], f"{where} step {i}")
        if do == "compute" and step.get("prim") not in PRIMITIVES:
            raise ProcedureError(f"{where} step {i}: unknown primitive {step.get('prim')!r}")


def _check_cond(cond: Any, where: str) -> None:
    if not isinstance(cond, dict) or len(cond) != 1:
        raise ProcedureError(f"{where}: a guard is one {{op: …}} pair, got {cond!r}")
    op, arg = next(iter(cond.items()))
    if op not in CONDITIONS:
        raise ProcedureError(f"{where}: unknown guard {op!r}")
    if op in ("not",):
        _check_cond(arg, where)
    if op in ("all", "any"):
        for c in arg:
            _check_cond(c, where)


# ------------------------------------------------------------------ templates


_TEMPLATE = re.compile(r"\{([a-zA-Z_][\w.]*)((?:\|[a-z_]+(?::[^}|]*)?)*)\}")


def q(s: str) -> str:
    return shlex.quote(str(s))


def show(path: str) -> str:
    path = str(path)
    return "~" + path[len(HOME):] if path == HOME or path.startswith(HOME + "/") else path


def base(path: str) -> str:
    return str(path).rstrip("/").rsplit("/", 1)[-1]


def parent(path: str) -> str:
    return str(path).rsplit("/", 1)[0] or "/"


def sentence(text: str) -> str:
    """Tidy a casually typed phrase into a sentence: 'im happy' -> "I'm happy"."""
    fixes = {"im": "I'm", "i": "I", "ive": "I've", "ill": "I'll", "id": "I'd", "dont": "don't",
             "cant": "can't", "wont": "won't", "thats": "that's", "youre": "you're", "its": "it's"}
    words = [fixes.get(w.lower(), w) if w.isalpha() else w for w in str(text).strip().split()]
    out = " ".join(words)
    return out[:1].upper() + out[1:] if out else out


def listing(names: Any, limit: int = 25) -> str:
    names = list(names or [])
    shown = ", ".join(str(n) for n in names[:limit])
    return shown + (f", … and {len(names) - limit} more" if len(names) > limit else "")


FILTERS: dict[str, Callable[..., Any]] = {
    "show": show,
    "q": q,
    "qlist": lambda v, *_: " ".join(q(x) for x in (v or [])),
    "base": base,
    "parent": parent,
    "sentence": sentence,
    "count": lambda v, *_: len(v or []) if not isinstance(v, str) else len(v),
    "listing": lambda v, limit="25": listing(v, int(limit)),
    "numbered": lambda v, *_: "\n".join(f"{i + 1}. {o}" for i, o in enumerate(v or [])),
    "lines": lambda v, *_: "\n".join(str(x) for x in (v or [])),
    "first": lambda v, *_: (list(v)[0] if v else ""),
    "plural": lambda v, word="item": f"{v} {word}{'s' * (v != 1)}",
    "s": lambda v, *_: "s" * (int(v or 0) != 1),
    "verb_s": lambda v, *_: "s" * (int(v or 0) == 1),  # subject-verb agreement: "1 file mentions", "2 files mention"
    "trim": lambda v, *_: str(v).rstrip(),
    "tail": lambda v, n="2500": str(v)[-int(n):],
    "head_lines": lambda v, n="20": "\n".join(str(v).splitlines()[: int(n)]),
    "json": lambda v, *_: json.dumps(v),
    "lower": lambda v, *_: str(v).lower(),
}


def lookup(env: dict, path: str) -> Any:
    """`out`, `stats.path`, `hits.0` — a dotted read from the environment."""
    cur: Any = env
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, (list, tuple)):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return cur


def resolve_value(spec: Any, env: dict) -> Any:
    """A template that is exactly one reference keeps the referenced object; otherwise it renders text."""
    if isinstance(spec, (list, tuple)):
        return [resolve_value(x, env) for x in spec]
    if isinstance(spec, dict):
        return {k: resolve_value(v, env) for k, v in spec.items()}
    if not isinstance(spec, str):
        return spec
    whole = _TEMPLATE.fullmatch(spec)
    if whole and not whole[2]:
        return lookup(env, whole[1])
    return render(spec, env)


def render(template: str, env: dict) -> str:
    def sub(m: re.Match) -> str:
        value = lookup(env, m[1])
        for f in [f for f in m[2].split("|") if f]:
            name, _, arg = f.partition(":")
            if name not in FILTERS:
                raise ProcedureError(f"unknown filter {name!r}")
            value = FILTERS[name](value, *( [arg] if arg else [] ))
        return "" if value is None else str(value)

    return _TEMPLATE.sub(sub, template)


# ------------------------------------------------------------------- guards


def _truthy(v: Any) -> bool:
    return bool(v) and v != "0"


CONDITIONS: dict[str, Callable[[Any, dict], bool]] = {
    "truthy": lambda a, env: _truthy(resolve_value(a, env)),
    "falsy": lambda a, env: not _truthy(resolve_value(a, env)),
    "eq": lambda a, env: resolve_value(a[0], env) == resolve_value(a[1], env),
    "ne": lambda a, env: resolve_value(a[0], env) != resolve_value(a[1], env),
    "gt": lambda a, env: _num(resolve_value(a[0], env)) > _num(resolve_value(a[1], env)),
    "is": lambda a, env: resolve_value(a[0], env) == resolve_value(a[1], env),
    "missing": lambda a, env: resolve_value(a, env) in (None, "", [], {}),
    "exists": lambda a, env: resolve_value(a, env) not in (None, "", [], {}),
    "contains": lambda a, env: str(resolve_value(a[1], env)) in str(resolve_value(a[0], env)),
    "matches": lambda a, env: bool(re.search(str(resolve_value(a[1], env)), str(resolve_value(a[0], env) or ""), re.I)),
    "not": lambda a, env: not holds(a, env),
    "all": lambda a, env: all(holds(c, env) for c in a),
    "any": lambda a, env: any(holds(c, env) for c in a),
}


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def holds(cond: Any, env: dict) -> bool:
    if cond is None:
        return True
    op, arg = next(iter(cond.items()))
    return CONDITIONS[op](arg, env)


# --------------------------------------------------------------- primitives

PRIMITIVES: dict[str, Callable[..., dict]] = {}


def primitive(name: str) -> Callable[[Callable[..., dict]], Callable[..., dict]]:
    """Register a named computation. Procedures may only compute through these."""

    def wrap(fn: Callable[..., dict]) -> Callable[..., dict]:
        PRIMITIVES[name] = fn
        return fn

    return wrap


ERROR = re.compile(
    r"No such file|not found|cannot |can't |File exists|Permission denied|Not a directory|Is a directory|"
    r"not a git repository|fatal:|error:|invalid option|unrecognized option|unknown object|not installed|"
    r"unable to|denied|not implemented|Directory not empty|nothing to commit", re.I)

PLACES = {
    "desktop": "~/Desktop", "documents": "~/Documents", "docs": "~/Documents", "downloads": "~/Downloads",
    "home": "~", "home folder": "~", "home directory": "~", "projects": "~/Projects", "pictures": "~/Pictures",
    "music": "~/Music", "videos": "~/Videos", "tmp": "/tmp", "temp": "/tmp",
}


def expand(p: str, cwd: str = HOME) -> str:
    p = str(p).strip()
    if p == "~" or p.startswith("~/"):
        return (HOME + p[1:]).rstrip("/") or "/"
    if p.startswith("/"):
        return p.rstrip("/") or "/"
    return f"{str(cwd).rstrip('/')}/{p}".rstrip("/")


@primitive("expand")
def _expand(path: str = "", cwd: str = HOME) -> dict:
    return {"path": expand(path, cwd)}


@primitive("strip_ref")
def _strip_ref(ref: str = "") -> dict:
    """'@recipes' -> 'recipes'; also says whether the word names a standard place."""
    name = str(ref)[1:] if str(ref).startswith("@") else str(ref)
    place = PLACES.get(name.lower())
    return {"name": name, "place_path": place, "looks_like_path": bool(re.search(r"^[~/]|/", name))}


@primitive("stat_map")
def _stat_map(out: str = "") -> dict:
    """Parse `stat -c '%F|%s|%n'` output into path -> [kind, size]."""
    found: dict[str, list] = {}
    for line in str(out).splitlines():
        if m := re.match(r"^(directory|regular file|symbolic link|[a-z ]+)\|(\d+)\|(.+)$", line.strip()):
            found[m[3]] = [m[1], int(m[2])]
    return {"stats": found}


@primitive("kind_of")
def _kind_of(stats: dict | None = None, path: str = "") -> dict:
    entry = (stats or {}).get(path)
    return {"kind": entry[0] if entry else None, "size": entry[1] if entry else None}


@primitive("confirmed")
def _confirmed(answer: str = "", answer_act: str | None = None) -> dict:
    """A yes is a yes however it is worded; anything else (including silence) is not."""
    from .programs import YES

    return {"yes": answer_act == "confirm" or bool(YES.match(str(answer or "").strip()))}


@primitive("command_risk")
def _command_risk(command: str = "") -> dict:
    """Does a command the user typed only read, or could it change the machine?"""
    from .programs import changes_things

    return {"risky": changes_things(str(command or ""))}


@primitive("protected_path")
def _protected_path(path: str = "") -> dict:
    """The standing folders are never deleted themselves, only emptied on request."""
    from .programs import PROTECTED

    return {"protected": path in PROTECTED}


@primitive("candidates")
def _candidates(name: str = "", place: str | None = None, cwd: str = HOME, known: dict | None = None) -> dict:
    """Where a bare name might live: the stated place, then this conversation, then the usual folders."""
    cands = [expand(f"{place}/{name}", cwd)] if place else []
    cands += [p for p in reversed(list(known or {})) if base(p) == name]
    cands += [f"{cwd}/{name}", f"{HOME}/{name}", f"{HOME}/Desktop/{name}", f"{HOME}/Documents/{name}"]
    return {"candidates": list(dict.fromkeys(cands))}


@primitive("hits")
def _hits(stats: dict | None = None, candidates: list | None = None, want: str | None = None,
          place: str | None = None, known: dict | None = None) -> dict:
    stats = stats or {}
    hits = [p for p in (candidates or []) if stats.get(p) and (want is None or stats[p][0] == want)]
    settled = bool(hits) and (bool(place) or len(hits) == 1 or hits[0] in (known or {}))
    return {"hits": hits, "settled": settled, "n_hits": len(hits)}


@primitive("paths_in_output")
def _paths_in_output(out: str = "", limit: int = 60) -> dict:
    """Absolute paths a find/grep printed, hidden ones dropped."""
    hits = [ln.strip() for ln in str(out).splitlines() if ln.strip().startswith("/") and "/." not in ln]
    return {"hits": hits[:limit], "n_hits": len(hits), "truncated": len(str(out).splitlines()) >= limit}


@primitive("at")
def _at(items: list | None = None, index: Any = 0) -> dict:
    items = list(items or [])
    try:
        i = int(index)
    except (TypeError, ValueError):
        return {"item": None}
    return {"item": items[i] if 0 <= i < len(items) else None}


@primitive("options")
def _options(hits: list | None = None, limit: int = 6) -> dict:
    return {"options": [show(h) for h in (hits or [])[:limit]]}


@primitive("choice")
def _choice(answer: str = "", options: list | None = None) -> dict:
    """Read '1', 'the second one', or a pasted path as a pick from the options offered."""
    options = list(options or [])
    words = str(answer or "").strip().lower()
    ordinals = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "fifth": 4,
                "sixth": 5, "last": len(options) - 1}
    pick: int | None = None
    if m := re.search(r"\b(\d+)\b", words):
        i = int(m[1]) - 1
        pick = i if 0 <= i < len(options) else None
    if pick is None:
        for word, i in ordinals.items():
            if re.search(rf"\b{word}\b", words):
                pick = i
                break
    if pick is None:
        for i, o in enumerate(options):
            if words and (words in o.lower() or o.lower().endswith(words)):
                pick = i
                break
    return {"pick": pick}


@primitive("output_facts")
def _output_facts(out: str = "", timed_out: bool = False) -> dict:
    """What a command's output tells us: its lines, whether it errored, and a one-line summary."""
    text = str(out or "")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    errors = [ln for ln in lines if ERROR.search(ln)]
    return {"out": text, "lines": lines, "errors": errors, "ok": not errors,
            "error_summary": (errors or lines or ["no output"])[0][:200], "timed_out": bool(timed_out)}


@primitive("ls_facts")
def _ls_facts(out: str = "") -> dict:
    names = [ln for ln in str(out).splitlines() if ln.strip()]
    folders = sum(n.endswith("/") for n in names)
    return {"names": names, "n_names": len(names), "folders": folders, "files": len(names) - folders}


@primitive("near_misses")
def _near_misses(name: str = "", names: list | None = None, known: dict | None = None) -> dict:
    """Names close to the one you asked for — a wrong name is usually a near miss, and naming it
    repairs the belief instead of ending the exchange (tensorcode.social.near_names)."""
    from tensorcode.social import near_names

    pool = [str(n).rstrip("/") for n in (names or [])]
    pool += [str(k).rsplit("/", 1)[-1] for k in (known or {})]
    near = near_names(str(name), dict.fromkeys(pool))
    joined = near[0] if len(near) == 1 else (", ".join(near[:-1]) + f" or {near[-1]}" if near else "")
    return {"near": near, "n_near": len(near), "near_joined": joined}


@primitive("name_stem")
def _name_stem(name: str = "", keep: int = 3) -> dict:
    """The first few characters of a name as a glob — a cheap net for a misspelling.

    Two-stage retrieval: the shell proposes candidates by prefix, ``near_names`` scores them.
    It is honest about its reach — a name whose opening characters are wrong is not caught.
    """
    stem = re.sub(r"[^\w.-]", "", str(name).rsplit("/", 1)[-1])[:keep]
    return {"stem_glob": f"{stem}*" if stem else "", "stem_wide": bool(stem)}


@primitive("basenames")
def _basenames(out: str = "", names: list | None = None) -> dict:
    """The pool of names something might have meant: what a find printed, plus a listing."""
    pool = [ln.strip().rsplit("/", 1)[-1] for ln in str(out).splitlines()
            if ln.strip().startswith("/") and "/." not in ln]
    pool += [str(n).rstrip("/") for n in (names or [])]
    return {"pool": list(dict.fromkeys(p for p in pool if p))}


@primitive("shared_prefix")
def _shared_prefix(subject: str = "", predicate: str = "", turn: object = 0, store: object = None) -> dict:
    """"As I mentioned, " when I have already said this, and nothing when it is new to you.

    The mind is passed in by the interpreter so grounding stays claims (tensorcode.social).
    """
    from tensorcode.social import I_SAID, CommonGround

    if store is None or not subject or not predicate:
        return {"prefix": "", "again": False}
    records = store.claims(__import__("tensorcode").Ref(str(subject)), str(predicate))
    if not records:
        return {"prefix": "", "again": False}
    ground, about = CommonGround(store), records[0].id
    again = ground.again(about)
    n = int(turn or 0)
    ground.add(about, I_SAID, turn=n, source=f"reply:{n}")  # saying it now is itself common ground
    return {"prefix": "As I mentioned, " if again else "", "again": again, "ground_about": about}


@primitive("goal_options")
def _goal_options(goal: str = "", names: list | None = None) -> dict:
    """The ways an underdetermined goal could be meant, priced for tensorcode.social.ask_or_act."""
    from tensorcode.social import Reading, ask_or_act

    readings = [Reading("by_type", 0.45, "group them into folders by kind", cost_if_wrong=1.0),
                Reading("by_date", 0.35, "group them into folders by month", cost_if_wrong=1.0),
                Reading("just_list", 0.20, "just show me what's there", cost_if_wrong=0.2)]
    decision = ask_or_act(readings, question="How would you like it organized?", about=str(goal))
    options = list(decision.clarification.options) if decision.clarification else []
    return {"decision": decision.choose, "options": options, "n_options": len(options),
            "options_numbered": "\n".join(f"{i + 1}. {o}" for i, o in enumerate(options)),
            "gain": round(decision.expected_gain, 3), "why_asking": decision.reason}


@primitive("organize_command")
def _organize_command(path: str = "", names: list | None = None, how: str = "by_type") -> dict:
    """One compound command that groups a folder's files, so the step language needs no loop."""
    import shlex

    files = [str(n) for n in (names or []) if not str(n).endswith("/")]
    groups: dict[str, list[str]] = {}
    for name in files:
        ext = name.rsplit(".", 1)[-1].lower() if "." in name[1:] else ""
        key = (ext or "other") if how == "by_type" else "dated"
        groups.setdefault(key, []).append(name)
    parts, moved = [], 0
    for key, members in sorted(groups.items()):
        folder = f"{path}/{key}"
        parts.append(f"mkdir -p {shlex.quote(folder)}")
        for name in members:
            parts.append(f"mv {shlex.quote(f'{path}/{name}')} {shlex.quote(f'{folder}/{name}')}")
            moved += 1
    return {"command": " && ".join(parts), "n_moved": moved, "n_groups": len(groups),
            "group_names": sorted(groups), "groups_joined": ", ".join(sorted(groups))}


@primitive("rm_all_command")
def _rm_all_command(path: str = "", names: list | None = None) -> dict:
    return {"command": " && ".join(f"rm -r {q(path + '/' + str(n).rstrip('/'))}" for n in (names or []))}


@primitive("place_for_new")
def _place_for_new(place: str | None = None, focus: str | None = None, focus_kind: str | None = None, cwd: str = HOME) -> dict:
    """Where a new file or folder goes when the request didn't say: the folder in focus, else the terminal's folder."""
    if place is not None:
        return {"place": place, "needs_resolving": True}
    return {"place": focus if focus_kind == "directory" and focus else cwd, "needs_resolving": False}


@primitive("info_reply")
def _info_reply(topic: str = "", out: str = "", lines: list | None = None) -> dict:
    """Turn one machine question's output into a sentence, or nothing if it doesn't parse."""
    lines = [ln for ln in (lines or []) if str(ln).strip()]
    first = lines[0].strip() if lines else ""
    text = None
    if topic == "date" and first:
        text = f"It's {first} on this machine."
    elif topic == "user" and first:
        text = f"You're logged in as “{first}”."
    elif topic == "disk" and len(lines) >= 2 and len(lines[-1].split()) >= 6:
        f = lines[-1].split()
        text = f"{f[3]} free of {f[1]} ({f[4]} used) on {f[5]}."
    elif topic == "ip":
        ips = [m[1] for ln in lines if (m := re.search(r"\binet (\d+\.\d+\.\d+\.\d+)", ln))]
        text = f"This machine's IP address is {', '.join(ips)}." if ips else None
    elif topic == "hostname" and first:
        text = f"This computer is called “{first}”."
    elif topic == "uptime" and first:
        text = f"Uptime: {first}"
    elif topic == "processes" and len(lines) > 1:
        text = f"{len(lines) - 1} processes are running, including {listing([ln.split()[-1] for ln in lines[1:]], 12)}."
    elif topic == "cpus" and first:
        text = f"This machine has {first} CPU cores."
    elif topic == "os" and first:
        text = first
    elif topic == "cwd" and first:
        text = f"The terminal is in {show(first)}."
    return {"reply": text}


INFO_COMMANDS = {
    "date": "date", "user": "whoami", "disk": "df -h ~", "ip": "ip addr", "hostname": "hostname",
    "uptime": "uptime", "processes": "ps -eo comm", "cpus": "nproc", "os": "uname -a", "cwd": "pwd",
}


@primitive("info_command")
def _info_command(topic: str = "") -> dict:
    return {"command": INFO_COMMANDS.get(topic, "true"), "topic": topic}


@primitive("first_number")
def _first_number(lines: list | None = None) -> dict:
    n = next((ln.strip() for ln in (lines or []) if str(ln).strip().isdigit()), None)
    return {"number": n}


@primitive("first_field")
def _first_field(lines: list | None = None) -> dict:
    m = re.match(r"^(\S+)\s", (lines or [""])[0] or "") if lines else None
    return {"field": m[1] if m else None}


@primitive("first_path")
def _first_path(lines: list | None = None) -> dict:
    return {"path_found": next((ln.strip() for ln in (lines or []) if str(ln).strip().startswith("/")), None)}


@primitive("text_or")
def _text_or(value: Any = None, fallback: Any = None) -> dict:
    return {"text": value if value not in (None, "") else fallback}


@primitive("truncated_note")
def _truncated_note(out: str = "", limit: int = 40) -> dict:
    n = len(str(out).splitlines())
    return {"more": f" (first {limit} lines)" if n >= limit else "", "empty": not str(out).strip()}


@primitive("contains_text")
def _contains_text(haystack: str = "", needle: Any = None) -> dict:
    return {"found": bool(needle) and str(needle) in str(haystack)}


@primitive("project_note")
def _project_note(note: str = "") -> dict:
    """Read a project-setup note and the shell steps that would build it (shared with the desktop task)."""
    from ..tasks.desktop import ProjectTask, plan_for, read_note

    task = read_note(str(note))
    if not isinstance(task, ProjectTask):
        return {"readable": False, "why": getattr(task, "detail", "not a project note")}
    plan = plan_for(task)
    return {"readable": True, "name": task.name, "parent": task.parent, "title": task.title,
            "message": task.message, "items": list(task.items), "n_items": len(task.items),
            "root": expand(f"{task.parent}/{task.name}"),
            "commands": [s.action.command for s in plan.steps], "step_ids": [s.id for s in plan.steps]}


@primitive("next_command")
def _next_command(commands: list | None = None, index: Any = 0, step_ids: list | None = None) -> dict:
    commands, index = list(commands or []), int(index or 0)
    return {"command": commands[index] if index < len(commands) else None,
            "step_id": (step_ids or [])[index] if index < len(step_ids or []) else None,
            "next_index": index + 1, "last": index >= len(commands) - 1}


@primitive("project_verified")
def _project_verified(out: str = "", title: str = "", message: str = "") -> dict:
    return {"verified": bool(title) and title in str(out) and bool(message) and message in str(out)}


@primitive("bare_path")
def _bare_path(words: str = "") -> dict:
    """Is a stray message just a path or filename? (then it is a request to show it)"""
    text = str(words).strip().strip("'\"`")
    ok = bool(re.fullmatch(r"~?/?[\w.@+ /-]*\.[A-Za-z0-9]{1,8}|~(?:/[\w.@+ -]+)*|/[\w.@+ /-]+", text))
    return {"is_path": ok, "path_text": text}


@primitive("find_on_screen")
def _find_on_screen(texts: list | None = None, value: str = "") -> dict:
    """'ada' -> 'Ada Kernel' when a text on screen starts a capitalized name with it."""
    for text in texts or []:
        if m := re.search(rf"\b((?i:{re.escape(str(value))})[\w'-]*(?:\s+[A-Z][a-z][\w'-]*)*)", str(text)):
            if m[1][:1].isupper():
                return {"full_name": m[1]}
    return {"full_name": None}


@primitive("mentions")
def _mentions(texts: list | None = None, labels: list | None = None, needle: str = "") -> dict:
    n = str(needle).lower()
    return {"mentioned": any(n in str(t).lower() for t in (texts or [])) or any(n in str(t).lower() for t in (labels or []))}


@primitive("plural_words")
def _plural_words(n: Any = 0, unit: str = "lines") -> dict:
    unit = str(unit)
    return {"unit": unit[:-1] if str(n) == "1" and unit.endswith("s") else unit}


@primitive("wc_flag")
def _wc_flag(unit: str = "lines") -> dict:
    return {"flag": "w" if str(unit) == "words" else "l"}


@primitive("find_flag")
def _find_flag(want: str | None = None) -> dict:
    return {"find_flag": " -type d" if want == "directory" else " -type f" if want == "regular file" else ""}


@primitive("wants_everything")
def _wants_everything(words: str = "") -> dict:
    return {"everything": bool(re.search(r"\b(?:everything|all (?:the )?(?:files|things|stuff))\b", str(words), re.I))}


@primitive("contents_note")
def _contents_note(kind: str | None = None, n: Any = 0) -> dict:
    if kind != "directory":
        return {"contents": "", "noun": "file"}
    n = int(n or 0)
    return {"contents": f" It contains {n} item{'s' * (n != 1)}." if n else " It's empty.", "noun": "folder"}


@primitive("rm_flag")
def _rm_flag(kind: str | None = None) -> dict:
    return {"rm": "-r " if kind == "directory" else ""}


@primitive("deleted_note")
def _deleted_note(before: Any = 0, after: Any = 0, names: list | None = None) -> dict:
    left = list(names or [])
    return {"deleted": int(before or 0) - int(after or 0),
            "still": f" Still there: {listing(left)}" if left else ""}


@primitive("suffix_note")
def _suffix_note(text: Any = None) -> dict:
    return {"tail": f" It now ends with “{text}”." if text else ""}


@primitive("copy_command")
def _copy_command(copy: Any = False, kind: str | None = None) -> dict:
    copy = bool(copy) and str(copy) != "False"
    return {"cmd": "cp -r" if copy and kind == "directory" else "cp" if copy else "mv",
            "verb": "Copied" if copy else "Moved", "verbing": "copy" if copy else "move"}


@primitive("is_copy")
def _is_copy(act: str = "") -> dict:
    return {"copy": act == "copy"}


@primitive("join_path")
def _join_path(folder: str = "", name: str = "") -> dict:
    return {"path": f"{str(folder).rstrip('/')}/{base(name)}"}


@primitive("new_name_ok")
def _new_name_ok(new_name: Any = None) -> dict:
    name = str(new_name or "").strip()
    return {"new_name": name, "usable": bool(name) and "/" not in name}


@primitive("git_message")
def _git_message(text: str = "", message: Any = None) -> dict:
    return {"in_log": bool(message) and str(message) in str(text)}


@primitive("head_limit")
def _head_limit(out: str = "", limit: int = 40) -> dict:
    n = len(str(out).splitlines())
    return {"more": f" (first {limit} lines)" if n >= limit else "", "blank": not str(out).strip()}


@primitive("shown_paths")
def _shown_paths(paths: list | None = None) -> dict:
    return {"shown": [show(p) for p in (paths or [])]}


@primitive("new_file_guess")
def _new_file_guess(target: Any = None, focus: str | None = None, focus_kind: str | None = None,
                    cwd: str = HOME, known: dict | None = None) -> dict:
    """A plain filename can name a file that does not exist yet: guess where the conversation means."""
    name = str(target or "")
    guessable = bool(name) and name != "@it" and not name.startswith("@") and "/" not in name and "." in name
    if not guessable:
        return {"guessable": False, "guess": None, "known_path": None}
    seen = [p for p in (known or {}) if base(p) == name]
    here = focus if focus_kind == "directory" and focus else cwd
    return {"guessable": True, "guess": f"{str(here).rstrip('/')}/{name}", "known_path": seen[-1] if seen else None}


@primitive("set_values")
def _set_values(**values: Any) -> dict:
    """Bind literal or rendered values (the arguments are already resolved against the environment)."""
    return values


@primitive("skill_reply")
def _skill_reply(skill_id: str = "", slots: dict | None = None, reply: str = "") -> dict:
    """The learned skill's own reply template, filled, with its standing noted."""
    from .learning import LIBRARY, fill

    skill = next((s for s in LIBRARY.skills if s.id == skill_id), None)
    values = {k: v for k, v in (slots or {}).items() if k != "skill"}
    text = fill(reply, values) + " (learned skill, no model used)."
    if skill is not None and skill.status == "trial":
        learned_from = skill.provenance.get("learned_from_slots", {})
        novel = any(str(values.get(k, "")).lower() != str(v).lower() for k, v in learned_from.items())
        text += " It worked on a new case, so I'm now trusting this skill." if novel else " (Still on trial until it works on a different case.)"
    return {"reply_text": text}


@primitive("lift_slots")
def _lift_slots(slots: dict | None = None) -> dict:
    """Learned skills template on slot names directly ({message}), so put them in the environment."""
    return {k: v for k, v in (slots or {}).items() if k != "skill"}


# ------------------------------------------------- answering, not only acting
#
# What a question needs is not a command but a look at what is already believed: the
# primitives below read what a ``recall`` step pulled out of the mind (screen claims,
# things you were told, what the assistant itself did) and turn it into an answer, or
# say plainly that the belief is not there. Nothing here invents a fact.


CHROME = re.compile(
    r"^(?:Resize window \w+|Minimize|Maximi[sz]e|Close|New tab|Computers|Simulator settings|"
    r"Remove from tab|Activities|System menu|Show Applications|Back|Forward|Up|Grid|List|"
    r"\w{3}, \w{3} \d+ .*[AP]M)$")


def _boxed(items: list | None) -> list[dict]:
    return [i for i in (items or []) if isinstance(i, dict) and i.get("box")]


@primitive("screen_regions")
def _screen_regions(items: list | None = None, texts: list | None = None) -> dict:
    """Group what is on screen the way a person would: a launcher strip, windows, the rest.

    The strip is found by geometry, not by a hardcoded list: buttons in a narrow column at
    the left edge, tall enough to be a launcher. So it survives a different desktop.
    """
    boxed = _boxed(items)
    buttons = [i for i in boxed if i.get("role") == "button" and not CHROME.match(str(i.get("object") or ""))]
    left = [b for b in buttons if b["box"][0] < 90 and b["box"][2] < 90]
    # a launcher is a column of same-sized, same-x buttons; anything else at the left edge
    # (a tab label, a window button) belongs to a different group and is dropped
    lanes: dict[tuple[int, int], list[dict]] = {}
    for b in left:
        lanes.setdefault((b["box"][0] // 8, b["box"][2] // 8), []).append(b)
    biggest = max(lanes.values(), key=len) if lanes else []
    column = sorted(biggest if len(biggest) >= 3 else left, key=lambda b: b["box"][1])
    windows: dict[str, int] = {}
    for i in _boxed(texts) + boxed:  # a window announces itself through the text inside it
        w = str(i.get("section") or "")
        if w and not CHROME.match(w):
            windows[w] = windows.get(w, 0) + 1
    return {
        "dock_items": [str(b.get("object") or "") for b in column],
        "dock_n": len(column),
        "dock_box": [min((b["box"][0] for b in column), default=0), min((b["box"][1] for b in column), default=0),
                     max((b["box"][0] + b["box"][2] for b in column), default=0) - min((b["box"][0] for b in column), default=0),
                     max((b["box"][1] + b["box"][3] for b in column), default=0) - min((b["box"][1] for b in column), default=0)] if column else None,
        "windows": sorted(windows, key=lambda w: -windows[w]),
        "windows_n": len(windows),
        "controls_n": len(buttons),
    }


#: named colours, so a sampled pixel can be reported in words rather than in numbers
COLOR_NAMES = {
    "black": (16, 16, 18), "very dark grey": (40, 42, 46), "dark grey": (80, 82, 88), "grey": (128, 128, 132),
    "light grey": (190, 192, 196), "white": (248, 248, 250), "dark blue": (24, 42, 96), "blue": (48, 96, 200),
    "light blue": (130, 180, 235), "teal": (32, 132, 132), "green": (60, 150, 70), "dark green": (28, 82, 44),
    "yellow": (225, 205, 70), "orange": (230, 140, 50), "brown": (120, 80, 50), "red": (200, 60, 55),
    "dark red": (120, 34, 34), "pink": (235, 150, 175), "purple": (130, 70, 180), "magenta": (200, 60, 170),
    "aubergine": (72, 40, 62),
}


def color_word(rgb: Any) -> str:
    r, g, b = (int(x) for x in list(rgb)[:3])
    return min(COLOR_NAMES, key=lambda name: sum((c - v) ** 2 for c, v in zip(COLOR_NAMES[name], (r, g, b))))


@primitive("color_words")
def _color_words(colors: list | None = None, unavailable: Any = None) -> dict:
    """Name the colours a sample found, biggest share first, with the share kept."""
    if unavailable:
        return {"named": [], "main_color": None, "palette": "", "unavailable": str(unavailable)}
    named, seen = [], set()
    for entry in colors or []:
        rgb = list(entry)[:3]
        share = float(list(entry)[3]) if len(list(entry)) > 3 else 0.0
        word = color_word(rgb)
        if word in seen:
            continue
        seen.add(word)
        named.append({"name": word, "share": round(share, 3), "rgb": [int(x) for x in rgb]})
    return {"named": named, "main_color": named[0]["name"] if named else None,
            "palette": ", ".join(f"{n['name']} ({round(n['share'] * 100)}%)" for n in named[:4]),
            "unavailable": None}


def _telling_order(rows: list, store: object) -> list:
    """Oldest first, by when you told me it — read off common ground, not off storage order.

    "In order" is a fact about the exchange rather than about the database, which is why the
    listing consults the ground; with no ground to consult it falls back to the timestamps.
    """
    if store is None:
        return sorted(rows, key=lambda i: str(i.get("when") or ""))
    from tensorcode.social import CommonGround

    ground = CommonGround(store)

    def told_at(row: dict) -> tuple:
        shared = ground.status(str(row.get("id"))) if row.get("id") else None
        return (shared.turn if shared else 0, str(row.get("when") or ""))

    return sorted(rows, key=told_at)


@primitive("told_facts")
def _told_facts(items: list | None = None, topic: Any = None, store: object = None) -> dict:
    """What you told me: one topic's value (most recent wins), or all of it in telling order."""
    rows = [i for i in (items or []) if isinstance(i, dict)]
    rows.sort(key=lambda i: str(i.get("when") or ""), reverse=True)
    if topic:
        want = str(topic).strip().lower()
        hit = next((i for i in rows if str(i.get("predicate", "")).lower() == want), None)
        return {"value": hit.get("object") if hit else None, "told_in": hit.get("when") if hit else None,
                "n_facts": len(rows), "facts": "", "topics": [str(i.get("predicate")) for i in rows]}
    in_order = _telling_order(rows, store)
    return {"value": None, "told_in": None, "n_facts": len(rows),
            "facts": "\n".join(f"• your {i.get('predicate')} is {i.get('object')}" for i in in_order[:15]),
            "topics": [str(i.get("predicate")) for i in in_order]}


@primitive("match_function")
def _match_function(items: list | None = None, phrase: str = "") -> dict:
    """Which app is 'used for writing code'? Token overlap against what apps are known to be for."""
    stop = {"the", "a", "an", "for", "to", "is", "are", "used", "use", "using", "that", "which", "app",
            "application", "program", "something", "can", "with", "my", "i", "me", "do", "on", "in", "of"}
    want = {w for w in re.findall(r"[a-z]+", str(phrase).lower()) if w not in stop and len(w) > 1}
    scored = []
    for i in items or []:
        if not isinstance(i, dict):
            continue
        have = {w for w in re.findall(r"[a-z]+", str(i.get("object", "")).lower()) if w not in stop}
        if not have:
            continue
        overlap = len(want & have)
        if overlap:
            scored.append((overlap / len(want | have), overlap, str(i.get("subject", "")).split("app:")[-1]))
    scored.sort(reverse=True)
    best = scored[0] if scored else None
    tie = len(scored) > 1 and scored[1][0] == scored[0][0] if scored else False
    return {"app": None if not best or tie else best[2],
            "app_score": round(best[0], 3) if best else 0.0,
            "candidates": [name for _, _, name in scored[:4]],
            "ambiguous": bool(tie)}


@primitive("window_contents")
def _window_contents(texts: list | None = None, window: str = "") -> dict:
    """What one window says, in reading order, from the text perceived inside it."""
    want = str(window).strip().lower()
    rows = [i for i in _boxed(texts) if want and want in str(i.get("section") or "").lower()]
    rows.sort(key=lambda i: (i["box"][1], i["box"][0]))
    lines, seen = [], set()
    for i in rows:
        line = " ".join(str(i.get("object") or "").split())
        if line and line not in seen:
            seen.add(line)
            lines.append(line)
    return {"window_lines": lines[:20], "window_n_lines": len(lines),
            "window_found": bool(rows), "window_name": next((str(i.get("section")) for i in rows), None)}


@primitive("named_window")
def _named_window(windows: list | None = None, window: str = "") -> dict:
    """Is the window someone named actually there? Matched loosely, because people shorten names."""
    want = str(window).strip().lower()
    hit = next((w for w in (windows or []) if want and (want in str(w).lower() or str(w).lower() in want)), None)
    return {"matched_window": hit, "window_is_open": bool(hit)}


@primitive("own_actions")
def _own_actions(items: list | None = None, limit: int = 6) -> dict:
    """What I did, newest last, read off the commands and clicks I actually issued."""
    def order(item: dict) -> tuple:
        tail = re.findall(r"(\d+)", str(item.get("subject", "")))
        return (str(item.get("when") or ""), [int(t) for t in tail])

    rows = [i for i in (items or []) if isinstance(i, dict) and i.get("object")]
    rows.sort(key=order)
    done = [str(i["object"]) for i in rows][-int(limit):]
    n = len(done)
    return {"actions": done, "n_actions": len(rows), "last_action": done[-1] if done else None,
            "last_subject": rows[-1].get("subject") if rows else None,
            "phrase": f"{n} thing{'s' * (n != 1)}"}

