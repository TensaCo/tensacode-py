"""Learning new skills from one teacher-guided attempt, then replaying them without a model.

    unknown request ──► learn():  look ─► teacher proposes ONE step ─► validate ─► (ask if it has an effect) ─► act ─► look …
                                   on success: teacher proposes a request pattern with slots
                                   compile(): generalize the trace over the slots, check it round-trips, save with provenance
    matching request ─► run_skill(): the same steps, filled from the new slots, no model; still asks before effects,
                                   still verifies on screen; if the skill does not fit, it says why and learns again

The step language is deliberately small and checkable: open_app, click(label), fill(label, text),
run(command), find_name(slot) and expect(text). Labels must be visible on screen when used.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Generator

from . import programs as P
from .language import Frame

PROMPT_VERSION = "learn-steps-v1"


@dataclass
class Settings:
    """How hard the learning loop tries. The chat assistant uses the defaults."""

    max_steps: int = 12
    plan: bool = False  # ask for subgoals first and keep them in memory
    notes: bool = False  # let the teacher keep notes across steps
    files: bool = False  # write_file / read_file actions (long work needs to author and read files)
    long_output_hint: bool = False  # tell the teacher how to read output too long for one screen
    compile_skill: bool = True  # turn a success into a reusable skill


SETTINGS = Settings()


def configure(*, long_horizon: bool = False) -> None:
    """Switch the loop between chat-sized requests and long-horizon work."""
    global SETTINGS
    from . import language

    from . import agent

    SETTINGS = Settings(max_steps=80, plan=True, notes=True, files=True, long_output_hint=True, compile_skill=False) if long_horizon else Settings()
    language.TREAT_LONG_AS_TASK = long_horizon
    agent.BODY.run_timeout_s = 600.0 if long_horizon else agent.RUN_TIMEOUT_S
SKILLS_PATH = Path(os.environ.get("TENSORCODE_SKILLS", Path.home() / ".local" / "share" / "tensorcode" / "assistant-skills.json"))
MAX_STEPS = 12  # default; SETTINGS.max_steps is what the loop uses
EFFECT_WORDS = re.compile(r"\b(?:send|post|submit|delete|remove|trash|buy|pay|order|confirm|save|apply|publish|share|install|uninstall|reply|forward|archive|move)\b", re.I)  # marks steps in the trace; not a gate
DOCK = ("Files", "Firefox", "Chromium", "Mail", "Rhythmbox", "Terminal", "Text Editor", "Visual Studio Code", "Slack", "App Center", "Settings", "System Monitor", "Wireshark")

SYSTEM = """You operate a simulated Ubuntu desktop for a user, one step at a time, and only through these actions:
  {"do": "open_app", "app": "<dock app name>"}
  {"do": "click", "label": "<a label copied exactly from SCREEN>"}
  {"do": "fill", "label": "<a textbox label copied exactly from SCREEN>", "text": "<text to type>"}
  {"do": "run", "command": "<one bash command>"}
  {"do": "done", "reply": "<short message telling the user what you did>"}
  {"do": "give_up", "reply": "<why this cannot be done here>"}
EXTRA (long work only, when offered in ACTIONS below):
  {"do": "write_file", "path": "/app/...", "content": "<the whole file>"}
  {"do": "read_file", "path": "/app/...", "start": <first line>, "lines": <how many>}
  {"do": "note", "text": "<something worth remembering later>"}
  {"do": "subgoal_done", "id": <number>}
Reply with ONE JSON object: {"thought": "<one sentence>", ...the action fields..., "effect": true|false}.
effect=true for anything that sends, posts, deletes, buys or changes data. Only use labels that appear in SCREEN.
Look at WHAT CHANGED after each step: if a step changed nothing, try something else. Say done only after SCREEN shows the result."""

PLAN_SYSTEM = """You plan work on a Linux machine before starting.
Reply with ONE JSON object: {"plan": ["<subgoal 1>", "<subgoal 2>", ...]} - between 3 and 10 subgoals,
each one a concrete, checkable step. No commentary."""

PATTERN_SYSTEM = """You turn a request that was just carried out into a reusable request pattern.
Reply with ONE JSON object: {"pattern": "<the request with the variable parts replaced by {slot_name}>", "slots": {"slot_name": "<exact text from the request>"}}.
Slot names are lowercase words like person, message, folder, app. Keep fixed words (verbs, prepositions) in the pattern. Every slot value must be copied exactly from the request."""


# ------------------------------------------------------------------ skills


@dataclass
class Skill:
    id: str
    pattern: str
    slots: list[str]
    steps: list[dict]
    reply: str
    provenance: dict
    stats: dict = field(default_factory=lambda: {"uses": 0, "successes": 0, "failures": 0})
    status: str = "trial"  # trial -> adopted after succeeding on a request it was not learned from; retracted on a trial failure
    history: list = field(default_factory=list)  # (when, event, detail): adoption and retraction are recorded, never silent

    def regex(self) -> re.Pattern:
        tokens = self.pattern.strip().rstrip(".!?").split()
        slot_tokens = [i for i, t in enumerate(tokens) if re.fullmatch(r"\{\w+\}", t)]
        parts = []
        for i, t in enumerate(tokens):
            if i in slot_tokens:
                name = t[1:-1]
                # the last slot takes the rest; earlier slots take one or two words
                parts.append(f"(?P<{name}>.+)" if i == slot_tokens[-1] and i == len(tokens) - 1 else f"(?P<{name}>[\\w@.'’+-]+(?:\\s+[\\w@.'’+-]+)??)")
            else:
                parts.append(re.escape(t))
        return re.compile(r"^\s*(?:please\s+|can you\s+|could you\s+|would you\s+)?" + r"\s+".join(parts) + r"\s*[.!?]*\s*$", re.I)

    def match(self, text: str) -> dict[str, str] | None:
        m = self.regex().match(text)
        return {k: v.strip() for k, v in m.groupdict().items()} if m else None


class Library:
    def __init__(self, path: Path = SKILLS_PATH) -> None:
        self.path = path
        self.skills: list[Skill] = []
        if path.exists():
            self.skills = [Skill(**s) for s in json.loads(path.read_text())]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(s) for s in self.skills], indent=1))
        tmp.replace(self.path)

    def match(self, text: str) -> tuple[Skill, dict[str, str]] | None:
        found = [(s, slots) for s in self.skills if s.status != "retracted" and (slots := s.match(text)) is not None]
        # prefer the most specific pattern (most literal characters), then the most reliable
        found.sort(key=lambda x: (-len(re.sub(r"\{\w+\}", "", x[0].pattern)), -(x[0].stats["successes"] - x[0].stats["failures"])))
        return found[0] if found else None

    def add(self, skill: Skill) -> None:
        self.skills = [s for s in self.skills if s.id != skill.id] + [skill]
        self.save()


LIBRARY = Library()


# ------------------------------------------------------------ text helpers


def sentence(text: str) -> str:
    """Tidy a casually typed phrase into a sentence: 'im happy' -> "I'm happy"."""
    fixes = {"im": "I'm", "i": "I", "ive": "I've", "ill": "I'll", "id": "I'd", "dont": "don't", "cant": "can't", "wont": "won't", "thats": "that's", "youre": "you're", "its": "it's"}
    words = [fixes.get(w.lower(), w) if w.isalpha() else w for w in text.strip().split()]
    out = " ".join(words)
    return out[:1].upper() + out[1:] if out else out


def fill(template: str, values: dict[str, str]) -> str:
    def sub(m: re.Match) -> str:
        name, _, filt = m[1].partition("|")
        value = values.get(name, "")
        return sentence(value) if filt == "sentence" else value

    return re.sub(r"\{(\w+(?:\|\w+)?)\}", sub, template)


def _loose(value: str) -> re.Pattern:
    """Match a typed value inside tidied text: letters in order, optional apostrophes, flexible spaces."""
    parts = []
    for ch in value.strip():
        if ch.isspace():
            parts.append(r"\s+")
        elif ch.isalnum():
            parts.append(re.escape(ch) + r"['’]?")
        else:
            parts.append(re.escape(ch) + "?")
    return re.compile("".join(parts).replace(r"\s+\s+", r"\s+"), re.I)


def find_name(texts: list[str], value: str) -> str | None:
    """'ada' -> 'Ada Kernel' when a text on screen starts a capitalized name with it."""
    for text in texts:
        if m := re.search(rf"\b((?i:{re.escape(value)})[\w'-]*(?:\s+[A-Z][a-z][\w'-]*)*)", text):
            if m[1][:1].isupper():
                return m[1]
    return None


def _objects(text: str) -> list[dict]:
    """Every complete top-level {...} object in a reply (models wrap JSON in prose, fences or thinking)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    found, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
            start = i if start is None else start
        elif ch == "}" and start is not None:
            depth -= 1
            if depth == 0:
                try:
                    out = json.loads(text[start : i + 1])
                    if isinstance(out, dict):
                        found.append(out)
                except json.JSONDecodeError:
                    pass
                start = None
    return found


def _json(text: str | None, *, wants: tuple[str, ...] = ("do", "action", "plan", "pattern")) -> dict | None:
    """The first object that actually carries an action (or plan/pattern), unwrapped and normalised."""
    if not text:
        return None
    candidates = _objects(text)
    for obj in candidates:
        for wrapper in ("step", "tool_call", "function_call", "command_object"):
            if isinstance(obj.get(wrapper), dict):
                obj = {**obj, **obj[wrapper]}
        if "do" not in obj and isinstance(obj.get("action"), str):
            obj["do"] = obj.pop("action")
        if any(k in obj for k in wants):
            return obj
    return candidates[0] if candidates else None


def diff(before: P.Seen | None, after: P.Seen) -> str:
    if before is None:
        return "(first look)"
    gone = [t for t in before.texts if t not in after.texts]
    new = [t for t in after.texts if t not in before.texts]
    new_controls = [c for c in after.controls if c not in before.controls]
    if not gone and not new and not new_controls:
        return "nothing changed on screen"
    parts = []
    if new:
        parts.append("new text: " + " | ".join(" ".join(t.split())[:100] for _, t in new[:12]))
    if new_controls:
        parts.append("new controls: " + " | ".join(label for _, label in new_controls[:12] if label))
    if gone:
        parts.append(f"{len(gone)} texts disappeared")
    return "; ".join(parts)


# ------------------------------------------------------------------ learning


def learn(frame: Frame, ctx: P.Context, *, note: str = "") -> P.Program:
    request = frame.words
    cfg = SETTINGS
    yield P.Say(("" if not note else note + " ") + "I don't know how to do that yet, so I'll work it out once with my local teacher model and remember how. I'll tell you each step as I go.")
    trace: list[dict] = []  # executed steps, what the screen showed before each, and what each observably changed
    history: list[str] = []
    notes: list[str] = []
    plan: list[dict] = []
    invalid = 0
    front: str | None = None
    failed: set[str] = set()
    change = "(first look)"
    now = yield P.Look()

    if cfg.plan:
        answer = yield P.Teach(PLAN_SYSTEM, f"TASK:\n{request}\n\nWHAT IS ON SCREEN NOW:\n{now.render(dock=DOCK)}")
        proposed = (_json(answer) or {}).get("plan") or []
        plan = [{"id": i + 1, "text": str(t)[:200], "done": False} for i, t in enumerate(proposed) if str(t).strip()][:10]
        if plan:
            yield P.Remember(tuple((f"plan:{request[:40]}", "subgoal", f"{p['id']}. {p['text']}") for p in plan), why="plan the work")
            yield P.Say("Plan:\n" + "\n".join(f"  {p['id']}. {p['text']}" for p in plan))

    for _ in range(cfg.max_steps):
        prompt = (f"REQUEST: {request}\nDOCK APPS: {', '.join(DOCK)}\n"
                  + (f"ACTIONS: open_app, click, fill, run, write_file, read_file, note, subgoal_done, done, give_up\n" if cfg.files else "ACTIONS: open_app, click, fill, run, done, give_up\n")
                  + (_plan_text(plan) if plan else "")
                  + (("NOTES:\n" + "\n".join(f"  - {n}" for n in notes[-12:]) + "\n") if cfg.notes and notes else "")
                  + (LONG_OUTPUT_HINT if cfg.long_output_hint else "")
                  + "STEPS SO FAR:\n" + ("\n".join(history[-14:]) or "  (none)")
                  + f"\nWHAT CHANGED AFTER THE LAST STEP: {change}\nSCREEN:\n{now.render(dock=DOCK, front=front)}")
        answer = yield P.Teach(SYSTEM, prompt, max_new_tokens=1200 if cfg.files else 300)
        if answer is None:
            yield P.Say("My teacher model isn't reachable, so I can't learn this right now.")
            return
        step = _json(answer)
        problem = _validate(step, now, cfg) or (f"{describe(step)} already failed; try something different" if step and describe(step) in failed else None)
        if problem:
            invalid += 1
            history.append(f"  (rejected proposal: {problem})")
            if invalid >= (6 if cfg.files else 3):
                yield P.Say(f"I couldn't work out how to do “{request}”: the teacher's suggestions didn't fit the screen ({problem}).")
                return
            continue
        invalid = 0
        do = step["do"]
        if do == "give_up":
            yield P.Say(f"I couldn't do “{request}”. The teacher's reason: {step.get('reply', 'no way found')}")
            return
        if do == "note":
            notes.append(str(step.get("text", ""))[:300])
            yield P.Remember(((f"work:{request[:40]}", "note", notes[-1]),), why="remember something")
            history.append(f"  note: {notes[-1][:100]}")
            continue
        if do == "subgoal_done":
            for p in plan:
                if p["id"] == step.get("id"):
                    p["done"] = True
                    yield P.Remember(((f"plan:{request[:40]}", "done", f"{p['id']}. {p['text']}"),), why="subgoal finished")
                    history.append(f"  subgoal {p['id']} marked done")
            continue
        if do == "done":
            done = [t for t in trace if t["ok"]]
            if not done:
                yield P.Say(f"I didn't manage to do anything for “{request}”.")
                return
            left = [p for p in plan if not p["done"]]
            yield P.Say(observed_summary(done) + (f"\nSubgoals still open: {', '.join(str(p['id']) for p in left)}." if left else "")
                        + f"\n(The teacher's own summary, not verified: “{step.get('reply', '')}”)")
            if cfg.compile_skill:
                yield from _compile(request, trace, observed_summary(done))
            return
        effect = bool(step.get("effect")) or bool(EFFECT_WORDS.search(step.get("label", "") + " " + step.get("command", "")))
        result = yield from _execute(step)
        after = yield P.Look()
        change = diff(now, after)
        if result is True and do in ("click", "open_app") and change == "nothing changed on screen":
            result = "it went through, but nothing changed on screen"  # a click that does nothing is not progress
        if result is not True:
            failed.add(describe(step))
        if do == "open_app" and result is True:
            front = step["app"]
        history.append(f"  {describe(step)} -> {'ok' if result is True else result}")
        trace.append({**{k: v for k, v in step.items() if k in ("do", "app", "label", "text", "command", "path")}, "effect": effect, "ok": result is True,
                      "screen": [t for _, t in now.texts], "change": change})
        now = after
    yield P.Say(f"I gave up on “{request}” after {cfg.max_steps} steps without finishing.")


LONG_OUTPUT_HINT = ("READING LONG OUTPUT: only the last part of a long output stays on screen. Send output you need to read to a file "
                    "(`cmd > /app/tmp/out.txt 2>&1`) and read it in slices with read_file, or filter it (grep/tail/sed).\n")


def _plan_text(plan: list[dict]) -> str:
    return "PLAN:\n" + "\n".join(f"  {'[x]' if p['done'] else '[ ]'} {p['id']}. {p['text']}" for p in plan) + "\n"


def observed_summary(steps: list[dict]) -> str:
    """What was done and seen, from the trace alone (never from the teacher's claims)."""
    parts = [describe(t) for t in steps]
    typed = [t for t in steps if t["do"] == "fill"]
    seen_after = typed and any(t.get("text") and t["text"] in (s.get("change") or "") for s in steps[steps.index(typed[-1]) + 1:] for t in typed[-1:])
    tail = f" “{typed[-1]['text']}” now appears on screen." if seen_after else (" I did not see the typed text appear on screen." if typed else "")
    return "I " + ", then ".join(parts) + "." + tail


def _validate(step: dict | None, now: P.Seen, cfg: "Settings" = None) -> str | None:  # noqa: RUF013
    cfg = cfg or SETTINGS
    if step is None:
        return "not a JSON action"
    do = step.get("do")
    allowed = ["open_app", "click", "fill", "run", "done", "give_up"] + (["write_file", "read_file", "note", "subgoal_done"] if cfg.files else [])
    if do not in allowed:
        return f"unknown action {do!r}"
    if do == "write_file":
        if not str(step.get("path", "")).startswith("/app/"):
            return "write_file only works under /app/"
        if not isinstance(step.get("content"), str):
            return "write_file needs content"
    if do == "read_file" and not str(step.get("path", "")).startswith("/app/"):
        return "read_file only works under /app/"
    if do == "open_app" and step.get("app") not in DOCK:
        return f"no dock app called {step.get('app')!r}"
    if do in ("click", "fill"):
        label = step.get("label") or ""
        if not any(label == lab for _, lab in now.controls):
            close = [lab for _, lab in now.controls if label and label.lower() in lab.lower()]
            if len(close) != 1:
                return f"label {label!r} is not on screen"
            step["label"] = close[0]
    if do == "fill" and not isinstance(step.get("text"), str):
        return "fill needs text"
    if do == "run":
        command = str(step.get("command", ""))
        if not command.strip():
            return "run needs a command"
        if "\n" in command:
            return "a command must be one line" + (" - use write_file for file contents" if cfg.files else "")
    return None


def describe(step: dict) -> str:
    do = step.get("do")
    if do == "open_app":
        return f"open {step.get('app')}"
    if do == "click":
        return f"click “{step.get('label')}”"
    if do == "fill":
        return f"type “{step.get('text')}” into “{step.get('label')}”"
    if do == "run":
        return f"run `{step.get('command')}`"
    if do == "write_file":
        return f"write {len(str(step.get('content', '')))} bytes to {step.get('path')}"
    if do == "read_file":
        return f"read {step.get('path')} from line {step.get('start', 1)}"
    if do == "find_name":
        return f"find {step.get('slot')} on screen"
    if do == "expect":
        return f"check the screen shows “{step.get('text')}”"
    return str(do)


def _execute(step: dict) -> Generator[object, object, object]:
    do = step["do"]
    if do == "open_app":
        opened = yield P.OpenApp(step["app"])
        return True if opened else f"{step['app']} did not open"
    if do == "click":
        return (yield P.Click(step["label"]))
    if do == "fill":
        return (yield P.Fill(step["label"], step["text"], submit=bool(step.get("submit"))))
    if do == "run":
        out = yield P.Run(step["command"], "run a learned command")
        return True if not out.errors else P.summarize_errors(out)
    if do == "write_file":
        path, content = step["path"], step["content"]
        payload = base64.b64encode(content.encode()).decode()
        out = yield P.Run(f"mkdir -p $(dirname {P.q(path)}) && python -c \"import base64,pathlib; pathlib.Path({path!r}).write_bytes(base64.b64decode('{payload}'))\" && wc -l {P.q(path)}",
                          f"write {path}")
        return True if not out.errors else P.summarize_errors(out)
    if do == "read_file":
        start = max(1, int(step.get("start", 1) or 1))
        lines = max(1, min(200, int(step.get("lines", 40) or 40)))
        out = yield P.Run(f"sed -n '{start},{start + lines - 1}p' {P.q(step['path'])}", f"read {step['path']}")
        return True if not out.errors else P.summarize_errors(out)
    return f"cannot execute {do}"


def _compile(request: str, trace: list[dict], reply: str) -> P.Program:
    answer = yield P.Teach(PATTERN_SYSTEM, f"REQUEST: {request}\nSTEPS: " + json.dumps([{k: v for k, v in t.items() if k in ('do', 'app', 'label', 'text', 'command')} for t in trace]))
    proposal = _json(answer) or {}
    pattern, slots = proposal.get("pattern"), proposal.get("slots") or {}
    skill, problem = compile_skill(request, trace, reply, pattern, slots)
    if skill is None:
        yield P.Say(f"(I didn't save this as a skill: {problem}.)")
        return
    LIBRARY.add(skill)
    yield P.Say(f"Learned “{skill.pattern}” ({len(skill.steps)} steps). Next time I'll do it without the teacher model; it stays on trial until it works on a different case.")


def compile_skill(request: str, trace: list[dict], reply: str, pattern: str | None, slots: dict) -> tuple[Skill | None, str]:
    """Generalize an executed trace over request slots; accept only if it round-trips exactly."""
    ok_steps = [t for t in trace if t.get("ok")]
    if not ok_steps:
        return None, "no step succeeded"
    if not pattern or not isinstance(slots, dict) or not slots:
        return None, "no request pattern with slots"
    probe = Skill("probe", pattern, list(slots), [], "", {})
    try:
        matched = probe.match(request)
    except re.error as exc:
        return None, f"pattern is not usable ({exc})"
    if matched is None or {k: v.lower() for k, v in matched.items()} != {k: str(v).lower() for k, v in slots.items()}:
        return None, f"pattern {pattern!r} does not reproduce the slots {slots} from the request"
    steps: list[dict] = []
    bound: dict[str, str] = {}
    values = {k: str(v) for k, v in slots.items()}
    for t in ok_steps:
        step = {k: v for k, v in t.items() if k in ("do", "app", "label", "text", "command")} | {"effect": t["effect"]}
        for key in ("label", "text", "command"):
            if key not in step:
                continue
            s = step[key]
            for name, value in values.items():
                # a name seen on screen that the user abbreviated ('ada' -> 'Ada Kernel')
                full = find_name(t["screen"], value)
                if full and full.lower() != value.lower() and full in s:
                    if f"{name}_name" not in bound:
                        steps.append({"do": "find_name", "slot": name, "bind": f"{name}_name", "effect": False})
                        bound[f"{name}_name"] = full
                    s = s.replace(full, "{" + name + "_name}")
                if re.fullmatch(re.escape(value), s, re.I):
                    s = "{" + name + "}"
                elif (m := _loose(value).search(s)) and m[0] != value and sentence(value) == m[0]:
                    s = s[: m.start()] + "{" + name + "|sentence}" + s[m.end():]
                elif re.search(rf"(?<!\w){re.escape(value)}(?!\w)", s, re.I):
                    s = re.sub(rf"(?<!\w){re.escape(value)}(?!\w)", "{" + name + "}", s, flags=re.I)
            step[key] = s
        steps.append(step)
    # the typed text of the final effect should show up on screen afterwards
    effects = [s for s in steps if s.get("effect")]
    fills = [s for s in steps if s["do"] == "fill" and "{" in s.get("text", "")]
    if effects and fills:
        steps.append({"do": "expect", "text": fills[-1]["text"], "effect": False})
    used = {m for s in steps for v in s.values() if isinstance(v, str) for m in re.findall(r"\{(\w+?)(?:_name)?(?:\|\w+)?\}", v)}
    unused = [k for k in values if k not in used]
    if unused:
        return None, f"the steps never use {', '.join(unused)}, so the skill would ignore that part of a request"
    # round trip: re-instantiating with the original values must give back what was executed
    env = values | bound
    replay = [fill(s[k], env) for s in steps if s["do"] not in ("find_name", "expect") for k in ("label", "text", "command", "app") if k in s]
    original = [t[k] for t in ok_steps for k in ("label", "text", "command", "app") if k in t]
    if replay != original:
        return None, "generalized steps don't reproduce what I did"
    digest = hashlib.sha256(json.dumps([request, steps], sort_keys=True).encode()).hexdigest()[:10]
    return Skill(
        id=f"skill-{digest}",
        pattern=pattern,
        slots=list(values),
        steps=steps,
        reply=_generalize_text(reply, values, bound),
        provenance={"learned_from": request, "learned_from_slots": values, "teacher": "local teacher model", "prompt_version": PROMPT_VERSION, "learned_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    "trace": [{k: v for k, v in t.items() if k != "screen"} for t in trace]},
    ), ""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _generalize_text(text: str, values: dict[str, str], bound: dict[str, str]) -> str:
    for name, full in bound.items():
        text = text.replace(full, "{" + name + "}")
    for name, value in values.items():
        if (m := _loose(value).search(text)) and m[0] != value and sentence(value) == m[0]:
            text = text[: m.start()] + "{" + name + "|sentence}" + text[m.end():]
        text = re.sub(rf"(?<!\w){re.escape(value)}(?!\w)", "{" + name + "}", text, flags=re.I)
    return text


# ------------------------------------------------------------------ replay


def run_skill(frame: Frame, ctx: P.Context) -> P.Program:
    skill = next((s for s in LIBRARY.skills if s.id == frame.slots.get("skill")), None)
    if skill is None:
        yield from learn(frame, ctx)
        return
    values = {k: v for k, v in frame.slots.items() if k != "skill"}
    skill.stats["uses"] += 1
    for step in skill.steps:
        do = step["do"]
        if do == "find_name":
            now = yield P.Look()
            full = find_name([t for _, t in now.texts], values.get(step["slot"], ""))
            if full is None:
                problem = f"I couldn't find “{values.get(step['slot'])}” on screen"
                break
            values[step["bind"]] = full
            continue
        if do == "expect":
            want = fill(step["text"], values)
            now = yield P.Look(settle_ms=600)
            if not now.mentions(want):
                problem = f"after doing it I don't see “{want}” on screen"
                break
            continue
        concrete = {k: (fill(v, values) if isinstance(v, str) else v) for k, v in step.items()}
        result = yield from _execute(concrete)
        if result is not True:
            problem = f"the step “{describe(concrete)}” failed: {result}"
            break
    else:
        skill.stats["successes"] += 1
        learned_from = skill.provenance.get("learned_from_slots", {})
        novel = any(str(values.get(k, "")).lower() != str(v).lower() for k, v in learned_from.items())
        status_note = ""
        if skill.status == "trial" and novel:
            skill.status = "adopted"
            skill.history.append([_now(), "adopted", f"succeeded on new values {({k: values[k] for k in skill.slots})}"])
            status_note = " It worked on a new case, so I'm now trusting this skill."
        elif skill.status == "trial":
            status_note = " (Still on trial until it works on a different case.)"
        LIBRARY.save()
        yield P.Say(fill(skill.reply, values) + " (learned skill, no model used)." + status_note)
        return
    skill.stats["failures"] += 1
    if skill.status == "trial":
        skill.status = "retracted"
        skill.history.append([_now(), "retracted", problem])
    else:
        skill.history.append([_now(), "failed", problem])
    LIBRARY.save()
    yield from learn(frame, ctx, note=f"My learned way to do this didn't fit here ({problem}).")


# ---------------------------------------------------- skills as procedures


def _fail(reason_template: str, extra: dict | None = None) -> list[dict]:
    """The two steps every failing branch of a replay shares: record why, then hand back to learning."""
    return [{"do": "compute", "prim": "set_values", "args": {"outcome": "failure", "reason": reason_template, **(extra or {})}},
            {"do": "say", "text": "My learned way to do this didn't fit here ({reason}).", "and": "stop"}]


def procedure_for_skill(skill_id: str | None) -> "P_Procedure | None":
    """A learned skill, as the same kind of procedure the hand-written ones are."""
    from .procedure import Procedure

    skill = next((s for s in LIBRARY.skills if s.id == skill_id), None)
    if skill is None:
        return None
    steps: list[dict] = [{"do": "compute", "prim": "set_values", "args": {"skill_id": skill.id, "outcome": "running"}},
                         {"do": "compute", "prim": "lift_slots", "args": {"slots": "{slot}"}}]
    for raw in skill.steps:
        do = raw["do"]
        if do == "find_name":
            steps += [
                {"do": "look"},
                {"do": "compute", "prim": "find_on_screen", "args": {"texts": "{screen_texts}", "value": "{slot." + raw["slot"] + "}"}},
                {"when": {"missing": "{full_name}"}, "do": "compute", "prim": "set_values",
                 "args": {"outcome": "failure", "reason": "I couldn't find “{slot." + raw["slot"] + "}” on screen"}},
                {"when": {"missing": "{full_name}"}, "do": "say", "text": "My learned way to do this didn't fit here ({reason}).", "and": "stop"},
                {"do": "compute", "prim": "set_values", "args": {raw["bind"]: "{full_name}"}},
            ]
        elif do == "expect":
            steps += [
                {"do": "look", "settle_ms": 600},
                {"do": "compute", "prim": "mentions", "args": {"texts": "{screen_texts}", "labels": "{screen_labels}", "needle": raw["text"]}},
                {"when": {"falsy": "{mentioned}"}, "do": "compute", "prim": "set_values",
                 "args": {"outcome": "failure", "reason": "after doing it I don't see “" + raw["text"] + "” on screen"}},
                {"when": {"falsy": "{mentioned}"}, "do": "say", "text": "My learned way to do this didn't fit here ({reason}).", "and": "stop"},
            ]
        elif do == "open_app":
            steps += [{"do": "open_app", "app": raw["app"]},
                      {"when": {"falsy": "{opened}"}, "do": "compute", "prim": "set_values",
                       "args": {"outcome": "failure", "reason": "“" + raw["app"] + "” did not open"}},
                      {"when": {"falsy": "{opened}"}, "do": "say", "text": "My learned way to do this didn't fit here ({reason}).", "and": "stop"}]
        elif do in ("click", "fill"):
            step = {"do": do, "label": raw["label"]}
            if do == "fill":
                step |= {"text": raw.get("text", ""), "submit": bool(raw.get("submit"))}
            steps += [step,
                      {"when": {"falsy": "{ok}"}, "do": "compute", "prim": "set_values",
                       "args": {"outcome": "failure", "reason": "the step “" + describe(raw) + "” failed: {problem}"}},
                      {"when": {"falsy": "{ok}"}, "do": "say", "text": "My learned way to do this didn't fit here ({reason}).", "and": "stop"}]
        elif do == "run":
            steps += [{"do": "run", "command": raw["command"], "why": "run a learned command"},
                      {"when": {"falsy": "{ok}"}, "do": "compute", "prim": "set_values",
                       "args": {"outcome": "failure", "reason": "the step “run `" + raw["command"] + "`” failed: {error_summary}"}},
                      {"when": {"falsy": "{ok}"}, "do": "say", "text": "My learned way to do this didn't fit here ({reason}).", "and": "stop"}]
        else:  # a step kind this replay does not know: say so rather than skip it silently
            steps += [{"do": "compute", "prim": "set_values", "args": {"outcome": "failure", "reason": f"I don't know how to replay a “{do}” step"}},
                      {"do": "say", "text": "My learned way to do this didn't fit here ({reason}).", "and": "stop"}]
    steps += [
        {"do": "compute", "prim": "set_values", "args": {"outcome": "success"}},
        {"do": "compute", "prim": "skill_reply", "args": {"skill_id": "{skill_id}", "slots": "{slot}", "reply": skill.reply}},
        {"do": "say", "text": "{reply_text}"},
    ]
    return Procedure(id=f"skill:{skill.id}", steps=steps, author="learned-from-trace", version=1,
                     provenance=dict(skill.provenance, skill=skill.id, status=skill.status, pattern=skill.pattern),
                     fixtures=[skill.provenance.get("learned_from_slots", {})])


def record_outcome(mind, req, env: dict, cycle: int):
    """After a learned skill runs: update its statistics, adopt it, or retract it — recorded, never silent."""
    from tensorcode.cognition import Thought

    skill_id, outcome = env.get("skill_id"), env.get("outcome")
    if not skill_id or outcome not in ("success", "failure"):
        return Thought()
    skill = next((s for s in LIBRARY.skills if s.id == skill_id), None)
    if skill is None:
        return Thought()
    values = {k: v for k, v in (env.get("slot") or {}).items() if k != "skill"}
    if outcome == "success":
        skill.stats["successes"] += 1
        learned_from = skill.provenance.get("learned_from_slots", {})
        if skill.status == "trial" and any(str(values.get(k, "")).lower() != str(v).lower() for k, v in learned_from.items()):
            skill.status = "adopted"
            skill.history.append([_now(), "adopted", f"succeeded on new values {values}"])
    else:
        skill.stats["failures"] += 1
        if skill.status == "trial":
            skill.status = "retracted"
            skill.history.append([_now(), "retracted", str(env.get("reason"))[:200]])
        else:
            skill.history.append([_now(), "failed", str(env.get("reason"))[:200]])
    skill.stats["uses"] += 1
    LIBRARY.save()
    return Thought()


# ------------------------------------------------- the learner, as data too

from .procedure import primitive as _prim  # noqa: E402  (registered where the learner's primitives are defined)


def _screen(texts: list | None, labels: list | None, windows: list | None) -> P.Seen:
    """Rebuild the compact screen view the teacher prompt wants from what the look step bound."""
    texts, labels, windows = list(texts or []), list(labels or []), list(windows or [])
    pairs = tuple(zip(windows + [""] * len(texts), texts))[: len(texts)]
    return P.Seen(tuple(("", lab) for lab in labels), pairs)


@_prim("learn_settings")
def _learn_settings(note: str = "") -> dict:
    cfg = SETTINGS
    opening = ((note + " ") if note else "") + ("I don't know how to do that yet, so I'll work it out once with my local "
                                                "teacher model and remember how. I'll tell you each step as I go.")
    return {"opening": opening, "max_steps": cfg.max_steps, "compile_enabled": cfg.compile_skill,
            "trace": [], "history": [], "invalid": 0, "change": "(first look)", "front": None, "step_no": 1,
            "system": SYSTEM}


@_prim("learn_prompt")
def _learn_prompt(request: str = "", history: list | None = None, change: str = "", front: str | None = None,
                  screen_texts: list | None = None, screen_labels: list | None = None, screen_windows: list | None = None) -> dict:
    cfg = SETTINGS
    seen = _screen(screen_texts, screen_labels, screen_windows)
    prompt = (f"REQUEST: {request}\nDOCK APPS: {', '.join(DOCK)}\n"
              + "ACTIONS: open_app, click, fill, run, done, give_up\n"
              + "STEPS SO FAR:\n" + ("\n".join(list(history or [])[-14:]) or "  (none)")
              + f"\nWHAT CHANGED AFTER THE LAST STEP: {change}\nSCREEN:\n{seen.render(dock=DOCK, front=front)}")
    return {"prompt": prompt, "system": SYSTEM, "max_new_tokens": 1200 if cfg.files else 300}


@_prim("learn_read_step")
def _learn_read_step(reply: str | None = None, screen_labels: list | None = None, history: list | None = None,
                     invalid: Any = 0, failed: list | None = None) -> dict:
    """Read the teacher's proposal, check it against the screen, and say what it asked for."""
    history, failed, invalid = list(history or []), list(failed or []), int(invalid or 0)
    step = _json(reply)
    seen = _screen([], screen_labels, [])
    problem = _validate(step, seen, SETTINGS)
    if not problem and step and describe(step) in failed:
        problem = f"{describe(step)} already failed; try something different"
    if problem:
        return {"problem": problem, "invalid": invalid + 1, "history": history + [f"  (rejected proposal: {problem})"],
                "do": None, "app": None, "label": None, "text": None, "command": None, "reply_text": None}
    return {"problem": None, "invalid": 0, "history": history, "do": step.get("do"), "app": step.get("app"),
            "label": step.get("label"), "text": step.get("text", ""), "command": step.get("command"),
            "submit": bool(step.get("submit")), "effect": bool(step.get("effect")), "reply_text": step.get("reply", "")}


@_prim("learn_record")
def _learn_record(do: str | None = None, app: str | None = None, label: str | None = None, text: Any = None,
                  command: str | None = None, effect: Any = False, trace: list | None = None, history: list | None = None,
                  failed: list | None = None, step_no: Any = 1, front: str | None = None,
                  before_texts: list | None = None, before_labels: list | None = None,
                  after_texts: list | None = None, after_labels: list | None = None,
                  opened: Any = None, ok: Any = None, problem: Any = None, out: str = "", error_summary: str = "") -> dict:
    """One executed step: what changed on screen, whether it counts as progress, and the growing trace."""
    trace, history, failed = list(trace or []), list(history or []), list(failed or [])
    before, after = _screen(before_texts, before_labels, []), _screen(after_texts, after_labels, [])
    change = diff(before, after)
    if do is None:  # the proposal was rejected; nothing ran
        return {"trace": trace, "history": history, "failed": failed, "change": change, "front": front, "step_no": int(step_no) + 1}
    if do == "open_app":
        result: object = True if opened else f"{app} did not open"
    elif do == "run":
        result = True if str(error_summary or "") in ("", "no output") or ok else str(error_summary)
        result = True if ok else str(error_summary)
    else:
        result = True if ok else str(problem)
    if result is True and do in ("click", "open_app") and change == "nothing changed on screen":
        result = "it went through, but nothing changed on screen"
    shown = describe({"do": do, "app": app, "label": label, "text": text, "command": command})
    return {"trace": trace + [{"do": do, "app": app, "label": label, "text": text, "command": command,
                               "effect": bool(effect), "ok": result is True, "screen": list(before_texts or []), "change": change}],
            "history": history + [f"  {shown} -> {'ok' if result is True else result}"],
            "failed": failed if result is True else failed + [shown],
            "change": change, "front": app if do == "open_app" and result is True else front,
            "step_no": int(step_no) + 1}


@_prim("learn_summary")
def _learn_summary(trace: list | None = None) -> dict:
    done = [t for t in (trace or []) if t.get("ok")]
    return {"summary": observed_summary(done) if done else "", "any_ok": bool(done)}


@_prim("learn_pattern_prompt")
def _learn_pattern_prompt(request: str = "", trace: list | None = None) -> dict:
    steps = [{k: v for k, v in t.items() if k in ("do", "app", "label", "text", "command")} for t in (trace or [])]
    return {"system": PATTERN_SYSTEM, "prompt": f"REQUEST: {request}\nSTEPS: " + json.dumps(steps)}


@_prim("learn_compile")
def _learn_compile(request: str = "", trace: list | None = None, summary: str = "", reply: str | None = None) -> dict:
    proposal = _json(reply) or {}
    skill, problem = compile_skill(request, list(trace or []), summary, proposal.get("pattern"), proposal.get("slots") or {})
    if skill is None:
        return {"saved": False, "problem": problem, "pattern": None, "n_steps": 0}
    LIBRARY.add(skill)
    return {"saved": True, "problem": None, "pattern": skill.pattern, "n_steps": len(skill.steps)}


def _learner_procedures() -> dict:
    from .procedure import Procedure

    learn = Procedure(id="learn", takes=["request", "note"], steps=[
        {"do": "compute", "prim": "learn_settings", "args": {"note": "{note}"}},
        {"do": "say", "text": "{opening}"},
        {"do": "call", "proc": "learn_turn", "with": {
            "request": "{request}", "trace": "{trace}", "history": "{history}", "failed": [],
            "invalid": "{invalid}", "change": "{change}", "front": "{front}", "step_no": "{step_no}",
            "max_steps": "{max_steps}", "compile_enabled": "{compile_enabled}"}, "bind": {}},
    ])
    turn = Procedure(id="learn_turn", takes=["request", "trace", "history", "failed", "invalid", "change", "front", "step_no", "max_steps", "compile_enabled"], steps=[
        {"when": {"gt": ["{step_no}", "{max_steps}"]}, "do": "say",
         "text": "I gave up on “{request}” after {max_steps} steps without finishing.", "and": "stop"},
        {"do": "look", "as": "before"},
        {"do": "compute", "prim": "learn_prompt", "args": {
            "request": "{request}", "history": "{history}", "change": "{change}", "front": "{front}",
            "screen_texts": "{before_screen_texts}", "screen_labels": "{before_screen_labels}", "screen_windows": "{before_screen_windows}"}},
        {"do": "teach", "system": "{system}", "user": "{prompt}", "max_new_tokens": 300},
        {"when": {"missing": "{reply}"}, "do": "say", "text": "My teacher model isn't reachable, so I can't learn this right now.", "and": "stop"},
        {"do": "compute", "prim": "learn_read_step", "args": {
            "reply": "{reply}", "screen_labels": "{before_screen_labels}", "history": "{history}",
            "invalid": "{invalid}", "failed": "{failed}"}},
        {"when": {"all": [{"exists": "{problem}"}, {"gt": ["{invalid}", 2]}]}, "do": "say",
         "text": "I couldn't work out how to do “{request}”: the teacher's suggestions didn't fit the screen ({problem}).", "and": "stop"},
        {"when": {"eq": ["{do}", "give_up"]}, "do": "say", "text": "I couldn't do “{request}”. The teacher's reason: {reply_text}", "and": "stop"},
        {"when": {"eq": ["{do}", "done"]}, "do": "call", "proc": "learn_finish", "with": {
            "request": "{request}", "trace": "{trace}", "reply_text": "{reply_text}", "compile_enabled": "{compile_enabled}"}, "bind": {}},
        {"when": {"eq": ["{do}", "done"]}, "do": "stop"},
        {"when": {"eq": ["{do}", "open_app"]}, "do": "open_app", "app": "{app}"},
        {"when": {"eq": ["{do}", "click"]}, "do": "click", "label": "{label}"},
        {"when": {"eq": ["{do}", "fill"]}, "do": "fill", "label": "{label}", "text": "{text}"},
        {"when": {"eq": ["{do}", "run"]}, "do": "run", "command": "{command}", "why": "run a step the teacher proposed"},
        {"do": "look", "as": "after"},
        {"do": "compute", "prim": "learn_record", "args": {
            "do": "{do}", "app": "{app}", "label": "{label}", "text": "{text}", "command": "{command}", "effect": "{effect}",
            "trace": "{trace}", "history": "{history}", "failed": "{failed}", "step_no": "{step_no}", "front": "{front}",
            "before_texts": "{before_screen_texts}", "before_labels": "{before_screen_labels}",
            "after_texts": "{after_screen_texts}", "after_labels": "{after_screen_labels}",
            "opened": "{opened}", "ok": "{ok}", "problem": "{problem}", "out": "{out}", "error_summary": "{error_summary}"}},
        {"do": "call", "proc": "learn_turn", "with": {
            "request": "{request}", "trace": "{trace}", "history": "{history}", "failed": "{failed}",
            "invalid": "{invalid}", "change": "{change}", "front": "{front}", "step_no": "{step_no}",
            "max_steps": "{max_steps}", "compile_enabled": "{compile_enabled}"}, "bind": {}},
    ])
    finish = Procedure(id="learn_finish", takes=["request", "trace", "reply_text", "compile_enabled"], steps=[
        {"do": "compute", "prim": "learn_summary", "args": {"trace": "{trace}"}},
        {"when": {"falsy": "{any_ok}"}, "do": "say", "text": "I didn't manage to do anything for “{request}”.", "and": "return"},
        {"do": "say", "text": "{summary}\n(The teacher's own summary, not verified: “{reply_text}”)"},
        {"when": {"falsy": "{compile_enabled}"}, "do": "return", "values": {}},
        {"do": "compute", "prim": "learn_pattern_prompt", "args": {"request": "{request}", "trace": "{trace}"}},
        {"do": "teach", "system": "{system}", "user": "{prompt}", "max_new_tokens": 300},
        {"do": "compute", "prim": "learn_compile", "args": {"request": "{request}", "trace": "{trace}", "summary": "{summary}", "reply": "{reply}"}},
        {"when": {"falsy": "{saved}"}, "do": "say", "text": "(I didn't save this as a skill: {problem}.)", "and": "return"},
        {"do": "say", "text": "Learned “{pattern}” ({n_steps} steps). Next time I'll do it without the teacher model; it stays on trial until it works on a different case."},
    ])
    return {p.id: p for p in (learn, turn, finish)}


LEARNER_PROCS = _learner_procedures()


def learner_procedure(frame, note: str = ""):
    """The teacher-guided loop, as a procedure: its state is claims like every other procedure's."""
    return LEARNER_PROCS["learn"]
