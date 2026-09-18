"""A promptable Ubuntu assistant, as a mind with no model calls.

    hear       : chat text -> request frames (forgiving grammar) -> claims about the conversation
    perceive   : the desktop's accessible structure -> scene graph (snapshot scope)
    think      : rules read the terminal transcript into command outputs and file-system facts
    deliberate : pick the next act for the oldest open request (start, advance, answer, drop)
    act        : procedures (procedures.py, data) are walked by interpreter.py: commands, clicks, questions, replies
    answer     : a question is a procedure too — its steps query the mind (``recall``) or a
                 modality (``sample``, pixels) instead of running a command, and every answer
                 names where it came from. What it has no belief for it says so, telling apart
                 "didn't understand", "can't do that" and "don't know yet".

Memory persists across turns, so "it", "there" and "that folder" mean what the conversation
last touched, and a pending question ("Delete ~/x?") is answered by the next message.
"""

from __future__ import annotations

import itertools
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Generator

import tensorcode as tc
from tensorcode.cognition import Fragment, Rule, Thought, integrate
from tensorcode.records import Evidence, Patch, Retract, Tell

from tensorcode.change import WINDOW_CLOSED, Watcher, attribute_windows, items_from_claims, snapshot
from tensorcode.memory import Memory, MemoryPolicy
from tensorcode.permanence import OBJECTS, Objects

from ..mind import SCREEN, BY_PRIORITY, Finish, MindSpec, Outcome, Press, Wait, controls, knowledge, one, run_mind
from . import interpreter as I
from . import procedure as L
from . import procedures as PR
from . import programs as P
from .language import _RELATIVE, Frame, act_is_affordable, indirect_frame, parse_message
from .memory import derive, note, set_state  # noqa: F401  (re-exported: other modules import them from here)

from tensorcode import control as C

ME = tc.Ref("agent:self")
PROMPT = re.compile(r"^(\S+@[\w.-]+):(\S*)\$\s?(.*)$", re.S)
TERMINAL_HEADERS = re.compile(r"^(?:Terminal|bash · .*)$", re.S)
RUN_TIMEOUT_S = 20.0  # chat default; long-horizon work raises it (see BODY.run_timeout_s)
V = tc.Var


# ------------------------------------------------------------ memory helpers


# ------------------------------------------------------------------ hearing


def hear(mind: tc.Store, text: str, turn: int) -> Thought:
    """Language -> graph: the utterance, its request frames, and whether it answers a pending question."""
    frames = parse_message(text)
    utterance = tc.Ref(f"utterance:{turn}")
    claims = [tc.Claim(utterance, "text", text), tc.Claim(utterance, "from", "user")]
    awaiting = one(mind, ME, "awaiting")
    if awaiting is not None and len(frames) == 1 and frames[0].act == "read" and frames[0].slots.get("target") == text.strip().rstrip(".!?"):
        frames = [Frame("choose", text.strip())]  # a bare path while a "which one?" is open is the answer
    from . import learning

    for i, f in enumerate(frames):
        whole = text if len(frames) == 1 else f.words
        if f.act != "unknown":
            continue
        if (grammar := _read_with_grammar(whole)) and act_is_affordable(whole, grammar):
            frames[i] = grammar  # the symbolic parser: reported speech, negation, modality
        elif hit := learning.LIBRARY.match(whole):  # skills learned earlier cover the rest
            skill, slots = hit
            frames[i] = Frame("learned", f.words, {"skill": skill.id, **slots})
        elif indirect := indirect_frame(whole):  # a request in a form that is not an order
            frames[i] = indirect
    for k, frame in enumerate(frames):
        req = tc.Ref(f"request:{turn}.{k}")
        claims += [tc.Claim(req, "part_of", utterance), tc.Claim(req, "order", (turn, k)), tc.Claim(req, "act", frame.act), tc.Claim(req, "words", frame.words)]
        claims += [tc.Claim(req, f"slot:{name}", value) for name, value in frame.slots.items()]
        if k == 0 and awaiting is not None and frame.act in ("confirm", "cancel", "choose"):
            claims.append(tc.Claim(req, "answers", awaiting))
    thought = note(mind, claims, f"utterance:{turn}")
    for k, frame in enumerate(frames):
        req = tc.Ref(f"request:{turn}.{k}")
        thought += set_state(mind, req, "status", "new", f"utterance:{turn}")
        # what arbitration weighs: how much work this ask is, read off the procedure that answers it
        proc = procedure_for(frame)
        C.weigh(mind, req, source=f"utterance:{turn}", what=frame.words,
                cost_to_go=float(len(proc.steps)) if proc is not None else 0.0)
    return thought


def _read_with_grammar(text: str) -> Frame | None:
    """Second reader: a unification grammar + chart parser (src/tensorcode/language), used only where the rules abstain."""
    try:
        from tensorcode.language.domains.desktop import read_request
    except Exception:  # noqa: BLE001 - the grammar is optional
        return None
    try:
        acts, _ = read_request(text)
    except Exception:  # noqa: BLE001 - a parser bug must not take down a turn
        return None
    # a demonstrative or pronoun, not relative "that" ("the app that is used ..."), which points at nothing
    refers = re.search(r"\b(?:it|this|these|those|them|there|here)\b|\b(?:that|those)\s*(?:$|[.,?!])|\bthat (?:one|file|folder|directory|repo|project)\b", text, re.I)
    for act in acts:
        if act.act not in PR.BY_ACT or act.act == "unknown":
            continue
        slots = {k: v for k, v in act.slots.items() if v is not None}
        if REQUIRED_SLOTS.get(act.act, frozenset()) - set(slots):
            continue  # the reading dropped the thing the act acts on: saying nothing beats acting on a default
        if not refers and any(isinstance(v, str) and v.startswith("@it") for v in slots.values()):
            continue  # a reading that invents a reference the sentence never made is a guess
        if _RELATIVE.search(text) and any(isinstance(v, str) and re.fullmatch(r"[a-z]+", v) for v in slots.values()):
            continue  # "the app that is used for writing code": a described thing, not a named one
        return Frame(act.act, text, slots)
    return None


#: an act may only come from the grammar when the reading actually filled what it acts on
REQUIRED_SLOTS = {
    "list": frozenset({"place"}), "read": frozenset({"target"}), "delete": frozenset({"target"}),
    "create_folder": frozenset({"name"}), "create_file": frozenset({"name"}), "write": frozenset({"target"}),
    "move": frozenset({"target", "dest"}), "copy": frozenset({"target", "dest"}), "rename": frozenset({"target", "new_name"}),
    "find": frozenset({"pattern"}), "grep": frozenset({"needle"}), "count": frozenset({"target"}), "size": frozenset({"target"}),
    "which": frozenset({"program"}), "cd": frozenset({"target"}), "open_app": frozenset({"app"}), "run": frozenset({"command"}),
    "install": frozenset({"package"}), "info": frozenset({"topic"}),
}


def frame_of(mind: tc.Store, req: tc.Ref) -> Frame:
    slots = {r.claim.predicate[5:]: r.claim.object for r in mind.claims(req) if r.claim.predicate.startswith("slot:")}
    return Frame(one(mind, req, "act"), one(mind, req, "words"), slots)


def context_of(mind: tc.Store) -> P.Context:
    known = {r.claim.subject.id[5:]: r.claim.object for r in sorted(mind.claims(predicate="is_a"), key=lambda r: r.evidence[-1].observed_at) if r.claim.subject.id.startswith("path:")}
    focus = one(mind, ME, "focus")
    return P.Context(cwd=one(mind, ME, "cwd", P.HOME), focus=focus, focus_kind=known.get(focus) if focus else None, known=known)


# ------------------------------------------------------------ the terminal


def terminal_items(mind: tc.Store) -> list[str]:
    rows = [(mind.get(r.claim.subject).box[1], r.claim.object) for r in mind.claims(predicate="reads") if r.claim.subject.id.startswith("text:Terminal#")]
    return [text for _, text in sorted(rows, key=lambda row: row[0]) if not TERMINAL_HEADERS.match(text)]


def screen_state(items: list[str]) -> tuple[str, str | None, str]:
    """('prompt_only' | 'done' | 'busy', cwd from the last prompt, output text)."""
    if not items:
        return "busy", None, ""
    last = PROMPT.match(items[-1])
    cwd = last[2] if last else None
    if last and not last[3].strip() and len(items) == 1:
        return "prompt_only", cwd, ""
    if last and not last[3].strip() and not any(i.strip() == "…" for i in items):
        output = "\n".join(i for i in items[:-1] if not PROMPT.match(i))
        return "done", cwd, output
    return "busy", cwd, ""


def _read_terminal(b, mind):
    """While a command is running, notice when its prompt returns and remember what it printed."""
    cmd = b["cmd"]
    if one(mind, cmd, "output") is not None:
        return
    state, cwd, output = screen_state(terminal_items(mind))
    if state == "done":
        claims = [(tc.Claim(cmd, "output", output), None)]
        if cwd:
            claims.append((tc.Claim(cmd, "cwd", P.HOME + cwd[1:] if cwd.startswith("~") else cwd), None))
        yield knowledge(claims, "obs:terminal", "read-terminal")


def _learn_from_stat(b, mind):
    """Outputs of `stat -c '%F|%s|%n'` are facts about the file system."""
    command = one(mind, b["cmd"], "command") or ""
    if not command.startswith("stat -c"):
        return
    seen = {m[3]: m[1] for line in b["out"].splitlines() if (m := re.match(r"^([a-z ]+)\|(\d+)\|(.+)$", line))}
    for path, kind in seen.items():
        yield tc.Claim(tc.Ref(f"path:{path}"), "is_a", kind)
    for m in re.finditer(r"cannot statx? '([^']+)': No such file", b["out"]):
        yield tc.Claim(tc.Ref(f"path:{m[1]}"), "missing", True)


RULES = [
    Rule("read_terminal_while_running", ((V("cmd"), "phase", "sent"),), _read_terminal, reacts_to="any_change"),
    Rule("learn_file_system_facts", ((V("cmd"), "output", V("out")),), _learn_from_stat),
]


# ------------------------------------------------------------- intentions


@dataclass(frozen=True)
class Start:
    request: tc.Ref
    why: str
    priority: float = 0.0


@dataclass(frozen=True)
class Advance:
    request: tc.Ref
    value: object
    why: str
    priority: float = 0.0

    def __hash__(self) -> int:
        return hash((self.request, self.why))


@dataclass(frozen=True)
class Drop:
    request: tc.Ref
    why: str
    priority: float = 0.0


@dataclass(frozen=True)
class Suspend:
    """Set a request aside without losing it: its frame, position and bindings stay put."""

    request: tc.Ref
    why: str
    priority: float = 0.0


@dataclass(frozen=True)
class Resume:
    """Pick a suspended request back up where it stopped, not at the beginning."""

    request: tc.Ref
    why: str
    priority: float = 0.0
    asked_by: tc.Ref | None = None  # an utterance whose whole content was "pick that back up"


@dataclass
class Body:
    """The few things that are not beliefs: the chat channel, model calls in flight, and budgets."""

    say: Callable[[str], None] = lambda text: None
    seq: itertools.count = field(default_factory=itertools.count)
    turn: int = 0
    jobs: dict[str, dict] = field(default_factory=dict)  # teacher calls in flight (threads), so the screen keeps streaming
    model_calls: int = 0
    teacher_url: str = "http://127.0.0.1:8790/"
    watcher: "Watcher" = field(default_factory=Watcher)  # snapshots of the screen, so "what changed" is answerable
    objects: "Objects" = field(default_factory=Objects)  # things that keep existing when they leave view
    memory: "Memory | None" = None  # episodic/semantic dynamics; set by new_mind so a fresh mind gets fresh memory
    last_memory: object = None  # what the last turn's dynamics did (episode, consolidation, forgetting)
    run_timeout_s: float = RUN_TIMEOUT_S  # how long to wait for one command before giving up on it

    def start_teacher(self, key: str, ask: "P.Teach") -> None:
        import json
        import os
        import threading
        import urllib.request

        job: dict = {"done": False, "asked_at": time.monotonic()}
        self.jobs[key] = job
        self.model_calls += 1

        def call() -> None:
            try:
                body = json.dumps({"system": ask.system, "user": ask.user, "max_new_tokens": ask.max_new_tokens}).encode()
                out = json.loads(urllib.request.urlopen(urllib.request.Request(self.teacher_url, data=body, method="POST"), timeout=600).read())
                job.update(text=out["text"], seconds=out.get("seconds", 0), model=out.get("model"))
                log = os.environ.get("TENSORCODE_TEACHER_LOG", os.path.expanduser("~/.local/share/tensacode/teacher-log.jsonl"))
                os.makedirs(os.path.dirname(log), exist_ok=True)
                with open(log, "a") as fh:  # every teacher exchange is kept: it is the provenance of what gets learned
                    fh.write(json.dumps({"at": time.time(), "model": out.get("model"), "system": ask.system, "user": ask.user, "reply": out["text"],
                                         "seconds": out.get("seconds"), "prompt_tokens": out.get("prompt_tokens"), "new_tokens": out.get("new_tokens"), "batch": out.get("batch")}) + "\n")
            except Exception as exc:  # noqa: BLE001 - the program hears None and says the teacher is unavailable
                job.update(text=None, error=f"{type(exc).__name__}: {exc}")
            job["done"] = True

        threading.Thread(target=call, daemon=True).start()


@dataclass
class Control:
    """How this assistant weighs its goals. One object so a test can change the stance."""

    stance: C.Stance = field(default_factory=lambda: C.Stance(stickiness=0.35, aging=0.25, cost_weight=0.05))

    def arbitrate(self, mind: tc.Store, *, among, now: float, current=None) -> C.Choice:
        return C.arbitrate(mind, stance=self.stance, among=list(among), now=now, current=current, source="control")


CONTROL = Control()
BODY = Body()


def open_requests(mind: tc.Store) -> list[tc.Ref]:
    live = [r.claim.subject for r in mind.claims(predicate="status") if r.claim.object in ("new", "running", "awaiting", "suspended")]
    return sorted(live, key=lambda r: one(mind, r, "order"))


#: "carry on" is control, not vocabulary: it names no act, it says which goal to go back to
_CARRY_ON = re.compile(r"^\s*(?:carry on|carry on then|continue|go on|go ahead|keep going|resume|back to (?:it|that|what)\b.*|where were we|as you were|finish (?:it|that)|and (?:then )?\?*)\s*[.!?]*\s*$", re.I)


def suspended_requests(mind: tc.Store) -> list[tc.Ref]:
    return [r for r in open_requests(mind) if one(mind, r, "status") == "suspended"]


def cost_to_go(mind: tc.Store, req: tc.Ref) -> float:
    """How much is left to do, from the procedure itself: steps not yet reached."""
    frame = one(mind, req, "frame")
    proc = find_procedure(one(mind, frame, "procedure")) if frame is not None else procedure_for(frame_of(mind, req))
    if proc is None:
        return 0.0
    pc = int(one(mind, frame, "pc") or 0) if frame is not None else 0
    return float(max(0, len(proc.steps) - pc))


def intentions(mind: tc.Store) -> list[object]:
    reqs = open_requests(mind)
    awaiting = one(mind, ME, "awaiting")
    fresh = [r for r in reqs if one(mind, r, "status") == "new"]
    running = [r for r in reqs if one(mind, r, "status") == "running"]
    asleep = [r for r in reqs if one(mind, r, "status") == "suspended"]
    if running:
        return body_intentions(mind, running[0])
    if awaiting is not None:
        # when the question was last put to the user: a question re-asked on resumption is
        # not "already answered by silence", so a later utterance means after *that* asking
        asked_in = max(one(mind, awaiting, "order")[0], int(one(mind, awaiting, "re_asked_at") or 0))
        answers = [r for r in fresh if one(mind, r, "answers") == awaiting]
        if answers:
            return [Advance(awaiting, frame_of(mind, answers[0]), f"take “{one(mind, answers[0], 'words')}” as the answer", priority=80)]
        later = [r for r in fresh if one(mind, r, "order")[0] > asked_in]
        if later and all(_CARRY_ON.match(one(mind, r, "words") or "") for r in later):
            # "carry on" while a question is open asks for the question, not for a new goal
            return [Resume(awaiting, "you asked me to carry on, and this is still open", priority=88, asked_by=later[0])]
        if later:
            # the user said something else while a question was open: set the question aside,
            # do not destroy it — everything it had done is still in its frame
            return [Suspend(awaiting, "you said something else first, so I'll come back to this", priority=85)]
        return [Finish("waiting for your answer", priority=100)]
    if fresh:
        # "carry on" chooses a goal rather than describing one: answer it from the task set
        if asleep and _CARRY_ON.match(one(mind, fresh[0], "words") or ""):
            back = _pick_suspended(mind, asleep)
            return [Resume(back, f"carry on with “{one(mind, back, 'words')[:50]}”", priority=90, asked_by=fresh[0])]
        current = _pick_fresh(mind, fresh, asleep)
        if one(mind, current, "status") == "suspended":
            return [Resume(current, f"back to “{one(mind, current, 'words')[:50]}”", priority=75)]
        return [Start(current, f"start: {one(mind, current, 'act')} “{one(mind, current, 'words')[:60]}”", priority=70)]
    if asleep:
        back = _pick_suspended(mind, asleep)
        return [Resume(back, f"back to “{one(mind, back, 'words')[:50]}”", priority=70)]
    return [Finish("all requests answered", priority=100)]


def _pick_fresh(mind: tc.Store, fresh: list[tc.Ref], asleep: list[tc.Ref]) -> tc.Ref:
    """Which goal to pursue now.

    Requests from one utterance keep the order they were said in, because nothing in the graph
    says whether the second depends on the first — "make a folder then put a file in it" is two
    requests and one dependency, and reordering them breaks it. Arbitration therefore decides
    *between* utterances (independent asks) and between a fresh ask and a suspended one, where
    stated order carries no such promise.
    """
    first = fresh[0]
    turn = one(mind, first, "order")[0]
    same_turn = [r for r in fresh if one(mind, r, "order")[0] == turn]
    if len(same_turn) > 1:
        return first  # a multi-clause utterance: its stated order is the only dependency we have
    # a request suspended while awaiting an answer is not competing for effort, it is waiting on
    # the user: resuming it can only re-ask, which would leave the fresh ask unserved
    runnable = [r for r in asleep if one(mind, r, "suspended_from") != "awaiting"]
    candidates = [first, *runnable]
    if len(candidates) == 1:
        return first
    chosen = CONTROL.arbitrate(mind, among=candidates, now=float(BODY.turn), current=None)
    return chosen.goal or first


def _pick_suspended(mind: tc.Store, asleep: list[tc.Ref]) -> tc.Ref:
    chosen = CONTROL.arbitrate(mind, among=asleep, now=float(BODY.turn), current=None)
    return chosen.goal or asleep[-1]


def body_intentions(mind: tc.Store, req: tc.Ref) -> list[object]:
    """The running program is waiting on the body: a terminal command or a dock click."""
    act = one(mind, req, "doing")
    kind = one(mind, act, "kind")
    if kind == "open_app":
        return app_intentions(mind, req, act)
    if kind in ("click", "fill", "look", "teach", "sample"):
        return gui_intentions(mind, req, act, kind)
    shell = controls(mind, role="textbox", label="Shell input")
    visible = shell and mind.get(shell[0]).point is not None
    if not visible:
        dock = [c for c in controls(mind, role="button", label="Terminal") if mind.get(c).point is not None]
        if one(mind, act, "raised_terminal") and time.monotonic() - one(mind, act, "raised_terminal") < 2:
            return [Wait(60, "waiting for Terminal to open", priority=2)]
        return [Press(dock[0], "open Terminal from the dock", priority=60, records=(tc.Claim(act, "raised_terminal", time.monotonic()),))] if dock else [Wait(60, "waiting for the desktop", priority=1)]
    phase = one(mind, act, "phase")
    state, _, _ = screen_state(terminal_items(mind))
    command = one(mind, act, "command")
    if (failed := one(mind, act, "typing_failed")) is not None:
        return [Advance(req, P.Output(f"(could not type into the terminal: {failed})", timed_out=True), f"typing failed: {failed}", priority=45)]
    if not command:  # a step asked for a command that rendered to nothing: a bug in the procedure, not something to type
        return [Advance(req, P.Output("(the step asked for an empty command, so nothing ran)", timed_out=True), "empty command from a procedure step", priority=45)]
    if phase == "want":
        if state == "prompt_only":
            return [Type(req, act, shell[0], command, "sent", f"$ {command[:90]}", priority=50)]
        return [Type(req, act, shell[0], "clear", "cleared", "clear the terminal", priority=50)]
    if phase == "cleared":
        if state == "prompt_only":
            return [Type(req, act, shell[0], command, "sent", f"$ {command[:90]}", priority=50)]
        return [Wait(25, "waiting for the screen to clear", priority=2)]
    if phase == "sent":
        output = one(mind, act, "output")
        if output is not None:
            return [Advance(req, P.Output(output, one(mind, act, "cwd")), f"read the output of `{command[:60]}`", priority=45)]
        sent_at = one(mind, act, "sent_at")
        if time.monotonic() - sent_at > BODY.run_timeout_s:
            return [Advance(req, P.Output("\n".join(terminal_items(mind)[:-1]), timed_out=True), "stop waiting: command still running", priority=44)]
        return [Wait(patience_ms(sent_at), f"waiting for `{command[:50]}`", priority=2)]
    return [Wait(30, "…", priority=1)]


def visible_controls(mind: tc.Store) -> list[tc.Ref]:
    return [c for c in controls(mind) if getattr(mind.get(c), "point", None) is not None]


def best_control(mind: tc.Store, label: str, roles: tuple[str, ...] | None = None) -> tuple[tc.Ref | None, str]:
    """The visible control whose label best matches; refuses ties and weak matches rather than guessing."""
    from ..browser import label_similarity

    scored = sorted(((label_similarity(label, one(mind, c, "label") or ""), c) for c in visible_controls(mind) if roles is None or one(mind, c, "is_a") in roles), key=lambda x: -x[0])
    if not scored or scored[0][0] < 0.6:
        near = ", ".join(repr(one(mind, c, "label")) for _, c in scored[:5] if one(mind, c, "label"))
        return None, f"no visible control labelled {label!r}" + (f" (closest: {near})" if near else "")
    if len(scored) > 1 and scored[1][0] == scored[0][0] and one(mind, scored[1][1], "label") != one(mind, scored[0][1], "label"):
        return None, f"{label!r} is ambiguous: {one(mind, scored[0][1], 'label')!r} or {one(mind, scored[1][1], 'label')!r}"
    return scored[0][1], ""


def seen(mind: tc.Store) -> P.Seen:
    ctrls = tuple((one(mind, c, "is_a"), one(mind, c, "label") or "") for c in visible_controls(mind))
    texts = []
    for r in mind.claims(predicate="reads"):
        t = mind.get(r.claim.subject)
        texts.append((getattr(t, "box", (0, 0))[1], getattr(t, "box", (0, 0))[0], getattr(t, "section", ""), r.claim.object))
    return P.Seen(ctrls, tuple((w, text) for _, _, w, text in sorted(texts)))


def patience_ms(started: float | None) -> int:
    """How long to wait between looks: snappy at first, then coarse, so long waits do not burn cycles."""
    waited = time.monotonic() - (started or time.monotonic())
    return 100 if waited < 2 else 400 if waited < 15 else 1500


def region_box(mind: tc.Store, ui, region: str) -> tuple[tuple[int, int, int, int] | None, str]:
    """Where to look, by name: the whole display, a window by title, or the launcher strip.

    Geometry comes from what was perceived (each control and text carries its box), so a
    region the assistant has never heard of is refused with a reason rather than guessed at.
    """
    size = getattr(ui, "page", None) and ui.page.viewport_size or {"width": 1280, "height": 800}
    whole = (0, 0, int(size["width"]), int(size["height"]))
    want = (region or "screen").strip().lower()
    if want in ("screen", "display", "desktop", "everything", "all", "background", "wallpaper", "whole screen"):
        return whole, ""
    boxes: list[tuple[int, int, int, int]] = []
    for r in mind.claims(predicate="label") + mind.claims(predicate="reads"):
        entity = mind.entities.get(r.claim.subject)
        box = getattr(entity, "box", None)
        section = (getattr(entity, "section", "") or "").lower()
        label = str(r.claim.object or "").lower()
        if box and (want in section or (want and want in label)):
            boxes.append(tuple(int(v) for v in box))
    if want in ("dock", "sidebar", "launcher", "taskbar"):
        boxes = [b for b in (tuple(int(v) for v in getattr(mind.entities.get(r.claim.subject), "box", None) or ())
                             for r in mind.claims(predicate="label")) if b and b[0] < 90 and b[2] < 90]
    if not boxes:
        return None, f"I don't know which part of the screen “{region}” is"
    x = min(b[0] for b in boxes)
    y = min(b[1] for b in boxes)
    w = max(b[0] + b[2] for b in boxes) - x
    h = max(b[1] + b[3] for b in boxes) - y
    return (max(0, x), max(0, y), max(1, min(w, whole[2] - x)), max(1, min(h, whole[3] - y))), ""


def sample_pixels(ui, box: tuple[int, int, int, int]) -> dict:
    """Read the pixels of one region: the colours actually there, biggest share first."""
    try:
        import io

        import numpy as np
        from PIL import Image

        image = Image.open(io.BytesIO(ui.screenshot(box))).convert("RGB")
        pixels = np.asarray(image).reshape(-1, 3)
        buckets = (pixels // 32 * 32 + 16).astype(int)
        colors, counts = np.unique(buckets, axis=0, return_counts=True)
        order = np.argsort(counts)[::-1][:6]
        total = float(counts.sum()) or 1.0
        return {"colors": [[int(colors[i][0]), int(colors[i][1]), int(colors[i][2]), float(counts[i]) / total] for i in order],
                "mean": [int(v) for v in pixels.mean(axis=0)], "box": list(box), "unavailable": None}
    except Exception as exc:  # noqa: BLE001 - no pixels available is an honest answer, not a crash
        return {"colors": [], "mean": None, "box": list(box), "unavailable": f"{type(exc).__name__}: {exc}"}


@dataclass(frozen=True)
class Sample:
    request: tc.Ref
    act: tc.Ref
    region: str
    why: str
    priority: float = 0.0


def decode_sample(i: Sample, mind: tc.Store, ui, cycle: int) -> Thought:
    box, why_not = region_box(mind, ui, i.region)
    facts = {"unavailable": why_not, "colors": [], "mean": None, "box": None} if box is None else sample_pixels(ui, box)
    return I.advance(mind, i.request, facts, cycle, HOST)


def gui_intentions(mind: tc.Store, req: tc.Ref, act: tc.Ref, kind: str) -> list[object]:
    phase = one(mind, act, "phase")
    if kind == "sample":
        region = one(mind, act, "region") or "screen"
        return [Sample(req, act, region, f"look at the pixels of {region}", priority=50)]
    if kind == "teach":
        job = BODY.jobs.get(act.id)
        if job is None:
            return [Advance(req, None, "teacher job missing", priority=45)]
        if job.get("done"):
            BODY.jobs.pop(act.id, None)
            return [Advance(req, job.get("text"), f"teacher answered in {job.get('seconds', 0):.1f} s", priority=45)]
        return [Wait(patience_ms(job.get("asked_at")), "asking the teacher model", priority=2)]
    if kind == "look":
        started = one(mind, act, "started_at")
        if time.monotonic() - started * 1 < one(mind, act, "settle_ms") / 1000:
            return [Wait(60, "letting the screen settle", priority=2)]
        return [Advance(req, seen(mind), "look at the screen", priority=45)]
    label = one(mind, act, "label")
    if (failed := one(mind, act, "typing_failed")) is not None:
        return [Advance(req, f"could not type into {label!r}: {failed}", f"typing failed: {failed}", priority=45)]
    if phase == "acted":
        return [Advance(req, True, f"{kind} {label!r} done", priority=45)]
    roles = ("textbox", "combobox", "searchbox") if kind == "fill" else None
    control, why_not = best_control(mind, label, roles)
    if control is None:
        return [Advance(req, why_not, why_not, priority=45)]
    if kind == "click":
        return [Press(control, f"click {one(mind, control, 'label')!r}", priority=55, records=(tc.Claim(act, "phase", "acted"),))]
    return [Type(req, act, control, one(mind, act, "text"), "acted", f"type into {one(mind, control, 'label')!r}", priority=55, submit=bool(one(mind, act, "submit")))]


def app_intentions(mind: tc.Store, req: tc.Ref, act: tc.Ref) -> list[object]:
    app = one(mind, act, "app")
    clicked = one(mind, act, "clicked_at")
    if clicked is None:
        dock = [c for c in controls(mind, role="button", label=app) if mind.get(c).point is not None]
        if not dock:
            return [Advance(req, False, f"no {app} in the dock", priority=45)]
        before = tuple(sorted({one(mind, r.claim.subject, "in") or "" for r in mind.claims(predicate="is_a")}))
        return [Press(dock[-1], f"click {app} in the dock", priority=55, records=(tc.Claim(act, "clicked_at", time.monotonic()), tc.Claim(act, "sections_before", before)))]
    sections = {one(mind, r.claim.subject, "in") or "" for r in mind.claims(predicate="is_a")} | {mind.get(r.claim.subject).section for r in mind.claims(predicate="reads") if hasattr(mind.get(r.claim.subject), "section")}
    appeared = sections - set(one(mind, act, "sections_before") or ())
    if any(app.lower() in s.lower() for s in sections) or appeared:
        return [Advance(req, True, f"{app} window is showing", priority=45)]
    if time.monotonic() - clicked > 3:
        return [Advance(req, False, f"no {app} window after 3 s", priority=45)]
    return [Wait(60, f"waiting for {app} to open", priority=2)]


@dataclass(frozen=True)
class Type:
    request: tc.Ref
    act: tc.Ref
    control: tc.Ref
    text: str
    phase: str
    why: str
    priority: float = 0.0
    submit: bool = True


# ------------------------------------------------------------------- motor


def teacher_available() -> bool:
    import urllib.request

    try:
        urllib.request.urlopen(BODY.teacher_url, timeout=0.5).read()
        return True
    except Exception:  # noqa: BLE001
        return False


def find_procedure(proc_id: str | None) -> L.Procedure | None:
    """Hand-written procedures and learned skills are the same kind of object, looked up the same way."""
    from . import learning

    if proc_id is None:
        return None
    if proc_id in PR.BY_ID:
        return PR.BY_ID[proc_id]
    if proc_id in learning.LEARNER_PROCS:
        return learning.LEARNER_PROCS[proc_id]
    return learning.procedure_for_skill(proc_id.split("skill:")[-1] if proc_id.startswith("skill:") else proc_id)


def procedure_for(frame: Frame) -> L.Procedure | None:
    from . import learning

    if frame.act == "learned":
        return learning.procedure_for_skill(frame.slots.get("skill"))
    return PR.BY_ACT.get(frame.act)


def _env_for(mind: tc.Store, frame: Frame) -> dict:
    ctx = context_of(mind)
    return {"slot": dict(frame.slots), "words": frame.words, "act": frame.act, "turn": BODY.turn,
            "cwd": ctx.cwd, "focus": ctx.focus, "focus_kind": ctx.focus_kind, "known": dict(ctx.known)}


def _on_finish(mind: tc.Store, req: tc.Ref, env: dict, cycle: int) -> Thought:
    from . import learning

    return learning.record_outcome(mind, req, env, cycle)


def changes_since(since: str) -> dict:
    """What changed on screen since a turn boundary, as a procedure can use it.

    "last_message" means the screen as it was when the previous message arrived — the snapshot
    tagged with the previous turn, not the previous frame. Volatile things (a clock) are counted
    separately rather than dropped, so a reply can say what it ignored.
    """
    tag = f"turn:{max(1, BODY.turn - 1)}" if since in ("last_message", "", None) else str(since)
    changes = BODY.watcher.since_first(tag)
    if changes is None:
        return {"any": False, "n": 0, "lines": [], "volatile_n": 0,
                "unavailable": "I have no snapshot of the screen from before your last message"}
    lines = [c.describe() for c in changes.steady]
    return {"any": bool(lines), "n": len(lines), "lines": lines, "summary": changes.summary(),
             "volatile_n": len(changes.volatile),
             "appeared": [c.label or c.key for c in changes.of("appeared", "window_opened")],
             "disappeared": [c.label or c.key for c in changes.of("disappeared", "window_closed")],
             "since": changes.since.isoformat(timespec="seconds")}


HOST = I.Host(say=lambda mind, req, text, cycle: say(mind, req, text, cycle), find=find_procedure,
              changes_since=changes_since,
              start_teacher=lambda key, ask: BODY.start_teacher(key, P.Teach(**ask)), on_finish=_on_finish)


def decode_start(i: Start, mind: tc.Store, ui, cycle: int) -> Thought:
    frame = frame_of(mind, i.request)
    thought = set_state(mind, i.request, "status", "running", f"decision:{cycle}")
    proc = procedure_for(frame)
    if proc is None and frame.act == "unknown" and teacher_available():
        from . import learning

        proc = learning.learner_procedure(frame)
        thought += I.begin(mind, i.request, proc, {**_env_for(mind, frame), "request": frame.words, "note": ""}, cycle)
        return thought + I.advance(mind, i.request, None, cycle, HOST)
    if proc is None:
        proc = PR.BY_ID["unknown"]
    thought += I.begin(mind, i.request, proc, _env_for(mind, frame), cycle)
    return thought + I.advance(mind, i.request, None, cycle, HOST)


def decode_advance(i: Advance, mind: tc.Store, ui, cycle: int) -> Thought:
    thought = Thought()
    if (doing := one(mind, i.request, "doing")) is not None and one(mind, doing, "phase") == "sent":
        thought += set_state(mind, doing, "phase", "done", f"decision:{cycle}")
    answering = [r for r in open_requests(mind) if one(mind, r, "answers") == i.request and one(mind, r, "status") == "new"]
    for r in answering:
        thought += set_state(mind, r, "status", "done", f"decision:{cycle}")
    if one(mind, i.request, "status") == "awaiting":
        thought += set_state(mind, i.request, "status", "running", f"decision:{cycle}")
        thought += _clear_awaiting(mind, cycle)
    return thought + I.advance(mind, i.request, i.value, cycle, HOST)


def decode_drop(i: Drop, mind: tc.Store, ui, cycle: int) -> Thought:
    asked_in = one(mind, i.request, "order")[0]
    held = [r for r in open_requests(mind) if one(mind, r, "status") == "new" and one(mind, r, "order")[0] == asked_in]
    thought = set_state(mind, i.request, "status", "dropped", f"decision:{cycle}") + _clear_awaiting(mind, cycle)
    for r in held:
        thought += set_state(mind, r, "status", "dropped", f"decision:{cycle}")
    skipped = [one(mind, r, "words") for r in [i.request, *held]]
    thought += say(mind, i.request, "Okay, skipping " + ", ".join(f"“{w}”" for w in skipped) + ".", cycle)
    return thought


def decode_suspend(i: Suspend, mind: tc.Store, ui, cycle: int) -> Thought:
    """Set a request aside. Nothing it has done is undone, and the question it asked is remembered."""
    was = one(mind, i.request, "status")
    asked = one(mind, i.request, "doing")
    thought = set_state(mind, i.request, "suspended_from", was, f"decision:{cycle}")
    if asked is not None and one(mind, asked, "kind") == "ask":
        thought += set_state(mind, i.request, "unanswered_question", one(mind, asked, "question"), f"decision:{cycle}")
    thought += set_state(mind, i.request, "status", "suspended", f"decision:{cycle}")
    thought += _clear_awaiting(mind, cycle)
    C.waiting_since(mind, i.request, float(BODY.turn), i.why, source=f"decision:{cycle}")
    return thought


def decode_resume(i: Resume, mind: tc.Store, ui, cycle: int) -> Thought:
    """Pick a request back up where it stopped: same frame, same position, same bindings.

    Nothing is re-run. If it had asked a question, the question is asked again — the user never
    answered it — but the work that produced it is not repeated.
    """
    was = one(mind, i.request, "suspended_from") or one(mind, i.request, "status") or "running"
    asked = one(mind, i.request, "doing")
    question = one(mind, i.request, "unanswered_question") or (
        one(mind, asked, "question") if asked is not None and one(mind, asked, "kind") == "ask" else None)
    thought = set_state(mind, i.request, "resumed", int(one(mind, i.request, "resumed") or 0) + 1, f"decision:{cycle}")
    if i.asked_by is not None:  # "carry on" asked for nothing else, and it is being done
        thought += set_state(mind, i.asked_by, "status", "done", f"decision:{cycle}")
    if was == "awaiting" and question:
        thought += say(mind, i.request, f"Back to it: {question}", cycle)
        thought += set_state(mind, i.request, "re_asked_at", int(BODY.turn), f"decision:{cycle}")
        thought += set_state(mind, i.request, "status", "awaiting", f"decision:{cycle}")
        return thought + set_state(mind, ME, "awaiting", i.request, f"decision:{cycle}")
    thought += set_state(mind, i.request, "status", "running", f"decision:{cycle}")
    return thought + I.advance(mind, i.request, None, cycle, HOST)


def decode_type(i: Type, mind: tc.Store, ui, cycle: int) -> Thought:
    thought = set_state(mind, i.act, "phase", i.phase, f"intent:{cycle}")
    if i.phase == "sent":
        thought += set_state(mind, i.act, "sent_at", time.monotonic(), f"intent:{cycle}")
    receipt = ui.fill(mind.get(i.control), i.text, submit=i.submit)
    if getattr(receipt, "status", "applied") == "rejected":  # the body could not type (no keyboard focus): the procedure hears about it
        return thought + set_state(mind, i.act, "typing_failed", receipt.error or "typing rejected", f"intent:{cycle}")
    return thought


def _clear_awaiting(mind: tc.Store, cycle: int) -> Thought:
    old = mind.claims(ME, "awaiting")
    if not old:
        return Thought()
    commit = mind.apply(Patch(tuple(Retract(r.id, "answered or dropped") for r in old), mind.revision))
    thought = Thought((), tuple(mind.claim(c) for c in commit.retracted))
    mind.forget(commit.retracted)
    return thought


def say(mind: tc.Store, req: tc.Ref, text: str, cycle: int) -> Thought:
    n = next(BODY.seq)
    BODY.say(text)
    return note(mind, [tc.Claim(ME, "said", (n, text)), tc.Claim(tc.Ref(f"reply:{n}"), "answers", req)], f"decision:{cycle}")


def watch_screen(ui, mind: tc.Store) -> Fragment | None:
    """Perceive-and-remember: snapshot what is on screen, learn what is volatile, keep object files.

    Runs as a perceiver, so it sees exactly what the scene graph just integrated. It adds no claims
    of its own about *now* (the screen scope already has those); it keeps the record of what was,
    which the snapshot scope by design does not.
    """
    records = [r for r in mind.claims() if r.claim.scope == SCREEN]
    if not records:
        return None
    items = items_from_claims(records, lambda ref: mind.entities.get(ref))
    changes = BODY.watcher.see(items, tag=f"turn:{BODY.turn}")
    snap = BODY.watcher.latest()
    if snap is not None:
        # object files need to know which window a thing is in, and the DOM does not attribute
        # chrome to its window; the diff does its own containment, so only this side needs it
        BODY.objects.observe(snapshot(attribute_windows(items), at=snap.at, tag=snap.tag))
        for closing in changes.of(WINDOW_CLOSED):  # seen to close: its contents are gone, not hidden
            BODY.objects.closed(closing.window or closing.label, at=snap.at)
        # existence and last-seen go in their own scope, so they survive the screen scope's retraction
        BODY.objects.remember(mind)
    return None


SPEC = MindSpec(
    "assistant",
    RULES,
    intentions,
    BY_PRIORITY,
    max_cycles=20000,  # cycles are cheap; waits are coarse (patience_ms), so this is a safety net, not a budget
    decoders={Start: decode_start, Advance: decode_advance, Drop: decode_drop, Type: decode_type, Sample: decode_sample,
              Suspend: decode_suspend, Resume: decode_resume},
    repeatable=(Type, Press),
    perceivers=(watch_screen,),
)


def new_mind() -> tc.Store:
    """A fresh mind already holds its prior knowledge (what apps are for), as claims with provenance."""
    from . import world_knowledge

    mind = SPEC.new_memory()
    integrate(mind, knowledge([(c, None) for c in world_knowledge.claims()], "prior:apps", "prior-knowledge"))
    BODY.watcher = Watcher()
    BODY.objects = Objects()
    # told facts and what the agent said are protected from forgetting; stale perception is not
    BODY.memory = Memory(mind, MemoryPolicy(protect_predicates=frozenset({"said", "told", "name", "act", "words", "status"})))
    return mind


def respond(ui, mind: tc.Store, text: str, *, on_say: Callable[[str], None], on_cycle=None) -> object:
    """One conversational turn: hear the message, then act until every request is answered or awaiting you."""
    BODY.turn += 1
    BODY.say = on_say
    hear(mind, text, BODY.turn)
    laid_down: list = []  # what this turn actually established, for the episode it becomes

    def watch_cycle(m: tc.Store, thought, intention) -> None:
        laid_down.extend(thought.added)
        if on_cycle is not None:
            on_cycle(m, thought, intention)

    try:
        outcome = run_mind(ui, SPEC, on_cycle=watch_cycle, mind=mind)
    except Exception as exc:  # noqa: BLE001 - one malformed request must not make the assistant unusable
        stuck = open_requests(mind)
        for r in stuck:
            set_state(mind, r, "status", "failed", "crash")
            set_state(mind, r, "failed_because", f"{type(exc).__name__}: {exc}", "crash")
        _clear_awaiting(mind, 0)
        BODY.say(f"Something in my own handling of that went wrong ({type(exc).__name__}: {exc}). I've dropped that request; the rest of the conversation still works.")
        outcome = Outcome("failed", f"{type(exc).__name__}: {exc}", 0)
    # the terminal's working folder, as last seen in a prompt
    cwds = [r for r in mind.claims(predicate="cwd") if r.claim.subject.id.startswith("command:")]
    if cwds:
        set_state(mind, ME, "cwd", max(cwds, key=lambda r: r.evidence[-1].observed_at).claim.object, "obs:terminal")
    if BODY.memory is not None:
        # the turn becomes an episode. Perception is left out on purpose: a retina's worth of claims
        # per turn would bury the handful that record what happened, and those are what recall needs.
        events = [r for r in laid_down if r.claim.scope not in (SCREEN, OBJECTS)]
        BODY.last_memory = BODY.memory.turn(events, summary=f"turn {BODY.turn}: {text[:80]}",
                                            keep_scopes=(SCREEN, OBJECTS))
    return outcome
