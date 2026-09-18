"""A cognitively shaped browser agent, with no model calls.

    perceive   : tc.parse(screen, Fragment)               vision -> scene graph (snapshot scope)
    integrate  : integrate(mind, frame)                   working memory update -> Thought
    think      : think(mind, rules, since=thought)        bottom-up spontaneous thoughts
    deliberate : tc.choose(intentions, objective, ...)    what to do next, under constraints
    decode     : intention -> mouse / keyboard            motor
    act        : Browser                                  real input events

A task is a *mind*: rules, intention generators, an objective, and constraints.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import tensacode as tc
from tensacode.backends.builtin import IN_PROCESS
from tensacode.cognition import Corroboration, Fragment, Rule, Thought, ThinkStats, integrate, think

from .browser import Browser, Control, Screen

SCREEN = tc.Ref("scope:screen")


# ------------------------------------------------------------------ program


@dataclass(frozen=True)
class Outcome:
    status: str  # "done" | "escalated"
    reason: str
    cycles: int
    think_ms: float = 0.0
    beliefs: int = 0


def run_mind(ui: Browser, spec: MindSpec, on_cycle: Callable[[tc.Store, Thought, object], None] | None = None, *, mind: tc.Store | None = None) -> Outcome:
    """Run until the mind finishes or escalates. Pass ``mind`` to keep memory across runs (e.g. a conversation)."""
    mind = mind if mind is not None else spec.new_memory()
    stats, last, pending = ThinkStats(), [], Thought()
    policy = spec.corroboration() if spec.corroboration else None       # glances stay tentative until corroborated
    for cycle in range(spec.max_cycles):
        thought = pending
        if not last or not isinstance(last[-1], Note):  # a purely mental act changes nothing on screen
            frame = tc.parse(ui.observe(), Fragment, frame=cycle)          # perceive
            thought += integrate(mind, frame, corroboration=policy)         # integrate awareness
            for look in spec.perceivers:                                    # closer looks (e.g. pixels)
                if (extra := look(ui, mind)) is not None:
                    thought += integrate(mind, extra, corroboration=policy)
        thought += think(mind, spec.rules, since=thought, stats=stats, corroboration=policy)  # spontaneous thoughts
        beliefs = policy.view(mind) if policy else mind                    # deliberate on established beliefs only
        intention = tc.choose(spec.intentions(beliefs), objective=spec.priority, given=mind, constraints=spec.constraints)
        (on_cycle or spec.on_cycle)(mind, thought, intention)
        if isinstance(intention, (tc.Unknown, Escalate)):
            return Outcome("escalated", intention.reason, cycle, stats.ms, len(mind._claims))
        if isinstance(intention, Finish):
            return Outcome("done", intention.why, cycle, stats.ms, len(mind._claims))
        last = (last + [intention])[-8:]
        if not isinstance(intention, (Wait, *spec.repeatable)) and last.count(intention) >= 4:  # repeats and oscillations
            return Outcome("escalated", f"no progress: {intention.why!r} chosen {last.count(intention)} times in 8 cycles", cycle, stats.ms, len(mind._claims))
        ui.caption(intention.why)
        decode = spec.decoders.get(type(intention), decode_and_act)
        try:
            pending = decode(intention, mind, ui, cycle)                   # decode + act
        except MotorFailure as failure:                                    # the body could not do it: say so, don't hang
            return Outcome("escalated", f"could not act on {intention.why!r}: {failure}", cycle, stats.ms, len(mind._claims))
    return Outcome("escalated", "cycle limit", spec.max_cycles, stats.ms, len(mind._claims))


# --------------------------------------------------------------- intentions


@dataclass(frozen=True)
class Press:
    control: tc.Ref
    why: str
    records: tuple[tc.Claim, ...] = ()  # written to memory *before* the click (durable intent before effect)
    priority: float = 0.0


@dataclass(frozen=True)
class Enter:
    control: tc.Ref
    text: str
    why: str
    priority: float = 0.0
    submit: bool = False  # press Enter afterwards
    records: tuple[tc.Claim, ...] = ()  # written to memory before typing


@dataclass(frozen=True)
class Note:
    """A cognitive act: adopt claims (a decision or a conclusion), citing what they rest on."""

    claims: tuple[tc.Claim, ...]
    premises: tuple[str, ...]
    why: str
    priority: float = 0.0


@dataclass(frozen=True)
class Wait:
    ms: int
    why: str
    priority: float = 0.0


@dataclass(frozen=True)
class Finish:
    why: str
    priority: float = 0.0


@dataclass(frozen=True)
class Escalate:
    reason: str
    why: str = ""
    priority: float = 0.0


class MotorFailure(RuntimeError):
    """An action the body reports it could not perform (e.g. typing without keyboard focus)."""


def decode_and_act(intention: object, mind: tc.Store, ui: Browser, cycle: int) -> Thought:
    if isinstance(intention, Note):
        now = datetime.now(timezone.utc)
        edits = tuple(tc.Tell(c, (tc.Evidence(tc.Ref(f"decision:{cycle}"), now, method="deliberate", derived_from=intention.premises),)) for c in intention.claims)
        commit = mind.apply(tc.Patch(edits, mind.revision))
        return Thought(tuple(mind.claim(i) for i in commit.added))
    if isinstance(intention, Press):
        if intention.records:
            integrate(mind, Fragment(tc.Ref(f"intent:{cycle}"), tuple((c, None) for c in intention.records), method="intend"))
        ui.click(mind.get(intention.control))
    elif isinstance(intention, Enter):
        if intention.records:
            integrate(mind, Fragment(tc.Ref(f"intent:{cycle}"), tuple((c, None) for c in intention.records), method="intend"))
        receipt = ui.fill(mind.get(intention.control), intention.text, submit=intention.submit)
        if receipt.status == "rejected":
            raise MotorFailure(receipt.error or "typing rejected")
    elif isinstance(intention, Wait):
        ui.page.wait_for_timeout(intention.ms)
    return Thought()


# ------------------------------------------------------------------- vision


@tc.implementation("parse", name="dom-scene-graph", version="1", accepts=lambda r: isinstance(r.subject, Screen) and r.target is Fragment, profile=IN_PROCESS)
def scene_graph(request: tc.Request) -> Fragment:
    """The page's accessible structure as claims in the screen scope (plus geometry as entity payloads)."""
    screen: Screen = request.subject
    claims: list[tuple[tc.Claim, str | None]] = []
    entities: list[tuple[tc.Ref, object]] = []
    seen: Counter[str] = Counter()

    def say(subject: tc.Ref, predicate: str, obj: Any, where: str | None = None) -> None:
        claims.append((tc.Claim(subject, predicate, obj, scope=SCREEN), where))

    for c in screen.controls:
        key = f"{c.section}/{c.role}/{c.name}"
        seen[key] += 1
        ref = tc.Ref(f"ui:{key}#{seen[key]}")
        entities.append((ref, c))
        where = f"box{c.box}"
        say(ref, "is_a", c.role, where)
        say(ref, "in", c.section, where)
        if c.name:
            say(ref, "label", c.name, where)
        if c.hint:
            say(ref, "hint", c.hint, where)
        if c.role == "textbox":
            say(ref, "value", c.value, where)
        if c.checked is not None:
            say(ref, "checked", c.checked, where)
        if c.current:
            say(ref, "current", True, where)
        if c.shown:
            say(ref, "shows", c.shown, where)
        if c.input_type in ("email", "number", "date"):
            say(ref, "input_type", c.input_type, where)
    sections: Counter[str] = Counter()
    for t in screen.texts:
        if t.seq:
            say(tc.Ref(f"announcement:{t.seq}"), "announces", t.text, f"box{t.box}")
        else:
            sections[t.section] += 1
            ref = tc.Ref(f"text:{t.section}#{sections[t.section]}")
            entities.append((ref, t))
            say(ref, "reads", t.text, f"box{t.box}")
    for table in screen.tables:
        for i, row in enumerate(table.rows):
            ref = tc.Ref(f"row:{table.label}#{i}")
            say(ref, "in_table", table.label)
            say(ref, "cells", tuple(zip(table.header, row)))
    for g in screen.graphics:
        ref = tc.Ref(f"graphic:{g.label or 'unlabeled'}")
        entities.append((ref, g))
        say(ref, "is_a", "graphic", f"box{g.box}")
        say(ref, "at", g.box)
    if screen.dialog:
        say(tc.Ref("ui:dialog"), "shows", screen.dialog)
    return Fragment(tc.Ref(f"obs:frame-{request.params.get('frame', 0)}"), tuple(claims), tuple(entities), snapshot_of=SCREEN, method="dom-scene-graph@1")


# ------------------------------------------------------------------ mind API


@dataclass
class MindSpec:
    name: str
    rules: list[Rule]
    intentions: Callable[[tc.Store], list[object]]
    priority: tc.Objective
    constraints: tuple[tc.Constraint, ...] = ()
    functional: tuple[str, ...] = ()
    max_cycles: int = 400
    on_cycle: Callable[[tc.Store, Thought, object], None] = lambda mind, thought, intention: None
    perceivers: tuple[Callable[[Browser, tc.Store], Fragment | None], ...] = ()
    decoders: dict[type, Callable[[object, tc.Store, Browser, int], Thought]] = field(default_factory=dict)  # motor programs for task-specific intentions
    repeatable: tuple[type, ...] = ()  # intention types that may legitimately recur (not counted as no-progress loops)
    corroboration: Callable[[], Corroboration] | None = None  # fresh policy per run; intentions then see established beliefs only

    def new_memory(self) -> tc.Store:
        mind = tc.Store()
        for p in self.functional:
            mind.declare(p, functional=True)
        return mind


# ------------------------------------------------------- helpers for minds


def objects(mind: tc.Store, subject: tc.Ref, predicate: str) -> list[Any]:
    return [r.claim.object for r in mind.claims(subject, predicate)]


def one(mind: tc.Store, subject: tc.Ref, predicate: str, default: Any = None) -> Any:
    found = objects(mind, subject, predicate)
    return found[0] if found else default


def subjects(mind: tc.Store, predicate: str, obj: Any) -> list[tc.Ref]:
    return [r.claim.subject for r in mind.claims(predicate=predicate, object=obj)]


def controls(mind: tc.Store, *, role: str | None = None, section: str | None = None, label: str | None = None) -> list[tc.Ref]:
    refs = subjects(mind, "is_a", role) if role else [r.claim.subject for r in mind.claims(predicate="is_a")]
    return sorted(
        (r for r in refs if (section is None or one(mind, r, "in") == section) and (label is None or one(mind, r, "label") == label)),
        key=lambda r: (mind.get(r).box[1], mind.get(r).box[0]),
    )


def knowledge(claims: list[tuple[tc.Claim, str | None]], source: str, method: str) -> Fragment:
    """What a reading act adds to long-term knowledge (not tied to the screen snapshot)."""
    return Fragment(tc.Ref(source), tuple(claims), method=method)


def order(ref: tc.Ref, mind: tc.Store) -> float:
    """Small tie-breaker: top-to-bottom, left-to-right."""
    x, y, *_ = mind.get(ref).box
    return -(y * 10_000 + x) / 1e9




BY_PRIORITY = tc.Objective("priority", "Highest-priority applicable intention (ties broken by screen order upstream)", lambda i, mind: i.priority)


def describe(thought: Thought, *, limit: int = 8) -> list[str]:
    """Human-readable lines for what the mind newly believes (perceptual claims omitted)."""
    lines = []
    for rec in thought.added:
        c = rec.claim
        if c.scope == SCREEN:
            continue
        obj = c.object.id if isinstance(c.object, tc.Ref) else getattr(c.object, "value", c.object)
        text = f"{c.subject.id} {c.predicate} {obj}"
        ev = rec.evidence[0] if rec.evidence else None
        if ev and ev.confidence:
            text += f" ({ev.confidence.kind} {ev.confidence.value:.2f})"
        if ev and ev.derived_from:
            text += f"  ← {ev.source.id}"
        lines.append(text if len(text) < 140 else text[:137] + "…")
    return lines[:limit]
