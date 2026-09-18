"""Action priming: procedures warmed by what is aware, firing when activation crosses.

Dispatch asks "which procedure handles this act?" and gets one answer. Priming asks nothing:
every procedure whose cues appear in the aware set gains activation, several are partly
warm at once, and one fires when it crosses its threshold. Lookup falls out as the special
case where a single cue (the parsed act) matches.

The synfire part is the chain. A procedure may declare ordered stages, and a stage only
accumulates once the stage before it has fired within a window of cycles; a gap lets the
chain cool and fall back. So "the terminal is open, then a command was typed, then the
prompt returned" is a sequence the priming can require, rather than three conditions a
guard happens to check together.

Nothing here executes: ``ready()`` reports what is warm enough, and ``why()`` says which
claims warmed it, so a caller keeps the decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .records import ClaimRecord, Ref

_ANY = object()


@dataclass(frozen=True)
class Cue:
    """A claim pattern that warms a procedure, and how much it contributes."""

    predicate: str | None = None
    subject: Ref | None = None
    object: Any = _ANY
    weight: float = 1.0

    def matches(self, rec: ClaimRecord) -> bool:
        c = rec.claim
        return ((self.predicate is None or c.predicate == self.predicate)
                and (self.subject is None or c.subject == self.subject)
                and (self.object is _ANY or c.object == self.object))

    def describe(self) -> str:
        parts = [f"{self.subject}" if self.subject else "?", self.predicate or "?",
                 "?" if self.object is _ANY else repr(self.object)]
        return " ".join(parts)


@dataclass(frozen=True)
class Primeable:
    """Something that can be primed: an id, its cues, and optionally an ordered chain."""

    id: str
    cues: tuple[Cue, ...] = ()
    threshold: float = 0.6
    chain: tuple[tuple[Cue, ...], ...] = ()  # ordered stages; a stage opens only after the one before
    window: int = 3  # cycles a stage stays open before the chain cools
    payload: Any = None  # whatever the caller wants back (a Procedure, a callable, a name)

    @property
    def staged(self) -> bool:
        return bool(self.chain)


def primeable_from(obj: Any, *, threshold: float = 0.6) -> Primeable:
    """Read a primeable off an object that may already describe its own cues.

    Honours ``cues``, ``prime_threshold``, ``chain`` and ``window`` when present. With none
    of them, a procedure that declares ``act`` gets the single cue that reproduces dispatch:
    a claim ``(request, "act", <act>)``. So a body of procedures becomes primeable without
    being rewritten, and declaring cues is how one stops being merely dispatchable.
    """
    ident = str(getattr(obj, "id", obj))
    cues = tuple(getattr(obj, "cues", ()) or ())
    chain = tuple(tuple(stage) for stage in (getattr(obj, "chain", ()) or ()))
    act = getattr(obj, "act", None)
    if not cues and not chain and act:
        cues = (Cue(predicate="act", object=act),)
    return Primeable(ident, cues, float(getattr(obj, "prime_threshold", threshold)), chain,
                     int(getattr(obj, "window", 3)), payload=obj)


@dataclass
class Priming:
    """Activation over a set of primeables, driven by the aware set."""

    primeables: list[Primeable] = field(default_factory=list)
    decay: float = 0.5
    floor: float = 0.02
    activation: dict[str, float] = field(default_factory=dict)
    stage: dict[str, int] = field(default_factory=dict)
    _last_stage_cycle: dict[str, int] = field(default_factory=dict)
    _matched: dict[str, list[tuple[str, str, float]]] = field(default_factory=dict)  # id -> (cue, claim id, weight)
    cycle: int = 0

    def add(self, primeable: Primeable) -> None:
        self.primeables.append(primeable)

    def observe(self, aware: Any, *, saliences: dict[str, float] | None = None) -> None:
        """One cycle: warm what the aware set matches, cool what it does not."""
        records, salience_of = _records_and_salience(aware, saliences)
        self.cycle += 1
        for p in self.primeables:
            before = self.activation.get(p.id, 0.0) * self.decay
            matched: list[tuple[str, str, float]] = []
            if p.staged:
                stage_index = self.stage.get(p.id, 0)
                gained = 0.0
                if stage_index < len(p.chain):
                    for cue in p.chain[stage_index]:
                        for rec in records:
                            if cue.matches(rec):
                                gained += cue.weight * (salience_of(rec.id) or 1e-3)
                                matched.append((cue.describe(), rec.id, cue.weight))
                    if gained > 0:
                        self.stage[p.id] = stage_index + 1
                        self._last_stage_cycle[p.id] = self.cycle
                    elif self.cycle - self._last_stage_cycle.get(p.id, self.cycle) > p.window:
                        self.stage[p.id] = 0  # the chain cooled: back to the beginning
                        before *= 0.0
                self.activation[p.id] = min(1.5, before + gained)
            else:
                gained = 0.0
                for cue in p.cues:
                    for rec in records:
                        if cue.matches(rec):
                            gained += cue.weight * (salience_of(rec.id) or 1e-3)
                            matched.append((cue.describe(), rec.id, cue.weight))
                self.activation[p.id] = min(1.5, before + gained)
            if self.activation[p.id] < self.floor:
                self.activation.pop(p.id, None)
            if matched:
                self._matched[p.id] = matched

    def partial(self) -> dict[str, float]:
        """Everything warm, whether or not it is ready — several procedures are usually partly on."""
        return dict(sorted(self.activation.items(), key=lambda kv: (-kv[1], kv[0])))

    def ready(self) -> list[tuple[Primeable, float]]:
        """What has crossed its threshold (and finished its chain), strongest first."""
        out = []
        for p in self.primeables:
            level = self.activation.get(p.id, 0.0)
            if level < p.threshold:
                continue
            if p.staged and self.stage.get(p.id, 0) < len(p.chain):
                continue
            out.append((p, level))
        return sorted(out, key=lambda pair: (-pair[1], pair[0].id))

    def why(self, primeable_id: str) -> list[str]:
        level = self.activation.get(primeable_id, 0.0)
        p = next((q for q in self.primeables if q.id == primeable_id), None)
        if p is None:
            return [f"{primeable_id} is not primed here"]
        lines = [f"{primeable_id} at {level:.3f} (threshold {p.threshold:.2f})"]
        if p.staged:
            lines.append(f"  chain stage {self.stage.get(primeable_id, 0)} of {len(p.chain)}")
        for cue, claim_id, weight in self._matched.get(primeable_id, []):
            lines.append(f"  {cue}  matched {claim_id} (+{weight:.2f})")
        return lines

    def reset(self, primeable_id: str) -> None:
        """After firing, or after the situation changed: this one starts cold."""
        self.activation.pop(primeable_id, None)
        self.stage.pop(primeable_id, None)
        self._last_stage_cycle.pop(primeable_id, None)
        self._matched.pop(primeable_id, None)


def prime(primeables: Sequence[Any], aware: Any, *, threshold: float = 0.6) -> Priming:
    """Build priming over a set of procedure-like objects and observe one cycle."""
    p = Priming([q if isinstance(q, Primeable) else primeable_from(q, threshold=threshold) for q in primeables])
    p.observe(aware)
    return p


def _records_and_salience(aware: Any, saliences: dict[str, float] | None) -> tuple[list[ClaimRecord], Any]:
    """Read the aware set and its saliences once; a cue match must not recompute them."""
    if hasattr(aware, "aware") and hasattr(aware, "salience"):  # an Awareness
        records = aware.aware()
        table = {rec.id: max(aware.salience(rec.id), 1e-3) for rec in records}
        return records, table.get
    records = list(aware.claims() if hasattr(aware, "claims") else aware)
    table = saliences or {}
    return records, lambda cid: table.get(cid, 1.0)


def cues_for_sequence(*stages: Iterable[Cue]) -> tuple[tuple[Cue, ...], ...]:
    """Spell a chain: ``cues_for_sequence([Cue(...)], [Cue(...)])`` in the order they must occur."""
    return tuple(tuple(stage) for stage in stages)
